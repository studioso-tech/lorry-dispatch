"""事象「見込みより売れた」。当日発注の引き金（連休・値上げ前・寒波）。

当日発注が来るのは、そのSSが計画より売れたから（実務）。生成器は見込みどおりに
売れた世界しか作らず、計画配送がタンクを満たすので、当日発注を入れる空きが
残らなかった（2026-09-12 のデモの下見）。
"""

from __future__ import annotations

from functools import lru_cache

import pytest
from fastapi.testclient import TestClient

from app.main import app
from core.domain import Product
from core.events import MAX_EXTRA_DAYS, SoldMore, apply_sold_more, parse_sold_more, sell_more
from core.generator import generate
from core.replan import NewOrder, replan
from core.solver import SolveMode, solve
from core.travel import build_matrix

DATE = "2026-10-16"
NOW = 9 * 60 + 12        # FAX の受信時刻
FAX = [NewOrder("c02", Product.REGULAR, 4_000, "fax", "FAX:test"),
       NewOrder("c02", Product.KEROSENE, 2_000, "fax", "FAX:test")]


def _customer(s, cid):
    return next(c for c in s.customers if c.id == cid)


@lru_cache(maxsize=None)
def _morning():
    """朝の計画は見込みのまま組む（事象はまだ誰も知らない）。"""
    s = generate(seed=20261015, date=DATE, season="winter")
    return s, solve(s, matrix=build_matrix(s.depot, s.customers), mode=SolveMode.REPRODUCIBLE)


# --- 在庫の下げ方 ------------------------------------------------------------


def test_sell_more_lowers_each_tank_by_its_daily_use_times_days():
    c = _customer(generate(seed=20261015, date=DATE, season="winter"), "c02")
    out = sell_more(c.tanks, [SoldMore("c02", 2.0)])
    for before, after in zip(c.tanks, out):
        assert after.level_l == max(0, before.level_l - round(before.daily_use_l * 2.0))


def test_product_specific_event_touches_only_that_tank():
    """寒波の灯油のように油種が決まっているときは、ほかのタンクは触らない。"""
    c = _customer(generate(seed=20261015, date=DATE, season="winter"), "c02")
    out = sell_more(c.tanks, [SoldMore("c02", 1.5, Product.KEROSENE)])
    for before, after in zip(c.tanks, out):
        if before.product is Product.KEROSENE:
            assert after.level_l < before.level_l
        else:
            assert after.level_l == before.level_l


def test_level_never_goes_below_zero():
    c = _customer(generate(seed=20261015, date=DATE, season="winter"), "c02")
    out = sell_more(c.tanks, [SoldMore("c02", MAX_EXTRA_DAYS)] * 3)
    assert all(t.level_l >= 0 for t in out)


# --- 生成器（朝の計画より前に分かっていた） ----------------------------------


def test_generator_event_leaves_every_other_customer_identical():
    """乱数を引いたあとに当てるので、事象の無い先は事象なしの日と完全に同じ。"""
    plain = generate(seed=20261015, date=DATE, season="winter")
    event = generate(seed=20261015, date=DATE, season="winter", sold_more=(SoldMore("c02", 2.0),))
    for a, b in zip(plain.customers, event.customers):
        if a.id == "c02":
            assert all(tb.level_l < ta.level_l for ta, tb in zip(a.tanks, b.tanks) if ta.daily_use_l)
        else:
            assert a == b


def test_selling_far_more_than_planned_turns_into_an_order_or_a_stockout():
    """計画配送の日でない先が発注点を割れば、発注が入る（実際の当日発注の形）。"""
    plain = _customer(generate(seed=20261015, date=DATE, season="winter"), "c02")
    assert plain.mandatory_basis is None          # 今日は計画配送の日ではない
    hot = _customer(generate(seed=20261015, date=DATE, season="winter",
                             sold_more=(SoldMore("c02", MAX_EXTRA_DAYS),)), "c02")
    assert hot.mandatory_basis in ("order", "stockout")


@pytest.mark.parametrize("event, match", [
    (SoldMore("c99", 1.0), "名簿に無い"),
    (SoldMore("c02", 0.0), "日以下"),
    (SoldMore("c02", MAX_EXTRA_DAYS + 1), "日以下"),
])
def test_bad_events_are_rejected_by_the_generator(event, match):
    with pytest.raises(ValueError, match=match):
        generate(seed=20261015, date=DATE, season="winter", sold_more=(event,))


def test_event_for_a_product_without_a_tank_is_rejected():
    s = generate(seed=20261015, date=DATE, season="winter")
    depot = next(c for c in s.customers if not any(t.product is Product.PREMIUM for t in c.tanks))
    with pytest.raises(ValueError, match="タンクは無い"):
        apply_sold_more(s.customers, (SoldMore(depot.id, 1.0, Product.PREMIUM),))


def test_cli_notation():
    assert parse_sold_more("c02:2") == SoldMore("c02", 2.0)
    assert parse_sold_more("c02:kerosene:1.5") == SoldMore("c02", 1.5, Product.KEROSENE)
    for bad in ("c02", "c02:two", "c02:diesel:1", "a:b:c:d"):
        with pytest.raises(ValueError, match="書き方"):
            parse_sold_more(bad)


# --- 組み直し（日中に分かった）＝デモの幕2〜3 --------------------------------


def test_without_the_event_the_fax_order_has_nowhere_to_go():
    """見込みのままでは空きが無く、4kLの発注に車を出せない（下見で見つけた姿）。"""
    s, base = _morning()
    r = replan(s, base, now_min=NOW, orders=FAX, time_limit_s=30)
    assert len(r.shortfalls) == 2
    assert not any("宇品港" in c for c in r.changes)


def test_learning_that_the_station_sold_more_lets_the_order_land():
    """FAXの但し書き（連休で来客が多い）を事象として当てると、発注どおり入る。"""
    s, base = _morning()
    r = replan(s, base, now_min=NOW, orders=FAX, time_limit_s=30,
               sold_more=(SoldMore("c02", 2.0),))
    assert r.ok and not r.shortfalls
    assert sum("宇品港" in c for c in r.changes) == 2


def test_a_holiday_rush_on_gasoline_does_not_make_room_for_kerosene():
    """連休で出るのはガソリン。灯油の空きは増えないので、灯油の発注はSSに確認を返す。"""
    s, base = _morning()
    r = replan(s, base, now_min=NOW, orders=FAX, time_limit_s=30,
               sold_more=(SoldMore("c02", 2.0, Product.REGULAR),))
    assert any("レギュラー 4.0kL" in c for c in r.changes)
    assert len(r.shortfalls) == 1 and "灯油" in r.shortfalls[0]


def test_the_event_does_not_change_what_the_morning_trips_carried():
    """積込済みの便の先に事象を当てても、朝の車が積んだ量の差し引きは朝の見込みで行う。"""
    s, base = _morning()
    frozen = next(st.customer_id for vd in base.vehicle_days for t in vd.trips
                  if t.load_start_min < NOW for st in t.stops)
    plain = replan(s, base, now_min=NOW, orders=FAX, time_limit_s=30)
    hot = replan(s, base, now_min=NOW, orders=FAX, time_limit_s=30,
                 sold_more=(SoldMore(frozen, 1.0),))
    assert [t for t in plain.frozen_trips] == [t for t in hot.frozen_trips]


# --- API ----------------------------------------------------------------------


def test_replan_api_takes_the_event_learned_during_the_day():
    client = TestClient(app)
    r = client.post("/api/replan", json={
        "seed": 20261015, "date": DATE, "now": "09:12",
        "orders": [{"customer_id": "c02", "product": "regular", "volume_l": 4000,
                    "source": "fax", "evidence_ref": "FAX:test"}],
        "sold_more_now": [{"customer_id": "c02", "extra_days": 2, "product": "regular"}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sold_more_now"] and "レギュラー" in body["sold_more_now"][0]
    assert any("宇品港" in c for c in body["changes"])
    assert not body["shortfalls"]


def test_replan_api_rejects_an_event_beyond_a_week():
    client = TestClient(app)
    r = client.post("/api/replan", json={
        "seed": 20261015, "date": DATE, "now": "09:12",
        "orders": [{"customer_id": "c02", "product": "regular", "volume_l": 4000,
                    "source": "fax", "evidence_ref": "FAX:test"}],
        "sold_more_now": [{"customer_id": "c02", "extra_days": 30}],
    })
    assert r.status_code == 422
