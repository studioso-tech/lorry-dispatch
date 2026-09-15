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


def _target():
    """朝の計画に無い市内SS（当日発注の相手）。前提はデータ任せにせず、ここで組み立てる。"""
    s, base = _morning()
    served = {st.customer_id for t in base.trips for st in t.stops}
    c = next(_customer(s, cid) for cid in ("c02", "c01", "c03", "c05", "c04")
             if cid not in served and _customer(s, cid).open_today)
    return s, base, c


def _over_room(c, product):
    """空きより4kL槽1本ぶん多い発注（8kL以上＝小口ではない）と、それが入るのに要る「売れた日数」。"""
    from core.domain import whole_drops_l

    t = next(t for t in c.tanks if t.product is product)
    vol = max(8_000, whole_drops_l(product, t.topup_l) + 4_000)
    days = round((vol - t.topup_l) / t.daily_use_l + 0.5, 1)
    return vol, days


def _order(c, product, vol):
    return NewOrder(c.id, product, vol, "fax", "FAX:test")


def test_without_the_event_the_order_has_nowhere_to_go():
    """見込みのままでは空きが足りず、発注どおりには入らない（下見で見つけた姿）。"""
    s, base, c = _target()
    vol, _days = _over_room(c, Product.REGULAR)
    r = replan(s, base, now_min=NOW, orders=[_order(c, Product.REGULAR, vol)], time_limit_s=30)
    assert len(r.shortfalls) == 1 and "空き" in r.shortfalls[0]


def test_learning_that_the_station_sold_more_lets_the_order_land():
    """FAXの但し書き（連休で来客が多い）を事象として当てると、発注どおり入る。"""
    s, base, c = _target()
    vol, days = _over_room(c, Product.REGULAR)
    r = replan(s, base, now_min=NOW, orders=[_order(c, Product.REGULAR, vol)], time_limit_s=30,
               sold_more=(SoldMore(c.id, days, Product.REGULAR),))
    assert r.ok and not r.shortfalls
    assert any(c.name in x and "着" in x for x in r.changes)


def test_a_holiday_rush_on_gasoline_does_not_make_room_for_kerosene():
    """連休で出るのはガソリン。灯油の空きは増えないので、灯油の発注はSSに確認を返す。"""
    s, base, c = _target()
    vol_r, days = _over_room(c, Product.REGULAR)
    vol_k, _ = _over_room(c, Product.KEROSENE)
    r = replan(s, base, now_min=NOW, time_limit_s=30,
               orders=[_order(c, Product.REGULAR, vol_r), _order(c, Product.KEROSENE, vol_k)],
               sold_more=(SoldMore(c.id, days, Product.REGULAR),))
    assert any(f"レギュラー {vol_r / 1000:.1f}kL" in x for x in r.changes)
    assert len(r.shortfalls) == 1 and "灯油" in r.shortfalls[0]


def test_the_event_does_not_change_what_the_morning_trips_carried():
    """積込済みの便の先に事象を当てても、朝の車が積んだ量の差し引きは朝の見込みで行う。"""
    s, base, c = _target()
    vol, _ = _over_room(c, Product.REGULAR)
    orders = [_order(c, Product.REGULAR, vol)]
    frozen = next(st.customer_id for vd in base.vehicle_days for t in vd.trips
                  if t.load_start_min < NOW for st in t.stops)
    plain = replan(s, base, now_min=NOW, orders=orders, time_limit_s=30)
    hot = replan(s, base, now_min=NOW, orders=orders, time_limit_s=30,
                 sold_more=(SoldMore(frozen, 1.0),))
    assert [t for t in plain.frozen_trips] == [t for t in hot.frozen_trips]


# --- 小口の発注（翌日の計画配送を前倒しして合わせる） --------------------------


def _small_order_case():
    """明日が計画配送の日で、朝の計画に無い先を探す。"""
    import datetime

    for k in range(7):
        d = (datetime.date(2026, 10, 12) + datetime.timedelta(days=k)).isoformat()
        s = generate(seed=20261015, date=d, season="winter")
        base = solve(s, matrix=build_matrix(s.depot, s.customers), mode=SolveMode.FAST, time_limit_s=20)
        served = {st.customer_id for t in base.trips for st in t.stops}
        for c in s.customers:
            if c.tomorrow_planned and c.id not in served and c.open_today and c.accepts_delivery:
                return s, base, c
    raise AssertionError("明日が計画配送の日で朝の計画に無い先が1週間に無い")


def test_small_order_pulls_tomorrows_planned_delivery_forward():
    """**小口の発注には、翌日の計画配送を組み合わせて当日に運ぶ**（実務）。

    需要の多い油種は4kL槽の単位でしか運ばない。小口だけのために車を出すのではなく、
    明日運ぶはずの計画配送を前倒しして、まとまった量にして運ぶ。
    """
    s, base, c = _small_order_case()
    p, _ = c.tomorrow_planned[0]
    r = replan(s, base, now_min=7 * 60, time_limit_s=30,
               orders=[NewOrder(c.id, p, 4_000, "system", "ORD-SMALL")])
    assert any("前倒し" in x and c.name in x for x in r.changes), r.changes


def test_a_large_order_does_not_pull_tomorrow_forward():
    s, base, c = _small_order_case()
    p, _ = c.tomorrow_planned[0]
    t = next(t for t in c.tanks if t.product is p)
    r = replan(s, base, now_min=7 * 60, time_limit_s=30,
               orders=[NewOrder(c.id, p, max(8_000, t.topup_l // 4_000 * 4_000), "system", "ORD-BIG")])
    assert not any("前倒し" in x for x in r.changes)


# --- API ----------------------------------------------------------------------


def test_replan_api_takes_the_event_learned_during_the_day():
    client = TestClient(app)
    _s, _base, c = _target()
    vol, days = _over_room(c, Product.REGULAR)
    r = client.post("/api/replan", json={
        "seed": 20261015, "date": DATE, "now": "09:12",
        "orders": [{"customer_id": c.id, "product": "regular", "volume_l": vol,
                    "source": "fax", "evidence_ref": "FAX:test"}],
        "sold_more_now": [{"customer_id": c.id, "extra_days": days, "product": "regular"}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sold_more_now"] and "レギュラー" in body["sold_more_now"][0]
    assert any(c.name in x for x in body["changes"])
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
