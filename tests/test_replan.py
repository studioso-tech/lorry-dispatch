"""当日発注の組み直し。

組み直しで壊れてはいけないものは3つ:

1. **積込済みの便を動かさない**（油はもう車に載っている）
2. **発注は証拠の残る形でしか受けない**（発注システムかFAX。電話は不可）
3. **入らないものを入ったように見せない**（発注量どおりに入らなければ確認を求める）

そのうえで、入らない発注への手の打ち方が「専属の余り → 非専属 → 逃げ道
（計画配送を明日へ）」の順になっていることを確かめる。
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from core.domain import Product
from core.generator import generate
from core.replan import MIN_DROP_L, NewOrder, format_replan_ja, replan
from core.solver import solve
from core.travel import build_matrix

TIME_LIMIT_S = 20.0


@lru_cache(maxsize=None)
def _base(seed: int, date: str, fleet: tuple[str, ...]):
    s = generate(seed=seed, date=date, season="winter")
    s.vehicles = tuple(v for v in s.vehicles if v.id in fleet)
    return s, solve(s, matrix=build_matrix(s.depot, s.customers), time_limit_s=TIME_LIMIT_S)


def _surge(s, base, n=4):
    """朝の計画に無い先から、当日発注を n 件（連休の高速SSのような重なり）。"""
    served = {st.customer_id for t in base.trips for st in t.stops}
    free = [c for c in s.customers if c.id not in served and c.open_today and c.accepts_delivery]
    orders = []
    for i, c in enumerate(free[:n]):
        t = max(c.tanks, key=lambda t: t.topup_l)
        orders.append(NewOrder(c.id, t.product, min(6_000, t.topup_l), "system", f"ORD-{i + 1:03d}"))
    return orders


def test_phone_is_not_an_order():
    """電話は発注ではない。証拠の残る形（発注システムかFAX）が要る（実務）。"""
    with pytest.raises(ValueError, match="証拠"):
        NewOrder("c01", Product.REGULAR, 8_000, "phone", "電話 9:30")
    with pytest.raises(ValueError, match="証拠"):
        NewOrder("c01", Product.REGULAR, 8_000, "fax", "")


def test_loaded_trips_are_not_moved():
    """積込を始めた便は、組み直しても一切動かない。"""
    s, base = _base(20261016, "2026-10-16", ("v4", "v5"))
    now = 8 * 60
    r = replan(s, base, now_min=now, orders=_surge(s, base))
    before = [
        (vd.vehicle_id, t.load_start_min, tuple((st.customer_id, st.arrive_min) for st in t.stops))
        for vd in base.vehicle_days for t in vd.trips if t.load_start_min < now
    ]
    after = [
        (vd.vehicle_id, t.load_start_min, tuple((st.customer_id, st.arrive_min) for st in t.stops))
        for vd in r.vehicle_days for t in vd.trips if t.load_start_min < now
    ]
    assert before and before == after
    # 組み直した便は、いまより前には積込を始めない
    for vd in r.sub_plan.vehicle_days:
        for t in vd.trips:
            assert t.load_start_min >= now


def test_duty_is_counted_through_the_whole_day():
    """拘束13時間・運転9時間は、午前の便も含めて1日通しで守る。"""
    s, base = _base(20261016, "2026-10-16", ("v4", "v5"))
    r = replan(s, base, now_min=8 * 60, orders=_surge(s, base))
    for vd in r.vehicle_days:
        drv = s.driver(next(v for v in s.vehicles if v.id == vd.vehicle_id).driver_id)
        assert vd.duty_end_min - vd.duty_start_min <= drv.max_duty_min, vd.vehicle_id
        assert vd.drive_min <= drv.max_drive_min, vd.vehicle_id


def test_without_spot_a_planned_delivery_is_deferred_not_the_order():
    """専属の余りで入らず非専属もいないなら、**計画配送を明日へずらして発注を入れる**。

    発注は需要家の約束、計画配送は運送会社側の都合で1日ずらせる。ずらしたら
    逃げ道として記録する（恒常化の監視が数える）。
    """
    s, base = _base(20261016, "2026-10-16", ("v4", "v5"))
    orders = _surge(s, base)
    r = replan(s, base, now_min=8 * 60, orders=orders, spot_standby=0)
    assert r.ok, format_replan_ja(r)
    assert r.escapes, "計画配送をずらさずに入った（この日は入らないはず）"
    assert all(e.kind == "defer_planned" for e in r.escapes)
    placed = {st.customer_id for vd in r.sub_plan.vehicle_days for t in vd.trips for st in t.stops}
    assert {o.customer_id for o in orders} <= placed


def test_spot_is_called_before_deferring_planned_deliveries():
    """待機の非専属がいれば、計画配送をずらす前に非専属を呼ぶ。"""
    s, base = _base(20261016, "2026-10-16", ("v4", "v5"))
    r = replan(s, base, now_min=8 * 60, orders=_surge(s, base), spot_standby=2)
    assert r.ok
    assert r.spot_called
    assert not r.escapes


def test_order_that_does_not_fit_is_flagged_not_trimmed():
    """発注量どおりに入らないなら、黙って削らずに確認を求める。

    最初の実装は、積込済みの便が同じ先に降ろす直後で空きが0.8kLしか無いのに、
    8kLの発注を「8.0kL…着」と表示し、0.8kLのために1台出していた。
    """
    s, base = _base(20261015, "2026-10-15", ("v1", "v4", "v5"))
    now = 9 * 60 + 30
    loaded = [
        st for vd in base.vehicle_days for t in vd.trips
        if t.load_start_min < now for st in t.stops
    ]
    assert loaded, "積込済みの便が無い（この確認が成り立たない）"
    target = loaded[-1].customer_id
    c = next(x for x in s.customers if x.id == target)
    product = max(c.tanks, key=lambda t: t.capacity_l).product
    r = replan(s, base, now_min=now,
               orders=[NewOrder(target, product, 8_000, "fax", "FAX-TEST")])
    assert r.shortfalls, "入らない発注を確認せずに通した"
    assert "確認" in r.shortfalls[0]
    visits = [st for vd in r.sub_plan.vehicle_days for t in vd.trips for st in t.stops
              if st.customer_id == target]
    assert all(st.delivered_l >= MIN_DROP_L for st in visits)
    assert "確認が要るもの" in format_replan_ja(r)


def test_orders_carry_their_evidence_into_the_explanation():
    """説明文に発注の証拠（伝票番号・FAX）が出る。後から辿れるようにする。"""
    s, base = _base(20261016, "2026-10-16", ("v4", "v5"))
    orders = _surge(s, base, n=1)
    r = replan(s, base, now_min=8 * 60, orders=orders, spot_standby=1)
    assert orders[0].evidence_ref in format_replan_ja(r)


def test_unknown_customer_or_product_is_rejected():
    s, base = _base(20261016, "2026-10-16", ("v4", "v5"))
    with pytest.raises(ValueError, match="名簿に無い"):
        replan(s, base, now_min=480, orders=[NewOrder("c99", Product.REGULAR, 1_000, "system", "X")])
    carrier = next(c for c in s.customers if all(t.product is Product.GASOIL for t in c.tanks))
    with pytest.raises(ValueError, match="タンクは無い"):
        replan(s, base, now_min=480, orders=[NewOrder(carrier.id, Product.KEROSENE, 1_000, "system", "X")])
