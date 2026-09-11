"""持ち戻り（計画配送の先で、持って行った量を降ろせない）への手の打ち方。

優先順は実務どおり: 近くの降ろし先 → クッションの大型SS（大竹・岩国IC近く）
→ 積場へ持ち戻り（買い戻し）。費用は元売の管理責任で、運送会社・SSへの請求は無い。

**持ち戻り品の処分が最優先**。積んだままでは後の配送がその制約を受けるので、
次の配送の時刻がずれても先に降ろす。間に合わなくなった先は他のローリーで持つ。

**量は槽の単位**。槽の全量が入らなければその槽は降ろさないので、3kLのような
半端な持ち戻りは起こり得ない。
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import pytest

from core.generator import generate
from core.returns import (
    COMPARTMENT_UNIT_L,
    RESPONSIBILITY_JA,
    ReturnEvent,
    format_return_ja,
    handle_return,
)
from core.solver import SolveMode, solve
from core.travel import build_matrix


@lru_cache(maxsize=None)
def _day():
    s = generate(seed=20261015, date="2026-10-15", season="winter")
    plan = solve(s, matrix=build_matrix(s.depot, s.customers), mode=SolveMode.REPRODUCIBLE)
    by_id = {c.id: c for c in s.customers}
    for vd in plan.vehicle_days:
        for t in vd.trips:
            for st in t.stops:
                c = by_id[st.customer_id]
                if c.mandatory_basis == "planned":
                    p, v = max(c.demand_by_product().items(), key=lambda kv: kv[1])
                    # 持ち戻りは槽の単位。槽の全量が入らなければその槽は降ろさない
                    vol = min(4_000, v // COMPARTMENT_UNIT_L * COMPARTMENT_UNIT_L)
                    if vol:
                        return s, plan, ReturnEvent(vd.vehicle_id, c.id, p, vol)
    raise AssertionError("計画配送の先が今日の計画に無い")


def _vehicle_day(s, plan, e):
    vd = next(v for v in plan.vehicle_days if v.vehicle_id == e.vehicle_id)
    trip = next(t for t in vd.trips if any(st.customer_id == e.customer_id for st in t.stops))
    idx = next(i for i, st in enumerate(trip.stops) if st.customer_id == e.customer_id)
    return vd, trip, idx


def test_nearby_redelivery_is_preferred_when_a_whole_compartment_fits():
    """近くにその槽が丸ごと入る先があれば、クッションより先にそちらへ回す（実務の優先順）。"""
    s, plan, e = _day()
    r = handle_return(s, plan, replace(e, volume_l=2_000))   # 2kL槽ぶん
    assert r.chosen.kind == "redeliver"
    assert any(o.kind == "cushion" and o.feasible for o in r.options)


def test_a_bigger_compartment_can_stop_fitting_where_a_smaller_one_did():
    """槽は丸ごとでしか降ろせない。2kLなら入る近くの先でも、4kLでは入らないことがある。

    SSの空きは1日の売上ぶんしかない。半端に降ろして残りを持ち帰ることはできないので
    （実務）、槽が大きいほど近くでは捌けず、クッションへ回ることになる。
    """
    s, plan, e = _day()
    small = handle_return(s, plan, replace(e, volume_l=2_000))
    big = handle_return(s, plan, replace(e, volume_l=4_000))
    assert small.chosen.kind == "redeliver"
    assert big.chosen.kind == "cushion"
    assert all(not o.feasible for o in big.options if o.kind == "redeliver")


def test_cushion_when_no_nearby_customer_has_room():
    """近くに空きのある先が無ければ、大竹・岩国IC近くの大型SSが受け入れる。"""
    s, plan, e = _day()
    full = tuple(
        c if c.id == e.customer_id else replace(
            c, tanks=tuple(replace(t, level_l=t.capacity_l - t.ullage_margin_l) for t in c.tanks))
        for c in s.customers
    )
    s2 = replace(s, customers=full)
    r = handle_return(s2, plan, e)
    assert r.chosen.kind == "cushion"


def test_later_stop_that_misses_its_window_is_handed_off_not_a_reason_to_refuse():
    """**降ろすのが先。** 後の先が受入時間に間に合わなくなっても手を捨てず、他のローリーへ渡す。

    以前はここで「後の予定が間に合わない」としてその手を捨てていた。持ち戻りを積んだままでは
    後の配送がずっとその制約を受けるので、先に降ろすほうを取る（実務）。
    """
    s, plan, e = _day()
    vd, trip, idx = _vehicle_day(s, plan, e)
    if len(trip.stops) <= idx + 1:
        pytest.skip("この便には降ろせなかった先より後の先が無い")
    nxt = trip.stops[idx + 1]
    # 1分でもずれたら間に合わない受入枠にする
    tight = replace(nxt, window_close_min=nxt.arrive_min + 1)
    stops = trip.stops[:idx + 1] + (tight,) + trip.stops[idx + 2:]
    plan2 = replace(plan, vehicle_days=tuple(
        replace(v, trips=tuple(replace(t, stops=stops) if t is trip else t for t in v.trips))
        if v is vd else v
        for v in plan.vehicle_days
    ))
    r = handle_return(s, plan2, e)
    assert r.chosen.feasible
    assert r.chosen.kind != "buyback"
    assert nxt.customer_name in r.chosen.handoff
    assert "他のローリー" in r.summary_ja()


def test_buyback_when_even_going_straight_home_breaks_the_duty_limit():
    """後を全部他のローリーへ渡しても拘束の上限を超えるなら、積場へ持ち戻って買い戻し。"""
    s, plan, e = _day()
    vd, trip, idx = _vehicle_day(s, plan, e)
    here = trip.stops[idx]
    driver_id = next(v.driver_id for v in s.vehicles if v.id == e.vehicle_id)
    # 降ろせなかった先を出た時点で上限。ここから1分も動けない
    cap = here.depart_min - vd.duty_start_min
    tight = tuple(replace(d, max_duty_min=cap) if d.id == driver_id else d for d in s.drivers)
    s2 = replace(s, drivers=tight)
    r = handle_return(s2, plan, e)
    assert r.chosen.kind == "buyback"
    assert "買い戻し" in r.summary_ja()


def test_feasible_options_stay_within_the_duty_limit():
    """成り立つとした手は、拘束の上限（法令）を守っている。後の予定のずれは理由にしない。"""
    s, plan, e = _day()
    vd, _trip, _idx = _vehicle_day(s, plan, e)
    cap = s.driver(next(v.driver_id for v in s.vehicles if v.id == e.vehicle_id)).max_duty_min
    r = handle_return(s, plan, e)
    for o in r.options:
        if o.feasible and o.kind != "buyback":
            assert vd.duty_min + o.extra_min <= cap
    # ずれを受け入れた手が実際に並んでいる（ずれが失格の理由になっていない）
    assert any(o.feasible and o.delay_min > 0 for o in r.options)


def test_responsibility_is_stated():
    """費用は元売の管理責任で、運送会社・SSへの請求は無いと明記する（実務）。"""
    s, plan, e = _day()
    text = format_return_ja(handle_return(s, plan, e))
    assert RESPONSIBILITY_JA in text


def test_half_a_compartment_cannot_be_returned():
    """槽の全量が入らなければその槽は降ろさない。3kLのような半端な持ち戻りは起きない（実務）。"""
    s, plan, e = _day()
    assert e.volume_l % COMPARTMENT_UNIT_L == 0
    with pytest.raises(ValueError, match="槽の単位"):
        handle_return(s, plan, replace(e, volume_l=3_000))


def test_bad_events_are_rejected():
    s, plan, e = _day()
    with pytest.raises(ValueError, match="名簿に無い"):
        handle_return(s, plan, replace(e, customer_id="c99"))
    with pytest.raises(ValueError, match="計画に"):
        handle_return(s, plan, replace(e, vehicle_id="v9"))
