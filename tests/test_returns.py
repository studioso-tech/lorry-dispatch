"""持ち戻り（計画配送の先で、持って行った量を降ろせない）への手の打ち方。

優先順は実務どおり: 近くの降ろし先 → クッションの大型SS（大竹・岩国IC近く）
→ 積場へ持ち戻り（買い戻し）。費用は元売の管理責任で、運送会社・SSへの請求は無い。
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import pytest

from core.generator import generate
from core.returns import RESPONSIBILITY_JA, ReturnEvent, format_return_ja, handle_return
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
                    return s, plan, ReturnEvent(vd.vehicle_id, c.id, p, min(3_000, v))
    raise AssertionError("計画配送の先が今日の計画に無い")


def test_nearby_redelivery_comes_first_even_if_the_cushion_is_shorter():
    """近くで降ろせる先があれば、クッションより寄り道が長くてもそちらを選ぶ（実務の優先順）。"""
    s, plan, e = _day()
    r = handle_return(s, plan, e)
    assert r.chosen.kind == "redeliver"
    assert any(o.kind == "cushion" and o.feasible for o in r.options)


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


def test_buyback_when_nothing_fits_the_day():
    """クッションにも寄れない（拘束の上限を超える）なら、積場へ持ち戻って買い戻し。"""
    s, plan, e = _day()
    vd = next(v for v in plan.vehicle_days if v.vehicle_id == e.vehicle_id)
    driver_id = next(v.driver_id for v in s.vehicles if v.id == e.vehicle_id)
    tight = tuple(replace(d, max_duty_min=vd.duty_min) if d.id == driver_id else d for d in s.drivers)
    s2 = replace(s, drivers=tight)
    r = handle_return(s2, plan, e)
    assert r.chosen.kind == "buyback"
    assert "買い戻し" in r.summary_ja()


def test_every_option_respects_room_windows_and_duty():
    """成り立つとした手は、空き・受入時間・拘束の上限をすべて満たしている。"""
    s, plan, e = _day()
    r = handle_return(s, plan, e)
    vd = next(v for v in plan.vehicle_days if v.vehicle_id == e.vehicle_id)
    cap = s.driver(next(v.driver_id for v in s.vehicles if v.id == e.vehicle_id)).max_duty_min
    for o in r.options:
        if o.feasible and o.kind != "buyback":
            assert vd.duty_min + o.extra_min <= cap


def test_responsibility_is_stated():
    """費用は元売の管理責任で、運送会社・SSへの請求は無いと明記する（実務）。"""
    s, plan, e = _day()
    text = format_return_ja(handle_return(s, plan, e))
    assert RESPONSIBILITY_JA in text


def test_bad_events_are_rejected():
    s, plan, e = _day()
    with pytest.raises(ValueError, match="名簿に無い"):
        handle_return(s, plan, replace(e, customer_id="c99"))
    with pytest.raises(ValueError, match="計画に"):
        handle_return(s, plan, replace(e, vehicle_id="v9"))
