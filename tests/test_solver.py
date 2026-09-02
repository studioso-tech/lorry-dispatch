"""決定論コアが返す計画が、制約を本当に守っているかを検算する。

なぜ検算するのか:
    ソルバーが「OPTIMAL」と言うのは、**モデルに書いた制約**を満たしたという
    意味でしかない。制約の書き間違い（時間の連鎖を1本落とす、容量を足し忘れる）
    があっても堂々と OPTIMAL が返る。だから解のほうを、モデルと独立に
    素の Python で読み直す。

    これは引き継ぎ文書の検証原則「合格率でなく現物を読む」と同じ姿勢。
"""

from __future__ import annotations

import pytest

from core.generator import generate
from core.solver import solve
from core.travel import build_matrix

SEEDS = (1, 42, 777, 20261015)


def _plan(seed: int = 20261015, season: str = "winter", **kw):
    s = generate(seed=seed, season=season, **kw)
    m = build_matrix(s.depot, s.customers)
    return s, m, solve(s, matrix=m, time_limit_s=20.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_plan_is_produced(seed):
    """どの日でも計画が返る。解けない日があってもよいが、黙って落ちてはいけない。"""
    _, _, plan = _plan(seed=seed)
    assert plan.status in ("OPTIMAL", "FEASIBLE")


@pytest.mark.parametrize("seed", SEEDS)
def test_time_windows_are_respected(seed):
    """到着はすべて受入時間枠の中。"""
    _, _, plan = _plan(seed=seed)
    for r in plan.routes:
        for s in r.stops:
            assert s.window_open_min <= s.arrive_min <= s.window_close_min, (
                f"{r.vehicle_id} {s.customer_name} 着 {s.arrive_min}"
            )


@pytest.mark.parametrize("seed", SEEDS)
def test_capacity_is_respected(seed):
    """積んだ量が積載量を超えない。"""
    _, _, plan = _plan(seed=seed)
    for r in plan.routes:
        assert r.load_l <= r.capacity_l
        assert r.load_l == sum(s.delivered_l for s in r.stops)


@pytest.mark.parametrize("seed", SEEDS)
def test_driver_hours_are_respected(seed):
    """拘束時間・運転時間が改善基準告示の上限に収まる。

    ここが破れた計画は、そもそも走らせてはいけない。
    """
    scenario, _, plan = _plan(seed=seed)
    for r in plan.routes:
        d = scenario.driver(r.driver_id)
        assert r.duty_min <= d.max_duty_min, f"{r.vehicle_id} 拘束 {r.duty_min}分"
        assert r.drive_min <= d.max_drive_min, f"{r.vehicle_id} 運転 {r.drive_min}分"
        # 連続運転4時間を超えるなら休憩の余裕が確保されている
        if r.drive_min > d.continuous_drive_limit_min:
            assert r.break_min >= d.break_min


@pytest.mark.parametrize("seed", SEEDS)
def test_travel_chain_is_consistent(seed):
    """到着時刻の連鎖を、行列から素の Python で足し直して検算する。

    ソルバーのモデルではなく行列を直接読むのが要点。モデル側の制約を
    1本落としていても、この計算は騙されない。
    """
    scenario, matrix, plan = _plan(seed=seed)
    index = {c.id: i + 1 for i, c in enumerate(scenario.customers)}

    for r in plan.routes:
        prev_node = 0
        prev_ready = r.depart_min          # 油槽所を出た時刻
        for s in r.stops:
            node = index[s.customer_id]
            earliest = prev_ready + matrix.minutes[prev_node][node]
            assert s.arrive_min >= earliest, (
                f"{r.vehicle_id} {s.customer_name}: "
                f"着 {s.arrive_min} < 最短 {earliest}"
            )
            prev_node = node
            prev_ready = s.depart_min
        # 帰着も同様
        assert r.return_min >= prev_ready + matrix.minutes[prev_node][0]


@pytest.mark.parametrize("seed", SEEDS)
def test_each_customer_visited_at_most_once(seed):
    """同じ需要家を2台が回らない。"""
    _, _, plan = _plan(seed=seed)
    visited = [s.customer_id for r in plan.routes for s in r.stops]
    assert len(visited) == len(set(visited))


@pytest.mark.parametrize("seed", SEEDS)
def test_served_and_unserved_partition_the_demand(seed):
    """需要のある需要家は「回った」か「回れなかった」のどちらかに必ず入る。

    集計から漏れる需要家がいると、切れかけの軒を黙って落とせてしまう。
    """
    scenario, _, plan = _plan(seed=seed)
    with_demand = {c.id for c in scenario.customers if c.demand_l > 0}
    served = {s.customer_id for r in plan.routes for s in r.stops}
    unserved = {u.customer_id for u in plan.unserved}
    assert served | unserved == with_demand
    assert not (served & unserved)


def _signature(plan) -> tuple:
    """計画を丸ごと比べるための署名。

    目的値だけを比べるのでは足りない。同点の最適解が複数あると、
    目的値は同じまま**どの車がどの需要家をどの順で回るか**が変わる。
    実際にそれが起きたので、割当・順序・時刻まで含めて比べる。
    """
    return tuple(
        (
            r.vehicle_id,
            r.depart_min,
            r.return_min,
            tuple((s.customer_id, s.arrive_min) for s in r.stops),
        )
        for r in plan.routes
    )


@pytest.mark.parametrize("seed", (20261015, 42, 777))
def test_solution_is_reproducible(seed):
    """同じ入力なら**同じ計画**。回帰テスト（W6）と審査員の再現の前提。

    目的関数のタイブレーク（到着時刻の合計 → 車と需要家の組 → 回る順序）で
    最適解を一意に絞っている。ここが崩れると、同じ要求に別の割当が返る。
    """
    runs = [_plan(seed=seed)[2] for _ in range(3)]
    assert len({r.objective for r in runs}) == 1
    assert len({_signature(r) for r in runs}) == 1


def test_absent_driver_is_not_dispatched():
    """欠勤した乗務員の車は計画に出てこない（デモ4の前提）。"""
    _, _, plan = _plan(absent_drivers=("d1", "d2"))
    assert plan.is_usable
    assert {r.driver_id for r in plan.routes}.isdisjoint({"d1", "d2"})


def test_no_drivers_returns_explained_infeasible():
    """全員欠勤なら、例外ではなく理由つきの INFEASIBLE を返す。

    エージェントが配車係に「なぜ組めないか」を言えなければ意味がない。
    """
    _, _, plan = _plan(absent_drivers=("d1", "d2", "d3", "d4", "d5"))
    assert plan.status == "INFEASIBLE"
    assert plan.routes == ()
    assert "ローリー" in str(plan.diagnostics.get("reason", ""))


def test_full_tanks_are_not_visited():
    """需要ゼロの需要家には行かない。行けるのに行かない、を明示する。"""
    scenario, _, plan = _plan()
    zero = {c.id for c in scenario.customers if c.demand_l == 0}
    served = {s.customer_id for r in plan.routes for s in r.stops}
    assert not (zero & served)
