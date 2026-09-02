"""決定論コアが返す計画が、制約を本当に守っているかを検算する。

なぜ検算するのか:
    ソルバーが「OPTIMAL」と言うのは、**モデルに書いた制約**を満たしたという
    意味でしかない。制約の書き間違い（時間の連鎖を1本落とす、容量を足し忘れる）
    があっても堂々と OPTIMAL が返る。だから解のほうを、モデルと独立に
    素の Python で読み直す。

    これは引き継ぎ文書の検証原則「合格率でなく現物を読む」と同じ姿勢。

求解時間について:
    複数便にしてから、1件あたり数十秒かかる。テストの seed を絞ってあるのは
    そのため。全 seed を舐めたい場合は W6 の回帰テスト一式のほうでやる。
"""

from __future__ import annotations

import pytest

from core.generator import generate
from core.solver import solve
from core.travel import build_matrix

#: 通常のテストで使う seed。ここを増やすと1回の実行が分単位で伸びる。
SEEDS = (20261015, 42)

#: 求解の打ち切り。既定より短くして、テスト全体を現実的な長さに保つ。
#: 最適性を証明できない seed が混ざるが、**制約を守っているかの検算には
#: 影響しない**（FEASIBLE な解も制約は満たしている）。
TEST_TIME_LIMIT_S = 25.0


def _plan(seed: int = 20261015, season: str = "winter", **kw):
    s = generate(seed=seed, season=season, **kw)
    m = build_matrix(s.depot, s.customers)
    return s, m, solve(s, matrix=m, time_limit_s=TEST_TIME_LIMIT_S)


@pytest.mark.parametrize("seed", SEEDS)
def test_plan_is_produced(seed):
    """どの日でも計画が返る。解けない日があってもよいが、黙って落ちてはいけない。"""
    _, _, plan = _plan(seed=seed)
    assert plan.status in ("OPTIMAL", "FEASIBLE")
    assert plan.vehicle_days


@pytest.mark.parametrize("seed", SEEDS)
def test_time_windows_are_respected(seed):
    """到着はすべて受入時間枠の中。"""
    _, _, plan = _plan(seed=seed)
    for tr in plan.trips:
        for s in tr.stops:
            assert s.window_open_min <= s.arrive_min <= s.window_close_min, (
                f"{tr.vehicle_id} {tr.trip_index}便 {s.customer_name} 着 {s.arrive_min}"
            )


@pytest.mark.parametrize("seed", SEEDS)
def test_capacity_is_respected_per_trip(seed):
    """1便ごとに積載量を超えない。

    便ごとに見るのが要点。1日の合計で見ると、3便で42kL 運んだ 14kL 車が
    「合計が容量を超えている」ことになって誤検知するし、逆に1便で
    容量を超えた計画を見逃す。
    """
    _, _, plan = _plan(seed=seed)
    for tr in plan.trips:
        assert tr.load_l <= tr.capacity_l, f"{tr.vehicle_id} {tr.trip_index}便"
        assert tr.load_l == sum(s.delivered_l for s in tr.stops)


@pytest.mark.parametrize("seed", SEEDS)
def test_driver_hours_are_respected(seed):
    """拘束時間・運転時間が改善基準告示の上限に収まる。

    上限は便ではなく**人**にかかる。1便ずつ見ていると、3便で拘束15時間の
    計画が「どの便も短いから大丈夫」に見えてしまう。
    """
    scenario, _, plan = _plan(seed=seed)
    for vd in plan.vehicle_days:
        d = scenario.driver(vd.driver_id)
        assert vd.duty_min <= d.max_duty_min, f"{vd.vehicle_id} 拘束 {vd.duty_min}分"
        assert vd.drive_min <= d.max_drive_min, f"{vd.vehicle_id} 運転 {vd.drive_min}分"
        assert vd.drive_min == sum(t.drive_min for t in vd.trips)
        if vd.drive_min > d.continuous_drive_limit_min:
            assert vd.break_min >= d.break_min


@pytest.mark.parametrize("seed", SEEDS)
def test_trips_do_not_overlap_and_allow_reloading(seed):
    """同じ車の便は時間順に並び、前の便が帰ってから積込ぶん空けて次が出る。

    ここが緩いと、まだ帰っていない車が次の便を出す計画になる。
    """
    scenario, _, plan = _plan(seed=seed)
    load_min = scenario.depot.load_min
    for vd in plan.vehicle_days:
        indices = [t.trip_index for t in vd.trips]
        assert indices == sorted(indices)
        for a, b in zip(vd.trips, vd.trips[1:]):
            assert b.depart_min >= a.return_min + load_min, (
                f"{vd.vehicle_id}: {a.trip_index}便 帰庫{a.return_min} → "
                f"{b.trip_index}便 出庫{b.depart_min}（積込{load_min}分）"
            )
        # 拘束の始まりと終わりが、実際の便と整合している
        assert vd.duty_start_min == vd.trips[0].load_start_min
        assert vd.duty_end_min == max(t.return_min for t in vd.trips)


@pytest.mark.parametrize("seed", SEEDS)
def test_travel_chain_is_consistent(seed):
    """到着時刻の連鎖を、行列から素の Python で足し直して検算する。

    ソルバーのモデルではなく行列を直接読むのが要点。モデル側の制約を
    1本落としていても、この計算は騙されない。
    """
    scenario, matrix, plan = _plan(seed=seed)
    index = {c.id: i + 1 for i, c in enumerate(scenario.customers)}

    for tr in plan.trips:
        prev_node = 0
        prev_ready = tr.depart_min          # 油槽所を出た時刻
        for s in tr.stops:
            node = index[s.customer_id]
            earliest = prev_ready + matrix.minutes[prev_node][node]
            assert s.arrive_min >= earliest, (
                f"{tr.vehicle_id} {tr.trip_index}便 {s.customer_name}: "
                f"着 {s.arrive_min} < 最短 {earliest}"
            )
            prev_node = node
            prev_ready = s.depart_min
        assert tr.return_min >= prev_ready + matrix.minutes[prev_node][0]


@pytest.mark.parametrize("seed", SEEDS)
def test_each_customer_visited_at_most_once(seed):
    """同じ需要家を2度回らない。便をまたいでも1回きり。"""
    _, _, plan = _plan(seed=seed)
    visited = [s.customer_id for tr in plan.trips for s in tr.stops]
    assert len(visited) == len(set(visited))


@pytest.mark.parametrize("seed", SEEDS)
def test_served_and_unserved_partition_the_demand(seed):
    """需要のある需要家は「回った」か「回れなかった」のどちらかに必ず入る。

    集計から漏れる需要家がいると、切れかけの軒を黙って落とせてしまう。
    """
    scenario, _, plan = _plan(seed=seed)
    with_demand = {c.id for c in scenario.customers if c.demand_l > 0}
    served = {s.customer_id for tr in plan.trips for s in tr.stops}
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
            vd.vehicle_id,
            tuple(
                (
                    t.trip_index,
                    t.depart_min,
                    t.return_min,
                    tuple((s.customer_id, s.arrive_min) for s in t.stops),
                )
                for t in vd.trips
            ),
        )
        for vd in plan.vehicle_days
    )


def test_solution_is_reproducible_when_both_phases_prove_optimality():
    """両段が最適性を証明できたなら、同じ入力は同じ計画を返す。

    決定性は目的関数の3段のタイブレーク（到着時刻の合計 → 便と需要家の組 →
    回る順序）で担保している。ただし**それは最適解が証明できた場合の話**で、
    時間切れで打ち切った解は同じ保証を持たない。
    その区別は diagnostics["reproducible"] に出ている。

    ここでは上限1便に落として、必ず証明できる条件で確かめる。
    複数便の求解を速くして、既定の条件でもこれが言えるようにするのが
    W2 に残っている課題。
    """
    runs = []
    for _ in range(3):
        s = generate(seed=20261015, season="winter")
        m = build_matrix(s.depot, s.customers)
        runs.append(solve(s, matrix=m, time_limit_s=30.0, max_trips=1))

    assert all(r.diagnostics["reproducible"] for r in runs), [
        (r.diagnostics["phase1_status"], r.diagnostics["phase2_status"]) for r in runs
    ]
    assert len({r.objective for r in runs}) == 1
    assert len({_signature(r) for r in runs}) == 1


def test_more_trips_serve_more_customers():
    """便を増やせば回れる軒が増える。複数便を入れた意味の確認。

    1台1日1便では、全車が午前中に帰庫して以降ずっと空くのに、
    切れかけの需要家が残っていた。
    """
    s = generate(seed=20261015, season="winter")
    m = build_matrix(s.depot, s.customers)
    one = solve(s, matrix=m, time_limit_s=30.0, max_trips=1)
    two = solve(s, matrix=m, time_limit_s=45.0, max_trips=2)

    served_one = sum(vd.stop_count for vd in one.vehicle_days)
    served_two = sum(vd.stop_count for vd in two.vehicle_days)
    assert served_two > served_one
    assert len(two.mandatory_missed) <= len(one.mandatory_missed)


def test_absent_driver_is_not_dispatched():
    """欠勤した乗務員の車は計画に出てこない（デモ4の前提）。"""
    _, _, plan = _plan(absent_drivers=("d1", "d2"))
    assert plan.is_usable
    assert {vd.driver_id for vd in plan.vehicle_days}.isdisjoint({"d1", "d2"})


def test_no_drivers_returns_explained_infeasible():
    """全員欠勤なら、例外ではなく理由つきの INFEASIBLE を返す。

    エージェントが配車係に「なぜ組めないか」を言えなければ意味がない。
    """
    _, _, plan = _plan(absent_drivers=("d1", "d2", "d3", "d4", "d5"))
    assert plan.status == "INFEASIBLE"
    assert plan.vehicle_days == ()
    assert "ローリー" in str(plan.diagnostics.get("reason", ""))


def test_full_tanks_are_not_visited():
    """需要ゼロの需要家には行かない。行けるのに行かない、を明示する。"""
    scenario, _, plan = _plan()
    zero = {c.id for c in scenario.customers if c.demand_l == 0}
    served = {s.customer_id for tr in plan.trips for s in tr.stops}
    assert not (zero & served)


def test_objective_stays_in_range():
    """目的値が変数の定義域に収まっている。

    2段階求解の1段目と2段目で同じ solver を使い回していたとき、2段目が
    時間切れになると1段目の解が消え、定義域を遥かに超えた目的値
    （1,760,776,892,128）や、必須訪問を6軒落とした目的値0が返っていた。
    段ごとに solver を分けて直した。その退行を捕まえる。
    """
    for seed in (20261015, 42):
        _, _, plan = _plan(seed=seed)
        assert 0 <= plan.objective <= 10**9, (seed, plan.objective)
        # 必須訪問を落としていないなら、目的値は罰1件ぶんより小さいはず
        if not plan.mandatory_missed:
            assert plan.objective < 100_000
