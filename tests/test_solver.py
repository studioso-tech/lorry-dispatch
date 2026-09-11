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

from functools import lru_cache

import pytest

from core.generator import generate
from core.solver import format_plan_ja, solve
from core.travel import build_matrix

#: 通常のテストで使う seed。ここを増やすと1回の実行が分単位で伸びる。
SEEDS = (20261015, 42)

#: 求解の打ち切り。既定より短くして、テスト全体を現実的な長さに保つ。
#: 最適性を証明できない seed が混ざるが、**制約を守っているかの検算には
#: 影響しない**（FEASIBLE な解も制約は満たしている）。
TEST_TIME_LIMIT_S = 25.0


@lru_cache(maxsize=None)
def _plan(seed: int = 20261015, season: str = "winter", absent_drivers: tuple = ()):
    """同じ条件は1度しか解かない。

    テストごとに解き直していたとき、同じ問題を8回解いていてテスト全体が
    10分を超えた。求解は入力が同じなら結果も同じ（それがこのコアの性質）
    なので、使い回して差し支えない。
    """
    s = generate(seed=seed, season=season, absent_drivers=absent_drivers)
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


def test_served_count_matches_actual_stops():
    """診断の「回れた軒数」が、実際の立ち寄り数と一致する。

    引き算（全需要家 − 回れなかった需要家）で出していたとき、満タンで
    行く必要のない軒が「回った」側に混ざり、本番で立ち寄り9軒の計画が
    served_count=10 と申告していた。数字が1つでもずれると、
    「今日は何軒回れたのか」という一番基本的な問いに答えられなくなる。
    """
    scenario, _, plan = _plan()
    actual = sum(len(t.stops) for t in plan.trips)
    assert plan.diagnostics["served_count"] == actual

    # 回った軒 ＋ 回れなかった軒 ＋ 需要ゼロの軒 ＝ 全需要家
    assert (
        actual + len(plan.unserved) + plan.diagnostics["no_demand_count"]
        == len(scenario.customers)
    )


def test_reproducible_flag_never_lies():
    """`reproducible` が真なら、両段とも最適性を証明できている。

    2段目を決定的時間で打ち切って最適性の証明を省く案を試したとき、
    この条件を緩めた。結果、**計画が違うのに reproducible が真を返す**
    という最悪の形になった（5件中3件）。フラグが嘘をつくくらいなら、
    遅いほうがましである。

    構造として守るべきなのは「真を返すなら根拠がある」ことなので、
    診断の中身どうしの整合をここで固定する。
    """
    for max_trips in (1, 2):
        s = generate(seed=777, season="winter")
        m = build_matrix(s.depot, s.customers)
        plan = solve(s, matrix=m, time_limit_s=TEST_TIME_LIMIT_S, max_trips=max_trips)
        d = plan.diagnostics
        if d["reproducible"]:
            assert d["phase1_status"] == "OPTIMAL", d
            assert d["phase2_status"] == "OPTIMAL", d


def test_skipping_the_tiebreak_phase_keeps_the_cost():
    """2段目を省いても、コストと実行可能性は変わらない。

    体制設計の掃引は「何台で回るか」しか要らないので、2段目を丸ごと
    省いて時間を稼ぐ。省いた計画は再現性を持たないので、その旨が
    診断に出ていることも確かめる。
    """
    s = generate(seed=99, season="winter")
    m = build_matrix(s.depot, s.customers)
    full = solve(s, matrix=m, time_limit_s=TEST_TIME_LIMIT_S)
    quick = solve(s, matrix=m, time_limit_s=TEST_TIME_LIMIT_S, tiebreak_phase=False)

    if full.diagnostics["phase1_status"] == "OPTIMAL" and quick.diagnostics["phase1_status"] == "OPTIMAL":
        assert full.objective == quick.objective
    assert quick.diagnostics["phase2_status"] == "skipped"
    assert quick.diagnostics["reproducible"] is False


def test_trailers_never_enter_customers_that_cannot_take_them():
    """トレーラーが入れない需要家に、トレーラーを配車しない。

    **積載量の制約ではない。** 狭い道・小さい構内・切り返しの余地の無さで
    決まる物理的な可否で、荷役条件のマスターに載っている情報。
    単車がそういう先へ優先的に回る。
    """
    for seed in SEEDS:
        scenario, _, plan = _plan(seed=seed)
        by_id = {c.id: c for c in scenario.customers}
        fleet = {v.id: v for v in scenario.vehicles}
        for tr in plan.trips:
            if not fleet[tr.vehicle_id].is_trailer:
                continue
            for s in tr.stops:
                assert by_id[s.customer_id].accepts_trailer, (
                    f"{tr.vehicle_id}（トレーラー）が {s.customer_name} に入っている"
                )


def test_rigid_only_customers_exist_in_the_scenario():
    """単車しか入れない需要家が実際にいること。

    全軒がトレーラーを受け入れる名簿だと、この制約が一度も効かない飾りに
    なる。デモで見せる制約は、デモのデータで発火しなければ意味がない。
    """
    s = generate(seed=20261015, season="winter")
    rigid_only = [c for c in s.customers if not c.accepts_trailer]
    assert rigid_only, "単車のみの需要家が1軒も無い"
    assert any(v.is_trailer for v in s.vehicles), "トレーラーが1台も無い"
    assert any(not v.is_trailer for v in s.vehicles), "単車が1台も無い"


def test_only_real_capacities_exist():
    """実在しない容量のローリーを作らない。

    最初の実装には 10kL の車が居た。そんなローリーは無い。作り物の車で
    組んだ計画は、作り物の制約しか検証しない。
    """
    from core.domain import CAPACITIES_L, Vehicle, VehicleClass

    s = generate(seed=1)
    for v in s.vehicles:
        assert v.capacity_l in CAPACITIES_L[v.vehicle_class], (v.id, v.capacity_l)

    # 型の側でも弾いていること（槽は正しいが合計が実在しない）
    with pytest.raises(ValueError, match="kL 積は無い"):
        Vehicle(
            id="bad", name="10kL", vehicle_class=VehicleClass.RIGID,
            compartments_l=(4_000, 4_000, 2_000), driver_id="d1",
        )


# ---------------------------------------------------------------------------
# 休憩（労基法34条 ＋ 改善基準告示の430休憩）
# ---------------------------------------------------------------------------


def test_labor_break_steps():
    """労基法34条の階段が正しいこと。**上に段を作らない。**

    「拘束12時間を超えたらさらに30分」という上乗せは法令に無い。12時間で
    1時間30分、16時間で2時間といった比例付与は義務づけられていない
    （厚労省の解釈も6時間超45分・8時間超1時間の二段のみ）。
    ここに勝手な段を足すと、法令上不要な休憩で拘束を膨らませ、
    「回れない」と嘘をつく計画が出る。

    判定の基準が**労働時間**（＝拘束−休憩）であることも押さえておく。
    閾値 525分 は 480分（労働8時間）＋ 45分（休憩）の逆算。
    """
    from core.domain import labor_break_min

    assert labor_break_min(0) == 0
    assert labor_break_min(360) == 0        # 拘束6時間ちょうど → 労働6時間 → 不要
    assert labor_break_min(361) == 45
    assert labor_break_min(525) == 45       # 労働8時間ちょうど → 45分で足りる
    assert labor_break_min(526) == 60
    assert labor_break_min(780) == 60       # 拘束13時間 → 60分。ここで打ち止め
    assert labor_break_min(900) == 60       # 15時間でも60分のまま


@pytest.mark.parametrize("seed", SEEDS)
def test_break_satisfies_both_laws(seed):
    """休憩が両方の法令を満たしていること。**足し算ではなく大きいほう。**

    430休憩（連続運転4時間ごとに30分）は、2024年4月の改正で「原則として
    休憩」とされ、荷積み・荷卸しを算入できなくなった。つまり430の30分は
    労基法34条の休憩の内数であって、別枠ではない。

    当初は430だけを入れていて、拘束13時間の計画に休憩が30分しか積まれて
    いなかった。**労基法上30分足りない計画**を出していたことになる。
    """
    from core.domain import labor_break_min

    _s, _m, plan = _plan(seed)
    for vd in plan.vehicle_days:
        if not vd.trips:
            continue
        need_labor = labor_break_min(vd.duty_min)
        need_430 = 0
        d = 240
        while vd.drive_min > d:
            need_430 += 30
            d += 240
        assert vd.break_min >= need_labor, (
            f"{vd.vehicle_id}: 拘束{vd.duty_min}分に休憩{vd.break_min}分。"
            f"労基法34条は{need_labor}分を要求する"
        )
        assert vd.break_min >= need_430, (
            f"{vd.vehicle_id}: 運転{vd.drive_min}分に休憩{vd.break_min}分。"
            f"430休憩は{need_430}分を要求する"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_duty_contains_every_component(seed):
    """拘束時間が、走行・積込・荷卸・休憩をすべて含んでいること。

    どれか1つを落とすと、拘束13時間の枠に入りきらない計画が
    「入った」ことになって出てくる。
    """
    s, _m, plan = _plan(seed)
    svc = {c.id: c.service_min for c in s.customers}
    for vd in plan.vehicle_days:
        if not vd.trips:
            continue
        load = len(vd.trips) * s.depot.load_min
        service = sum(svc[st.customer_id] for t in vd.trips for st in t.stops)
        assert vd.duty_min >= vd.drive_min + load + service + vd.break_min, (
            f"{vd.vehicle_id}: 拘束{vd.duty_min} < "
            f"走行{vd.drive_min}+積込{load}+荷卸{service}+休憩{vd.break_min}"
        )
        assert vd.duty_min <= s.driver(
            next(v for v in s.vehicles if v.id == vd.vehicle_id).driver_id
        ).max_duty_min


def test_break_location_risk_detects_stranded_trip():
    """休憩を取る場所が無い便を検出できること。

    **ローリーはどこででも休憩できない。** 停められるのは油槽所の待機場、
    高速道路のSA/PA、運送会社の車庫のいずれか。コンビニや路上に停めると
    クレームになる（ローリーは元売りの看板を背負って走っている）。
    需要家の構内も、荷役中は労働時間なので休憩に算入できない。

    危ないのは「高速に乗らない便で連続運転が4時間を超える」形。
    いまの15軒の名簿では発火しないが、名簿が増えて1便5〜6軒の郊外ループが
    出てくると境目に来る。**発火しない検出器は、壊れていても分からない。**
    """
    from core.domain import LegClass
    from core.solver import Stop, Trip, VehicleDay, _break_location_risk
    from core.travel import TravelMatrix

    customers = generate(seed=1).customers[:2]
    n = 3
    arterial = tuple(tuple(LegClass.ARTERIAL for _ in range(n)) for _ in range(n))
    express = tuple(
        tuple(LegClass.EXPRESSWAY if (i, j) == (0, 1) else LegClass.ARTERIAL
              for j in range(n))
        for i in range(n)
    )
    zeros = tuple(tuple(0 for _ in range(n)) for _ in range(n))
    zerosf = tuple(tuple(0.0 for _ in range(n)) for _ in range(n))

    def matrix(leg):
        return TravelMatrix(
            labels=("depot",) + tuple(c.name for c in customers),
            minutes=zeros, km=zerosf, leg_class=leg,
        )

    def day(drive_min):
        stops = tuple(
            Stop(customer_id=c.id, customer_name=c.name, arrive_min=0,
                 depart_min=0, delivered_l=1000,
                 window_open_min=c.window_open_min,
                 window_close_min=c.window_close_min)
            for c in customers
        )
        trip = Trip(
            vehicle_id="v1", vehicle_name="1号車", trip_index=1,
            load_start_min=0, depart_min=0, return_min=drive_min,
            drive_min=drive_min, load_l=2000, capacity_l=16000, stops=stops,
        )
        return [VehicleDay(
            vehicle_id="v1", vehicle_name="1号車", driver_id="d1",
            driver_name="乗務員A", duty_start_min=0, duty_end_min=drive_min,
            drive_min=drive_min, break_min=60, trips=(trip,),
        )]

    # 4時間以内 → 油槽所へ戻って取れる
    assert _break_location_risk(day(200), matrix(arterial), customers, 240) == []
    # 4時間超 かつ 一般道だけ → 停める場所が無い
    risk = _break_location_risk(day(300), matrix(arterial), customers, 240)
    assert len(risk) == 1 and risk[0]["vehicle"] == "v1"
    # 4時間超 だが 高速に乗る → SA/PA が使える
    assert _break_location_risk(day(300), matrix(express), customers, 240) == []


# ---------------------------------------------------------------------------
# 再現性（SolveMode）
# ---------------------------------------------------------------------------


def test_reproducible_mode_never_sets_a_wall_clock_limit():
    """REPRODUCIBLE では実時間の上限を置かない。並列数も固定する。

    **「念のため」の安全網が決定性を壊した。** 決定的時間と実時間を両方
    渡すと、実時間の側が探索に干渉し、同じ設定・同じマシンの2回で別の
    計画が返った（2026-09-10）。善意で上限を足し直す変更をここで止める。

    並列数を固定するのは、並列数が違えば探索が違い別の計画になるから。
    環境変数（SOLVE_WORKERS）や vCPU 数に引きずられてはいけない。
    """
    from core.solver import (
        DETERMINISTIC_BUDGET,
        REPRODUCIBLE_WORKERS,
        SolveMode,
        _new_solver,
    )

    s = _new_solver(SolveMode.REPRODUCIBLE, time_limit_s=5.0, workers=2, seed=1, log=False)
    assert s.parameters.max_time_in_seconds > 1e9, "再現モードに実時間の上限が入っている"
    assert s.parameters.max_deterministic_time == DETERMINISTIC_BUDGET
    assert s.parameters.interleave_search
    assert s.parameters.num_workers == REPRODUCIBLE_WORKERS, "並列数が引数に引きずられている"

    f = _new_solver(SolveMode.FAST, time_limit_s=5.0, workers=2, seed=1, log=False)
    assert f.parameters.max_time_in_seconds == 5.0
    assert f.parameters.num_workers == 2


def _signature(plan):
    """計画そのもの。どの車がどの軒をどの順で回るか。"""
    return tuple(
        (vd.vehicle_id, tuple(tuple(st.customer_id for st in t.stops) for t in vd.trips))
        for vd in sorted(plan.vehicle_days, key=lambda x: x.vehicle_id)
    )


def _solve_reproducible(seed):
    from core.solver import SolveMode

    s = generate(seed=seed, season="winter")
    s.vehicles = tuple(v for v in s.vehicles if v.id in {"v1", "v4", "v5"})
    return solve(s, matrix=build_matrix(s.depot, s.customers), mode=SolveMode.REPRODUCIBLE)


def test_reproducible_mode_returns_the_same_plan_twice():
    """同じ入力を2回解いて、同じ計画が返ること。フラグもそう申告すること。"""
    a = _solve_reproducible(20261015)
    b = _solve_reproducible(20261015)
    assert a.status in ("OPTIMAL", "FEASIBLE")
    assert _signature(a) == _signature(b)
    assert a.objective == b.objective
    assert a.diagnostics["reproducible"] is True
    assert a.diagnostics["reproducible_basis"] == "deterministic_search"


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_reproducible_mode_holds_without_an_optimality_proof():
    """**最適性を証明できない日でも、同じ計画が返ること。** これが本丸。

    seed 20261016（3台）は、実時間なら90秒かけても1段目の最適性を証明
    できない日。以前の「両段が最適なら再現できる」では、この日は再現を
    保証できなかった。探索量で打ち切れば、証明の有無と再現性は無関係になる。
    """
    a = _solve_reproducible(20261016)
    b = _solve_reproducible(20261016)
    assert _signature(a) == _signature(b)
    assert a.objective == b.objective
    assert a.diagnostics["tiebreak"] == b.diagnostics["tiebreak"]



# ---------------------------------------------------------------------------
# 受入見送り
# ---------------------------------------------------------------------------


def test_declined_customer_is_never_visited():
    """受入を見送った需要家には、発注があっても配車しない。

    **断られても配車してしまう問題**の回帰テスト。見送った先は「回れなかった
    必須訪問」にも数えない（行けなかったのではなく、行かないのが正しい）。
    代わりに診断と説明文で見送りを知らせる。
    """
    base = generate(seed=20261015, season="winter")
    target = next(c for c in base.customers if c.mandatory_basis == "order")

    s = generate(seed=20261015, season="winter", declined_customers=(target.id,))
    s.vehicles = tuple(v for v in s.vehicles if v.id in {"v1", "v4", "v5"})
    p = solve(s, matrix=build_matrix(s.depot, s.customers), time_limit_s=TEST_TIME_LIMIT_S)

    visited = {st.customer_id for vd in p.vehicle_days for t in vd.trips for st in t.stops}
    assert target.id not in visited
    assert target.id not in {u.customer_id for u in p.mandatory_missed}
    assert target.id not in p.diagnostics["mandatory_basis"]
    assert [d["customer_id"] for d in p.diagnostics["declined"]] == [target.id]
    assert "受入見送り" in format_plan_ja(p)
