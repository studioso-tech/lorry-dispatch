"""決定論コア: 単槽 CVRPTW ＋ 乗務員拘束時間（v0）。

この層の役割:
    **数字を出すのはここだけ。**エージェント層（W4）は「何を計算させるか」
    「いつ組み直すか」「どう説明するか」を決めるが、所要時間も積載量も
    到着時刻も一切計算しない。散文と数字が食い違う事故は、モデルに算術を
    させたときにしか起きない。

v0 の範囲（M1a）:
    - 容量制約つき配送計画（CVRP）
    - 需要家の受入時間枠（TW）
    - 乗務員の拘束時間・運転時間・連続運転の休憩（改善基準告示）
    - 発注点割れの需要家は必須訪問、それ以外は任意（切迫度で重み付け）

v0 に**まだ無いもの**（意図的に外してある）:
    - 多コンパートメント割当（槽ごとに単一油種）        → W2 / M1b
    - 油槽所の積込回数・回転（1台1日1便に固定）        → W2 / M1b
    - 有向の時間帯係数、高速 vs 一般道、p80 で解く      → W3 / M2
    ここで単槽に落としてあるのは手抜きではなく、W2 で多槽に上げたときに
    「多槽にしたせいで解けなくなった」のか「元から解けていなかった」のかを
    切り分けられるようにするため。単槽の解が基準線になる。

なぜ CP-SAT か:
    時間枠・容量・拘束時間・必須訪問が混ざった問題で、しかも
    「解けなかった理由を説明する」必要がある。CP-SAT は AddCircuit で
    経路を素直に書け、目的関数に説明可能な項（未訪問のペナルティ、
    車両の固定費）を並べられる。ルーティング専用ソルバーより、
    後から制約を1本足すのが安い。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from core.domain import Customer, Scenario, minutes_to_hhmm
from core.travel import TravelMatrix, build_matrix

# ---------------------------------------------------------------------------
# 目的関数の重み
#
# 単位はすべて「分」に揃えてある。混ざった単位を重みで無理やり釣り合わせると、
# 重みを1つ動かしたときに何が起きるか誰にも説明できなくなる。
# 「この需要家を回るために何分の遠回りまで許すか」で読める形にしてある。
# ---------------------------------------------------------------------------

#: 必須訪問（発注点割れ）を落としたときの罰。実質的な禁止。
#: それでも有限にしてあるのは、**解無しで黙るより「3軒回れなかった」と
#: 言えるほうが役に立つ**から。エージェントはこれを読んで代替案を出す。
PENALTY_MANDATORY_MIN = 100_000

#: 任意訪問を落としたときの罰の基準値。240分＝「4時間の遠回りまでなら
#: 今日ついでに回ったほうが得」という判断。
PENALTY_OPTIONAL_BASE_MIN = 240

#: 切迫度による上乗せ。あと何日で空になるかが3日を切ると効き始める。
PENALTY_URGENCY_PER_DAY_MIN = 90
URGENCY_HORIZON_DAYS = 3.0

#: ローリーを1台出すことの固定費（分換算）。人件費と車両費の代理。
#: これが無いと、遠回りが1分でも減るなら5台全部出す計画が出る。
VEHICLE_FIXED_COST_MIN = 90

#: 拘束時間1分あたりの重み。
#:
#: **これが無いと帰庫時刻が意味を失う。** 目的関数が走行時間しか見ていないと、
#: 帰着時刻の変数は「拘束13時間の上限までなら何時でもよい」ことになり、
#: 10:48 に最後の荷卸を終えた車の帰庫が 17:05 と出る。制約としては正しいが、
#: 配車係が読む表としては嘘である。
#:
#: 重み1は「待ち時間も人件費」という意味。走行はこれに加えて燃料と車両摩耗が
#: かかるので、走行1分のほうが拘束1分より重い（走行は両方に計上される）。
DUTY_COST_PER_MIN = 1

#: 主目的をこの倍率で持ち上げ、下位の桁をタイブレークに使う。
#:
#: **なぜ必要か。** 同点の最適解が複数あるとき、CP-SAT の並列探索は
#: そのうちどれを返すかを保証しない。目的値は毎回同じなのに、どの車が
#: どの需要家を回るかが実行のたびに変わる。実際に、同じ要求を2回投げると
#: 別の割当が返ってきた。
#:
#: これは審査上そのまま問題になる。提出要件は「審査員が動かせる状態」であり、
#: 手元で再現できない計画は説明の裏づけにならない。LM-as-Judge の回帰テスト
#: （W6）も、計画が毎回変わるなら品質の劣化を検出できない。
#:
#: 対策として、ワーカー数を1に落とす案は捨てた。単一ワーカーでは
#: seed=20261015 が60秒かけても最適性を証明できず、時間切れによる
#: 別種の非決定性を招く。代わりに**最適解を一意に近づける**。
#:
#: CP-SAT の interleave_search=True でも決定性は得られた（10 seed × 3回で
#: 完全一致）が、求解が 0.3〜8秒から 7〜20秒に鈍る。タイブレークで
#: 同じ決定性が得られたので採らなかった。目的関数側で閉じるほうが、
#: なぜその解が選ばれたかを説明できる点でも優る。
OBJECTIVE_SCALE = 1_000_000


@dataclass(frozen=True)
class Stop:
    """1つの立ち寄り。"""

    customer_id: str
    customer_name: str
    arrive_min: int
    depart_min: int
    delivered_l: int
    window_open_min: int
    window_close_min: int

    @property
    def slack_min(self) -> int:
        """締切までの余裕。ここが薄い立ち寄りが、W3 で p95 警告の対象になる。"""
        return self.window_close_min - self.arrive_min


@dataclass(frozen=True)
class Route:
    """ローリー1台ぶんの1日。"""

    vehicle_id: str
    vehicle_name: str
    driver_id: str
    driver_name: str
    load_start_min: int      # 積込開始＝拘束の始まり
    depart_min: int          # 油槽所を出る時刻
    return_min: int          # 油槽所に戻る時刻＝拘束の終わり
    drive_min: int           # 運転時間の合計
    service_min: int         # 荷卸時間の合計
    break_min: int           # 連続運転の上限で要る休憩
    load_l: int              # 積んだ量
    capacity_l: int
    stops: tuple[Stop, ...]

    @property
    def duty_min(self) -> int:
        """拘束時間。積込開始から帰着まで。"""
        return self.return_min - self.load_start_min

    @property
    def utilization(self) -> float:
        """積載率。低い便が並ぶなら台数を減らせる余地がある。"""
        return self.load_l / self.capacity_l if self.capacity_l else 0.0


@dataclass(frozen=True)
class Unserved:
    """今日回れなかった需要家と、その重さ。

    これを黙って捨てないことが、このシステムの存在理由に直結する。
    「回れなかった」ではなく「回れなかった、あと1.4日で切れる、
    代替はこれ」まで言えて初めて配車係の役に立つ。
    """

    customer_id: str
    customer_name: str
    demand_l: int
    mandatory: bool
    days_to_empty: float


@dataclass
class Plan:
    """求解の結果一式。エージェント層と画面はこれしか見ない。"""

    status: str                      # OPTIMAL / FEASIBLE / INFEASIBLE / UNKNOWN
    routes: tuple[Route, ...]
    unserved: tuple[Unserved, ...]
    total_drive_min: int
    objective: int
    wall_time_s: float
    scenario_date: str
    scenario_seed: int
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        """配車係に出してよい解か。"""
        return self.status in ("OPTIMAL", "FEASIBLE")

    @property
    def mandatory_missed(self) -> tuple[Unserved, ...]:
        """必須訪問なのに落ちたもの。ここが空でないなら計画は失敗している。"""
        return tuple(u for u in self.unserved if u.mandatory)

    def summary_ja(self) -> str:
        """配車係の言葉での1行要約。OR の用語は出さない。"""
        if not self.is_usable:
            return "今日の配車を組めませんでした。制約を見直してください。"
        vehicles = len(self.routes)
        stops = sum(len(r.stops) for r in self.routes)
        missed = len(self.mandatory_missed)
        head = (
            f"{vehicles}台で{stops}軒。走行は合計"
            f"{self.total_drive_min // 60}時間{self.total_drive_min % 60}分。"
        )
        if missed:
            return head + f" ただし在庫が切れかけの{missed}軒に本日は届けられません。"
        return head + " 在庫が切れかけの需要家はすべて回れます。"


def _optional_penalty(customer: Customer) -> int:
    """任意訪問を落としたときの罰。切迫しているほど重い。

    切迫度は「一番早く切れるタンクがあと何日もつか」。3日を切ったところから
    1日あたり90分ぶん重くなる。0日（すでに空）なら基準値＋270分。
    """
    if customer.demand_l <= 0:
        return 0
    days = customer.min_days_to_empty()
    if math.isinf(days):
        urgency = 0.0
    else:
        urgency = max(0.0, URGENCY_HORIZON_DAYS - days)
    return PENALTY_OPTIONAL_BASE_MIN + int(round(urgency * PENALTY_URGENCY_PER_DAY_MIN))


def solve(
    scenario: Scenario,
    *,
    matrix: TravelMatrix | None = None,
    time_limit_s: float = 30.0,
    workers: int = 8,
    log: bool = False,
) -> Plan:
    """1日ぶんの配車を解く。

    Args:
        scenario: 生成器またはエージェント層が用意した当日の問題。
        matrix: 所要時間行列。省略時は幾何モデル（v0）で組む。
            W2 で実測のスナップショットを渡すようになるが、ここは変えない。
        time_limit_s: 求解の打ち切り。デモの応答性のために既定30秒。
        workers: 並列探索数。Cloud Run の vCPU に合わせる。
        log: CP-SAT の探索ログを出すか。障害調査用。

    Returns:
        Plan。解けなかった場合も status と診断を入れて必ず返す。
        例外で落とさないのは、エージェントが「なぜ組めないか」を
        配車係に説明する必要があるため。
    """
    depot = scenario.depot
    customers = scenario.customers
    vehicles = scenario.usable_vehicles()
    matrix = matrix or build_matrix(depot, customers)

    n = len(customers) + 1          # 0 = 油槽所
    travel = matrix.minutes
    horizon = depot.close_min

    diagnostics: dict[str, object] = {
        "fleet_capacity_l": scenario.fleet_capacity_l,
        "mandatory_demand_l": scenario.mandatory_demand_l,
        "total_demand_l": sum(c.demand_l for c in customers),
        "mandatory_count": sum(1 for c in customers if c.is_mandatory),
        "usable_vehicles": [v.id for v in vehicles],
        "absent_drivers": [d.id for d in scenario.drivers if not d.available],
    }

    if not vehicles:
        # 車が1台も出せない。解こうとするだけ無駄なので、理由をつけて返す。
        return Plan(
            status="INFEASIBLE",
            routes=(),
            unserved=tuple(
                Unserved(c.id, c.name, c.demand_l, c.is_mandatory, c.min_days_to_empty())
                for c in customers
                if c.demand_l > 0
            ),
            total_drive_min=0,
            objective=0,
            wall_time_s=0.0,
            scenario_date=scenario.date,
            scenario_seed=scenario.seed,
            diagnostics={**diagnostics, "reason": "出せるローリーが無い（乗務員欠勤または資格不足）"},
        )

    model = cp_model.CpModel()
    K = len(vehicles)

    # --- 変数 --------------------------------------------------------------
    # visit[k][i]: 車 k が節点 i を訪れる（i=0 は油槽所＝その車を使う）
    visit = [[model.NewBoolVar(f"visit_k{k}_n{i}") for i in range(n)] for k in range(K)]
    used = [visit[k][0] for k in range(K)]

    # x[k][(i,j)]: 車 k が i→j を走る
    x: list[dict[tuple[int, int], cp_model.IntVar]] = [{} for _ in range(K)]

    # 時刻。t[k][i] は「車 k が節点 i に着く時刻」。
    # 車ごとに持つのは、同じ需要家を別の車が別の時刻に訪れる可能性を
    # 型の上で許してしまわないため（訪問は1回に制約するが、時間の整合を
    # 車をまたいで共有すると、未訪問の車の時刻変数が経路を歪める）。
    t = [
        [
            model.NewIntVar(
                depot.open_min if i == 0 else customers[i - 1].window_open_min,
                horizon if i == 0 else customers[i - 1].window_close_min,
                f"t_k{k}_n{i}",
            )
            for i in range(n)
        ]
        for k in range(K)
    ]
    depart = [
        model.NewIntVar(depot.open_min + depot.load_min, horizon, f"depart_k{k}")
        for k in range(K)
    ]
    ret = [model.NewIntVar(depot.open_min, horizon, f"return_k{k}") for k in range(K)]

    # --- 経路（AddCircuit） -------------------------------------------------
    #
    # 自己ループのリテラルに visit の否定を置くと、
    # 「訪れない節点は circuit から外れる」が素直に書ける。
    # 全節点が自己ループなら空の circuit として成立するので、
    # 使わない車もそのまま表現できる。
    for k in range(K):
        arcs: list[tuple[int, int, cp_model.IntVar]] = []
        for i in range(n):
            arcs.append((i, i, visit[k][i].Not()))
            for j in range(n):
                if i == j:
                    continue
                lit = model.NewBoolVar(f"x_k{k}_{i}_{j}")
                x[k][(i, j)] = lit
                arcs.append((i, j, lit))
        model.AddCircuit(arcs)

    # --- 訪問の一意性 ------------------------------------------------------
    #
    # 必須も任意も「served」という同じ形にして、落ちたときの罰の重さだけを
    # 変える。必須を hard 制約にすると、1軒でも届かない日に解無しになり、
    # 「何軒がどう届かないのか」を説明できなくなる。
    served: list[cp_model.IntVar] = []
    for idx, c in enumerate(customers, start=1):
        s = model.NewBoolVar(f"served_{c.id}")
        served.append(s)
        model.Add(sum(visit[k][idx] for k in range(K)) == s)
        if c.demand_l <= 0:
            # 満タンの需要家に行く意味はない。回れるのに回らない、を明示する。
            model.Add(s == 0)

    # --- 容量 --------------------------------------------------------------
    load = []
    for k, v in enumerate(vehicles):
        l = model.NewIntVar(0, v.capacity_l, f"load_k{k}")
        model.Add(
            l == sum(customers[i - 1].demand_l * visit[k][i] for i in range(1, n))
        )
        load.append(l)

    # --- 時間の連鎖 --------------------------------------------------------
    for k in range(K):
        for (i, j), lit in x[k].items():
            if i == 0:
                # 油槽所を出て最初の1軒へ
                model.Add(t[k][j] >= depart[k] + travel[0][j]).OnlyEnforceIf(lit)
            elif j == 0:
                # 最後の1軒から帰着
                model.Add(
                    ret[k] >= t[k][i] + customers[i - 1].service_min + travel[i][0]
                ).OnlyEnforceIf(lit)
            else:
                model.Add(
                    t[k][j] >= t[k][i] + customers[i - 1].service_min + travel[i][j]
                ).OnlyEnforceIf(lit)

    # --- 乗務員の拘束時間・運転時間 ----------------------------------------
    drive = []
    breaks = []
    for k, v in enumerate(vehicles):
        d = scenario.driver(v.driver_id)

        drive_k = model.NewIntVar(0, d.max_drive_min, f"drive_k{k}")
        model.Add(drive_k == sum(travel[i][j] * lit for (i, j), lit in x[k].items()))
        drive.append(drive_k)

        service_k = sum(customers[i - 1].service_min * visit[k][i] for i in range(1, n))

        # 連続運転の休憩（改善基準告示: 連続4時間で30分）。
        #
        # v0 の近似: 休憩を経路上のどこに挟むかは決めず、「運転時間が
        # 4時間の区切りを越えるごとに30分ぶんの余裕を1日の中に確保する」
        # という形で拘束時間に効かせる。挟む位置まで最適化するのは
        # W2 で多便化するときに一緒に入れる。
        #
        # 近似の向きは**安全側**にしてある。余裕を先に取るので、実際に
        # 休憩を挟んでも計画が破れない。逆向き（余裕を取らない）にすると
        # 法令上走れない計画が「解けた」ことになる。
        n_breaks = model.NewIntVar(0, 3, f"nbreak_k{k}")
        over1 = model.NewBoolVar(f"over1_k{k}")
        over2 = model.NewBoolVar(f"over2_k{k}")
        limit = d.continuous_drive_limit_min
        model.Add(drive_k > limit).OnlyEnforceIf(over1)
        model.Add(drive_k <= limit).OnlyEnforceIf(over1.Not())
        model.Add(drive_k > 2 * limit).OnlyEnforceIf(over2)
        model.Add(drive_k <= 2 * limit).OnlyEnforceIf(over2.Not())
        model.Add(n_breaks == over1 + over2)
        break_k = model.NewIntVar(0, 3 * d.break_min, f"break_k{k}")
        model.AddMultiplicationEquality(break_k, [n_breaks, d.break_min])
        breaks.append(break_k)

        # 経過時間は「運転＋荷卸＋休憩」を必ず飲み込めるだけ空けておく。
        model.Add(ret[k] - depart[k] >= drive_k + service_k + break_k).OnlyEnforceIf(
            used[k]
        )
        # 拘束時間は積込開始から帰着まで。上限は改善基準告示の13時間。
        model.Add(ret[k] - depart[k] + depot.load_min <= d.max_duty_min)

        # 使わない車の時刻を遊ばせない。目的関数に効かない変数が動くと
        # 解が毎回変わって見え、再現性の確認が難しくなる。
        model.Add(depart[k] == depot.open_min + depot.load_min).OnlyEnforceIf(used[k].Not())
        model.Add(ret[k] == depot.open_min + depot.load_min).OnlyEnforceIf(used[k].Not())
        model.Add(ret[k] >= depart[k])

    # --- 目的関数 ----------------------------------------------------------
    penalties = []
    for idx, c in enumerate(customers):
        p = PENALTY_MANDATORY_MIN if c.is_mandatory else _optional_penalty(c)
        if p:
            penalties.append(p * (1 - served[idx]))

    # 拘束時間（積込開始〜帰着）。走行と別に計上する。
    duty = []
    for k in range(K):
        d_k = model.NewIntVar(0, 24 * 60, f"duty_k{k}")
        model.Add(d_k == ret[k] - depart[k] + depot.load_min).OnlyEnforceIf(used[k])
        model.Add(d_k == 0).OnlyEnforceIf(used[k].Not())
        duty.append(d_k)

    primary = model.NewIntVar(0, 10**9, "primary")
    model.Add(
        primary
        == sum(drive)
        + DUTY_COST_PER_MIN * sum(duty)
        + sum(penalties)
        + VEHICLE_FIXED_COST_MIN * sum(used)
    )

    # --- タイブレーク（決定性のため） --------------------------------------
    #
    # 主目的が同点のとき、どれを返すかをここで一意に決める。2項ある。
    #
    # 1. 到着時刻の合計を最小化する。
    #    余裕のある立ち寄りは「窓の中ならいつでもよい」ので時刻が浮く。
    #    早い側に寄せる。副作用も望ましい方向で、着けるうちに着いておく計画に
    #    なり、W3 で確率的な所要時間を入れたときに時間枠を割りにくくなる。
    #
    # 2. 車と需要家の組を、車番と需要家番号の**積**で決める。
    #    1 でも同点なら、若い番号の車が若い番号の需要家を持つ側を選ぶ。
    #    意味のある基準ではないが、**基準が無いことよりはるかにまし**。
    #
    #    積にしてあるのが要点。最初は (k*100 + i) と書いていたが、これは
    #    k の項と i の項に分離できるため、2軒を2台の間で入れ替えても総和が
    #    変わらない。3号車と4号車は槽構成も乗務員の上限も同じで完全に対称
    #    なので、実際にこの入れ替えが実行のたびに起きていた。
    #    (k+1)*i^2 なら、入れ替えで必ず値が動く。
    arrivals = []
    for k in range(K):
        for i in range(1, n):
            a = model.NewIntVar(0, horizon, f"arr_k{k}_n{i}")
            model.Add(a == t[k][i]).OnlyEnforceIf(visit[k][i])
            model.Add(a == 0).OnlyEnforceIf(visit[k][i].Not())
            arrivals.append(a)

    lexical = sum(
        (k + 1) * (i * i) * visit[k][i] for k in range(K) for i in range(1, n)
    )

    # 3. 回る順序を決める。
    #    2 は「どの車がどの需要家を持つか」しか見ておらず、**同じ2軒を回る順**は
    #    区別できない。実際に、5号車が周東→玖珂と玖珂→周東の間で揺れていた
    #    （到着時刻の集合まで同じで、目的値もタイブレークも一致していた）。
    #
    #    重みを i*i*j にしてあるのは、経路が閉じた輪だから。
    #    i+j や i*j のような対称・分離可能な重みは、輪を一周すると
    #    どの順で回っても総和が同じになり、順序を区別できない。
    arc_order = sum(
        (i * i * j) * lit for k in range(K) for (i, j), lit in x[k].items()
    )

    tiebreak = model.NewIntVar(0, OBJECTIVE_SCALE - 1, "tiebreak")
    model.Add(tiebreak == sum(arrivals) + lexical + arc_order)

    model.Minimize(primary * OBJECTIVE_SCALE + tiebreak)

    # --- 求解 --------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = log
    # 同じ入力で同じ解が出ることを優先する。並列探索は既定で非決定的なので、
    # 明示的に決定性を要求する（回帰テストの前提）。
    solver.parameters.random_seed = scenario.seed % 2_147_483_647
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Plan(
            status=status_name,
            routes=(),
            unserved=tuple(
                Unserved(c.id, c.name, c.demand_l, c.is_mandatory, c.min_days_to_empty())
                for c in customers
                if c.demand_l > 0
            ),
            total_drive_min=0,
            objective=0,
            wall_time_s=solver.WallTime(),
            scenario_date=scenario.date,
            scenario_seed=scenario.seed,
            diagnostics={**diagnostics, "reason": "制約を同時に満たす計画が見つからない"},
        )

    # --- 解の取り出し ------------------------------------------------------
    routes: list[Route] = []
    for k, v in enumerate(vehicles):
        if not solver.BooleanValue(used[k]):
            continue
        d = scenario.driver(v.driver_id)

        # 経路を油槽所から辿る。AddCircuit なので後続は一意に決まる。
        order: list[int] = []
        cur = 0
        for _ in range(n):
            nxt = None
            for j in range(n):
                if j != cur and solver.BooleanValue(x[k][(cur, j)]):
                    nxt = j
                    break
            if nxt is None or nxt == 0:
                break
            order.append(nxt)
            cur = nxt

        stops = tuple(
            Stop(
                customer_id=customers[i - 1].id,
                customer_name=customers[i - 1].name,
                arrive_min=solver.Value(t[k][i]),
                depart_min=solver.Value(t[k][i]) + customers[i - 1].service_min,
                delivered_l=customers[i - 1].demand_l,
                window_open_min=customers[i - 1].window_open_min,
                window_close_min=customers[i - 1].window_close_min,
            )
            for i in order
        )

        routes.append(
            Route(
                vehicle_id=v.id,
                vehicle_name=v.name,
                driver_id=d.id,
                driver_name=d.name,
                load_start_min=solver.Value(depart[k]) - depot.load_min,
                depart_min=solver.Value(depart[k]),
                return_min=solver.Value(ret[k]),
                drive_min=solver.Value(drive[k]),
                service_min=sum(s.depart_min - s.arrive_min for s in stops),
                break_min=solver.Value(breaks[k]),
                load_l=solver.Value(load[k]),
                capacity_l=v.capacity_l,
                stops=stops,
            )
        )

    unserved = tuple(
        Unserved(
            customer_id=c.id,
            customer_name=c.name,
            demand_l=c.demand_l,
            mandatory=c.is_mandatory,
            days_to_empty=c.min_days_to_empty(),
        )
        for idx, c in enumerate(customers)
        if c.demand_l > 0 and not solver.BooleanValue(served[idx])
    )

    return Plan(
        status=status_name,
        routes=tuple(routes),
        unserved=unserved,
        total_drive_min=sum(r.drive_min for r in routes),
        # 外へ出すのは主目的だけ。タイブレークは決定性のための内部事情であって、
        # 「この計画のコスト」として読まれると桁が意味を失う。
        objective=solver.Value(primary),
        wall_time_s=solver.WallTime(),
        scenario_date=scenario.date,
        scenario_seed=scenario.seed,
        diagnostics={
            **diagnostics,
            "served_count": len(customers) - len(unserved),
            "tiebreak": solver.Value(tiebreak),
            "best_bound": solver.BestObjectiveBound() / OBJECTIVE_SCALE,
        },
    )


def format_plan_ja(plan: Plan) -> str:
    """計画を配車係の言葉で書き出す。

    OR の用語（目的関数、緩和、双対）は一切出さない。出すのは
    「何号車が、何時に出て、どこへ、何kL、何時に戻る」だけ。
    """
    lines = [f"【{plan.scenario_date} の配車】{plan.summary_ja()}", ""]
    for r in plan.routes:
        lines.append(
            f"■ {r.vehicle_name}（{r.driver_name}）"
            f"　積込 {minutes_to_hhmm(r.load_start_min)}"
            f"／出庫 {minutes_to_hhmm(r.depart_min)}"
            f"／帰庫 {minutes_to_hhmm(r.return_min)}"
            f"　拘束 {r.duty_min // 60}時間{r.duty_min % 60}分"
            f"　積載 {r.load_l / 1000:.1f}kL / {r.capacity_l / 1000:.0f}kL"
            f"（{r.utilization * 100:.0f}%）"
        )
        for i, s in enumerate(r.stops, start=1):
            lines.append(
                f"   {i}. {minutes_to_hhmm(s.arrive_min)} {s.customer_name}"
                f"　{s.delivered_l / 1000:.1f}kL"
                f"　（受入 {minutes_to_hhmm(s.window_open_min)}"
                f"〜{minutes_to_hhmm(s.window_close_min)}"
                f"／余裕 {s.slack_min}分）"
            )
        lines.append("")

    if plan.unserved:
        lines.append("■ 今日は回れない需要家")
        for u in sorted(plan.unserved, key=lambda u: u.days_to_empty):
            mark = "【要手配】" if u.mandatory else "　　　　"
            days = "∞" if math.isinf(u.days_to_empty) else f"{u.days_to_empty:.1f}日"
            lines.append(
                f"   {mark}{u.customer_name}　必要 {u.demand_l / 1000:.1f}kL"
                f"　在庫あと {days}"
            )
    return "\n".join(lines)
