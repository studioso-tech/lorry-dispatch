"""決定論コア: 複数便 CVRPTW ＋ 乗務員拘束時間。

この層の役割:
    **数字を出すのはここだけ。**エージェント層は「何を計算させるか」
    「いつ組み直すか」「どう説明するか」を決めるが、所要時間も積載量も
    到着時刻も一切計算しない。散文と数字が食い違う事故は、モデルに算術を
    させたときにしか起きない。

いま解けること:
    - 容量制約つき配送計画（CVRP）
    - 需要家の受入時間枠（TW）
    - 乗務員の拘束時間・運転時間・連続運転の休憩（改善基準告示）
    - 発注点割れの需要家は必須訪問、それ以外は任意（切迫度で重み付け）
    - 1台が1日に複数便を回る（油槽所へ戻って積み直す）

まだ無いもの:
    - 車両と乗務員の分離（二交代）。いまは1台に1人が固定されている
    - 有人／無人荷卸しによる受入枠の分離
    - 多コンパートメント割当（槽ごとに単一油種）
    - 有向の時間帯係数、高速 vs 一般道、p80 で解く

なぜ CP-SAT か:
    時間枠・容量・拘束時間・必須訪問が混ざった問題で、しかも
    「解けなかった理由を説明する」必要がある。CP-SAT は AddCircuit で
    経路を素直に書け、目的関数に説明可能な項（未訪問のペナルティ、
    車両の固定費）を並べられる。ルーティング専用ソルバーより、
    後から制約を1本足すのが安い。

モデルの組み方（**油槽所の節点を複製して1台1周回にする**）:
    最初は「便ごとに輪を1本」張っていた。5台×2便で10本の輪になり、
    輪はCP-SATの中で最も重い大域制約なので、そのまま探索の重さになった。
    60秒かけても最適性を証明できない日があった。

    いまは油槽所の節点を便の数だけ複製し、**1台につき輪を1本**張る。

        d0 → 需要家 → 需要家 → d1 → 需要家 → d0
        └─── 1便目 ───┘       └─ 2便目 ─┘

    複製した油槽所 d_r が r 便目の出発点になり、次の複製に着いた時点で
    前の便が終わる。輪の本数が台数と同じになり、時刻変数も半分になる。

    積載量は「便ごと」に効かせる必要があるが、便の区切りを表す変数を
    別に持たなくても、**積み荷の累積量を経路に沿って伝播させる**だけで済む。
    油槽所の複製を出た直後に累積がリセットされるので、複製をまたぐと
    自然に別便になる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from core.domain import Customer, Scenario, Vehicle, minutes_to_hhmm
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
DUTY_COST_PER_MIN = 1

#: 1台が1日に回れる便の上限。油槽所の節点をこの数だけ複製する。
MAX_TRIPS_PER_VEHICLE = 2

#: 既定の求解時間。
DEFAULT_TIME_LIMIT_S = 60.0

# 2段目を速くする試みで、**採らなかった**案を2つ記録しておく。どちらも
# 一度は良さそうに見えたので、次に同じ道を通らないために残す。
#
# 案A: 2段目を決定的時間（max_deterministic_time）＋ interleave_search で
#   切り、最適性の証明を省く。理屈は通っていた——2段目の入力は「1段目が
#   出した目的値」という数字1つだけで、1段目がどの解を見つけたかには
#   依存しないので、2段目の探索が決定的なら計画全体が決定的になる。
#   実際には、決定的時間の枠内で実行可能解を1つも見つけられない日があり
#   （UNKNOWN）、1段目の解に落ちて決定性が崩れた。
#
# 案B: 案Aに加えて、1段目の解を初期解（AddHint）として渡す。UNKNOWN は
#   消えたが、**2段目の結果が1段目の解に依存するようになり**、案Aの
#   決定性の根拠そのものを壊した。5件中3件で計画が一致しなくなり、しかも
#   reproducible が真を返しながら計画が違う、という最悪の形になった。
#
# よって2段目は最適性を証明させる。そのぶん時間はかかるが、
# **フラグが嘘をつかない**ことのほうが重要である。


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
        """締切までの余裕。ここが薄い立ち寄りが、M2 で p95 警告の対象になる。"""
        return self.window_close_min - self.arrive_min


@dataclass(frozen=True)
class Trip:
    """1便。油槽所を出て、何軒か回って、油槽所へ帰るまで。"""

    vehicle_id: str
    vehicle_name: str
    trip_index: int          # 0 が1便目
    load_start_min: int      # 積込開始
    depart_min: int          # 油槽所を出る時刻
    return_min: int          # 油槽所に戻る時刻
    drive_min: int           # この便の運転時間
    load_l: int              # この便で積んだ量
    capacity_l: int
    stops: tuple[Stop, ...]

    @property
    def service_min(self) -> int:
        """この便の荷卸時間の合計。"""
        return sum(s.depart_min - s.arrive_min for s in self.stops)

    @property
    def elapsed_min(self) -> int:
        """積込開始から帰庫まで。1便の長さ。"""
        return self.return_min - self.load_start_min

    @property
    def utilization(self) -> float:
        """この便の積載率。"""
        return self.load_l / self.capacity_l if self.capacity_l else 0.0


@dataclass(frozen=True)
class VehicleDay:
    """1台の1日。便をまとめて、拘束時間と運転時間をここで見る。

    上限は便ではなく人にかかる。1便ずつ見ていると、3便で拘束15時間の
    計画が「どの便も短いから大丈夫」に見えてしまう。
    """

    vehicle_id: str
    vehicle_name: str
    driver_id: str
    driver_name: str
    duty_start_min: int      # 最初の便の積込開始＝拘束の始まり
    duty_end_min: int        # 最後の便の帰庫＝拘束の終わり
    drive_min: int           # 1日の運転時間の合計
    break_min: int           # 連続運転の上限で要る休憩
    trips: tuple[Trip, ...]

    @property
    def duty_min(self) -> int:
        """1日の拘束時間。"""
        return self.duty_end_min - self.duty_start_min

    @property
    def load_l(self) -> int:
        """1日に運んだ量の合計。"""
        return sum(t.load_l for t in self.trips)

    @property
    def stop_count(self) -> int:
        return sum(len(t.stops) for t in self.trips)


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
    vehicle_days: tuple[VehicleDay, ...]
    unserved: tuple[Unserved, ...]
    total_drive_min: int
    objective: int
    wall_time_s: float
    scenario_date: str
    scenario_seed: int
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def trips(self) -> tuple[Trip, ...]:
        """全便を平らに並べたもの。"""
        return tuple(t for vd in self.vehicle_days for t in vd.trips)

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
        vehicles = len(self.vehicle_days)
        trips = len(self.trips)
        stops = sum(vd.stop_count for vd in self.vehicle_days)
        missed = len(self.mandatory_missed)
        head = (
            f"{vehicles}台・{trips}便で{stops}軒。走行は合計"
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


def _all_unserved(customers) -> tuple[Unserved, ...]:
    return tuple(
        Unserved(c.id, c.name, c.demand_l, c.is_mandatory, c.min_days_to_empty())
        for c in customers
        if c.demand_l > 0
    )


def solve(
    scenario: Scenario,
    *,
    matrix: TravelMatrix | None = None,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    workers: int = 8,
    max_trips: int = MAX_TRIPS_PER_VEHICLE,
    tiebreak_phase: bool = True,
    log: bool = False,
) -> Plan:
    """1日ぶんの配車を解く。

    Args:
        scenario: 生成器またはエージェント層が用意した当日の問題。
        matrix: 所要時間行列。省略時は幾何モデルで組む。
        time_limit_s: 求解の打ち切り。
        workers: 並列探索数。Cloud Run の vCPU に合わせる。
        max_trips: 1台が1日に回れる便の上限＝油槽所の節点を複製する数。
        tiebreak_phase: 2段目（タイブレークの最小化）を回すか。
            False にすると同じ入力でも別の計画が返り得るが、**コストと
            実行可能性は変わらない**。体制設計の掃引のように「何台で回るか」
            だけを知りたい用途では、2段目は要らないぶん丸ごと省ける。
            配車係や審査員に見せる計画では True のままにすること。
        log: CP-SAT の探索ログを出すか。障害調査用。

    Returns:
        Plan。解けなかった場合も status と診断を入れて必ず返す。
        例外で落とさないのは、エージェントが「なぜ組めないか」を
        配車係に説明する必要があるため。
    """
    depot = scenario.depot
    customers = scenario.customers
    fleet: tuple[Vehicle, ...] = scenario.usable_vehicles()
    matrix = matrix or build_matrix(depot, customers)

    C = len(customers)
    R = max_trips
    N = R + C                    # 0..R-1 が油槽所の複製、R.. が需要家
    travel = matrix.minutes
    horizon = depot.close_min
    load_min = depot.load_min

    def mat(node: int) -> int:
        """節点 → 所要時間行列の添字。油槽所の複製はすべて 0 を指す。"""
        return 0 if node < R else node - R + 1

    def cust(node: int) -> Customer:
        return customers[node - R]

    diagnostics: dict[str, object] = {
        "fleet_capacity_l": scenario.fleet_capacity_l,
        "fleet_capacity_with_trips_l": scenario.fleet_capacity_l * R,
        "mandatory_demand_l": scenario.mandatory_demand_l,
        "total_demand_l": sum(c.demand_l for c in customers),
        "mandatory_count": sum(1 for c in customers if c.is_mandatory),
        "usable_vehicles": [v.id for v in fleet],
        "absent_drivers": [d.id for d in scenario.drivers if not d.available],
        "max_trips_per_vehicle": R,
    }

    if not fleet:
        # 車が1台も出せない。解こうとするだけ無駄なので、理由をつけて返す。
        return Plan(
            status="INFEASIBLE",
            vehicle_days=(),
            unserved=_all_unserved(customers),
            total_drive_min=0,
            objective=0,
            wall_time_s=0.0,
            scenario_date=scenario.date,
            scenario_seed=scenario.seed,
            diagnostics={
                **diagnostics,
                "reason": "出せるローリーが無い（乗務員欠勤または資格不足）",
            },
        )

    model = cp_model.CpModel()
    V = len(fleet)

    # --- 到達できない区間は最初から作らない -------------------------------
    #
    # i を受入開始と同時に着いて荷卸し、そのまま j へ向かっても j の締切に
    # 間に合わないなら、その区間は解の中に現れ得ない。変数として置いておくと
    # 探索空間だけが広がる。
    def reachable(i: int, j: int) -> bool:
        if i < R and j < R:
            return False                      # 空の便は作らない
        if j < R:
            return True                       # 帰庫はいつでもできる
        latest_j = cust(j).window_close_min
        if i < R:
            earliest = depot.open_min + load_min + travel[0][mat(j)]
        else:
            earliest = (
                cust(i).window_open_min + cust(i).service_min + travel[mat(i)][mat(j)]
            )
        return earliest <= latest_j

    # --- 変数 --------------------------------------------------------------
    # active[v][node]: 車 v の周回にその節点が含まれる。
    #   油槽所の複製 d_r が含まれる = r 便目を出す
    #   需要家が含まれる = その車がその軒を回る
    active = [[model.NewBoolVar(f"act_v{v}_n{i}") for i in range(N)] for v in range(V)]
    used = [active[v][0] for v in range(V)]      # 1便目を出す = その車を使う

    x: list[dict[tuple[int, int], cp_model.IntVar]] = [{} for _ in range(V)]

    # 時刻。油槽所の複製は「その便の出発時刻」、需要家は「到着時刻」。
    t = [
        [
            model.NewIntVar(
                depot.open_min + load_min if i < R else cust(i).window_open_min,
                horizon if i < R else cust(i).window_close_min,
                f"t_v{v}_n{i}",
            )
            for i in range(N)
        ]
        for v in range(V)
    ]
    duty_end = [model.NewIntVar(0, horizon, f"dutyend_v{v}") for v in range(V)]

    # --- 周回（1台につき1本） ----------------------------------------------
    pruned = 0
    for v, veh in enumerate(fleet):
        arcs: list[tuple[int, int, cp_model.IntVar]] = []
        for i in range(N):
            if i >= R:
                # 積載量を超える需要家には、その車では行けない
                if cust(i).demand_l > veh.capacity_l:
                    model.Add(active[v][i] == 0)
                # **トレーラーが物理的に入れない需要家がある。**
                # 狭い道・小さい構内・切り返しの余地の無さで決まるので、
                # 積載量とは別の制約。単車がそういう先へ優先的に回る。
                elif veh.is_trailer and not cust(i).accepts_trailer:
                    model.Add(active[v][i] == 0)
            arcs.append((i, i, active[v][i].Not()))
            for j in range(N):
                if i == j:
                    continue
                if not reachable(i, j):
                    pruned += 1
                    continue
                lit = model.NewBoolVar(f"x_v{v}_{i}_{j}")
                x[v][(i, j)] = lit
                arcs.append((i, j, lit))
        model.AddCircuit(arcs)
    diagnostics["pruned_arcs"] = pruned

    # --- 便は順番に使う ----------------------------------------------------
    #
    # 「1便目を出さずに2便目だけ出す」を禁じる。禁じないと同じ計画が
    # 便番号の付け替えだけで何通りにも表現でき、探索が無駄に広がる。
    # 出さない便の時刻は地平線に固定して、遊ばせない。
    for v in range(V):
        for r in range(R - 1):
            model.AddImplication(active[v][r + 1], active[v][r])
            model.Add(t[v][r] <= t[v][r + 1])
        for r in range(1, R):
            model.Add(t[v][r] == horizon).OnlyEnforceIf(active[v][r].Not())
        model.Add(t[v][0] == depot.open_min + load_min).OnlyEnforceIf(used[v].Not())
        model.Add(duty_end[v] == t[v][0]).OnlyEnforceIf(used[v].Not())

    # --- 訪問の一意性 ------------------------------------------------------
    #
    # 必須も任意も「served」という同じ形にして、落ちたときの罰の重さだけを
    # 変える。必須を hard 制約にすると、1軒でも届かない日に解無しになり、
    # 「何軒がどう届かないのか」を説明できなくなる。
    served: list[cp_model.IntVar] = []
    for c_idx, c in enumerate(customers):
        node = R + c_idx
        s = model.NewBoolVar(f"served_{c.id}")
        served.append(s)
        model.Add(sum(active[v][node] for v in range(V)) == s)
        if c.demand_l <= 0:
            # 満タンの需要家に行く意味はない。回れるのに回らない、を明示する。
            model.Add(s == 0)

    # --- 同型の車の対称性を潰す --------------------------------------------
    #
    # 1号車と2号車、3号車と4号車は槽構成も乗務員の上限も同じで完全に対称。
    # 何もしないと、同じ計画が車の入れ替えだけで何通りにも表現でき、探索が
    # そのぶん無駄に広がる。
    #
    # 「若い番号の車が、若い番号の需要家を先に取る」形にする。どんな解も
    # 車を並べ替えれば必ずこの形にできるので、最適解を切り落とさない。
    def same_kind(a: Vehicle, b: Vehicle) -> bool:
        da, db = scenario.driver(a.driver_id), scenario.driver(b.driver_id)
        return (
            sorted(a.compartments_l) == sorted(b.compartments_l)
            and (da.max_duty_min, da.max_drive_min) == (db.max_duty_min, db.max_drive_min)
            and da.available == db.available
        )

    symmetric_pairs = 0
    for v in range(V - 1):
        if not same_kind(fleet[v], fleet[v + 1]):
            continue
        symmetric_pairs += 1
        for c_idx in range(C):
            model.AddBoolOr(
                [active[v][R + k] for k in range(c_idx)]
                + [active[v + 1][R + c_idx].Not()]
            )
    diagnostics["symmetric_vehicle_pairs"] = symmetric_pairs

    # --- 時間の連鎖 --------------------------------------------------------
    for v in range(V):
        for (i, j), lit in x[v].items():
            if i < R and j >= R:
                # 油槽所（r便目）を出て最初の1軒へ
                model.Add(t[v][j] >= t[v][i] + travel[0][mat(j)]).OnlyEnforceIf(lit)
            elif i >= R and j >= R:
                model.Add(
                    t[v][j] >= t[v][i] + cust(i).service_min + travel[mat(i)][mat(j)]
                ).OnlyEnforceIf(lit)
            elif i >= R and j == 0:
                # 最後の便から帰庫。ここで1日が終わる
                model.Add(
                    duty_end[v] >= t[v][i] + cust(i).service_min + travel[mat(i)][0]
                ).OnlyEnforceIf(lit)
            elif i >= R and j >= 1:
                # 帰庫して積み直し、次の便へ。積込時間をここで飲み込む
                model.Add(
                    t[v][j]
                    >= t[v][i] + cust(i).service_min + travel[mat(i)][0] + load_min
                ).OnlyEnforceIf(lit)

    # --- 積載量（便ごとに積み直す） ----------------------------------------
    #
    # 便の区切りを表す変数は持たない。積み荷の累積量を経路に沿って伝播させ、
    # **油槽所の複製を出た直後にリセットする**だけで、便ごとの容量制約になる。
    load = []
    for v, veh in enumerate(fleet):
        cap = veh.capacity_l
        lv = [None] * N
        for c_idx in range(C):
            node = R + c_idx
            lv[node] = model.NewIntVar(0, cap, f"load_v{v}_n{node}")
            model.Add(lv[node] == 0).OnlyEnforceIf(active[v][node].Not())
        for (i, j), lit in x[v].items():
            if j < R:
                continue
            if i < R:
                model.Add(lv[j] == cust(j).demand_l).OnlyEnforceIf(lit)
            else:
                model.Add(lv[j] == lv[i] + cust(j).demand_l).OnlyEnforceIf(lit)
        load.append(lv)

    # --- 乗務員の拘束時間・運転時間（車＝人ごと） ---------------------------
    duty = []
    breaks = []
    drive = []
    for v, veh in enumerate(fleet):
        d = scenario.driver(veh.driver_id)

        drive_v = model.NewIntVar(0, d.max_drive_min, f"drive_v{v}")
        model.Add(
            drive_v == sum(travel[mat(i)][mat(j)] * lit for (i, j), lit in x[v].items())
        )
        drive.append(drive_v)

        # 連続運転の休憩（改善基準告示: 連続4時間で30分）。
        #
        # 近似: 休憩を経路上のどこに挟むかは決めず、「1日の運転時間が
        # 4時間の区切りを越えるごとに30分ぶんの余裕を拘束時間の中に
        # 確保する」という形で効かせる。
        #
        # 近似の向きは**安全側**にしてある。余裕を先に取るので、実際に
        # 休憩を挟んでも計画が破れない。逆向き（余裕を取らない）にすると
        # 法令上走れない計画が「解けた」ことになる。
        limit = d.continuous_drive_limit_min
        n_breaks = model.NewIntVar(0, 3, f"nbreak_v{v}")
        overs = []
        for step in (1, 2, 3):
            o = model.NewBoolVar(f"over{step}_v{v}")
            model.Add(drive_v > step * limit).OnlyEnforceIf(o)
            model.Add(drive_v <= step * limit).OnlyEnforceIf(o.Not())
            overs.append(o)
        model.Add(n_breaks == sum(overs))
        break_v = model.NewIntVar(0, 3 * d.break_min, f"break_v{v}")
        model.AddMultiplicationEquality(break_v, [n_breaks, d.break_min])
        breaks.append(break_v)

        duty_v = model.NewIntVar(0, 24 * 60, f"duty_v{v}")
        model.Add(duty_v == duty_end[v] - (t[v][0] - load_min)).OnlyEnforceIf(used[v])
        model.Add(duty_v == 0).OnlyEnforceIf(used[v].Not())
        model.Add(duty_v <= d.max_duty_min)

        # 休憩と積込のぶんの余裕を、拘束時間の中に確保しておく。
        service_v = sum(
            cust(R + c_idx).service_min * active[v][R + c_idx] for c_idx in range(C)
        )
        loads_v = sum(active[v][r] for r in range(R)) * load_min
        model.Add(duty_v >= drive_v + service_v + break_v + loads_v).OnlyEnforceIf(used[v])
        duty.append(duty_v)

    # --- 目的関数 ----------------------------------------------------------
    penalties = []
    for c_idx, c in enumerate(customers):
        p = PENALTY_MANDATORY_MIN if c.is_mandatory else _optional_penalty(c)
        if p:
            penalties.append(p * (1 - served[c_idx]))

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
    # 1. 需要家ごとに重みを変えた到着時刻の和。余裕のある立ち寄りは
    #    「窓の中ならいつでもよい」ので時刻が浮く。早い側に寄せる。
    #    副作用も望ましい方向で、着けるうちに着いておく計画になり、
    #    確率的な所要時間を入れたときに時間枠を割りにくくなる。
    #
    # 2. 車と需要家の組を (v+1)*i^2 で決める。最初は (v*100 + i) と書いていたが、
    #    これは v の項と i の項に分離できるため、2軒を2台の間で入れ替えても
    #    総和が変わらない。同型の車が実際にこの入れ替えを起こしていた。
    #
    # 3. 上の2つを、**下位が上位を打ち消せない大きさに分離する**（下記）。
    arrivals = []
    for c_idx in range(C):
        node = R + c_idx
        a = model.NewIntVar(0, horizon, f"arrive_{customers[c_idx].id}")
        for v in range(V):
            model.Add(a == t[v][node]).OnlyEnforceIf(active[v][node])
        model.Add(a == 0).OnlyEnforceIf(served[c_idx].Not())
        arrivals.append(a)

    # 需要家ごとに重みを変えた到着時刻の和。
    # 重みが全部1（素の合計）だと、時刻の集合が同じで割り当てだけ違う解を
    # 区別できない。重みを変えれば「どの軒が早い時刻を取るか」で値が動く。
    # 重みはすべて正なので、早く着く計画が選ばれる性質は保たれる。
    weighted_arrivals = sum((c_idx + 1) * arrivals[c_idx] for c_idx in range(C))

    lexical = sum(
        (v + 1) * ((c_idx + 1) ** 2) * active[v][R + c_idx]
        for v in range(V)
        for c_idx in range(C)
    )

    # **段を数値で分離する。** 下位の段が上位の段を打ち消せない大きさにする。
    #
    # 以前は「到着時刻の素の合計」と「区間ごとの重み ((i+1)^2*(j+1))」を
    # そのまま足していた。桁が同程度だったため、実際に打ち消し合った:
    #
    #     c02→c03 の順: 到着の和 730 ＋ 区間 55 = 785
    #     c03→c02 の順: 到着の和 724 ＋ 区間 61 = 785
    #
    # 到着時刻では後者が6小さく、区間の重みでは前者が6小さい。ちょうど
    # 相殺して同点になり、どちらが返るかが実行のたびに変わっていた。
    #
    # 区間ごとの重みは廃止した。「どの軒がどの時刻を取るか」を見れば
    # 訪問順は決まるので、重み付きの到着時刻1本で足りる。
    LEXICAL_MAX = V * sum((i + 1) ** 2 for i in range(C))
    ARRIVAL_SCALE = LEXICAL_MAX + 1

    tiebreak = model.NewIntVar(0, 10**12, "tiebreak")
    model.Add(tiebreak == weighted_arrivals * ARRIVAL_SCALE + lexical)

    # --- 求解（2段階の辞書式） ---------------------------------------------
    #
    # 主目的とタイブレークを1本の式に足し込むと、目的式が数千項の密な線形式に
    # なり、線形緩和が効かなくなって最適性を証明できなくなる。
    # 1回目は主目的だけ（項が疎で速い）。2回目は主目的を見つかった最適値に
    # 固定して、タイブレークだけを最小化する。解空間が「最適な計画すべて」に
    # 狭まっているので2回目は速い。
    def new_solver(limit: float) -> cp_model.CpSolver:
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = limit
        s.parameters.num_workers = workers
        s.parameters.log_search_progress = log
        s.parameters.random_seed = scenario.seed % 2_147_483_647
        return s

    # 2段目は決定的時間で切るので実時間の予算をほとんど食わない。
    # 残りは1段目に回す。**1段目が最適性を証明できるかが再現性の要**なので、
    # 予算はできるだけこちらに寄せる。
    phase1_budget = (
        time_limit_s if not tiebreak_phase else time_limit_s * 0.6
    )
    solver1 = new_solver(phase1_budget)
    model.Minimize(primary)
    status1 = solver1.Solve(model)
    phase1_status = solver1.StatusName(status1)
    phase1_time = solver1.WallTime()

    solver = solver1
    status = status1
    status_name = phase1_status
    phase2_status = "skipped"
    phase2_time = 0.0

    if tiebreak_phase and status1 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        best_primary = solver1.Value(primary)
        model.Add(primary == best_primary)
        model.Minimize(tiebreak)

        # **2段目には必ず別の solver を使う。**
        # 同じ solver を使い回して2段目が時間切れになると、1段目の解が
        # 上書きされて消える。それに気づかず値を読むと、変数の定義域を
        # 超えた数字（目的値 1,760,776,892,128 や、必須訪問を6軒落とした
        # 目的値 0）が平然と返ってきた。
        solver2 = new_solver(max(10.0, time_limit_s - phase1_time))
        status2 = solver2.Solve(model)
        phase2_status = solver2.StatusName(status2)
        phase2_time = solver2.WallTime()
        if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solver = solver2
            status = status2
            status_name = phase2_status
        else:
            status_name = phase1_status

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Plan(
            status=status_name,
            vehicle_days=(),
            unserved=_all_unserved(customers),
            total_drive_min=0,
            objective=0,
            wall_time_s=phase1_time + phase2_time,
            scenario_date=scenario.date,
            scenario_seed=scenario.seed,
            diagnostics={
                **diagnostics,
                "reason": "制約を同時に満たす計画が見つからない",
                "phase1_status": phase1_status,
                "phase2_status": phase2_status,
            },
        )

    # --- 解の取り出し ------------------------------------------------------
    #
    # 周回を d0 から辿る。油槽所の複製に着いたところが便の切れ目になる。
    vehicle_days: list[VehicleDay] = []
    for v, veh in enumerate(fleet):
        if not solver.BooleanValue(used[v]):
            continue
        d = scenario.driver(veh.driver_id)

        succ: dict[int, list[int]] = {i: [] for i in range(N)}
        for (i, j) in x[v]:
            succ[i].append(j)

        def next_node(cur: int) -> int | None:
            for j in succ[cur]:
                if solver.BooleanValue(x[v][(cur, j)]):
                    return j
            return None

        # (便の出発になった油槽所節点, その便で回った需要家節点, 帰着した油槽所節点)
        segments: list[tuple[int, list[int], int]] = []
        start_depot = 0
        current: list[int] = []
        cur = 0
        for _ in range(N + 1):
            nxt = next_node(cur)
            if nxt is None:
                break
            if nxt < R:
                segments.append((start_depot, current, nxt))
                current = []
                start_depot = nxt
                if nxt == 0:
                    break
            else:
                current.append(nxt)
            cur = nxt

        trips: list[Trip] = []
        for start_node, nodes, end_node in segments:
            stops = tuple(
                Stop(
                    customer_id=cust(node).id,
                    customer_name=cust(node).name,
                    arrive_min=solver.Value(t[v][node]),
                    depart_min=solver.Value(t[v][node]) + cust(node).service_min,
                    delivered_l=cust(node).demand_l,
                    window_open_min=cust(node).window_open_min,
                    window_close_min=cust(node).window_close_min,
                )
                for node in nodes
            )
            depart_min = solver.Value(t[v][start_node])
            # 次の便の出発時刻には積込時間が織り込まれている。最後の便は帰庫時刻。
            return_min = (
                solver.Value(duty_end[v])
                if end_node == 0
                else solver.Value(t[v][end_node]) - load_min
            )
            trip_drive = sum(
                travel[mat(a)][mat(b)]
                for a, b in zip([start_node] + nodes, nodes + [end_node])
            )
            trips.append(
                Trip(
                    vehicle_id=veh.id,
                    vehicle_name=veh.name,
                    trip_index=start_node,
                    load_start_min=depart_min - load_min,
                    depart_min=depart_min,
                    return_min=return_min,
                    drive_min=trip_drive,
                    load_l=sum(s.delivered_l for s in stops),
                    capacity_l=veh.capacity_l,
                    stops=stops,
                )
            )

        vehicle_days.append(
            VehicleDay(
                vehicle_id=veh.id,
                vehicle_name=veh.name,
                driver_id=d.id,
                driver_name=d.name,
                duty_start_min=solver.Value(t[v][0]) - load_min,
                duty_end_min=solver.Value(duty_end[v]),
                drive_min=solver.Value(drive[v]),
                break_min=solver.Value(breaks[v]),
                trips=tuple(trips),
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
        for c_idx, c in enumerate(customers)
        if c.demand_l > 0 and not solver.BooleanValue(served[c_idx])
    )

    return Plan(
        status=status_name,
        vehicle_days=tuple(vehicle_days),
        unserved=unserved,
        total_drive_min=sum(vd.drive_min for vd in vehicle_days),
        # 外へ出すのは主目的だけ。タイブレークは決定性のための内部事情であって、
        # 「この計画のコスト」として読まれると桁が意味を失う。
        objective=solver.Value(primary),
        # 2段の合計。最後の solver の時間だけを出すと、45秒かけた求解が
        # 「15秒で解けた」ことになり、性能の判断を誤らせる。
        wall_time_s=phase1_time + phase2_time,
        scenario_date=scenario.date,
        scenario_seed=scenario.seed,
        diagnostics={
            **diagnostics,
            # 実際に立ち寄った軒数。**引き算で出してはいけない。**
            # len(customers) - len(unserved) と書いていたが、unserved は
            # 需要のある需要家しか含まないので、満タンで行く必要のない軒が
            # 「回った」側に混ざる。本番で立ち寄り9軒の計画が
            # served_count=10 と申告していた。
            "served_count": sum(len(t.stops) for vd in vehicle_days for t in vd.trips),
            "no_demand_count": sum(1 for c in customers if c.demand_l <= 0),
            "trip_count": sum(len(vd.trips) for vd in vehicle_days),
            "tiebreak": solver.Value(tiebreak),
            "phase1_status": phase1_status,
            "phase1_time_s": round(phase1_time, 3),
            "phase2_status": phase2_status,
            "phase2_time_s": round(phase2_time, 3),
            # 両段が最適性を証明できたときだけ、この計画は完全に再現できる。
            # どちらかが時間切れなら、同じ入力でも別の計画が返り得る。
            # **ここを緩めてはいけない。** 緩めた版は、計画が違うのに
            # reproducible が真を返した。
            "tiebreak_phase": tiebreak_phase,
            "reproducible": phase1_status == "OPTIMAL" and phase2_status == "OPTIMAL",
        },
    )


def format_plan_ja(plan: Plan) -> str:
    """計画を配車係の言葉で書き出す。

    OR の用語（目的関数、緩和、双対）は一切出さない。出すのは
    「何号車が、何時に出て、どこへ、何kL、何時に戻る」だけ。
    """
    lines = [f"【{plan.scenario_date} の配車】{plan.summary_ja()}", ""]
    for vd in plan.vehicle_days:
        lines.append(
            f"■ {vd.vehicle_name}（{vd.driver_name}）"
            f"　拘束 {minutes_to_hhmm(vd.duty_start_min)}〜"
            f"{minutes_to_hhmm(vd.duty_end_min)}"
            f"（{vd.duty_min // 60}時間{vd.duty_min % 60}分）"
            f"　運転 {vd.drive_min // 60}時間{vd.drive_min % 60}分"
            + (f"　休憩 {vd.break_min}分" if vd.break_min else "")
            + f"　{len(vd.trips)}便で {vd.load_l / 1000:.1f}kL"
        )
        for trip in vd.trips:
            lines.append(
                f"  ‹{trip.trip_index + 1}便› 積込 {minutes_to_hhmm(trip.load_start_min)}"
                f"／出庫 {minutes_to_hhmm(trip.depart_min)}"
                f"／帰庫 {minutes_to_hhmm(trip.return_min)}"
                f"　積載 {trip.load_l / 1000:.1f}kL / {trip.capacity_l / 1000:.0f}kL"
                f"（{trip.utilization * 100:.0f}%）"
            )
            for i, s in enumerate(trip.stops, start=1):
                lines.append(
                    f"      {i}. {minutes_to_hhmm(s.arrive_min)} {s.customer_name}"
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
