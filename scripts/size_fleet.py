"""需要想定から、必要なローリー台数と乗務員数を求める。

**因果の向きがこの順序であることが要点。**

    需要想定（元売が季節ごとに予測）
      → 必要なローリー台数
      → 必要な乗務員数
      → 半期（4-9月／10-3月）ごとの専属契約台数 ＋ 非専属の確保

台数を入力として受け取って計画を組むのは、この鎖の**一番下だけ**を見ている
ことになる。当日の配車はそれでよいが、事業の判断はここから始まる。

求め方:
    需要（季節 × 複数日）を固定し、台数を1台ずつ増やしながら
    「欠品ゼロで回る最小の台数」を探す。台数が答えで、そこから乗務員数が出る。

    乗務員数は体制案ごとに決まる。**二交代は当日の需要では切り替えられない。**
    設備（夜間に降ろせる先）と人事（採用・シフト）が先に要るので、体制は
    **曜日の周期で固定**される。実務では日曜に解除する（SSの休業で配送量が
    減るため）。

        案A: 1交代（乗務員 = 台数）
        案B: 2交代（月〜土は乗務員 = 台数×2、日曜は解除）

    **いまは案Aしか評価できない。** 車と人が `Vehicle.driver_id` で
    1対1に固定されていて、二交代を表せないため（docs/業務前提.md §5.2）。

なぜ複数日で見るか:
    1日だけ見て決めた台数は、その日の需要のばらつきに引きずられる。
    在庫水準は日替わりなので、切迫する軒数も日によって変わる。
    複数日の**最悪の日**で足りる台数でないと、その日に欠品する。

    **本来の単位は1週間。** 体制が曜日の周期で固定される以上、日曜だけを
    見れば少ない台数で足りるが、それでは月曜が回らない。いまの掃引は
    曜日を持たない「N日ぶん」を見ているので、**ここはまだ正しくない**
    （生成器が曜日で需要を動かせないため。docs/業務前提.md §5.2）。

使い方:
    python -m scripts.size_fleet
    python -m scripts.size_fleet --season summer --days 8
    python -m scripts.size_fleet --season winter --time-limit 90
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.domain import VEHICLE_CLASS_JA, Vehicle  # noqa: E402
from core.generator import FLEET, generate  # noqa: E402
from core.solver import solve  # noqa: E402
from core.travel import build_matrix  # noqa: E402

#: 台数を増やすときに、どの順で足すか。
#:
#: **単車を先に置く。** 単車しか入れない需要家があるので、トレーラーだけの
#: 編成にするとその軒が最初から回れない。台数の答えが「足りないから増やす」
#: ではなく「そもそも行けない」で歪む。
ADD_ORDER: tuple[str, ...] = ("v4", "v5", "v1", "v2", "v3")

#: 拘束時間がこの割合を超えていたら「張り付いている」と見なす。
#:
#: **これは二交代の判定ではない。** 二交代は日ごとに選べる運用ではなく、
#: 設備と人事から決まる恒常的な体制なので、日ごとの逼迫から導けない。
#: ここで言えるのは「**1交代では余裕が無い**」までで、その先は体制案の
#: 比較（案A/案B）の仕事になる。
DUTY_TIGHT_RATIO = 0.92


@dataclass
class DayResult:
    seed: int
    vehicles_used: int
    served: int
    missed: int
    max_duty_min: int
    max_duty_cap_min: int
    total_km: float
    trips: int
    proven: bool

    @property
    def ok(self) -> bool:
        """欠品ゼロで回れたか。**これが台数を決める基準。**"""
        return self.missed == 0

    @property
    def duty_is_tight(self) -> bool:
        return self.max_duty_min >= self.max_duty_cap_min * DUTY_TIGHT_RATIO


def evaluate(fleet_size: int, seeds, season: str, time_limit: float) -> list[DayResult]:
    """その台数で、各日が回るかを見る。

    掃引ではコストと実行可能性しか要らないので tiebreak_phase=False。
    2段目（計画を一意に絞る処理）を丸ごと省けるぶん、倍近く速い。
    """
    keep = set(ADD_ORDER[:fleet_size])
    out: list[DayResult] = []
    for seed in seeds:
        s = generate(seed=seed, season=season)
        s.vehicles = tuple(v for v in s.vehicles if v.id in keep)
        m = build_matrix(s.depot, s.customers)
        p = solve(
            s, matrix=m, time_limit_s=time_limit, max_trips=4, tiebreak_phase=False
        )
        acts = p.diagnostics.get("per_vehicle_actuals", [])
        cap = max((s.driver(v.driver_id).max_duty_min for v in s.vehicles), default=0)
        out.append(
            DayResult(
                seed=seed,
                vehicles_used=len(acts),
                served=int(p.diagnostics.get("served_count", 0)),
                missed=len(p.mandatory_missed),
                max_duty_min=max((a["duty_min"] for a in acts), default=0),
                max_duty_cap_min=cap,
                total_km=sum(a["drive_km"] for a in acts),
                trips=int(p.diagnostics.get("trip_count", 0)),
                proven=p.diagnostics.get("phase1_status") == "OPTIMAL",
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="需要想定から必要なローリー台数と乗務員数を求める"
    )
    ap.add_argument(
        "--season", default="winter", choices=["winter", "spring", "summer", "autumn"]
    )
    ap.add_argument("--days", type=int, default=5, help="見る日数（需要のばらつき）")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--max-fleet", type=int, default=len(FLEET))
    args = ap.parse_args()

    seeds = [20261015 + i for i in range(args.days)]

    print(f"需要想定: {args.season} / {args.days}日ぶん（seed {seeds[0]}〜{seeds[-1]}）")
    print(f"名簿: 重要家15軒（麻里布油槽所の配送圏）\n")
    print(f"{'台数':>4} {'欠品ゼロ':>9} {'回れた軒':>9} {'総走行':>9} "
          f"{'最長拘束':>9} {'便':>4} {'最適性':>7}")
    print("-" * 60)

    answer = None
    for size in range(1, args.max_fleet + 1):
        rows = evaluate(size, seeds, args.season, args.time_limit)
        ok_days = sum(1 for r in rows if r.ok)
        served = sum(r.served for r in rows) / len(rows)
        km = sum(r.total_km for r in rows) / len(rows)
        duty = max(r.max_duty_min for r in rows)
        trips = sum(r.trips for r in rows) / len(rows)
        proven = sum(1 for r in rows if r.proven)
        print(
            f"{size:>4} {ok_days:>4}/{len(rows):<4} {served:>9.1f} {km:>8.0f}km "
            f"{duty // 60:>6}時間{duty % 60:02d}分 {trips:>4.1f} {proven:>3}/{len(rows)}",
            flush=True,
        )
        if ok_days == len(rows) and answer is None:
            answer = (size, rows)

    print()
    if answer is None:
        print(f"**{args.max_fleet}台でも、どこかの日に欠品が出ます。**")
        print("名簿を減らすか、配送圏の一部を隣の油槽所へ移すかの判断になります。")
        return 1

    size, rows = answer
    tight = [r for r in rows if r.duty_is_tight]

    print(f"【{args.season} の答え（1交代を前提）】")
    print(f"  必要なローリー: **{size}台**")
    for i, vid in enumerate(ADD_ORDER[:size]):
        v = next(x for x in FLEET if x[0] == vid)
        print(f"      {v[1]}  {VEHICLE_CLASS_JA[v[2]]} {sum(v[3]) // 1000}kL")
    print(f"  必要な乗務員: **{size}名**（1台1人）")
    if tight:
        worst = max(r.max_duty_min for r in tight)
        print()
        print(f"  ただし拘束が上限に張り付く日があります"
              f"（最長 {worst // 60}時間{worst % 60:02d}分 / 上限 "
              f"{rows[0].max_duty_cap_min // 60}時間）。")
        print("  **1交代では余裕がありません。** 需要が少し増えれば崩れます。")
        print("  二交代を敷くかは、ここではなく体制の判断になります。")
        print("  日ごとに選べるものではなく、夜間に降ろせる先の確保と")
        print("  乗務員の採用が先に要る、恒常的な決定です。")
    else:
        print("（拘束に余裕がある）")
    print(f"  1日あたり総走行: {sum(r.total_km for r in rows) / len(rows):.0f}km")
    print(f"  1台あたり: {sum(r.trips for r in rows) / len(rows) / size:.1f}回転 / "
          f"{sum(r.total_km for r in rows) / len(rows) / size:.0f}km")
    print()
    print("  ※ 実務の目安は 1台 3回転前後 / 500〜600km / 拘束13時間前後。")
    print("     ここから離れているなら、需要か名簿か所要時間のどれかがずれている。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
