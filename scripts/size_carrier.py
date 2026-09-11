"""モデルの運送会社（専属20台規模）の半期の体制を出す。

    需要の鎖（地区需要 × シェア × 引き当て）
      → 配送エリアの何個ぶんか
      → 各日の必要台数（エリアごとに欠品ゼロで回る最小台数を解いて合計）
      → 専属と非専属の配分（core/fleet_plan.py の規則）

**半期を通して見る。** 冬の半期（10〜3月）でも、灯油が跳ねるのは12〜2月で、
10〜11月と3月は穏やか。専属を冬の山に合わせて取ると秋口と春先に車が遊び、
少なすぎると単価の高い非専属を毎日呼ぶことになる。

各月から1週間（第2月曜から7日）を取り、エリアごとに別の乱数種で解く。
エリアの発注の振れは互いに独立なので、合計すると均される（まとめて持つ
ことの効果）。

**必要台数は幅で出す。** 当日の配車はエリア単位で組むので、1.2台ぶんの
仕事でもそのエリアでは2台要る。エリアをまたいで車を融通すればもっと
少なくて済むが、融通には回送がかかる。実際はその間にある:

    融通しない（上限）  エリアごとの最小台数の合計
    完全に融通（下限）  延べ拘束時間 ÷ 13時間 の切り上げ

使い方:
    python -m scripts.size_carrier
    python -m scripts.size_carrier --half summer_half
    python -m scripts.size_carrier --allocation 0.45     # 引き当てが増えたら
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.demand_chain import HALF_YEAR_MONTHS, DemandChain, zone_kl_per_day, zones_for  # noqa: E402
from core.fleet_plan import SPOT_PREMIUM, UNAVAILABILITY, split_fleet  # noqa: E402
from core.generator import EPOCH, generate  # noqa: E402
from core.solver import solve  # noqa: E402
from core.travel import build_matrix  # noqa: E402

#: エリアごとに台数を足していく順（単車を先に。scripts/size_fleet.py と同じ理由）。
ADD_ORDER: tuple[str, ...] = ("v4", "v5", "v1", "v2", "v3")

CACHE = Path(__file__).resolve().parents[1] / "data" / "carrier_cache_v2.json"

#: 1人1日の拘束の上限（改善基準告示の原則13時間）。融通した場合の下限に使う。
DUTY_CAP_MIN = 780


def _week(year: int, month: int) -> list[datetime.date]:
    """その月の第2月曜から7日。"""
    d = datetime.date(year, month, 1)
    first_monday = d + datetime.timedelta(days=(7 - d.weekday()) % 7)
    start = first_monday + datetime.timedelta(days=7)
    return [start + datetime.timedelta(days=i) for i in range(7)]


def min_fleet(task: tuple[int, str, str, float]) -> tuple[str, int, bool, int]:
    """1エリア・1日を、欠品ゼロで回る最小の台数で解く。

    Returns:
        (キャッシュのキー, 台数, 5台で足りたか, 使った車の拘束時間の合計（分）)。
        5台でも欠品が出るなら5を返し、足りなかったことを知らせる。
    """
    seed, date, season, limit = task
    key = f"{season}|{date}|{seed}"
    for size in range(1, len(ADD_ORDER) + 1):
        s = generate(seed=seed, date=date, season=season)
        keep = set(ADD_ORDER[:size])
        s.vehicles = tuple(v for v in s.vehicles if v.id in keep)
        if not any(c.demand_l for c in s.customers):
            return key, 0, True, 0
        p = solve(s, matrix=build_matrix(s.depot, s.customers), time_limit_s=limit,
                  workers=1, max_trips=4, tiebreak_phase=False)
        if p.is_usable and not p.mandatory_missed:
            return key, size, True, sum(vd.duty_min for vd in p.vehicle_days)
    return key, len(ADD_ORDER), False, len(ADD_ORDER) * DUTY_CAP_MIN


def main() -> int:
    ap = argparse.ArgumentParser(description="モデルの運送会社の半期の体制（専属と非専属）")
    ap.add_argument("--half", default="winter_half", choices=sorted(HALF_YEAR_MONTHS))
    ap.add_argument("--year", type=int, default=2026, help="半期の始まりの年")
    ap.add_argument("--district", type=float, default=DemandChain().district_kl_per_day)
    ap.add_argument("--share", type=float, default=DemandChain().eneos_share)
    ap.add_argument("--allocation", type=float, default=DemandChain().allocation)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    chain = DemandChain(args.district, args.share, args.allocation)
    zones = zones_for(chain)
    months = HALF_YEAR_MONTHS[args.half]

    print("【需要の鎖】")
    print(f"  地区 {chain.district_kl_per_day:,.0f}kL/日 × シェア {chain.eneos_share:.0%}"
          f" × 引き当て {chain.allocation:.0%} ＝ {chain.carrier_kl_per_day:,.0f}kL/日（冬の平日）")
    print(f"  配送エリア1つ {zone_kl_per_day():.1f}kL/日 → **{zones}エリアぶん**")
    print("  ※ 地区の需要は逆算した仮置（docs/業務前提.md §5.1.1b）\n", flush=True)

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    plan: list[tuple[int, datetime.date, str, list[tuple[int, str, str, float]]]] = []
    for month, season in months:
        year = args.year if month >= months[0][0] else args.year + 1
        for d in _week(year, month):
            n = (d - EPOCH).days
            tasks = [(z * 100_000 + n, d.isoformat(), season, args.time_limit) for z in range(zones)]
            plan.append((month, d, season, tasks))

    todo = [t for _, _, _, ts in plan for t in ts if f"{t[2]}|{t[1]}|{t[0]}" not in cache]
    short: set[str] = set()
    if todo:
        print(f"  {len(todo)}件（エリア×日）を解きます…", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for i, (key, size, ok, duty) in enumerate(ex.map(min_fleet, todo, chunksize=4), 1):
                cache[key] = [size, ok, duty]
                if i % 50 == 0:
                    print(f"    {i}/{len(todo)}", flush=True)
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")

    upper: list[int] = []
    lower: list[int] = []
    by_month: dict[int, list[tuple[int, int]]] = {}
    for month, d, season, tasks in plan:
        vals = [cache[f"{t[2]}|{t[1]}|{t[0]}"] for t in tasks]
        hi = sum(v[0] for v in vals)
        lo = -(-sum(v[2] for v in vals) // DUTY_CAP_MIN)
        if not all(v[1] for v in vals):
            short.add(d.isoformat())
        upper.append(hi)
        lower.append(lo)
        by_month.setdefault(month, []).append((lo, hi))

    print("【各日の必要台数】 融通した場合〜融通しない場合（欠品ゼロで回る最小）")
    for month, season in months:
        v = by_month[month]
        print(f"  {month:>2}月（{season:<6}）  月〜日 " + " ".join(f"{a:>2}〜{b:<2}" for a, b in v))
    if short:
        print(f"  ※ 5台でも欠品が出たエリアのある日: {', '.join(sorted(short))}")

    for label, daily in (("融通しない場合（上限）", upper), ("完全に融通した場合（下限）", lower)):
        _report(label, daily)

    print()
    print("  実際はこの間にある。エリアをまたぐ融通には回送がかかるため下限には届かない。")
    print("  乗務員は1交代を前提に台数と同数（二交代は体制の判断。docs/業務前提.md §5.2）")
    return 0


def _report(label: str, daily: list[int]) -> None:
    r = split_fleet(daily)
    print(f"\n【専属と非専属の配分 — {label}】（非専属は専属の{1 + SPOT_PREMIUM:.2f}倍の単価、"
          f"専属が抜ける割合 {UNAVAILABILITY:.0%}）")
    print(f"  専属にする境目: 半期のうち **{r.threshold:.0%}** を超える日に要る車")
    first_spot = r.dedicated
    lo = max(0, first_spot - 3)
    for k in range(lo, len(r.need_share)):
        mark = "専属" if k < r.dedicated else "非専属"
        print(f"    {k + 1:>3}台目  要る日 {r.need_share[k]:5.0%}  → {mark}")
    print(f"\n  **専属 {r.dedicated}台**")
    print(f"  非専属  需要の振れ {r.spot_days_swing}台日 ＋ 抜けた日の穴埋め"
          f" {r.spot_days_cover:.0f}台日（{r.days}日間）")
    print(f"          1日に最大 {r.spot_peak}台")
    saving = 1 - r.cost / r.cost_all_dedicated
    print(f"  費用  ピーク{max(daily)}台を全部専属で持つ場合より {saving:.1%} 安い")


if __name__ == "__main__":
    raise SystemExit(main())
