"""1日ぶんの配車を組んで、配車係の言葉で書き出す。

使い方:
    python -m scripts.plan_day
    python -m scripts.plan_day --season summer --seed 7
    python -m scripts.plan_day --absent d1 --absent d2

なぜ CLI を先に作るか:
    W1 の段階で確かめたいのは「制約を同時に満たす計画が本当に出るか」であって、
    画面ではない。UI を先に作ると、解が出ていないことを画面のせいにできて
    しまう。決定論コアが単体で正しいことを、テキストで読める形にしておく。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.domain import minutes_to_hhmm  # noqa: E402
from core.generator import generate  # noqa: E402
from core.solver import DEFAULT_TIME_LIMIT_S, SolveMode, format_plan_ja, solve  # noqa: E402
from core.travel import build_matrix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="石油ローリー配車エージェント — 当日計画")
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--date", default="2026-10-15")
    ap.add_argument(
        "--season", default="winter", choices=["winter", "spring", "summer", "autumn"]
    )
    ap.add_argument(
        "--absent", action="append", default=[], help="欠勤する乗務員ID（例 d1）"
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="実時間で打ち切る（--time-limit 秒）。既定は再現モード"
        "（どのマシンでも同じ計画が返る。時間は1〜2分かかることがある）",
    )
    ap.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT_S,
                    help="--fast のときだけ効く")
    ap.add_argument("--verbose", action="store_true", help="需要と車両の内訳も出す")
    args = ap.parse_args()

    scenario = generate(
        seed=args.seed,
        date=args.date,
        season=args.season,
        absent_drivers=tuple(args.absent),
    )
    matrix = build_matrix(scenario.depot, scenario.customers)

    if args.verbose:
        print("=== 当日の需要 ===")
        for c in scenario.customers:
            mark = "必須" if c.is_mandatory else "任意"
            days = c.min_days_to_empty()
            print(
                f"  [{mark}] {c.name:<20} {c.demand_l / 1000:5.1f}kL"
                f"　受入 {minutes_to_hhmm(c.window_open_min)}〜"
                f"{minutes_to_hhmm(c.window_close_min)}"
                f"　在庫あと {days:.1f}日"
            )
        print(
            f"\n  必須合計 {scenario.mandatory_demand_l / 1000:.1f}kL"
            f" / 全体 {sum(c.demand_l for c in scenario.customers) / 1000:.1f}kL"
            f" / 車両総容量 {scenario.fleet_capacity_l / 1000:.1f}kL"
        )
        print(f"  出せる車 {len(scenario.usable_vehicles())}台\n")

    plan = solve(
        scenario,
        matrix=matrix,
        time_limit_s=args.time_limit,
        mode=SolveMode.FAST if args.fast else SolveMode.REPRODUCIBLE,
    )
    print(format_plan_ja(plan))
    print()
    print(
        f"[求解] {plan.status}"
        f" / 目的値 {plan.objective}"
        f" / {plan.wall_time_s:.2f}秒"
        f" / 走行 {plan.total_drive_min}分"
    )
    return 0 if plan.is_usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
