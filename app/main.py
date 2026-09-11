"""HTTP の入口。

v0（W1・M1a）で出すのは3本だけ:
    GET  /api/healthz   — 生きているか、設定が読めているか、台帳はどちらか
    POST /api/plan      — 当日の配車を組む（合成データ → 決定論コア）
    GET  /api/usage     — その日のコスト使用状況

画面はまだ無い（W5・M4）。ここで API を先に固めておくのは、
UI を作り始めてから「そもそも解が出ていなかった」と分かる事態を避けるため。

エージェント層（W4）はまだ繋がっていない。今のところ /api/plan は
モデルを1回も呼ばず、決定論コアだけで完結する。
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import ledger
from app.config import settings
from core.domain import minutes_to_hhmm
from core.generator import generate
from core.solver import Plan, SolveMode, format_plan_ja, solve
from core.travel import build_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lorry.api")

app = FastAPI(
    title="石油ローリー配車エージェント",
    description="麻里布油槽所からの二次配送を組み、外の変化に応じて組み直す。",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# ヘルスチェック
# ---------------------------------------------------------------------------


@app.get("/api/healthz")
def healthz() -> dict:
    """生存確認。設定の読み込み確認も兼ねる。

    パスが `/healthz` ではなく `/api/healthz` なのは意図的。
    Cloud Run のフロントエンド（GFE）が `/healthz` を横取りしており、
    リクエストがコンテナに一度も到達せず Google の 404 ページが返る。
    ローカルでは正常に動くため、本番へ出して初めて分かる。

    台帳のバックエンドを返しているのは、**本番で memory になっていたら
    全体上限が効いていない**ということだから。画面に出す前にここで気づく。
    """
    return {
        "status": "ok",
        "project": settings.project_id or "(unset)",
        "location": settings.location,
        "model": settings.gemini_model,
        "ledger_backend": ledger.backend(),
        "solve_time_limit_s": settings.solve_time_limit_s,
    }


@app.get("/api/usage")
def usage(day: str | None = None) -> dict:
    """その日のコスト使用状況と上限。"""
    return ledger.usage(day)


@app.post("/api/model-check")
def model_check(subject: str = "modelcheck") -> dict:
    """**Cloud Run 上から**モデルを1回だけ呼び、実費を測って返す。

    なぜ画面でも配車でもない経路をわざわざ置くか:
        ローカルで呼べることと、Cloud Run から呼べることは別である。
        違うのは資格情報（実行サービスアカウントの権限）で、そこが
        通っているかは**本番から1回呼んでみるまで分からない**。
        エージェント層に着手する前に、ここで確かめる。

        あわせて、1回の呼び出しで思考トークンが何個出るかを測る。
        Gemini 3.x flash 系は思考トークンを出力として課金するので、
        コスト台帳の見積り（ledger.ESTIMATES）が現実と合っているかは
        これを見て判断する。

    モデルは必ず **genai_location（global）** で呼ぶ。us-central1 では
    404 になる（2026-09-03 実機確認）。
    """
    try:
        amount = ledger.reserve(subject, "trigger")
    except ledger.BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    try:
        # 起動を遅くしないよう、ここで初めて読み込む。
        from google import genai

        client = genai.Client(
            vertexai=True,
            project=settings.project_id,
            location=settings.genai_location,
        )
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents="Reply with the single word: ok",
        )
    except Exception as exc:
        ledger.release(subject, amount)
        logger.exception("モデル呼び出しに失敗")
        raise HTTPException(
            status_code=502,
            detail=f"{settings.gemini_model} を呼べません: {type(exc).__name__}: {exc}",
        ) from exc

    meta = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
    output_tokens = getattr(meta, "candidates_token_count", 0) or 0
    thought_tokens = getattr(meta, "thoughts_token_count", 0) or 0

    ledger.record(subject, "trigger", amount, detail="model-check")

    return {
        "ok": True,
        "model": settings.gemini_model,
        "location": settings.genai_location,
        "reply": (response.text or "").strip()[:100],
        "tokens": {
            "prompt": prompt_tokens,
            "output": output_tokens,
            # 思考トークンは出力として課金される。ここが実費の大半を占める。
            "thoughts": thought_tokens,
        },
        "reserved_usd": amount,
    }


# ---------------------------------------------------------------------------
# 配車
# ---------------------------------------------------------------------------


class PlanRequest(BaseModel):
    """当日の配車を組む要求。

    合成データ生成器のパラメータをそのまま受ける。審査員が手元で
    「寒波の日」「乗務員が2人欠勤した日」を再現できるようにするため、
    seed と季節と欠勤を外から指定できる形にしてある。
    """

    seed: int = Field(default=20261015, description="乱数種。同じ値なら同じ日")
    date: str = Field(default="2026-10-15")
    season: str = Field(default="winter", pattern="^(winter|spring|summer|autumn)$")
    absent_drivers: list[str] = Field(
        default_factory=list, description="欠勤する乗務員ID（例 ['d1']）"
    )
    subject: str = Field(default="demo", description="利用者の識別子。台帳の単位")
    mode: SolveMode = Field(
        default=SolveMode.REPRODUCIBLE,
        description=(
            "reproducible: どのマシンでも同じ計画を返す（探索量で打ち切る。"
            "時間は読めない）。fast: 時間で打ち切る（再現は保証しない）"
        ),
    )


def _plan_to_dict(plan: Plan) -> dict:
    """計画をそのまま JSON にする。

    時刻は分と HH:MM の両方を入れる。分は画面が計算に使い、HH:MM は
    人が読む。片方だけにすると、どちらかの側で必ず変換ミスが起きる。
    """
    return {
        "status": plan.status,
        "summary_ja": plan.summary_ja(),
        "text_ja": format_plan_ja(plan),
        "objective": plan.objective,
        "wall_time_s": round(plan.wall_time_s, 3),
        "total_drive_min": plan.total_drive_min,
        "vehicle_days": [
            {
                "vehicle_id": vd.vehicle_id,
                "vehicle_name": vd.vehicle_name,
                "driver_id": vd.driver_id,
                "driver_name": vd.driver_name,
                "duty_start_min": vd.duty_start_min,
                "duty_start": minutes_to_hhmm(vd.duty_start_min),
                "duty_end_min": vd.duty_end_min,
                "duty_end": minutes_to_hhmm(vd.duty_end_min),
                "duty_min": vd.duty_min,
                "drive_min": vd.drive_min,
                "break_min": vd.break_min,
                "load_l": vd.load_l,
                "trips": [
                    {
                        "trip_index": tr.trip_index,
                        "load_start_min": tr.load_start_min,
                        "load_start": minutes_to_hhmm(tr.load_start_min),
                        "depart_min": tr.depart_min,
                        "depart": minutes_to_hhmm(tr.depart_min),
                        "return_min": tr.return_min,
                        "return": minutes_to_hhmm(tr.return_min),
                        "drive_min": tr.drive_min,
                        "load_l": tr.load_l,
                        "capacity_l": tr.capacity_l,
                        "utilization": round(tr.utilization, 3),
                        "stops": [
                            {
                                "customer_id": s.customer_id,
                                "customer_name": s.customer_name,
                                "arrive_min": s.arrive_min,
                                "arrive": minutes_to_hhmm(s.arrive_min),
                                "depart_min": s.depart_min,
                                "depart": minutes_to_hhmm(s.depart_min),
                                "delivered_l": s.delivered_l,
                                "window": (
                                    f"{minutes_to_hhmm(s.window_open_min)}-"
                                    f"{minutes_to_hhmm(s.window_close_min)}"
                                ),
                                "slack_min": s.slack_min,
                            }
                            for s in tr.stops
                        ],
                    }
                    for tr in vd.trips
                ],
            }
            for vd in plan.vehicle_days
        ],
        "unserved": [
            {
                "customer_id": u.customer_id,
                "customer_name": u.customer_name,
                "demand_l": u.demand_l,
                "mandatory": u.mandatory,
                "days_to_empty": (
                    None if u.days_to_empty == float("inf") else round(u.days_to_empty, 2)
                ),
            }
            for u in sorted(plan.unserved, key=lambda u: u.days_to_empty)
        ],
        "diagnostics": plan.diagnostics,
    }


@app.post("/api/plan")
def make_plan(req: PlanRequest) -> dict:
    """当日の配車を組む。

    求解自体はモデルを呼ばないので費用はかからないが、台帳は通す。
    「課金の無い経路は台帳を素通りしてよい」を一度でも許すと、
    あとで課金が乗ったときに通し忘れる。
    """
    plan_id = uuid.uuid4().hex[:12]
    started = time.monotonic()

    try:
        amount = ledger.reserve(req.subject, "solve")
    except ledger.BudgetExceeded as exc:
        # 429 を返すのは「今は無理、あとでなら可」という意味だから。
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    try:
        scenario = generate(
            seed=req.seed,
            date=req.date,
            season=req.season,
            absent_drivers=tuple(req.absent_drivers),
        )
    except ValueError as exc:
        ledger.release(req.subject, amount)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        matrix = build_matrix(scenario.depot, scenario.customers)
        # 既定は REPRODUCIBLE。この API は審査員が手元で計画を再現する入口
        # なので、時間より再現性を取る。当日の組み直し（エージェント層）は
        # FAST を明示して呼ぶ。
        plan = solve(
            scenario,
            matrix=matrix,
            time_limit_s=settings.solve_time_limit_s,
            workers=settings.solve_workers,
            mode=req.mode,
        )
    except Exception as exc:
        ledger.release(req.subject, amount)
        logger.exception("求解に失敗")
        raise HTTPException(status_code=500, detail="配車を組めませんでした") from exc

    ledger.record(
        req.subject,
        "solve",
        amount,
        plan_id=plan_id,
        detail=f"seed={req.seed} season={req.season} mode={req.mode.value} status={plan.status}",
    )

    body = _plan_to_dict(plan)
    body["plan_id"] = plan_id
    body["scenario"] = {
        "date": scenario.date,
        "season": scenario.season,
        "seed": scenario.seed,
        "depot": scenario.depot.name,
        "customers": len(scenario.customers),
        "usable_vehicles": len(scenario.usable_vehicles()),
        "fleet_capacity_l": scenario.fleet_capacity_l,
        "mandatory_demand_l": scenario.mandatory_demand_l,
        "total_demand_l": sum(c.demand_l for c in scenario.customers),
    }
    body["elapsed_s"] = round(time.monotonic() - started, 3)
    return body


@app.get("/api/scenario")
def scenario_view(
    seed: int = 20261015,
    date: str = "2026-10-15",
    season: str = "winter",
) -> dict:
    """当日の需要側だけを見る。計画を組まずに「今どこが切迫しているか」を読む。

    再計画のトリガー判定（W4）が最初に見る面でもある。
    """
    try:
        scenario = generate(seed=seed, date=date, season=season)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "date": scenario.date,
        "season": scenario.season,
        "seed": scenario.seed,
        "depot": {
            "name": scenario.depot.name,
            "lat": scenario.depot.lat,
            "lon": scenario.depot.lon,
            "open": minutes_to_hhmm(scenario.depot.open_min),
            "close": minutes_to_hhmm(scenario.depot.close_min),
            "load_min": scenario.depot.load_min,
        },
        "customers": [
            {
                "id": c.id,
                "name": c.name,
                "type": c.type.value,
                "lat": c.lat,
                "lon": c.lon,
                "window": (
                    f"{minutes_to_hhmm(c.window_open_min)}-"
                    f"{minutes_to_hhmm(c.window_close_min)}"
                ),
                "service_min": c.service_min,
                "demand_l": c.demand_l,
                "mandatory": c.is_mandatory,
                "days_to_empty": (
                    None
                    if c.min_days_to_empty() == float("inf")
                    else round(c.min_days_to_empty(), 2)
                ),
                "tanks": [
                    {
                        "product": t.product.value,
                        "capacity_l": t.capacity_l,
                        "level_l": t.level_l,
                        "reorder_l": t.reorder_l,
                        "daily_use_l": t.daily_use_l,
                        "needs_topup": t.needs_topup,
                    }
                    for t in c.tanks
                ],
            }
            for c in scenario.customers
        ],
        "vehicles": [
            {
                "id": v.id,
                "name": v.name,
                "compartments_l": list(v.compartments_l),
                "capacity_l": v.capacity_l,
                "driver_id": v.driver_id,
            }
            for v in scenario.vehicles
        ],
        "drivers": [
            {
                "id": d.id,
                "name": d.name,
                "available": d.available,
                "max_duty_min": d.max_duty_min,
                "max_drive_min": d.max_drive_min,
            }
            for d in scenario.drivers
        ],
    }
