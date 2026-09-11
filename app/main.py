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

import base64

from app import escapes, intake, ledger
from app.config import settings
from app.meter import CostMeter
from core.domain import minutes_to_hhmm
from core.generator import generate
from core.domain import Product
from core.replan import NewOrder, format_replan_ja, replan
from core.returns import RESPONSIBILITY_JA, ReturnEvent, format_return_ja, handle_return
from core.solver import Plan, SolveMode, VehicleDay, format_plan_ja, solve
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
    declined_customers: list[str] = Field(
        default_factory=list,
        description="今日の受入を見送る需要家ID（例 ['c03']）。配車しない",
    )
    subject: str = Field(default="demo", description="利用者の識別子。台帳の単位")
    mode: SolveMode = Field(
        default=SolveMode.REPRODUCIBLE,
        description=(
            "reproducible: どのマシンでも同じ計画を返す（探索量で打ち切る。"
            "時間は読めない）。fast: 時間で打ち切る（再現は保証しない）"
        ),
    )


def _vehicle_days_to_list(vehicle_days: tuple[VehicleDay, ...]) -> list[dict]:
    """車ごとの1日を JSON にする。朝の計画と当日の組み直しで共有する。"""
    return [
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
        for vd in vehicle_days
    ]


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
        "vehicle_days": _vehicle_days_to_list(plan.vehicle_days),
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
            declined_customers=tuple(req.declined_customers),
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


class OrderIn(BaseModel):
    """当日の発注1件。**証拠の残る形でしか受けない**（発注システムかFAX）。"""

    customer_id: str
    product: Product
    volume_l: int = Field(gt=0)
    source: str = Field(pattern="^(system|fax)$", description="system（発注システム）か fax。電話は不可")
    evidence_ref: str = Field(min_length=1, description="伝票番号・FAX画像の保存先など")


class ReplanRequest(PlanRequest):
    """当日の組み直しの要求。朝の計画と同じ日の条件に、いまの時刻と発注を足す。"""

    now: str = Field(pattern="^([01][0-9]|2[0-3]):[0-5][0-9]$", description="いまの時刻 HH:MM")
    orders: list[OrderIn] = Field(min_length=1)
    spot_standby: int = Field(default=0, ge=0, le=10, description="いま呼べる待機の非専属の台数")


def _replan_body(req: PlanRequest, now: str, orders: list[NewOrder], spot_standby: int,
                 *, dry_run: bool, plan_id: str) -> dict:
    """朝の計画を再現モードで組み直し、当日の発注を差し込む。

    朝の計画を**再現モードで組み直す**のは、審査員が /api/plan で見た朝の計画と
    必ず同じものを土台にするため（再現モードは同じ入力から同じ計画を返す）。
    組み直しそのものは即応モード（時間で打ち切る）。

    試算（dry_run）でなければ、使った逃げ道を記録する（恒常化の監視が数える）。
    """
    scenario = generate(
        seed=req.seed,
        date=req.date,
        season=req.season,
        absent_drivers=tuple(req.absent_drivers),
        declined_customers=tuple(req.declined_customers),
    )
    base = solve(
        scenario,
        matrix=build_matrix(scenario.depot, scenario.customers),
        workers=settings.solve_workers,
        mode=SolveMode.REPRODUCIBLE,
    )
    hh, mm = now.split(":")
    result = replan(
        scenario,
        base,
        now_min=int(hh) * 60 + int(mm),
        orders=orders,
        spot_standby=spot_standby,
        time_limit_s=settings.solve_time_limit_s,
        dry_run=dry_run,
    )
    if not dry_run and result.escapes:
        escapes.record(scenario.date, result.escapes, plan_id=plan_id)
    return {
        "plan_id": plan_id,
        "dry_run": result.dry_run,
        "ok": result.ok,
        "summary_ja": result.summary_ja(),
        "text_ja": format_replan_ja(result),
        "changes": list(result.changes),
        "shortfalls": list(result.shortfalls),
        "escapes": [
            {"kind": e.kind, "customer_id": e.customer_id, "customer_name": e.customer_name,
             "volume_l": e.volume_l}
            for e in result.escapes
        ],
        "unfilled": [
            {"customer_id": u.customer_id, "customer_name": u.customer_name, "demand_l": u.demand_l}
            for u in result.unfilled
        ],
        "spot_called": list(result.spot_called),
        "frozen_trip_count": len(result.frozen_trips),
        "vehicle_days": _vehicle_days_to_list(result.vehicle_days),
        "base_summary_ja": base.summary_ja(),
        "escape_watch": escapes.summary(scenario.date),
        "diagnostics": {
            "base_reproducible": base.diagnostics.get("reproducible"),
            "replan_status": result.sub_plan.status,
            "replan_mode": result.sub_plan.diagnostics.get("solve_mode"),
        },
    }


@app.post("/api/replan")
def make_replan(req: ReplanRequest) -> dict:
    """朝の計画に、当日の発注（発注システムかFAX）を差し込んで組み直す。"""
    plan_id = uuid.uuid4().hex[:12]
    started = time.monotonic()
    try:
        amount = ledger.reserve(req.subject, "solve")
    except ledger.BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    try:
        orders = [
            NewOrder(o.customer_id, o.product, o.volume_l, o.source, o.evidence_ref)
            for o in req.orders
        ]
        body = _replan_body(req, req.now, orders, req.spot_standby, dry_run=False, plan_id=plan_id)
    except ValueError as exc:
        ledger.release(req.subject, amount)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        ledger.release(req.subject, amount)
        logger.exception("組み直しに失敗")
        raise HTTPException(status_code=500, detail="組み直せませんでした") from exc
    ledger.record(req.subject, "solve", amount, plan_id=plan_id,
                  detail=f"replan seed={req.seed} now={req.now} orders={len(orders)}")
    body["elapsed_s"] = round(time.monotonic() - started, 3)
    return body


# ---------------------------------------------------------------------------
# 当日発注の受け口（Gemini は読み取りと分類だけ）
# ---------------------------------------------------------------------------


class FaxIntakeRequest(PlanRequest):
    """FAX の発注書を受けて、読めたらそのまま組み直す。"""

    now: str = Field(pattern="^([01][0-9]|2[0-3]):[0-5][0-9]$")
    image_base64: str = Field(min_length=16, description="FAX 画像（base64）")
    mime: str = Field(default="image/png", pattern="^(image/(png|jpeg)|application/pdf)$")
    spot_standby: int = Field(default=0, ge=0, le=10)


class NoteIntakeRequest(PlanRequest):
    """配車係のメモ（「宇品港から電話でレギュラー8kL」等）を受ける。"""

    now: str = Field(pattern="^([01][0-9]|2[0-3]):[0-5][0-9]$")
    note: str = Field(min_length=2, max_length=500)
    spot_standby: int = Field(default=0, ge=0, le=10)


def _intake_response(req: PlanRequest, got: "intake.Intake", now: str, spot_standby: int,
                     plan_id: str) -> dict:
    """読み取った結果から、組み直す・試算する・確認を求める、を決める。

    **確認が要るものが1つでもあれば組み直さない。** 名前が絞れない、量が
    読めない等を推測で埋めて車を動かすより、配車係に確かめてもらうほうが安い。
    """
    base = {
        "plan_id": plan_id,
        "read": got.raw,
        "issues": got.issues,
        "notice": got.notice,
        "orders": [
            {"customer_id": o.customer_id, "product": o.product.value, "volume_l": o.volume_l,
             "source": o.source, "evidence_ref": o.evidence_ref}
            for o in got.orders
        ],
        "model_usd": round(got.usd, 6),
    }
    if not got.ready:
        base["action"] = "confirm"
        base["summary_ja"] = "確認が要ります: " + " / ".join(got.issues or ["発注が読み取れませんでした"])
        return base
    replanned = _replan_body(req, now, got.orders, spot_standby, dry_run=got.notice, plan_id=plan_id)
    base["action"] = "dry_run" if got.notice else "replanned"
    base["replan"] = replanned
    base["summary_ja"] = replanned["summary_ja"]
    return base


@app.post("/api/intake/fax")
def intake_fax(req: FaxIntakeRequest) -> dict:
    """FAX の発注書を読み、読めたらそのまま組み直す（証拠＝画像のハッシュ）。"""
    plan_id = uuid.uuid4().hex[:12]
    try:
        amount = ledger.reserve(req.subject, "intake")
    except ledger.BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    try:
        image = base64.b64decode(req.image_base64, validate=True)
        scenario = generate(seed=req.seed, date=req.date, season=req.season)
        meter = CostMeter(request_id=plan_id, ceiling_usd=ledger.MAX_USD_PER_REQUEST)
        got = intake.read_fax(image, req.mime, scenario.customers, caller=intake.gemini_caller, meter=meter)
        body = _intake_response(req, got, req.now, req.spot_standby, plan_id)
    except ValueError as exc:
        ledger.release(req.subject, amount)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        ledger.release(req.subject, amount)
        logger.exception("FAXの受け付けに失敗")
        raise HTTPException(status_code=502, detail=f"FAXを読めませんでした: {type(exc).__name__}") from exc
    ledger.record(req.subject, "intake", amount, plan_id=plan_id,
                  detail=f"fax action={body['action']} orders={len(body['orders'])}")
    return body


@app.post("/api/intake/note")
def intake_note(req: NoteIntakeRequest) -> dict:
    """配車係のメモを読む。電話の連絡は予告として試算だけ返し、確定させない。"""
    plan_id = uuid.uuid4().hex[:12]
    try:
        amount = ledger.reserve(req.subject, "intake")
    except ledger.BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    try:
        scenario = generate(seed=req.seed, date=req.date, season=req.season)
        meter = CostMeter(request_id=plan_id, ceiling_usd=ledger.MAX_USD_PER_REQUEST)
        got = intake.read_note(req.note, scenario.customers, caller=intake.gemini_caller, meter=meter)
        body = _intake_response(req, got, req.now, req.spot_standby, plan_id)
    except ValueError as exc:
        ledger.release(req.subject, amount)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        ledger.release(req.subject, amount)
        logger.exception("メモの受け付けに失敗")
        raise HTTPException(status_code=502, detail=f"メモを読めませんでした: {type(exc).__name__}") from exc
    ledger.record(req.subject, "intake", amount, plan_id=plan_id,
                  detail=f"note action={body['action']} orders={len(body['orders'])}")
    return body


class ReturnRequest(PlanRequest):
    """持ち戻りの連絡（計画配送の先で、持って行った量を降ろせない）。"""

    vehicle_id: str
    customer_id: str
    product: Product
    volume_l: int = Field(gt=0)


@app.post("/api/return")
def make_return(req: ReturnRequest) -> dict:
    """持ち戻りへの手を優先順に返す: 近くの降ろし先 → クッション → 買い戻し。

    朝の計画は再現モードで組み直す（/api/plan と同じ計画を土台にする）。
    費用は元売の管理責任で、運送会社・SSへの請求は無い（実務）。
    """
    try:
        scenario = generate(
            seed=req.seed, date=req.date, season=req.season,
            absent_drivers=tuple(req.absent_drivers),
            declined_customers=tuple(req.declined_customers),
        )
        plan = solve(scenario, matrix=build_matrix(scenario.depot, scenario.customers),
                     workers=settings.solve_workers, mode=SolveMode.REPRODUCIBLE)
        r = handle_return(scenario, plan, ReturnEvent(req.vehicle_id, req.customer_id,
                                                      req.product, req.volume_l))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    c = r.chosen
    return {
        "summary_ja": r.summary_ja(),
        "text_ja": format_return_ja(r),
        "responsibility_ja": RESPONSIBILITY_JA,
        "chosen": {"kind": c.kind, "destination": c.destination, "extra_min": c.extra_min,
                   "arrive": minutes_to_hhmm(c.arrive_min) if c.arrive_min is not None else None,
                   "delay_min": c.delay_min, "handoff": list(c.handoff)},
        "options": [
            {"rank": o.rank, "kind": o.kind, "destination": o.destination, "extra_min": o.extra_min,
             "feasible": o.feasible, "reason": o.reason, "delay_min": o.delay_min,
             "handoff": list(o.handoff)}
            for o in r.options
        ],
    }


@app.get("/api/escapes")
def escape_watch(as_of: str = "2026-10-15") -> dict:
    """直近30日の逃げ道と、恒常化の恐れ。"""
    try:
        return escapes.summary(as_of)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
                "mandatory_basis": c.mandatory_basis,
                "ordering": c.ordering,
                "orders": [{"product": p.value, "l": q} for p, q in c.orders],
                "accepts_delivery": c.accepts_delivery,
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
                        "below_reorder": t.is_below_reorder,
                        "stockout_before_next_delivery": t.stockout_before_next_delivery,
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
