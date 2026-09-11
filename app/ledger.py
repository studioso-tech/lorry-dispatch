"""コスト台帳と暴走ガードレール。

なぜ最初に作るか:
    エージェントを外に出す以上、URLを知った誰かが叩き続けるだけで請求が伸びる。
    別プロジェクトでこれを後付けにして怖い思いをしたので、**エージェント層
    （W4）を書く前に**枠組みだけ先に置く。あとから足す台帳は、必ずどこかの
    経路を通し忘れる。

四層で守る:
    1. 1リクエストの上限   — 1回の再計画が青天井に膨らまない
    2. 利用者ごとの1日上限 — 一人が使いすぎない
    3. 全体の1日上限       — 何人来ても1日で使い切らない
    4. **全体の1か月上限** — 請求書に載る単位で止める最後の砦

    4 が本命。GCP の予算アラートは**通知するだけで支出を止めない**。
    実際に止められるのはここだけである。日次上限だけだと、毎日上限まで
    使って月末に30倍になる。

台帳はリクエストを**またいだ**上限しか見ない:
    1リクエストの**中**で50回ループされても、台帳が知っているのは
    「1回予約した」ことだけで素通りする。そこは app/meter.py（実測）と
    app/guardrails.py（回数と時間の上限）が受け持つ。

先に確保してから使う:
    生成してから記録すると、上限を越えた分を後追いで知ることになる。
    トランザクションで先に枠を確保し、失敗したら返す。

保存先:
    GOOGLE_CLOUD_PROJECT があれば Firestore、無ければプロセス内メモリ。
    **メモリ版は開発用**。Cloud Run で複数インスタンスに増えると
    インスタンスごとに別の台帳を持つことになり、全体上限が意味を失う。
    本番では必ず Firestore を使う（/api/healthz が今どちらかを返す）。

    v0（W1）の時点ではモデルを1回も呼ばないので、台帳は動くが金額はゼロ。
    W4 でエージェント層を通したときに初めて数字が入る。
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

from app.config import settings

logger = logging.getLogger("lorry.ledger")

# --- 上限（USD） -----------------------------------------------------------
#
# 1リクエストの上限が要点。配車の再計画は「シグナルを読む→制約に翻訳する→
# 説明を書く」で数回モデルを呼ぶ設計なので、1回あたり $0.10 は十分な余裕が
# ありつつ、暴走した場合に1回で止まる水準。
# 上限の根拠は「月にいくらまで使ってよいか」ではない。
# **一度の暴走でいくらまで飛ぶか**である。
#
# 過去に ADK の暴走で、無料クレジットの半分（2万数千円分）を**一度に**
# 失ったことがある。上限が無ければ、残高がある限り走り続ける。
#
# GCP の予算アラート（¥500 に設定済み）は、無料クレジットが尽きて実課金に
# 切り替わったことを知らせるための警報であって、支出を止めるものではない。
# **止められるのはここだけ。**
#
# 実測の目安（2026-09-03）:
#   Gemini 3.8 Flash 1回              約 $0.002
#   求解1回（Cloud Run 4vCPU × 60秒） 約 $0.006  ← モデルより高い
#
# 既定値が意味するもの:
#   1リクエスト $0.10  … 1回の暴走が約15円で頭打ちになる
#   1日 全体 $2.00     … 丸1日暴走し続けても約300円
#   1か月 全体 $15.00  … 最悪の月で約2,250円。過去の事故の10分の1以下
#
# 審査期間の想定使用量は、審査員が触るぶんで $1〜4 程度（1往復 $0.02 × 数百回）。
# 通常の利用は妨げず、暴走したときだけ効く水準に置いてある。
MAX_USD_PER_REQUEST = float(os.getenv("MAX_USD_PER_REQUEST", "0.10"))
DAILY_USD_PER_SUBJECT = float(os.getenv("DAILY_USD_PER_SUBJECT", "0.50"))
DAILY_USD_GLOBAL = float(os.getenv("DAILY_USD_GLOBAL", "2.00"))
MONTHLY_USD_GLOBAL = float(os.getenv("MONTHLY_USD_GLOBAL", "15.00"))

# --- 操作ごとの想定コスト（USD） -------------------------------------------
#
# 前提モデル: gemini-3.8-flash
# 単価（2026-12-31まで）: 入力 $0.75 / 出力 $3.75 per 100万トークン。
# **出力単価には思考トークンが含まれる。2027-01-01 に倍額になる。**
#
# **思考トークンが支配的である。** 2026-09-03 の実測（34トークンの些細な
# 質問1つ）:
#
#     思考レベル   入力  出力   思考   実費
#     low           34    17    498   $0.00196
#     medium(既定)  34    23    522   $0.00207
#     high          34    21   1220   $0.00468
#
# 出力17〜23に対して思考が498〜1220。**実費のほぼ全部が思考**である。
# しかも 3.8-flash は思考レベル minimal を受け付けないので、思考を
# 完全に切ることはできない。最も安いのが low。
#
# 下の数字は、この実測に「実際のプロンプト（当日の需要家15軒とタンク在庫と
# 計画で数千トークン）」ぶんを足した見積り。**まだ実測ではない。**
# W4 でエージェント層を通したら usage_metadata.thoughts_token_count を
# 読んで置き直すこと。上限判定に使う数字なので、迷ったら高めに置く。
# 安く見積もると素通りする。
#
# モデルを変えたら、この表も必ず更新する。単価を据え置いたままモデルだけ
# 上げると、上限が実質的に外れる。
ESTIMATES: dict[str, float] = {
    "interview": 0.025,      # 診断面談の1往復（medium。会話が長くなる）
    "signal": 0.010,         # 外部シグナル→ソルバーのパラメータへの翻訳（low）
    "trigger": 0.006,        # 再計画すべきかの判定（low。短いが思考は出る）
    "explain": 0.020,        # 計画の理由説明・代替案の提示（medium。入力が長い）
    "draft": 0.022,          # 対外ドラフトの起草（medium。出力も長い）
    "absorb": 0.008,         # 配車係の手直しを制約として取り込む（low）
    "intake": 0.010,         # FAX・メモからの当日発注の読み取り（low。画像1枚）
}

#: 操作ごとの思考レベル。**既定の medium に任せない。**
#: 判定や翻訳のような短い仕事に high を使うと、実費が2倍以上になる（上の実測）。
THINKING_LEVEL: dict[str, str] = {
    "interview": "medium",
    "signal": "low",
    "trigger": "low",
    "explain": "medium",
    "draft": "medium",
    "absorb": "low",
    "intake": "low",
}

#: 求解そのものは無料。OR-Tools は Cloud Run 上で動くだけで従量課金がない。
#: 表に載せてあるのは「載っていない操作は0.01と見なす」既定に落とさないため。
ESTIMATES["solve"] = 0.0


def estimate(kind: str) -> float:
    """操作の想定コスト。表に無い操作は高めの既定値で弾く。"""
    return ESTIMATES.get(kind, 0.02)


class BudgetExceeded(Exception):
    """上限に達したので実行しない。

    scope は "request"（1回で使いすぎ）/ "subject"（その人が使いすぎ）/
    "global"（その日に全体で使いすぎ）/ "month"（今月使い切った）。
    どれかで配車係への伝え方が変わる。month だけは「明日まで待っても
    直らない」ので、伝え方が根本的に違う。
    """

    def __init__(self, scope: str, spent: float, cap: float):
        self.scope = scope
        self.spent = spent
        self.cap = cap
        super().__init__(f"{scope} の上限に達しました（${spent:.4f} / ${cap:.2f}）")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _this_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


# ---------------------------------------------------------------------------
# メモリ版（開発用）
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_memory: dict[str, dict[str, Any]] = {}
_memory_months: dict[str, float] = {}
_entries: list[dict[str, Any]] = []


def _memory_day(day: str) -> dict[str, Any]:
    return _memory.setdefault(day, {"day": day, "total_usd": 0.0, "subjects": {}})


# ---------------------------------------------------------------------------
# Firestore 版
# ---------------------------------------------------------------------------


def _day_ref(day: str):
    """その日の台帳ドキュメント。

    コレクション名は firestore_client.collection() を通す。相乗り先の
    Wayfinder が同じ "(default)" データベースの `ledger/{day}` を使っており、
    接頭辞なしで書くと互いの台帳を上書きし合う。
    """
    from app.firestore_client import collection, get_client

    return get_client().collection(collection("ledger")).document(day)


def _month_ref(month: str):
    """今月の合計。日次とは別のドキュメントに持つ。

    月をまたいだ集計を日次ドキュメントの走査で出すこともできるが、
    予約のたびに30件読むことになる。上限判定は毎回通る経路なので、
    1件で済む形にしておく。
    """
    from app.firestore_client import collection, get_client

    return get_client().collection(collection("ledger_month")).document(month)


def backend() -> str:
    """今どちらの台帳を使っているか。/api/healthz が返す。"""
    return "firestore" if settings.use_firestore_ledger else "memory"


# ---------------------------------------------------------------------------
# 枠の確保・返却・記録
# ---------------------------------------------------------------------------


def reserve(subject: str, kind: str, usd: float | None = None) -> float:
    """使う前に枠を確保する。上限を越えていれば BudgetExceeded。

    Args:
        subject: 利用者の識別子（配車係のID、またはデモのセッションID）。
        kind: 操作の種類。ESTIMATES のキー。
        usd: 金額を明示する場合。省略時は kind から見積もる。

    Returns:
        確保した金額（USD）。返却や記録に使う。

    Raises:
        BudgetExceeded: いずれかの上限に達した。
    """
    amount = estimate(kind) if usd is None else usd

    # 1リクエストの上限は保存先に関係なく効く。ここを最初に見る。
    if amount > MAX_USD_PER_REQUEST:
        raise BudgetExceeded("request", amount, MAX_USD_PER_REQUEST)

    day = _today()

    month = _this_month()

    if not settings.use_firestore_ledger:
        with _lock:
            d = _memory_day(day)
            total = float(d["total_usd"])
            mine = float(d["subjects"].get(subject, 0.0))
            month_total = _memory_months.get(month, 0.0)
            # 月を先に見る。明日まで待っても直らないほうから弾く。
            if month_total + amount > MONTHLY_USD_GLOBAL:
                raise BudgetExceeded("month", month_total, MONTHLY_USD_GLOBAL)
            if total + amount > DAILY_USD_GLOBAL:
                raise BudgetExceeded("global", total, DAILY_USD_GLOBAL)
            if mine + amount > DAILY_USD_PER_SUBJECT:
                raise BudgetExceeded("subject", mine, DAILY_USD_PER_SUBJECT)
            d["total_usd"] = total + amount
            d["subjects"][subject] = mine + amount
            _memory_months[month] = month_total + amount
        return amount

    from google.cloud import firestore

    from app.firestore_client import get_client

    ref = _day_ref(day)
    mref = _month_ref(month)

    @firestore.transactional
    def _txn(txn):
        # **読みは全部、書きより先に。** Firestore のトランザクションの規則。
        snap = ref.get(transaction=txn)
        msnap = mref.get(transaction=txn)

        data = snap.to_dict() if snap.exists else {}
        total = float(data.get("total_usd", 0.0))
        subjects = data.get("subjects", {})
        mine = float(subjects.get(subject, 0.0))
        month_total = float((msnap.to_dict() or {}).get("total_usd", 0.0)) if msnap.exists else 0.0

        # 広いほうから弾く。月が尽きているなら、明日まで待っても直らない。
        if month_total + amount > MONTHLY_USD_GLOBAL:
            raise BudgetExceeded("month", month_total, MONTHLY_USD_GLOBAL)
        if total + amount > DAILY_USD_GLOBAL:
            raise BudgetExceeded("global", total, DAILY_USD_GLOBAL)
        if mine + amount > DAILY_USD_PER_SUBJECT:
            raise BudgetExceeded("subject", mine, DAILY_USD_PER_SUBJECT)

        subjects[subject] = mine + amount
        txn.set(ref, {"day": day, "total_usd": total + amount, "subjects": subjects})
        txn.set(mref, {"month": month, "total_usd": month_total + amount})

    _txn(get_client().transaction())
    return amount


def release(subject: str, amount: float) -> None:
    """確保した枠を返す。処理が失敗したときに呼ぶ。

    失敗した処理で配車係の枠を食わせない。ここを忘れると、
    エラーが続いた利用者ほど早く上限に達することになる。
    """
    day = _today()

    month = _this_month()

    if not settings.use_firestore_ledger:
        with _lock:
            d = _memory_day(day)
            d["total_usd"] = max(0.0, float(d["total_usd"]) - amount)
            d["subjects"][subject] = max(
                0.0, float(d["subjects"].get(subject, 0.0)) - amount
            )
            _memory_months[month] = max(0.0, _memory_months.get(month, 0.0) - amount)
        return

    from google.cloud import firestore

    from app.firestore_client import get_client

    ref = _day_ref(day)
    mref = _month_ref(month)

    @firestore.transactional
    def _txn(txn):
        snap = ref.get(transaction=txn)
        msnap = mref.get(transaction=txn)
        if not snap.exists:
            return
        data = snap.to_dict()
        subjects = data.get("subjects", {})
        subjects[subject] = max(0.0, float(subjects.get(subject, 0.0)) - amount)
        txn.set(
            ref,
            {
                "day": day,
                "total_usd": max(0.0, float(data.get("total_usd", 0.0)) - amount),
                "subjects": subjects,
            },
        )
        if msnap.exists:
            mtotal = float((msnap.to_dict() or {}).get("total_usd", 0.0))
            txn.set(mref, {"month": month, "total_usd": max(0.0, mtotal - amount)})

    try:
        _txn(get_client().transaction())
    except Exception as exc:  # 返却の失敗で本処理を壊さない
        logger.warning("枠の返却に失敗: %s", exc)


def record(
    subject: str,
    kind: str,
    usd: float,
    *,
    plan_id: str = "",
    detail: str = "",
) -> dict[str, Any]:
    """1件を台帳に残す。

    ログにも COST 行として出す。Cloud Logging から
    `jsonPayload.message:"COST"` で拾えるようにしておくと、
    Firestore を開かなくても当日の傾向が読める。
    """
    day = _today()
    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "day": day,
        "subject": subject,
        "kind": kind,
        "usd": round(usd, 6),
        "plan_id": plan_id,
        "detail": detail,
    }
    logger.info("COST %s", entry)

    if not settings.use_firestore_ledger:
        with _lock:
            _entries.append(entry)
        return entry

    try:
        _day_ref(day).collection("entries").add(entry)
    except Exception as exc:
        logger.warning("台帳への記録に失敗: %s", exc)
    return entry


def usage(day: str | None = None) -> dict[str, Any]:
    """その日の使用状況。画面と運用の確認用。"""
    day = day or _today()

    if not settings.use_firestore_ledger:
        with _lock:
            d = _memory.get(day, {"day": day, "total_usd": 0.0, "subjects": {}})
            data = {**d, "subjects": dict(d["subjects"])}
    else:
        snap = _day_ref(day).get()
        data = snap.to_dict() if snap.exists else {"day": day, "total_usd": 0.0, "subjects": {}}

    month = _this_month()
    if not settings.use_firestore_ledger:
        with _lock:
            month_total = _memory_months.get(month, 0.0)
    else:
        msnap = _month_ref(month).get()
        month_total = float((msnap.to_dict() or {}).get("total_usd", 0.0)) if msnap.exists else 0.0

    return {
        "day": day,
        "month": month,
        "backend": backend(),
        "total_usd": round(float(data.get("total_usd", 0.0)), 6),
        "month_total_usd": round(month_total, 6),
        "subjects": data.get("subjects", {}),
        "caps": {
            "per_request_usd": MAX_USD_PER_REQUEST,
            "per_subject_daily_usd": DAILY_USD_PER_SUBJECT,
            "global_daily_usd": DAILY_USD_GLOBAL,
            "global_monthly_usd": MONTHLY_USD_GLOBAL,
        },
        # 予算アラートは通知するだけで支出を止めない。実際に止まるのは上の上限。
        "note": "GCPの予算アラートは通知のみ。支出を止めるのはこの上限だけ。",
    }


def reset_for_tests() -> None:
    """メモリ版の台帳を空にする。テスト専用。"""
    with _lock:
        _memory.clear()
        _memory_months.clear()
        _entries.clear()


__all__ = [
    "BudgetExceeded",
    "DAILY_USD_GLOBAL",
    "DAILY_USD_PER_SUBJECT",
    "ESTIMATES",
    "THINKING_LEVEL",
    "MAX_USD_PER_REQUEST",
    "MONTHLY_USD_GLOBAL",
    "backend",
    "estimate",
    "record",
    "release",
    "reserve",
    "reset_for_tests",
    "usage",
]
