"""コスト台帳と暴走ガードレール。

なぜ最初に作るか:
    エージェントを外に出す以上、URLを知った誰かが叩き続けるだけで請求が伸びる。
    別プロジェクトでこれを後付けにして怖い思いをしたので、**エージェント層
    （W4）を書く前に**枠組みだけ先に置く。あとから足す台帳は、必ずどこかの
    経路を通し忘れる。

三層で守る:
    1. 1リクエストの上限   — 1回の再計画が青天井に膨らまない
    2. 利用者ごとの1日上限 — 一人が使いすぎない
    3. 全体の1日上限       — 何人来ても請求が青天井にならない最後の砦

    3 が本命。2 だけだと、識別子を変えられれば素通りする。

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
MAX_USD_PER_REQUEST = float(os.getenv("MAX_USD_PER_REQUEST", "0.10"))
DAILY_USD_PER_SUBJECT = float(os.getenv("DAILY_USD_PER_SUBJECT", "1.00"))
DAILY_USD_GLOBAL = float(os.getenv("DAILY_USD_GLOBAL", "10.00"))

# --- 操作ごとの想定コスト（USD） -------------------------------------------
#
# **概算である。実測ではない。** W4 でエージェント層を通したら、実際の
# トークン数から測り直してここを更新すること。上限判定に使う数字なので、
# 迷ったら高めに置く。安く見積もると素通りする。
#
# モデルを変えたら、この表も必ず更新する。単価を据え置いたままモデルだけ
# 上げると、上限が実質的に外れる。
ESTIMATES: dict[str, float] = {
    "interview": 0.020,      # 診断面談の1往復（配車係から暗黙知を引き出す）
    "signal": 0.008,         # 外部シグナル→ソルバーのパラメータへの翻訳
    "trigger": 0.004,        # 再計画すべきかの判定
    "explain": 0.012,        # 計画の理由説明・代替案の提示
    "draft": 0.015,          # 対外ドラフト（スポット手配の依頼文など）の起草
    "absorb": 0.006,         # 配車係の手直しを制約として取り込む
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
    "global"（全体で使いすぎ）。どれかで配車係への伝え方が変わる。
    """

    def __init__(self, scope: str, spent: float, cap: float):
        self.scope = scope
        self.spent = spent
        self.cap = cap
        super().__init__(f"{scope} の上限に達しました（${spent:.4f} / ${cap:.2f}）")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# メモリ版（開発用）
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_memory: dict[str, dict[str, Any]] = {}
_entries: list[dict[str, Any]] = []


def _memory_day(day: str) -> dict[str, Any]:
    return _memory.setdefault(day, {"day": day, "total_usd": 0.0, "subjects": {}})


# ---------------------------------------------------------------------------
# Firestore 版
# ---------------------------------------------------------------------------


def _day_ref(day: str):
    from app.firestore_client import get_client

    return get_client().collection("ledger").document(day)


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

    if not settings.use_firestore_ledger:
        with _lock:
            d = _memory_day(day)
            total = float(d["total_usd"])
            mine = float(d["subjects"].get(subject, 0.0))
            if total + amount > DAILY_USD_GLOBAL:
                raise BudgetExceeded("global", total, DAILY_USD_GLOBAL)
            if mine + amount > DAILY_USD_PER_SUBJECT:
                raise BudgetExceeded("subject", mine, DAILY_USD_PER_SUBJECT)
            d["total_usd"] = total + amount
            d["subjects"][subject] = mine + amount
        return amount

    from google.cloud import firestore

    from app.firestore_client import get_client

    ref = _day_ref(day)

    @firestore.transactional
    def _txn(txn):
        snap = ref.get(transaction=txn)
        data = snap.to_dict() if snap.exists else {}
        total = float(data.get("total_usd", 0.0))
        subjects = data.get("subjects", {})
        mine = float(subjects.get(subject, 0.0))

        # 全体を先に見る。ここが最後の砦なので、個人の枠より優先して弾く。
        if total + amount > DAILY_USD_GLOBAL:
            raise BudgetExceeded("global", total, DAILY_USD_GLOBAL)
        if mine + amount > DAILY_USD_PER_SUBJECT:
            raise BudgetExceeded("subject", mine, DAILY_USD_PER_SUBJECT)

        subjects[subject] = mine + amount
        txn.set(ref, {"day": day, "total_usd": total + amount, "subjects": subjects})

    _txn(get_client().transaction())
    return amount


def release(subject: str, amount: float) -> None:
    """確保した枠を返す。処理が失敗したときに呼ぶ。

    失敗した処理で配車係の枠を食わせない。ここを忘れると、
    エラーが続いた利用者ほど早く上限に達することになる。
    """
    day = _today()

    if not settings.use_firestore_ledger:
        with _lock:
            d = _memory_day(day)
            d["total_usd"] = max(0.0, float(d["total_usd"]) - amount)
            d["subjects"][subject] = max(
                0.0, float(d["subjects"].get(subject, 0.0)) - amount
            )
        return

    from google.cloud import firestore

    from app.firestore_client import get_client

    ref = _day_ref(day)

    @firestore.transactional
    def _txn(txn):
        snap = ref.get(transaction=txn)
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

    return {
        "day": day,
        "backend": backend(),
        "total_usd": round(float(data.get("total_usd", 0.0)), 6),
        "subjects": data.get("subjects", {}),
        "caps": {
            "per_request_usd": MAX_USD_PER_REQUEST,
            "per_subject_daily_usd": DAILY_USD_PER_SUBJECT,
            "global_daily_usd": DAILY_USD_GLOBAL,
        },
    }


def reset_for_tests() -> None:
    """メモリ版の台帳を空にする。テスト専用。"""
    with _lock:
        _memory.clear()
        _entries.clear()


__all__ = [
    "BudgetExceeded",
    "DAILY_USD_GLOBAL",
    "DAILY_USD_PER_SUBJECT",
    "ESTIMATES",
    "MAX_USD_PER_REQUEST",
    "backend",
    "estimate",
    "record",
    "release",
    "reserve",
    "reset_for_tests",
    "usage",
]
