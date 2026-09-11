"""逃げ道の記録と、恒常化の監視。**このプロジェクトの新規性の核。**

逃げ道（計画配送を明日にずらす、拘束を延ばす、配送圏を越える、欠品）は
**一時的なら可、恒常的なら構造の欠陥**（実務）。1日ずつ見ていると、どの日も
「今日は仕方がない」で済んでしまう。日をまたいで数えて初めて、構造の問題が
見える。

    その日の組み直し → 使った逃げ道を記録 → 直近30日で数える
      → 目安を超えたら「恒常化の恐れ」として、体制の見直しを提案する

提案の中身は体制設計（scripts/size_carrier.py）につながる: 専属の積み増し、
非専属の確保、配送圏の見直し。**どれを選ぶかは元売と運送会社の判断**で、
ここはその判断を促す材料を出すところまで。

日付は計画の対象日（シナリオの日付）で数える。デモでひと月ぶんを模擬できる
ようにするため。
"""

from __future__ import annotations

import datetime
import threading
from collections import Counter
from typing import Any

from app.config import settings
from core.fleet_plan import ESCAPE_TOLERANCE
from core.replan import Escape

#: 恒常化を見る窓（日）。
WINDOW_DAYS = 30

_lock = threading.Lock()
_memory: dict[str, list[dict[str, Any]]] = {}


def _ref(date: str):
    from app.firestore_client import collection, get_client

    return get_client().collection(collection("escapes")).document(date)


def record(date: str, escapes: list[Escape] | tuple[Escape, ...], *, plan_id: str = "") -> None:
    """その日に使った逃げ道を足す。試算（dry_run）の結果は記録しないこと。"""
    rows = [
        {
            "kind": e.kind,
            "customer_id": e.customer_id,
            "customer_name": e.customer_name,
            "volume_l": e.volume_l,
            "plan_id": plan_id,
        }
        for e in escapes
    ]
    if not rows:
        return
    if not settings.use_firestore_ledger:
        with _lock:
            _memory.setdefault(date, []).extend(rows)
        return
    from google.cloud import firestore

    _ref(date).set({"date": date, "rows": firestore.ArrayUnion(rows)}, merge=True)


def _rows_between(start: str, end: str) -> dict[str, list[dict[str, Any]]]:
    if not settings.use_firestore_ledger:
        with _lock:
            return {d: list(r) for d, r in _memory.items() if start <= d <= end}
    from app.firestore_client import collection, get_client

    q = (
        get_client().collection(collection("escapes"))
        .where("date", ">=", start).where("date", "<=", end)
    )
    return {snap.get("date"): snap.get("rows") or [] for snap in q.stream()}


def summary(as_of: str, *, window_days: int = WINDOW_DAYS) -> dict[str, Any]:
    """直近 window_days 日の逃げ道を数え、恒常化の恐れがあるかを返す。"""
    end = datetime.date.fromisoformat(as_of)
    start = end - datetime.timedelta(days=window_days - 1)
    by_day = _rows_between(start.isoformat(), end.isoformat())
    days = sorted(d for d, rows in by_day.items() if rows)
    per_customer = Counter(r["customer_name"] for rows in by_day.values() for r in rows)
    tolerance_days = ESCAPE_TOLERANCE * window_days
    chronic = len(days) > tolerance_days
    repeat = [(n, k) for n, k in per_customer.most_common() if k >= 2]

    if chronic:
        message = (
            f"直近{window_days}日で、計画配送をずらした日が{len(days)}日あります"
            f"（目安は{tolerance_days:.1f}日まで）。**恒常化の恐れがあります。**"
            "一時的なら可、恒常的なら構造の欠陥です。専属の積み増し、非専属の確保、"
            "配送圏の見直しのいずれかを、元売と運送会社で検討してください。"
        )
    elif days:
        message = (
            f"直近{window_days}日で、計画配送をずらした日は{len(days)}日"
            f"（目安{tolerance_days:.1f}日以内）。一時的な範囲です。"
        )
    else:
        message = f"直近{window_days}日で、逃げ道は使っていません。"
    if repeat:
        message += " 繰り返し後回しになっている先: " + "、".join(f"{n}（{k}回）" for n, k in repeat) + "。"

    return {
        "as_of": as_of,
        "window_days": window_days,
        "escape_days": days,
        "escape_count": sum(per_customer.values()),
        "tolerance_days": tolerance_days,
        "chronic": chronic,
        "repeated_customers": [{"customer_name": n, "count": k} for n, k in repeat],
        "message_ja": message,
    }


def reset_for_tests() -> None:
    with _lock:
        _memory.clear()
