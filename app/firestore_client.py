"""Firestore アクセス層。

v0（W1）で使うのはコスト台帳だけ。エージェント層（W4）が入ると
決定ログ（agent_event）と、配車係の手直しから学んだ制約が加わる。

コレクション構成（予定）:
    lorry_ledger/{day}                  — 1日ぶんの使用額
    lorry_ledger/{day}/entries/{auto}   — 明細
    lorry_plans/{plan_id}               — 出した計画のスナップショット
    lorry_plans/{plan_id}/events/{auto} — agent_event（何を見て、なぜ組み直したか）
    lorry_rules/{rule_id}               — 配車係の手直しから学んだ制約

**接頭辞 lorry_ は飾りではない。**
このプロジェクトは既存の GCP プロジェクトに相乗りしており、同じ Firestore の
"(default)" データベースを別のアプリ（Wayfinder）と共有している。向こうも
`ledger/{day}` を使っているので、接頭辞を付けずに書くと両方の台帳が
同じ文書を上書きし合う。金額が混ざるだけでなく、相手の全体上限まで壊す。

専用プロジェクトに移すときは接頭辞を外してよいが、外す理由が無ければ
そのままでよい。COLLECTION_PREFIX を空にすれば従来名になる。

クライアントを lru_cache で使い回すのは、Cloud Run のインスタンスあたり
接続を1本に保つため。リクエストごとに作ると接続が積み上がる。
"""

from __future__ import annotations

import os
from functools import lru_cache

from google.cloud import firestore

from app.config import settings

#: このアプリのコレクション名に付ける接頭辞。相乗り先との衝突を防ぐ。
COLLECTION_PREFIX = os.getenv("FIRESTORE_COLLECTION_PREFIX", "lorry_")


def collection(name: str) -> str:
    """接頭辞を付けたコレクション名。直接文字列を書かず必ずこれを通す。"""
    return f"{COLLECTION_PREFIX}{name}"


@lru_cache(maxsize=1)
def get_client() -> firestore.Client:
    """Firestore クライアント。プロセス内で使い回す。"""
    return firestore.Client(
        project=settings.project_id or None,
        database=settings.firestore_database,
    )


__all__ = ["COLLECTION_PREFIX", "collection", "get_client"]
