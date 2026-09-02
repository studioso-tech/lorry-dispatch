"""Firestore アクセス層。

v0（W1）で使うのはコスト台帳だけ。エージェント層（W4）が入ると
決定ログ（agent_event）と、配車係の手直しから学んだ制約が加わる。

コレクション構成（予定）:
    ledger/{day}                     — 1日ぶんの使用額
    ledger/{day}/entries/{auto}      — 明細
    plans/{plan_id}                  — 出した計画のスナップショット
    plans/{plan_id}/events/{auto}    — agent_event（何を見て、なぜ組み直したか）
    rules/{rule_id}                  — 配車係の手直しから学んだ制約

クライアントを lru_cache で使い回すのは、Cloud Run のインスタンスあたり
接続を1本に保つため。リクエストごとに作ると接続が積み上がる。
"""

from __future__ import annotations

from functools import lru_cache

from google.cloud import firestore

from app.config import settings


@lru_cache(maxsize=1)
def get_client() -> firestore.Client:
    """Firestore クライアント。プロセス内で使い回す。"""
    return firestore.Client(
        project=settings.project_id or None,
        database=settings.firestore_database,
    )


__all__ = ["get_client"]
