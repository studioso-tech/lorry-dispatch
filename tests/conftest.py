"""テスト全体の前提。

**app.config を読み込む前に環境変数を置く必要がある。**
Settings は import 時に1回だけ環境を読むので、テストの中で os.environ を
書き換えても遅い。conftest.py はテストモジュールより先に読まれるので、
ここが唯一の置き場所になる。

求解時間を既定の60秒から縮めているのは、テストが確かめているのが
「制約を守っているか」であって「最適か」ではないから。時間切れで
打ち切った解も制約は満たしている。最適性そのものを見るテストは、
その場で上限を指定して条件を作っている。
"""

from __future__ import annotations

import os

os.environ.setdefault("SOLVE_TIME_LIMIT_S", "20")
os.environ.setdefault("SOLVE_WORKERS", "8")

# Firestore へ触りに行かせない。テストはメモリ台帳で動く。
os.environ.setdefault("USE_FIRESTORE_LEDGER", "FALSE")
