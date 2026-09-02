"""環境設定の一元管理。

Cloud Run 上ではランタイムの環境変数、ローカルでは .env から読む。
設定を各所で os.getenv するのをやめているのは、既定値がファイルごとに
食い違って「ローカルでは動くのに本番だけ違うモデルを呼んでいた」類の
事故を防ぐため。既定値はここにしか書かない。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """このプロセスの設定一式。"""

    project_id: str
    location: str
    genai_location: str
    gemini_model: str
    firestore_database: str
    use_vertexai: bool
    #: 求解の打ち切り（秒）。長くすれば解は良くなるが、画面が待たされる。
    solve_time_limit_s: float
    #: CP-SAT の並列探索数。Cloud Run に割り当てた vCPU に合わせる。
    solve_workers: int
    #: 台帳のバックエンドに Firestore を使うか。未設定ならプロセス内メモリ。
    use_firestore_ledger: bool

    @classmethod
    def from_env(cls) -> "Settings":
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        return cls(
            project_id=project,
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            # Gemini API は global エンドポイント経由で呼ぶ（提供状況が安定するため）
            genai_location=os.getenv("GOOGLE_GENAI_LOCATION", "global"),
            # エージェント層は W4。ここはまだ実際には呼ばれない。
            # **W4 で着手する前に models.list() で実在を確認すること。**
            # 別プロジェクトでは 2026-08-19 時点で gemini-3.5-pro が存在せず、
            # 3.5 以上は flash 系のみだった。モデル名を憶測で置くと、
            # 最初の1回目の呼び出しまで誤りに気づけない。
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.7-flash"),
            firestore_database=os.getenv("FIRESTORE_DATABASE", "(default)"),
            use_vertexai=os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE").upper() == "TRUE",
            solve_time_limit_s=float(os.getenv("SOLVE_TIME_LIMIT_S", "60")),
            solve_workers=int(os.getenv("SOLVE_WORKERS", "8")),
            use_firestore_ledger=bool(project)
            and os.getenv("USE_FIRESTORE_LEDGER", "TRUE").upper() == "TRUE",
        )


settings = Settings.from_env()

__all__ = ["Settings", "settings"]
