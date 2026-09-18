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
    #: 画面とAPIを使うための合言葉。**空なら関所を置かない**（手元と試験）。
    #: 審査ではURLを公開にするので、ここを置いて通りすがりの連打を止める。
    #: 秘密ではない（審査員に文字で渡す）。守るのは費用であって中身ではない。
    demo_key: str

    @classmethod
    def from_env(cls) -> "Settings":
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        return cls(
            project_id=project,
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            # **Gemini は必ず global エンドポイントで呼ぶ。**
            # 2026-09-03 の実機確認で、gemini-3.8-flash も 3.7-flash も
            # us-central1 では 404（Publisher model not found）だった。
            # global では両方とも通る。
            #
            # モデルを呼ぶときは location ではなく **genai_location** を使うこと。
            # location（us-central1）は Cloud Run と Firestore 用で、別物。
            genai_location=os.getenv("GOOGLE_GENAI_LOCATION", "global"),
            # エージェント層は W4。ここはまだ実際には呼ばれない。
            #
            # **2026-09-03 に studio-so-test で実機確認済み**
            # （scripts/check_model.py で再確認できる）。
            #   ・カタログに存在する
            #   ・global エンドポイントで実際に呼べる
            #   ・us-central1 では 404。**リージョン指定では呼べない**
            #
            # 3.8-flash は 3.7-flash と同じ基盤の上に構築されており、改善は
            # ソフトウェアエンジニアリングとエージェント的タスクに集中している。
            # 単価も同じ（$0.75 / $3.75 per 1M、2026-12-31まで）。本PJの用途に
            # 正面から合うので、こちらを既定にする。
            #
            # モデル名を憶測で置いてはいけない。別プロジェクトでは 2026-08-19
            # 時点で gemini-3.5-pro が存在せず、3.5 以上は flash 系のみだった。
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            firestore_database=os.getenv("FIRESTORE_DATABASE", "(default)"),
            use_vertexai=os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE").upper() == "TRUE",
            solve_time_limit_s=float(os.getenv("SOLVE_TIME_LIMIT_S", "60")),
            solve_workers=int(os.getenv("SOLVE_WORKERS", "8")),
            use_firestore_ledger=bool(project)
            and os.getenv("USE_FIRESTORE_LEDGER", "TRUE").upper() == "TRUE",
            demo_key=os.getenv("DEMO_KEY", ""),
        )


settings = Settings.from_env()

__all__ = ["Settings", "settings"]
