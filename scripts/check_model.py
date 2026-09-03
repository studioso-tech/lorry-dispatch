"""使おうとしているモデルが、このプロジェクトから本当に呼べるかを確かめる。

なぜ要るか:
    モデル名を憶測で置くと、**最初の1回目の呼び出しまで誤りに気づけない**。
    別プロジェクトでは 2026-08-19 時点で `gemini-3.5-pro` が存在せず、
    3.5 以上は flash 系のみだった。カタログに載っていることと、自分の
    プロジェクト・リージョンから呼べることは別である。

    エージェント層に着手する前に、必ずこれを通すこと。

何を見るか:
    1. models.list() に出てくるか（カタログ上の存在）
    2. 実際に generate_content が通るか（権限とリージョンを含む実証）
    3. **1回の呼び出しで何トークン使うか**（思考トークンを含む）

3 が本題になることがある。Gemini 3.x flash 系は思考トークンを出力として
課金し、しかも `minimal` の思考レベルは 3.8-flash では使えない。
エージェント1往復あたりの実費が、コスト台帳の見積り（app/ledger.py の
ESTIMATES）を超えていないかは、ここで測って初めて分かる。

使い方:
    python -m scripts.check_model
    python -m scripts.check_model --model gemini-3.8-flash --thinking low
    LOC=global python -m scripts.check_model
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 開発機は Norton が TLS を傍受するので、OS の証明書ストアを使う。
# これが無いと HTTPS が証明書エラーで落ちる（Cloud Run では不要だが無害）。
try:
    import truststore

    truststore.inject_into_ssl()
    _TLS = "truststore 有効"
except Exception as exc:  # pragma: no cover - 環境依存
    _TLS = f"truststore 無効: {exc}"

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from app.config import settings  # noqa: E402
from app.ledger import ESTIMATES  # noqa: E402

#: 2026-12-31 までの導入価格（USD / 100万トークン）。
#: **2027-01-01 に倍になる。** 提出後も動かすなら、そのとき単価表も直すこと。
#: 出力単価には**思考トークンが含まれる**。
PRICING_USD_PER_MTOK = {
    "gemini-3.8-flash": {"in": 0.75, "out": 3.75},
    "gemini-3.7-flash": {"in": 0.75, "out": 3.75},
}


def usd(model: str, in_tok: int, out_tok: int) -> float | None:
    p = PRICING_USD_PER_MTOK.get(model)
    if not p:
        return None
    return in_tok * p["in"] / 1e6 + out_tok * p["out"] / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description="モデルの実在と実費を確かめる")
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help="確かめるモデル。既定は 3.8-flash と 3.7-flash と設定値",
    )
    ap.add_argument(
        "--thinking",
        default=None,
        choices=["low", "medium", "high"],
        help="思考レベル。3.8-flash は minimal を受け付けない",
    )
    ap.add_argument("--project", default=settings.project_id or "studio-so-test")
    ap.add_argument("--location", default=os.getenv("LOC", settings.location))
    args = ap.parse_args()

    models = args.model or list(
        dict.fromkeys(["gemini-3.8-flash", "gemini-3.7-flash", settings.gemini_model])
    )

    print(_TLS)
    print(f"project={args.project} location={args.location}")
    print(f"設定値のモデル（app/config.py）: {settings.gemini_model}\n")

    try:
        client = genai.Client(
            vertexai=True, project=args.project, location=args.location
        )
    except Exception:
        traceback.print_exc()
        return 1

    print("=== カタログ（models.list() の gemini-3 系）===")
    try:
        found = sorted({
            n for m in client.models.list()
            if "gemini-3" in (n := (getattr(m, "name", "") or ""))
        })
        for n in found:
            print("  ", n)
        if not found:
            print("   （gemini-3 系は列挙に出ませんでした。列挙に出なくても"
                  "呼べる場合があるので、下の実証を見ること）")
    except Exception as exc:
        print(f"   models.list() 失敗: {type(exc).__name__}: {str(exc)[:200]}")

    print("\n=== 実証（1モデルにつき1回だけ呼ぶ）===")
    cfg = None
    if args.thinking:
        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=args.thinking)
        )

    ok_any = False
    for model in models:
        try:
            r = client.models.generate_content(
                model=model,
                contents=(
                    "あなたは石油ローリーの配車エージェントです。"
                    "浜田SSに本日届けられない場合の代替案を、40字以内で1つ挙げてください。"
                ),
                config=cfg,
            )
            u = getattr(r, "usage_metadata", None)
            pin = getattr(u, "prompt_token_count", 0) or 0
            pout = getattr(u, "candidates_token_count", 0) or 0
            think = getattr(u, "thoughts_token_count", 0) or 0
            total_out = pout + think
            cost = usd(model, pin, total_out)
            ok_any = True
            print(f"  {model}: OK")
            print(f"     応答: {(r.text or '').strip()[:80]}")
            print(
                f"     入力 {pin} / 出力 {pout} / 思考 {think}"
                + (f" → 実費 ${cost:.5f}" if cost is not None else " → 単価表に未登録")
            )
            if cost is not None:
                worst = max(ESTIMATES.values())
                if cost > worst:
                    print(
                        f"     ⚠ 1回で ${cost:.5f}。台帳の見積り最大 ${worst:.3f} を"
                        f"超えている。app/ledger.py の ESTIMATES を実測で直すこと"
                    )
        except Exception as exc:
            print(f"  {model}: NG  {type(exc).__name__}: {str(exc)[:200]}")

    if not ok_any:
        print(
            "\nどれも呼べませんでした。よくある原因:\n"
            "  1. 認証切れ → gcloud auth application-default login\n"
            "  2. Vertex AI API が無効 → gcloud services enable aiplatform.googleapis.com\n"
            "  3. 権限不足 → 実行主体に roles/aiplatform.user が要る\n"
            "  4. リージョン非対応 → LOC=global で試す"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
