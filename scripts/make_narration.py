"""デモ動画のナレーションを、カットごとに1本ずつ合成する。

**1本につなげず、カットごとに切る。** 組み立て（scripts/assemble_demo.py）が
各カットの実際の長さに合わせて静止画を保持するため。1本にすると、画面の切替と
読みのずれを後から直せない。

声は `Leda`（澄んだ女性の声）。過去のプロジェクト（Wayfinder・Greenlight Studio）
と同じ標準。日本語の審査なので読みも日本語。

**呼び方は REST（curl 相当）＋ gcloud のアクセストークン。** 開発機の ADC は
別アカウントで作られていて `studio-so-test` を使えず、SDK 経由だと
`serviceUsageConsumer` が無いと断られる（2026-10-14）。gcloud の資格情報は
代表のアカウントなので、そちらのトークンで直接叩く。

使い方:
    .venv/Scripts/python.exe scripts/make_narration.py
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402,F401  TLS の下準備（開発機の Norton 対策）

OUT_DIR = ROOT / "media" / "narration"

PROJECT = "studio-so-test"
TTS_LOCATION = "us-central1"      # 音声合成はリージョン限定。global には居ない
TTS_MODEL = "gemini-3.1-flash-tts-preview"
VOICE = "Leda"
SAMPLE_RATE = 24000

STYLE = (
    "落ち着いた、確信のある読み方で。配車係に向けて事実を伝える口調で、"
    "売り込みではない。1分あたり330文字程度のゆっくりした速さで、読点と句点で"
    "自然に間を置く。\n\n"
    "次の行をデモ動画のナレーションとして読んでください。記号は読まないこと。\n\n"
)

#: カット番号 → 読み上げる文。数字は**本番で実際に出た値**。
CUTS: dict[int, str] = {
    1: (
        "石油ローリーの配車は、毎朝一人の配車係が、紙と経験で組んでいます。"
        "槽の割り当て、受入時間、拘束時間、タンクの在庫。どれか一つが動くと、全部が動きます。"
    ),
    2: (
        "このエージェントは、朝の計画を八十秒で組みます。五台、九便、十四軒。"
        "必ず回るべき先は、すべて回れます。"
    ),
    3: (
        "必ず回る先には、理由が付きます。発注、発注点割れ、欠品の直前の三つだけ。"
        "休憩は、労働基準法と改善基準告示の、大きいほうを取っています。"
    ),
    4: (
        "九時十二分。宇品港エスエスからファックスが届きます。"
        "レギュラー八キロリットル、灯油四キロリットル。"
        "電話では受けません。発注書が届くまで、車は動かしません。"
    ),
    5: (
        "ジェミニが読むのは、紙に何と書いてあるかだけです。"
        "名簿の照合も、単位の換算も、積み込みの計算も、すべてコードが行います。"
        "読み取った一行ごとに、ファックス画像の証拠が付きます。"
    ),
    6: (
        "ここが山です。八キロリットル頼まれて、四キロリットルだけ積みます。"
        "空きは六点六キロリットル。ですがレギュラーは、四キロリットル槽の単位でしか運びません。"
        "残りは、理由を添えて突き返します。"
    ),
    7: (
        "但し書きの、連休で来客が多く、を在庫に当て直すと、空きが増え、八キロリットルがまるごと入ります。"
        "同じ発注書でも、在庫の見立てが違えば、運ぶ量が変わります。"
    ),
    8: (
        "降ろせない荷が出たときも、同じです。"
        "持ち戻りは、近くの降ろし先、クッション、積場へ戻す、の順に探します。"
        "選ばなかった手にも、空きが足りない、という理由が一件ずつ残ります。"
    ),
    9: (
        "そして、日をまたいで数えます。応えられない日が積み上がれば、それは体制の問題です。"
        "半期に専属を何台取るか、までを答えにします。"
    ),
    10: (
        "クラウドランの上で、ジェミニは読むだけ。計算はすべて、決定論のコアが行います。"
        "同じ入力なら、同じ計画が返ります。"
    ),
}


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def access_token() -> str:
    # Windows の gcloud は gcloud.cmd。shutil.which で実体を引く
    exe = shutil.which("gcloud") or "gcloud"
    out = subprocess.run(
        [exe, "auth", "print-access-token"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def synth(token: str, text: str) -> bytes:
    url = (
        f"https://{TTS_LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}"
        f"/locations/{TTS_LOCATION}/publishers/google/models/{TTS_MODEL}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": STYLE + text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE}}
            },
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as res:
        payload = json.load(res)
    part = payload["candidates"][0]["content"]["parts"][0]
    return base64.b64decode(part["inlineData"]["data"])


def main() -> None:
    # 引数でカット番号を指定すると、そのカットだけ作り直す。読みの速さは呼ぶ
    # たびに振れるので、長さを詰めたいカットだけを撮り直せるようにしてある。
    only = {int(a) for a in sys.argv[1:]} if len(sys.argv) > 1 else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = access_token()
    total = 0.0
    for cut, text in sorted(CUTS.items()):
        if only is not None and cut not in only:
            path = OUT_DIR / f"cut_{cut:02d}.wav"
            if path.exists():
                total += wav_seconds(path)
            continue
        pcm = synth(token, text)
        path = OUT_DIR / f"cut_{cut:02d}.wav"
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            f.writeframes(pcm)
        seconds = len(pcm) / 2 / SAMPLE_RATE
        total += seconds
        print(f"cut {cut:02d}: {seconds:5.1f}秒  {path.name}")
    print(f"合計 {total:.0f}秒（{total / 60:.1f}分）")


if __name__ == "__main__":
    main()
