"""デモ動画を組み立てる。**本番画面の静止画 ＋ 生成したナレーション。**

1カット＝1枚の静止画を、そのカットのナレーションの長さだけ保持する。動く画面を
撮って引き伸ばすより、実際に出た画面をその長さだけ見せるほうが正直で、読みと
画面がずれない。

使い方:
    .venv/Scripts/python.exe scripts/assemble_demo.py
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STILLS = ROOT / "media" / "stills"
NARR = ROOT / "media" / "narration"
OUT_DIR = ROOT / "media"
OUT = OUT_DIR / "lorry_dispatch_demo.mp4"

W, H = 1920, 1080

#: (静止画, ナレーション, 読み終わりの後に足す秒)
PLAN: list[tuple[str, str, float]] = [
    ("cut_01_top.png", "cut_01.wav", 0.4),
    ("cut_02_plan.png", "cut_02.wav", 0.4),
    ("cut_03_vehicle.png", "cut_03.wav", 0.4),
    ("cut_04_intake_top.png", "cut_04.wav", 0.4),
    ("cut_05_read.png", "cut_05.wav", 0.4),
    ("cut_06_shortfall.png", "cut_06.wav", 0.6),
    ("cut_07_filled.png", "cut_07.wav", 0.4),
    ("cut_08b_options.png", "cut_08.wav", 0.4),
    ("cut_09_fleet.png", "cut_09.wav", 0.4),
    ("cut_10_arch.png", "cut_10.wav", 0.8),
]


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def main() -> None:
    parts: list[Path] = []
    work = OUT_DIR / "work"
    work.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for i, (still, wav, hold) in enumerate(PLAN, start=1):
        img, aud = STILLS / still, NARR / wav
        if not img.exists():
            raise SystemExit(f"静止画が無い: {img}")
        if not aud.exists():
            raise SystemExit(f"ナレーションが無い: {aud}")
        dur = wav_seconds(aud) + hold
        total += dur
        out = work / f"part_{i:02d}.mp4"
        # 静止画を dur 秒の動画にし、音を重ねる。**画は引き伸ばさず、余白を足して
        # 1920x1080 に収める**（画面の文字が歪むとデモとして読めない）。
        subprocess.run(
            [
                "ffmpeg", "-y", "-loop", "1", "-i", str(img), "-i", str(aud),
                "-filter_complex",
                f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1[v];"
                f"[1:a]apad=pad_dur={hold}[a]",
                "-map", "[v]", "-map", "[a]",
                "-t", f"{dur:.3f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                str(out),
            ],
            check=True, capture_output=True,
        )
        parts.append(out)
        print(f"{still:28s} {dur:5.1f}秒")

    listing = work / "parts.txt"
    listing.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in parts), encoding="utf-8"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", str(OUT)],
        check=True, capture_output=True,
    )
    print(f"\n完成: {OUT}  合計 {total:.0f}秒（{total / 60:.1f}分）")
    if total > 180:
        print("**3分を超えている。** どれかのカットを削るか、読みを短くすること")


if __name__ == "__main__":
    main()
