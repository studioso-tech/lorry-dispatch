"""開発機（Windows）でだけ、OS の証明書ストアで TLS を検証させる。

開発機には Norton の SSL/TLS スキャンが入っていて、Google API への通信が
Norton のルート CA で再署名される。Python 3.13 以降は証明書の検証が厳格に
なり（VERIFY_X509_STRICT）、Norton の証明書は basicConstraints が critical で
ないため落ちる。CA バンドルを渡しても直らない。

truststore で Windows の証明書ストアを使わせると、検証を無効にせずに通る。
**検証を切る（verify=False 等）ことはしない。**

Cloud Run（Linux）には Norton が無いので何もしない。
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)
_done = False


def ensure() -> None:
    global _done
    if _done or sys.platform != "win32":
        return
    try:
        import truststore

        truststore.inject_into_ssl()
        logger.info("TLS: Windows の証明書ストアで検証する（truststore）")
    except Exception as exc:  # 入っていなければ既定の検証のまま
        logger.warning("TLS: truststore を使えない: %s", exc)
    _done = True
