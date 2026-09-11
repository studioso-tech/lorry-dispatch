"""専属と非専属の配分を決める。

**専属は毎日費用がかかり、非専属は使った日だけ費用がかかる。**（実務）

    専属    半期ごとの契約。最低保証があるので、使わない日も払う。単価は安い
    非専属  最低保証なし。動かした分だけ。単価は専属の2〜3割高い

すると「k台目の車」を専属にするか非専属にするかは、その車が**半期のうち
何割の日に要るか**で決まる。k台目が要る日の割合を f とすると:

    専属にした場合    1（毎日） ＋ 抜けた日の穴埋め (1+p)·u·f
    非専属にした場合  (1+p)·f  （要る日だけ呼ぶ）

    p = 非専属の割増（0.25）、u = 専属が点検・整備・故障で抜ける割合（0.05）

したがって **f > 1 / ((1+p)(1-u))** なら専属のほうが安い。p=0.25, u=0.05
で約84%。**8割5分以上の日に要る車は専属、それ未満は非専属で呼ぶ。**

非専属の必要量は2つの足し算になる（実務）:

    需要の振れ   専属の台数を超える日の超過分（寒波・連休・発注の山）
    穴埋め       専属が点検・整備・故障で抜ける日（需要と無関係に、専属の台数に比例）

**近似**: 最低保証の単位は実際には kL・km だが、ここでは「車1台・1日」で数える。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: 非専属の割増。**専属の2〜3割高**（実務・記憶ベース）の中をとった。
SPOT_PREMIUM = 0.25

#: 専属が点検・整備・故障で抜ける割合。**仮置**（20台で1台が抜けている水準）。
UNAVAILABILITY = 0.05


@dataclass(frozen=True)
class FleetSplit:
    """専属と非専属の配分の答え。

    Attributes:
        dedicated: 専属の台数。
        need_share: k台目（k=1,2,…）が要る日の割合。境目の説明に使う。
        threshold: 専属にする境目（要る日の割合がこれを超えたら専属）。
        spot_days_swing: 非専属の延べ台日のうち、需要の振れの分。
        spot_days_cover: 同じく、専属が抜けた日の穴埋めの分（期待値）。
        spot_peak: 非専属が1日に最大何台要るか（穴埋めは切り上げで見込む）。
        days: 評価した日数。
        cost: 配分したときの費用（専属1台1日＝1）。
        cost_all_dedicated: ピークの台数を全部専属で持った場合の費用。
    """

    dedicated: int
    need_share: tuple[float, ...]
    threshold: float
    spot_days_swing: int
    spot_days_cover: float
    spot_peak: int
    days: int
    cost: float
    cost_all_dedicated: float


def split_fleet(
    daily_need: list[int],
    *,
    premium: float = SPOT_PREMIUM,
    unavailability: float = UNAVAILABILITY,
) -> FleetSplit:
    """日ごとの必要台数から、専属と非専属の配分を決める。

    Args:
        daily_need: 評価した各日に必要な台数（欠品ゼロで回る最小の台数）。
        premium: 非専属の割増。
        unavailability: 専属が抜ける割合。
    """
    if not daily_need:
        raise ValueError("評価する日が無い")
    days = len(daily_need)
    peak = max(daily_need)
    need_share = tuple(sum(1 for n in daily_need if n >= k) / days for k in range(1, peak + 1))
    threshold = 1.0 / ((1.0 + premium) * (1.0 - unavailability))
    dedicated = sum(1 for f in need_share if f > threshold)

    def cost_of(n: int) -> tuple[float, int, float]:
        swing = sum(max(0, d - n) for d in daily_need)
        cover = unavailability * sum(min(d, n) for d in daily_need)
        return n * days + (1.0 + premium) * (swing + cover), swing, cover

    cost, swing, cover = cost_of(dedicated)
    cost_peak, _, _ = cost_of(peak)
    spot_peak = max(max(0, d - dedicated) for d in daily_need) + math.ceil(unavailability * dedicated)
    return FleetSplit(
        dedicated=dedicated,
        need_share=need_share,
        threshold=threshold,
        spot_days_swing=swing,
        spot_days_cover=cover,
        spot_peak=spot_peak,
        days=days,
        cost=cost,
        cost_all_dedicated=cost_peak,
    )
