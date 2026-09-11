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

---

**払う側の費用だけで決めてはいけない。**（実務・2026-09-11）

上の規則は元売の運賃だけを最小にする。すると需要の振れはすべて非専属に
押し付けられる。だが非専属を呼べるようにしておくには、**運送会社が車と
危険物の乗務員を待機させておく**必要があり、呼ばれない日の費用は運送会社が
被る。専属20台に対して非専属を1日9台呼べる前提の答え（最初の実装）は、
**元売が得をして運送会社が潰れる配分**だった。

    運送会社の負担が大きく、リスクをすべて運送会社が引き受ける形は成り立たない。
    元売が儲かっても運送会社が潰れては商売は成り立たない。利益は適正に配分されるべき。

そこで2つを入れた:

1. **非専属の上限**（規模に比例。専属20台で数台以下、実務）。上限を超える振れは
   専属で持つ。専属が抜けた日の穴埋めも、この上限の中から出す
2. **運送会社側の負担の見える化**。待機させている非専属のうち、実際に呼ばれる
   割合（稼働率）を出す。低ければ、運送会社が待機の費用を被っている。
   利益の額は分からないので、額ではなく割合で示す
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: 非専属の割増。**専属の2〜3割高**（実務・記憶ベース）の中をとった。
SPOT_PREMIUM = 0.25

#: 専属が点検・整備・故障で抜ける割合。**仮置**（20台で1台が抜けている水準）。
UNAVAILABILITY = 0.05

#: 非専属を1日に呼べる上限（専属の台数に対する比）。**規模に比例**し、専属20台
#: なら数台以下（実務）。**仮置**で 0.15（20台で3台）。
SPOT_CAP_RATIO = 0.15


@dataclass(frozen=True)
class FleetSplit:
    """専属と非専属の配分の答え。

    Attributes:
        dedicated: 専属の台数（非専属の上限を守ったうえで費用が最小）。
        dedicated_cost_only: 非専属を無制限に呼べるとした場合の専属の台数
            （払う側の費用だけで決めた答え。比較のために残す）。
        spot_cap: 非専属を1日に呼べる上限（専属の台数から決まる）。
        standby_utilization: 運送会社が待機させる非専属（spot_cap 台）のうち、
            実際に呼ばれる割合。**低いほど運送会社が待機の費用を被っている。**
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
    dedicated_cost_only: int
    spot_cap: int
    standby_utilization: float
    need_share: tuple[float, ...]
    threshold: float
    spot_days_swing: int
    spot_days_cover: float
    spot_peak: int
    days: int
    cost: float
    cost_all_dedicated: float


def spot_cap_for(dedicated: int, ratio: float = SPOT_CAP_RATIO) -> int:
    """専属 dedicated 台の運送会社が、1日に呼べる非専属の上限。"""
    return int(math.floor(ratio * dedicated + 1e-9))


def split_fleet(
    daily_need: list[int],
    *,
    premium: float = SPOT_PREMIUM,
    unavailability: float = UNAVAILABILITY,
    spot_cap_ratio: float = SPOT_CAP_RATIO,
) -> FleetSplit:
    """日ごとの必要台数から、専属と非専属の配分を決める。

    **非専属の上限を守る台数の中で、費用が最小の専属台数**を選ぶ。上限の中に
    専属が抜けた日の穴埋めも入る（穴埋めも非専属で出すため）。

    Args:
        daily_need: 評価した各日に必要な台数（欠品ゼロで回る最小の台数）。
        premium: 非専属の割増。
        unavailability: 専属が抜ける割合。
        spot_cap_ratio: 非専属を1日に呼べる上限（専属に対する比）。
            math.inf にすると上限なし（払う側の費用だけで決める）。
    """
    if not daily_need:
        raise ValueError("評価する日が無い")
    days = len(daily_need)
    peak = max(daily_need)
    need_share = tuple(sum(1 for n in daily_need if n >= k) / days for k in range(1, peak + 1))
    threshold = 1.0 / ((1.0 + premium) * (1.0 - unavailability))

    def cost_of(n: int) -> tuple[float, int, float]:
        swing = sum(max(0, d - n) for d in daily_need)
        cover = unavailability * sum(min(d, n) for d in daily_need)
        return n * days + (1.0 + premium) * (swing + cover), swing, cover

    def spot_peak_of(n: int) -> int:
        return max(max(0, d - n) for d in daily_need) + math.ceil(unavailability * n)

    def cap_of(n: int) -> float:
        if math.isinf(spot_cap_ratio):
            return math.inf
        return spot_cap_for(n, spot_cap_ratio)

    # 払う側の費用だけで決めた答え（上限なし）。境目の規則と同じになる。
    cost_only = sum(1 for f in need_share if f > threshold)

    # 上限を守れる台数の中で費用最小。台数は peak を超えて持つ意味がない
    # （穴埋めのぶんを上限から出せなくなる小さい規模では peak まで行く）。
    candidates = [n for n in range(0, peak + 1) if spot_peak_of(n) <= cap_of(n)]
    if not candidates:
        candidates = [peak]
    dedicated = min(candidates, key=lambda n: (cost_of(n)[0], n))

    cost, swing, cover = cost_of(dedicated)
    cost_peak, _, _ = cost_of(peak)
    cap = cap_of(dedicated)
    cap_int = spot_peak_of(dedicated) if math.isinf(cap) else int(cap)
    standby = (swing + cover) / (cap_int * days) if cap_int else 0.0
    return FleetSplit(
        dedicated=dedicated,
        dedicated_cost_only=cost_only,
        spot_cap=cap_int,
        standby_utilization=standby,
        need_share=need_share,
        threshold=threshold,
        spot_days_swing=swing,
        spot_days_cover=cover,
        spot_peak=spot_peak_of(dedicated),
        days=days,
        cost=cost,
        cost_all_dedicated=cost_peak,
    )
