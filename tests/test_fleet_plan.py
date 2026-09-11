"""需要の鎖と、専属・非専属の配分の規則。

どちらも計算そのものは単純だが、**体制設計の答え（専属何台）を直接決める**。
規則が「それらしい答え」を出していても、総当たりの最適と食い違っていたら
意味が無いので、総当たりと突き合わせる。
"""

from __future__ import annotations

import random

import pytest

from core.demand_chain import DemandChain, zone_kl_per_day, zones_for
from core.fleet_plan import SPOT_PREMIUM, UNAVAILABILITY, split_fleet


def _brute_force(daily, premium=SPOT_PREMIUM, unavailability=UNAVAILABILITY):
    days = len(daily)

    def cost(n):
        swing = sum(max(0, d - n) for d in daily)
        cover = unavailability * sum(min(d, n) for d in daily)
        return n * days + (1 + premium) * (swing + cover)

    return min(range(max(daily) + 1), key=lambda n: (cost(n), n))


def test_split_matches_brute_force():
    """「要る日の割合が境目を超えたら専属」が、費用の総当たりと一致すること。"""
    rng = random.Random(0)
    for _ in range(200):
        base = rng.randint(5, 25)
        daily = [max(0, base + rng.randint(-4, 6)) for _ in range(rng.randint(20, 60))]
        assert split_fleet(daily).dedicated == _brute_force(daily), daily


def test_threshold_follows_the_premium():
    """非専属の割増が大きいほど、専属を多く持つほうが得になる。"""
    daily = [10] * 60 + [12] * 25 + [14] * 15
    cheap = split_fleet(daily, premium=0.10)
    dear = split_fleet(daily, premium=0.50)
    assert cheap.threshold > dear.threshold
    assert cheap.dedicated <= dear.dedicated


def test_steady_demand_needs_spot_only_for_cover():
    """需要が毎日同じでも、専属が抜ける日の穴埋めに非専属が要る（実務）。"""
    r = split_fleet([20] * 100)
    assert r.dedicated == 20
    assert r.spot_days_swing == 0
    assert r.spot_days_cover > 0
    assert r.spot_peak >= 1


def test_split_is_cheaper_than_owning_the_peak():
    """配分した費用は、ピークの台数を全部専属で持つより高くならない。"""
    daily = [18] * 40 + [20] * 40 + [24] * 20
    r = split_fleet(daily)
    assert r.cost <= r.cost_all_dedicated


def test_carrier_demand_is_the_product_of_the_chain():
    """運送会社の需要 ＝ 地区の需要 × シェア × 引き当て。"""
    c = DemandChain(district_kl_per_day=2_000, eneos_share=0.5, allocation=0.4)
    assert c.carrier_kl_per_day == pytest.approx(400)


def test_more_allocation_means_more_zones():
    """引き当てが増えれば、担うエリアの数も増える（因果の向き）。"""
    base = DemandChain()
    more = DemandChain(allocation=base.allocation * 1.5)
    assert zones_for(more) > zones_for(base)


def test_default_chain_lands_near_the_model_carrier():
    """既定の鎖は、モデルの運送会社（専属20台規模）の桁に着地する。

    地区の需要は逆算した仮置なので、この着地点を守らないと、
    「需要が先、台数は結果」の鎖が何の運送会社も表さなくなる。
    """
    zones = zones_for(DemandChain())
    assert 8 <= zones <= 12
    assert zone_kl_per_day("winter") > zone_kl_per_day("summer")


def test_invalid_shares_are_rejected():
    with pytest.raises(ValueError):
        DemandChain(eneos_share=1.5)
    with pytest.raises(ValueError):
        DemandChain(allocation=0)
