"""需要の鎖と、専属・非専属の配分の規則。

どちらも計算そのものは単純だが、**体制設計の答え（専属何台）を直接決める**。
規則が「それらしい答え」を出していても、総当たりの最適と食い違っていたら
意味が無いので、総当たりと突き合わせる。
"""

from __future__ import annotations

import math
import random

import pytest

from core.demand_chain import DemandChain, zone_kl_per_day, zones_for
from core.fleet_plan import SPOT_PREMIUM, UNAVAILABILITY, spot_cap_for, split_fleet


def _brute_force(daily, premium=SPOT_PREMIUM, unavailability=UNAVAILABILITY):
    days = len(daily)

    def cost(n):
        swing = sum(max(0, d - n) for d in daily)
        cover = unavailability * sum(min(d, n) for d in daily)
        return n * days + (1 + premium) * (swing + cover)

    return min(range(max(daily) + 1), key=lambda n: (cost(n), n))


def test_cost_only_rule_matches_brute_force():
    """上限なしなら「要る日の割合が境目を超えたら専属」が、費用の総当たりと一致する。"""
    rng = random.Random(0)
    for _ in range(200):
        base = rng.randint(5, 25)
        daily = [max(0, base + rng.randint(-4, 6)) for _ in range(rng.randint(20, 60))]
        r = split_fleet(daily, spot_cap_ratio=math.inf)
        assert r.dedicated == r.dedicated_cost_only == _brute_force(daily), daily


def test_capped_split_is_the_cheapest_that_respects_the_cap():
    """上限ありでは、「非専属の上限を守れる台数」の中で費用最小を選ぶ。

    **払う側の費用だけで決めてはいけない。** 非専属を呼べるようにしておくには
    運送会社が車と乗務員を待機させる必要があり、呼ばれない日の費用は運送会社が
    被る。上限を超える振れは専属で持つ。
    """
    rng = random.Random(1)
    for _ in range(200):
        base = rng.randint(8, 25)
        daily = [max(0, base + rng.randint(-5, 6)) for _ in range(rng.randint(20, 60))]
        r = split_fleet(daily)
        # 上限を守っている
        assert r.spot_peak <= r.spot_cap or r.dedicated == max(daily)
        # 上限を守れる台数の中で費用最小（総当たり）
        days = len(daily)

        def cost(n):
            swing = sum(max(0, d - n) for d in daily)
            cover = UNAVAILABILITY * sum(min(d, n) for d in daily)
            return n * days + (1 + SPOT_PREMIUM) * (swing + cover)

        def ok(n):
            peak = max(max(0, d - n) for d in daily) + math.ceil(UNAVAILABILITY * n)
            return peak <= spot_cap_for(n)

        feasible = [n for n in range(max(daily) + 1) if ok(n)] or [max(daily)]
        assert r.dedicated == min(feasible, key=lambda n: (cost(n), n)), daily
        # 上限があれば、専属は上限なしの答え以上になる
        assert r.dedicated >= r.dedicated_cost_only


def test_twenty_dedicated_can_call_only_a_few_spot():
    """専属20台なら、呼べる非専属は数台以下（実務）。"""
    assert 1 <= spot_cap_for(20) <= 4


def test_standby_utilization_shows_the_carriers_burden():
    """運送会社が待機させる非専属のうち、実際に呼ばれる割合を出す。

    需要が毎日同じでも、専属が抜ける日の穴埋めのために待機は要る。そのとき
    待機の大半は呼ばれない＝運送会社が費用を被っている。
    """
    steady = split_fleet([20] * 100)
    swingy = split_fleet([16] * 50 + [20] * 50)
    assert 0 < steady.standby_utilization < swingy.standby_utilization <= 1


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


# ---------------------------------------------------------------------------
# 運送会社が成り立つ配分
# ---------------------------------------------------------------------------


def test_standby_never_drops_below_the_utilization_floor():
    """待機させる非専属は、稼働率70%を割らない台数まで（実務）。

    車両の維持費、減価償却（1台2,000万円）、乗務員の確保を考えると、
    5割前後では運送会社は持たない。
    """
    from core.fleet_plan import MIN_STANDBY_UTILIZATION, OTHER_BUSINESS_SHARE, fleet_options

    rng = random.Random(2)
    for _ in range(100):
        base = rng.randint(8, 22)
        daily = [max(0, base + rng.randint(-5, 5)) for _ in range(42)]
        for o in fleet_options(daily):
            if o.standby:
                # こちらから呼ぶ分 ＋ 系列外の仕事 で70%以上
                assert o.utilization + OTHER_BUSINESS_SHARE >= MIN_STANDBY_UTILIZATION - 1e-9


def test_breakdown_alone_does_not_justify_a_standby_vehicle():
    """故障だけに備えた待機は成り立たない。

    故障1%なら専属20台で1日平均0.2台。1台待機させても出番は5日に1日で、
    稼働率70%に届かない。
    """
    from core.fleet_plan import fleet_options

    # 系列外の仕事が無ければ、故障だけのための待機は成り立たない
    steady = fleet_options([20] * 42, other_share=0.0)
    assert steady[20].standby == 0


def test_enough_dedicated_leaves_no_escape_days():
    """ピークまで専属で持てば、逃げ道に頼る日は無い。"""
    from core.fleet_plan import fleet_options

    daily = [14] * 20 + [18] * 15 + [21] * 7
    opts = fleet_options(daily)
    assert opts[max(daily)].escape_days == 0
    assert opts[14].escape_days > 0 and opts[14].chronic


def test_recommend_avoids_chronic_reliance_on_escape_routes():
    """推奨は「恒常化の恐れが無い案」の中で元売の運賃が最小のもの。

    逃げ道（計画配送を1日ずらす、拘束を延ばす）は一時的なら可、恒常的なら
    構造の欠陥（実務）。運賃が安くても、逃げ道に恒常的に頼る案は選ばない。
    """
    from core.fleet_plan import fleet_options, recommend

    daily = [14] * 20 + [18] * 15 + [21] * 7
    opts = fleet_options(daily)
    r = recommend(opts)
    assert not r.chronic
    cheaper = [o for o in opts if o.cost < r.cost]
    assert all(o.chronic for o in cheaper)



def test_non_affiliate_work_makes_standby_viable():
    """系列外の仕事で稼げれば、非専属を待機させる案が成り立つ（実務）。

    ノーマーク（元売の看板を付けない）の非専属は、系列以外の仕事もできる。
    専属は看板を背負っているので系列の仕事しかできない。
    """
    from core.fleet_plan import fleet_options, recommend

    daily = [16] * 20 + [18] * 15 + [20] * 7
    without = recommend(fleet_options(daily, other_share=0.0))
    with_other = recommend(fleet_options(daily, other_share=0.5))
    assert with_other.standby >= without.standby
    assert with_other.cost <= without.cost
