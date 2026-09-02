"""合成データ生成器の性質テスト。

ここで確かめたいのは「値が正しいか」ではなく「**性質が保たれているか**」。
合成データの個々の数字に正解は無いが、再現性・季節の効き方・需要の
決まり方には、崩れたら後段が全部おかしくなる性質がある。
"""

from __future__ import annotations

import pytest

from core.domain import DELIVERY_LOOKAHEAD_DAYS, Product
from core.generator import ROSTER, generate


def test_same_seed_gives_identical_scenario():
    """同じ seed は完全に同じ日を返す。

    これが崩れると回帰テスト（W6）が成立しない。計画品質が落ちたのか、
    たまたま難しい日が生成されたのかを区別できなくなる。
    """
    a = generate(seed=42, season="winter")
    b = generate(seed=42, season="winter")
    assert [(c.id, c.tanks) for c in a.customers] == [
        (c.id, c.tanks) for c in b.customers
    ]


def test_different_seed_gives_different_day():
    """seed を変えれば別の日が出る。固定値を返していないことの確認。"""
    a = generate(seed=1, season="winter")
    b = generate(seed=2, season="winter")
    assert [c.tanks for c in a.customers] != [c.tanks for c in b.customers]


def test_roster_is_stable_across_seeds():
    """需要家の顔ぶれ・座標・時間枠は seed によらず固定。

    地図が日替わりで変わると、等時線マップ（W5）の見え方が安定せず、
    「なぜこの便を20時に回したか」の説明が成立しない。
    """
    a = generate(seed=1)
    b = generate(seed=999)
    assert [(c.id, c.lat, c.lon, c.window_open_min) for c in a.customers] == [
        (c.id, c.lat, c.lon, c.window_open_min) for c in b.customers
    ]
    assert len(a.customers) == len(ROSTER)


def test_unknown_season_raises():
    """季節の綴り違いは黙って通さない。

    フォールバックすると灯油の消費が2.6倍から1.0倍に化け、
    誰も気づかないまま計画が変わる。
    """
    with pytest.raises(ValueError):
        generate(season="Winter")   # 大文字は別物として弾く


def test_winter_raises_kerosene_demand():
    """冬は灯油の消費が跳ねる。季節の乗数が実際に効いていることの確認。"""
    winter = generate(seed=7, season="winter")
    summer = generate(seed=7, season="summer")

    def kerosene_use(scenario):
        return sum(
            t.daily_use_l
            for c in scenario.customers
            for t in c.tanks
            if t.product is Product.KEROSENE
        )

    assert kerosene_use(winter) > kerosene_use(summer) * 5


def test_demand_covers_only_tanks_needing_topup():
    """需要は「2日先に発注点を割るタンク」だけの合計。

    全タンクを毎日満タンにする定義にすると、1軒でローリー1台を使い切る
    需要になり、車が何台あっても足りない計画が出る（v0 の最初の実装がこれ）。
    """
    s = generate(seed=3, season="winter")
    for c in s.customers:
        expected = sum(t.topup_l for t in c.tanks if t.needs_topup)
        assert c.demand_l == expected
        for t in c.tanks:
            if not t.needs_topup:
                continue
            projected = t.level_l - t.daily_use_l * DELIVERY_LOOKAHEAD_DAYS
            assert projected <= t.reorder_l


def test_mandatory_means_already_below_reorder():
    """必須訪問は「すでに発注点を割っている」こと。先読みぶんは必須ではない。"""
    s = generate(seed=5, season="winter")
    for c in s.customers:
        assert c.is_mandatory == any(t.is_below_reorder for t in c.tanks)


def test_absent_driver_removes_vehicle():
    """乗務員が欠勤したら、その車は出せない。

    「車はあるが人がいない」を計画側で握りつぶさないための関門。
    デモ4（乗務員欠勤）が成立する前提でもある。
    """
    s = generate(seed=11, absent_drivers=("d1",))
    assert not s.driver("d1").available
    assert "v1" not in [v.id for v in s.usable_vehicles()]
    assert len(s.usable_vehicles()) == len(s.vehicles) - 1


def test_demand_fits_realistic_scale():
    """1日の需要が車両総容量と同じ桁に収まっている。

    ここが桁違いだと、ソルバーが正しくても計画は毎日破綻する。
    複数の seed で見るのは、たまたま緩い日を見て安心しないため。
    """
    for seed in (1, 42, 777, 20261015):
        s = generate(seed=seed, season="winter")
        total = sum(c.demand_l for c in s.customers)
        # 総需要は車両総容量の 1〜4 倍。1未満なら1日で片付いてしまい
        # 配車の問題にならず、4倍を超えると何をしても回らない。
        assert 1.0 <= total / s.fleet_capacity_l <= 4.0, (seed, total)
