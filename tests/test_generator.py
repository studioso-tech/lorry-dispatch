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


# ---------------------------------------------------------------------------
# 荷卸時間（運送会社への聞き取り 2026-09-11）
# ---------------------------------------------------------------------------


def test_unload_flow_matches_field_measurements():
    """流下時間が聞き取りの実測値を再現すること。

    **自然落下なので経路で一番細いところが律速する。**

        4インチ  4kL で5分 / 2kL で3分
        3インチ  4kL で8分 / 2kL で4分

    ここが崩れたら、係数を誰かが「それらしく」いじったということ。
    """
    from core.domain import UNLOAD_SETUP_MIN, unload_minutes
    from core.domain import Product as P

    flow = lambda v, bore: unload_minutes({P.REGULAR: v}, bore) - UNLOAD_SETUP_MIN
    assert flow(4_000, 4) == 5
    assert flow(2_000, 4) == 3
    assert flow(4_000, 3) == 8
    assert flow(2_000, 3) == 4


def test_unload_total_matches_carrier_allowance():
    """準備・手仕舞を含めた合計が、運送会社の見込み（30分／60分）と合うこと。

    流量の実測と現場の段取りが、準備20分という1つの数字で両方説明できる。
    **片方だけ合わせた係数は、もう片方で嘘をつく。**
    """
    from core.domain import Product as P, unload_minutes

    # 4インチで8kL（2槽）→ 見込み30分
    assert unload_minutes({P.REGULAR: 4_000, P.GASOIL: 4_000}, 4) == 30
    # 3インチで20kL（1台まるごと、5槽）→ 見込み60分
    assert unload_minutes({P.KEROSENE: 20_000}, 3) == 60


def test_unload_time_depends_on_volume():
    """荷卸時間が降ろす量で変わること。

    以前は類型ごとの定数で、2kL置くのも16kL置くのも同じ35分だった。
    """
    s = generate(seed=20261015, season="winter")
    by_type: dict = {}
    for c in s.customers:
        if c.demand_l:
            by_type.setdefault((c.type, c.bore_inch), set()).add(c.service_min)
    assert any(len(v) > 1 for v in by_type.values()), "同じ類型・口径で荷卸時間がすべて同じ"


def test_compartments_obey_hazmat_law():
    """1室は4,000L以下（危険物の規制に関する政令 第15条）。

    以前の生成器には 5,000L の槽を持つ車が3台いた。法令上作れない車で
    組んだ計画は、存在しない車の計画である。
    """
    from core.domain import COMPARTMENT_MAX_L, Vehicle, VehicleClass

    for v in generate(seed=1).vehicles:
        assert max(v.compartments_l) <= COMPARTMENT_MAX_L, (v.id, v.compartments_l)

    with pytest.raises(ValueError, match="第15条"):
        Vehicle(
            id="bad", name="5kL槽", vehicle_class=VehicleClass.TRAILER,
            compartments_l=(5_000, 5_000, 5_000, 5_000), driver_id="d1",
        )


def test_compartments_are_2kl_or_4kl():
    """槽は 2kL と 4kL の組合せだけ（実務）。

    法令は上限しか決めていないが、実際の車はこの2種類で作られている。
    以前は 3kL の槽を持つ車が2台いた。
    """
    from core.domain import COMPARTMENT_SIZES_L, Vehicle, VehicleClass

    for v in generate(seed=1).vehicles:
        assert set(v.compartments_l) <= set(COMPARTMENT_SIZES_L), (v.id, v.compartments_l)

    with pytest.raises(ValueError, match="槽は無い"):
        Vehicle(
            id="bad", name="3kL槽", vehicle_class=VehicleClass.RIGID,
            compartments_l=(4_000, 3_000, 3_000, 4_000), driver_id="d1",
        )


def test_bore_must_be_a_real_hose_size():
    """荷卸しの口径は実在する2種類（4インチ・3インチ）だけ。"""
    from dataclasses import replace

    c = generate(seed=1).customers[0]
    with pytest.raises(ValueError, match="口径"):
        replace(c, bore_inch=5)


def test_bore_is_the_narrower_of_hose_and_piping():
    """実効口径はホースと配管の細いほう（自然落下は一番細いところが律速する）。

    ホースは3インチが多い（乗務員の負担を軽くするため）。配管は系列で決まり、
    旧三菱系だけが4インチ。したがって:

        3インチホース → 旧三菱系でも3インチ
        4インチホース → 旧三菱系だけ4インチ、旧日石系は3インチのまま

    **ホースを替えても旧日石系は速くならない。** ここが崩れると、4インチ
    ホースに替える効果を過大に見積もる。
    """
    import core.generator as g
    from core.generator import ROSTER

    four = {spec.id for spec in ROSTER if spec.piping_inch == 4}
    assert four, "配管4インチ（旧三菱系）の先が1軒も無い"
    assert len(four) < len(ROSTER), "全軒が4インチ配管になっている"

    original = g.HOSE_INCH
    try:
        g.HOSE_INCH = 3
        assert {c.bore_inch for c in g.generate(seed=1).customers} == {3}
        g.HOSE_INCH = 4
        for c in g.generate(seed=1).customers:
            assert c.bore_inch == (4 if c.id in four else 3), c.id
    finally:
        g.HOSE_INCH = original
