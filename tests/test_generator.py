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


def test_planned_delivery_comes_every_three_operating_days():
    """計画配送の先には、営業日で数えてほぼ3日に1回、決まった周期で運ぶ。

    以前は毎日「2日先に発注点を割るタンク」を満タンまで補充していて、
    計画配送の量が日によってばらついていた。実態は月〜土ほぼ一定量。
    """
    import datetime

    from core.generator import PLANNED_INTERVAL_DAYS

    start = datetime.date(2026, 10, 12)          # 月曜
    due: dict[str, list[int]] = {}
    for k in range(21):
        d = start + datetime.timedelta(days=k)
        for c in generate(seed=k, date=d.isoformat(), season="winter").customers:
            if c.ordering or not c.open_today:
                continue
            due.setdefault(c.id, [])
            due[c.id].append(1 if c.planned else 0)
    for cid, marks in due.items():
        # 営業日だけを並べると、きっちり3日に1回
        assert sum(marks) == len(marks) // PLANNED_INTERVAL_DAYS, cid
        hits = [i for i, m in enumerate(marks) if m]
        assert all(b - a == PLANNED_INTERVAL_DAYS for a, b in zip(hits, hits[1:])), cid


def test_mandatory_depends_on_whose_promise_it_is():
    """必須訪問は「誰の約束か」で決まる。量が出る理由はどれも必須。

        受入見送り・休業 → 必須にしない（行っても降ろせない）
        計画配送の日     → 必須（運送会社側の約束）
        発注             → 必須（需要家の約束）
        欠品の直前       → 必須（元売の信用）
    """
    for seed in range(1, 11):
        s = generate(seed=seed, date=f"2026-10-{10 + seed:02d}", season="winter")
        for c in s.customers:
            stockout = any(t.stockout_before_next_delivery and t.topup_l > 0 for t in c.tanks)
            reason = stockout or bool(c.orders) or bool(c.planned)
            expected = c.accepts_delivery and c.open_today and c.demand_l > 0 and reason
            assert c.is_mandatory == expected, c.id


def test_ordering_customer_below_reorder_without_order_is_not_forced():
    """発注配送の先で、発注の無い油種が発注点を割っていても押しかけない。

    何を入れるかは需要家が決めている。値下げ前に在庫を下げている先がこれに
    あたる。**ただし欠品の直前なら行く**（別のテスト）。
    """
    from dataclasses import replace

    c = next(x for x in generate(seed=20261015).customers if x.ordering)
    t = c.tanks[0]
    low = replace(t, level_l=t.reorder_l - 1)            # 発注点は割っている
    assert low.is_below_reorder and not low.stockout_before_next_delivery
    c2 = replace(c, orders=(), tanks=(low,) + c.tanks[1:])
    assert not c2.is_mandatory
    assert c2.demand_l == 0


def test_declined_customer_gets_nothing():
    """受入見送りの先には何も降ろさない。発注があっても、欠品の直前でも。

    **在庫は需要家の意思を代理できない。** 見送りなのに在庫が危ない先は
    declined_at_risk で知らせる（計画では救えないので、配車はしない）。
    """
    base = {c.id: c for c in generate(seed=20261015, season="winter").customers}
    target = next(c for c in base.values() if c.is_mandatory)
    s = generate(seed=20261015, season="winter", declined_customers=(target.id,))
    c = next(x for x in s.customers if x.id == target.id)
    assert c.demand_l == 0
    assert not c.is_mandatory
    assert c.mandatory_basis is None
    assert c.declined_at_risk == any(
        t.is_below_reorder or t.stockout_before_next_delivery for t in c.tanks
    )
    # ほかの需要家は変わらない（見送りは1軒だけに効く）
    for x in s.customers:
        if x.id != target.id:
            assert x.demand_l == base[x.id].demand_l


def test_unknown_declined_customer_is_rejected():
    """見送りの綴り違いは落とす。黙って通すと、見送りが効かないまま
    「見送りを入れた計画」として読まれる。"""
    with pytest.raises(ValueError, match="名簿に無い"):
        generate(seed=1, declined_customers=("c99",))


def test_order_is_capped_by_ullage():
    """発注量が空き容量を超えていたら、入りきる分だけ降ろす。"""
    from dataclasses import replace

    c = next(x for x in generate(seed=20261015).customers if x.ordering)
    t = c.tanks[0]
    big = replace(c, orders=((t.product, t.capacity_l * 10),))
    assert big.demand_by_product()[t.product] == t.topup_l


def test_stockout_forces_a_visit_even_without_an_order():
    """発注配送の先でも、欠品の直前なら必須にする。**欠品は最上位。**

    何を頼むかは需要家が決めるが、頼み忘れで明日より先に空になるなら
    行く。これは元売の信用の問題なので、発注の有無より優先する。
    """
    from dataclasses import replace

    c = next(x for x in generate(seed=20261015).customers if x.ordering)
    t = c.tanks[0]
    dry = replace(t, level_l=t.daily_use_l // 2)          # 半日で空になる
    c2 = replace(c, orders=(), tanks=(dry,) + c.tanks[1:])
    assert dry.stockout_before_next_delivery
    assert c2.is_mandatory and c2.mandatory_basis == "stockout"
    assert c2.demand_by_product()[t.product] == dry.topup_l


def test_absent_driver_removes_vehicle():
    """乗務員が欠勤したら、その車は出せない。

    「車はあるが人がいない」を計画側で握りつぶさないための関門。
    デモ4（乗務員欠勤）が成立する前提でもある。
    """
    s = generate(seed=11, absent_drivers=("d1",))
    assert not s.driver("d1").available
    assert "v1" not in [v.id for v in s.usable_vehicles()]
    assert len(s.usable_vehicles()) == len(s.vehicles) - 1


def test_deliveries_balance_what_is_sold():
    """運ぶ量が売れる量と釣り合っていること。**売れる以上に運び続けることはできない。**

    以前の作り方（毎日「2日先に発注点を割るタンク」を満タンまで）は、名簿が
    1日に売る47kL（冬）に対して毎日約140kLを運ぶ需要を出していた。約3倍に
    膨らんだ需要の上で台数を数えていたことになる。2週間ならして、運ぶ量が
    売れる量の±25%に入ることを確かめる。
    """
    import datetime

    from core.generator import CLOSED_ON_SUNDAY, ROSTER, _planned_daily

    start = datetime.date(2026, 10, 12)
    delivered = 0
    sold = 0
    for k in range(14):
        d = start + datetime.timedelta(days=k)
        s = generate(seed=100 + k, date=d.isoformat(), season="winter")
        delivered += sum(c.demand_l for c in s.customers)
        for spec in ROSTER:
            if d.weekday() == 6 and CLOSED_ON_SUNDAY[spec.type]:
                continue                                   # 休業日は売れない
            sold += sum(_planned_daily(spec, "winter").values())
    assert 0.75 <= delivered / sold <= 1.25, (delivered, sold)


def test_sunday_is_about_sixty_percent_of_a_weekday():
    """日曜の配送量は平日の6割程度（実務）。休業のSSが多いため。"""
    import datetime
    import statistics

    start = datetime.date(2026, 10, 12)
    weekday, sunday = [], []
    for k in range(56):
        d = start + datetime.timedelta(days=k)
        total = sum(
            c.demand_l
            for c in generate(seed=1000 + k, date=d.isoformat(), season="winter").customers
        )
        (sunday if d.weekday() == 6 else weekday).append(total)
    ratio = statistics.mean(sunday) / statistics.mean(weekday)
    assert 0.45 <= ratio <= 0.70, ratio


def test_planned_volume_is_steady_monday_to_saturday():
    """計画配送の量は月〜土でほぼ一定（実務）。発注は上に乗って振れる。"""
    import datetime
    import statistics

    start = datetime.date(2026, 10, 12)
    planned = []
    for k in range(42):
        d = start + datetime.timedelta(days=k)
        if d.weekday() == 6:
            continue
        s = generate(seed=2000 + k, date=d.isoformat(), season="winter")
        planned.append(sum(c.demand_l for c in s.customers if c.planned))
    cv = statistics.pstdev(planned) / statistics.mean(planned)
    # 名簿が15軒と小さいので、周期の起点の割り付けで±1〜2割の凸凹は残る
    assert cv < 0.20, cv


def test_closed_customers_get_nothing_on_sunday():
    """日曜休業の先は、日曜に計画配送も発注も受けない。"""
    s = generate(seed=1, date="2026-10-18", season="winter")      # 日曜
    closed = [c for c in s.customers if not c.open_today]
    assert closed, "日曜に休業の先が1軒も無い"
    for c in closed:
        assert c.demand_l == 0 and not c.is_mandatory and not c.declined_at_risk


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
    from dataclasses import replace

    c = next(x for x in generate(seed=20261015, season="winter").customers if x.open_today)
    t = max(c.tanks, key=lambda t: t.topup_l)
    small = replace(c, orders=(), planned=((t.product, 2_000),))
    large = replace(c, orders=(), planned=((t.product, min(12_000, t.topup_l)),))
    assert small.demand_l < large.demand_l
    assert small.service_min < large.service_min


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
