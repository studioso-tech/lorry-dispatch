"""当日発注の組み直し。**このシステムの存在理由にあたる部分。**

前日に計画配送と発注を合わせて翌日の計画が固まる。当日になっても発注は入る
（実務）。引き金は3つ:

    連休      高速道路のSSが急に在庫を切らす
    価格改定  値下げ前は在庫を下げたがって足りなくなる／値上げ前は駆け込み
    天候      冬の灯油。寒波

**発注は証拠の残る形でしか受けない。** 発注システムかFAX。電話で連絡が入る
ことはあっても、発注そのものは必ずその形をとる（実務）。したがって
NewOrder は「証拠の所在」（source と evidence_ref）を必ず持つ。電話だけの
連絡は発注ではなく予告として扱い、組み直しには入れない。

組み直しは「朝の計画を捨てて解き直す」ではない。乗務員はもう朝の計画を
聞いて動いていて、積み込みを始めた便の油は車に載っている。**動かせるのは
まだ積んでいない便だけ**で、それもなるべく崩さない。

入らない発注があれば、**その日と翌日にどれだけ余力が残るか**で手を選ぶ（実務）:

    1. 専属の余り（午後の空き）で運ぶ
    2. 余力が20kLローリー1台ぶん以上残るなら、**計画配送を前後にずらして吸収**
       （在庫が切れない限り自由に動かせるので費用がかからない）
    3. 余力を割り込むなら、**費用を払ってでも非専属を動かす**（リスク対策）
    4. それでも入らない発注・欠品は「入らない」と正直に返す ← **ここを記録する**

実績は確定していても先は予定なので、不測の事態にどれだけ応えられるかで決める。
**欠品と、発注に応えられないことが積み上がったら**体制の見直しを提案する
（エージェント層の仕事）。一時的なら可、恒常的なら構造の欠陥（実務）。

**近似**:

- 午前にとった休憩の有無は追わない。1日ぶんの休憩を拘束の中に見込む
- 午前に降ろした先のタンクは、その量だけ上がったとする（午後の消費は見ない）
- 積み込みを始めた便は、まだ需要家に着いていなくても動かさない
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from core.domain import (
    COMPARTMENT_UNIT_L,
    RESERVE_UNIT_L,
    DROP_STEP_BY_PRODUCT,
    MIN_STOP_DROP_L,
    Customer,
    Driver,
    Product,
    Scenario,
    Vehicle,
    VehicleClass,
    minutes_to_hhmm,
    whole_drops_l,
)
from core.events import SoldMore, sell_more, validate as validate_events
from core.solver import Plan, SolveMode, Trip, Unserved, VehicleDay, solve
from core.travel import build_matrix

#: 非専属を呼んでから、油槽所で積込を始められるまでの時間（分）。**仮置。**
SPOT_LEAD_MIN = 60

#: 組み直しで、午後に回れる便の上限（車1台あたり）。
REPLAN_MAX_TRIPS = 3

#: これより少ない量のためには車を出さない。**2kLだけの荷卸しはしない**（実務）ので、
#: 1軒に降ろす合計がこれに満たなければ、SSに確認してからにする。
MIN_DROP_L = MIN_STOP_DROP_L

#: 予備能力がこれを下回るなら、費用を払ってでも非専属を動かす（実務・2026-09-16）。
#:
#: 判断は「小口かどうか」ではなく、**その日と翌日にどれだけ余力が残るか**。実績は
#: 確定していても先は予定なので、不測の事態にどれだけ応えられるかで決める。
#: 余力は**20kLローリー何台ぶんか**で数える（`domain.RESERVE_UNIT_L`）。
#:
#:     余力が1台ぶん以上残る → 計画配送を前後にずらして吸収（費用がかからない）
#:     余力を割り込む       → 非専属を呼ぶ（費用はかかるがリスク対策）
MIN_RESERVE_UNITS = 1

#: 小口の発注とみなす量（1軒ぶんの合計）。**仮置。** 実務: 特殊な油種でなければ
#: 2kL・4kLの発注はあり得ず、小口の発注は翌日の計画配送を前倒しして合わせて運ぶ。
SMALL_ORDER_MAX_L = 4_000

#: 発注として受け付ける経路。**証拠の残るものだけ。**（実務）
ORDER_SOURCES = ("system", "fax")

#: 予告（電話などの連絡）。**発注ではない。** 試算（dry_run）の中でだけ使える。
#: 「入る余地があるか」を先に見ておき、発注書が届いたら発注として入れ直す。
NOTICE_SOURCE = "notice"

PRODUCT_JA = {
    Product.KEROSENE: "灯油",
    Product.GASOIL: "軽油",
    Product.REGULAR: "レギュラー",
    Product.PREMIUM: "ハイオク",
}


@dataclass(frozen=True)
class NewOrder:
    """当日に入った発注。

    Attributes:
        customer_id: 需要家。
        product: 油種。
        volume_l: 量（L）。
        source: 受けた経路。system（発注システム）か fax。**電話は不可。**
        evidence_ref: 証拠の所在（発注システムの伝票番号、FAX画像の保存先など）。
            後から「どの発注で車を動かしたか」を辿れるようにする。
    """

    customer_id: str
    product: Product
    volume_l: int
    source: str
    evidence_ref: str

    def __post_init__(self) -> None:
        if self.source not in ORDER_SOURCES + (NOTICE_SOURCE,):
            raise ValueError(
                f"発注は証拠の残る形でしか受けない（{'/'.join(ORDER_SOURCES)}）。"
                f"{self.source!r} は不可。電話の連絡は予告（{NOTICE_SOURCE}）として扱う"
            )
        if not self.evidence_ref:
            raise ValueError("発注には証拠の所在（伝票番号・FAX画像）が要る")

    @property
    def is_notice(self) -> bool:
        """予告か（発注ではない）。"""
        return self.source == NOTICE_SOURCE


@dataclass(frozen=True)
class Escape:
    """応えられなかった先。**恒常化の監視はこれを数える。**

    **計画配送を明日にずらすのは、ここに入らない。** 在庫が切れない限り前後に
    動かせるので、ふつうの運用の幅にあたる（実務・2026-09-16）。数えるのは
    **欠品**（今日行かないと明日の配送より先に空になる）と、**発注に応えられない**
    こと。この2つが積み上がるなら、体制のほうに無理がある。
    """

    kind: str               # "stockout"（欠品の直前）/ "order"（発注に応えられない）
    customer_id: str
    customer_name: str
    volume_l: int


@dataclass
class ReplanResult:
    """組み直しの結果。"""

    now_min: int
    vehicle_days: tuple[VehicleDay, ...]     # 動かさなかった便＋組み直した便
    frozen_trips: tuple[Trip, ...]
    sub_plan: Plan                           # 組み直した部分だけの求解結果
    changes: tuple[str, ...]                 # 配車係の言葉での変更点
    escapes: tuple[Escape, ...]
    unfilled: tuple[Unserved, ...]           # 必須（発注・欠品）なのに入らなかった
    spot_called: tuple[str, ...]
    orders: tuple[NewOrder, ...] = ()
    #: 発注量どおりに入らなかった発注と、その理由。**配車係がSSに確認する材料。**
    shortfalls: tuple[str, ...] = ()
    #: 試算か。**試算の結果で車を動かしてはいけない**し、記録もしない。
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        """入れるべきものがすべて入ったか（逃げ道は使っていてもよい）。"""
        return self.sub_plan.is_usable and not self.unfilled

    def summary_ja(self) -> str:
        now = minutes_to_hhmm(self.now_min)
        if not self.sub_plan.is_usable:
            return f"{now} 時点で組み直せませんでした。"
        if self.dry_run:
            parts = [f"【試算・未確定】{now} 時点で入れるとしたら。発注書（発注システムかFAX）が届くまで車は動かしません。"]
        else:
            parts = [f"{now} 時点で組み直しました。"]
        if self.spot_called:
            parts.append(f"非専属を{len(self.spot_called)}台呼びます。")
        if self.escapes:
            parts.append(f"**{len(self.escapes)}軒に応えられません（欠品・発注）。**")
        if self.unfilled:
            parts.append(f"**{len(self.unfilled)}軒は今日中に入りません。**")
        if self.shortfalls:
            parts.append(f"**発注量どおりに入らないものが{len(self.shortfalls)}件あります（SSに確認）。**")
        if not (self.spot_called or self.escapes or self.unfilled or self.shortfalls):
            parts.append("専属の余りで入ります。")
        return "".join(parts)


def validate_orders(scenario: Scenario, orders: list[NewOrder]) -> None:
    """発注が名簿と噛み合っているかだけを見る。**組み直す前に呼べる。**

    朝の計画を組むのに1分以上かかるので、名簿に無い需要家のような
    そもそも成り立たない発注は、求解の前にここで弾く。
    """
    by_id = {c.id: c for c in scenario.customers}
    for o in orders:
        c = by_id.get(o.customer_id)
        if c is None:
            raise ValueError(f"名簿に無い需要家: {o.customer_id}")
        if o.volume_l <= 0:
            raise ValueError(f"{c.name}: 発注量は正の値（{o.volume_l}L）")
        if not any(t.product is o.product for t in c.tanks):
            raise ValueError(f"{c.name} に {PRODUCT_JA[o.product]} のタンクは無い")


def replan(
    scenario: Scenario,
    base: Plan,
    *,
    now_min: int,
    orders: list[NewOrder],
    spot_standby: int = 0,
    time_limit_s: float = 20.0,
    mode: SolveMode = SolveMode.FAST,
    dry_run: bool = False,
    sold_more: tuple[SoldMore, ...] = (),
) -> ReplanResult:
    """朝の計画 base に、当日の発注 orders を差し込む。

    Args:
        scenario: 朝の計画を組んだときの問題。
        base: 朝の計画。
        now_min: いまの時刻（0時からの分）。これより前に積込を始めた便は動かさない。
        orders: 当日に入った発注（発注システムかFAX。電話は不可）。
        spot_standby: いま呼べる待機の非専属の台数。
        time_limit_s: 求解の打ち切り（FAST のとき）。
        mode: 既定は FAST。組み直しは「いま組み直してほしい」要求なので時間で切る。
        dry_run: 試算か。予告（電話の連絡など）を入れられるのは試算だけ。
        sold_more: **日中に分かった**「見込みより売れた」（FAXの但し書き等）。
            朝の計画は見込みで組まれているので、ここで初めて在庫に当てる。
            積込済みの便が降ろした量を**朝の見込みで**差し引いてから当てる
            （先に当てると欠品扱いで降ろす量が変わり、朝の車が積んだ量と合わない）。

    Raises:
        ValueError: 名簿に無い需要家、タンクの無い油種、量が0以下の発注。
            黙って捨てると「発注を入れた計画」として読まれるので落とす。
    """
    validate_orders(scenario, orders)
    validate_events(scenario.customers, tuple(sold_more))
    events_by_customer: dict[str, list[SoldMore]] = {}
    for e in sold_more:
        events_by_customer.setdefault(e.customer_id, []).append(e)
    notices = [o for o in orders if o.is_notice]
    if notices and not dry_run:
        # **予告で車を動かさない。** 発注は証拠の残る形でしか受けない（実務）。
        raise ValueError(
            f"予告（{', '.join(o.customer_id for o in notices)}）は発注ではない。"
            "試算（dry_run）でしか入れられない。発注書が届いてから入れ直す"
        )

    # --- 1. 動かさない便 --------------------------------------------------
    frozen_by_vehicle: dict[str, list[Trip]] = {}
    for vd in base.vehicle_days:
        frozen_by_vehicle[vd.vehicle_id] = [t for t in vd.trips if t.load_start_min < now_min]
    frozen_trips = tuple(t for ts in frozen_by_vehicle.values() for t in ts)
    served_frozen = {st.customer_id for t in frozen_trips for st in t.stops}

    # **積込済みの便が実際に降ろす量**を油種ごとに数える。
    #
    # 1軒の1日ぶんが1台に載らない大型店は、2〜3回に分けて届ける（実務）。
    # その**1回目だけが積込済み**という状態は普通に起きる。以前はここで
    # 「その先の1日ぶんが全部届いた」ことにしていたので、**残りの回が需要ごと
    # 消えていた**（朝26.0kLの予定が、組み直すと12.0kLになって報告も出ない。
    # 2026-09-20 発見）。
    frozen_delivered: dict[str, dict[Product, int]] = {}
    for t in frozen_trips:
        for st in t.stops:
            d = frozen_delivered.setdefault(st.customer_id, {})
            for p, v in st.delivered_by_product:
                d[p] = d.get(p, 0) + v

    # --- 2. 需要家: 午前に降ろした先は済み、新しい発注を足す ---------------
    new_by_customer: dict[str, dict[Product, int]] = {}
    for o in orders:
        d = new_by_customer.setdefault(o.customer_id, {})
        d[o.product] = d.get(o.product, 0) + o.volume_l

    # 積込済みの便で、これから（または今朝）降ろす先。発注と重なると、
    # タンクの空きが足りず発注量どおりに入らないことがある。
    frozen_visit: dict[str, tuple[str, int, int]] = {}
    for vd in base.vehicle_days:
        for t in frozen_by_vehicle.get(vd.vehicle_id, []):
            for st in t.stops:
                frozen_visit[st.customer_id] = (vd.vehicle_name.split("（")[0], st.arrive_min, st.delivered_l)

    customers: list[Customer] = []
    pulled: dict[str, tuple[tuple[Product, int], ...]] = {}
    shortfalls: list[str] = []
    deliverable: dict[tuple[str, Product], int] = {}
    for c in scenario.customers:
        c2 = c
        if c.id in served_frozen:
            # **届いた量だけを在庫に足す。** 分けて届ける先は、残りの回がまだある
            delivered = frozen_delivered.get(c.id, {})
            tanks = tuple(
                replace(
                    t,
                    level_l=min(
                        t.capacity_l - t.ullage_margin_l,
                        t.level_l + delivered.get(t.product, 0),
                    ),
                )
                for t in c.tanks
            )
            # 計画配送・発注も、届いたぶんだけ減らす（丸ごと消さない）
            def _left(items: tuple[tuple[Product, int], ...]) -> tuple[tuple[Product, int], ...]:
                out = []
                for p, v in items:
                    rest = v - delivered.get(p, 0)
                    if rest > 0:
                        out.append((p, rest))
                return tuple(out)

            c2 = replace(c, tanks=tanks, planned=_left(c.planned), orders=_left(c.orders))
        if c.id in events_by_customer:
            # 日中に分かった「見込みより売れた」。積込済みの便のぶんを差し引いた後に当てる
            c2 = replace(c2, tanks=sell_more(c2.tanks, events_by_customer[c.id]))
        if c.id in new_by_customer:
            # **発注量どおりに入るとは限らない。** タンクの空き以上は降ろせない。
            # 積込済みの便がこれから同じ先に降ろすなら、空きはさらに小さい。
            # 黙って空きまで削ると、発注量で運んだように読まれる（最初の実装が
            # これで、8kLの発注に0.8kLを運ぶために1台出していた）。
            merged = dict(c2.orders)
            room = {t.product: t.topup_l for t in c2.tanks}
            new = dict(new_by_customer[c.id])
            # **小口の発注は、翌日の計画配送を前倒しして合わせ、当日1回で運ぶ**（実務）。
            # 明日運ぶはずの量を今日運ぶ（小口の発注はその中に含まれる）ので、
            # **足すのではなく大きいほうを取る**。足すと空きに入らない
            # （タンクの空きは配送の周期ぶんしかない。2026-09-16 に修正）。
            want = dict(new)
            if (sum(new.values()) <= SMALL_ORDER_MAX_L and c.tomorrow_planned
                    and c.id not in served_frozen):
                for p, v in c.tomorrow_planned:
                    want[p] = max(want.get(p, 0), v)
                pulled[c.id] = tuple(c.tomorrow_planned)
            for p, v in want.items():
                ordered_v = new.get(p, 0)
                fits = max(0, room.get(p, 0) - merged.get(p, 0))
                # **槽の単位でしか積めない**（実務）。主要3油種は4kL、ハイオクは2kL。
                # 刻みに満たなければ車は出さない
                take = whole_drops_l(p, min(v, fits))
                if take < ordered_v:
                    step = DROP_STEP_BY_PRODUCT[p]
                    if ordered_v % step and fits >= ordered_v:
                        why = (f"{PRODUCT_JA[p]}は{step // 1000}kL槽の単位で運ぶ"
                               "（明日が計画配送の日でないので、合わせて運べない）")
                    else:
                        why = f"タンクの空きが {fits / 1000:.1f}kL しかない"
                    if c.id in frozen_visit:
                        vname, at, vol = frozen_visit[c.id]
                        why += f"（{minutes_to_hhmm(at)} に{vname}が {vol / 1000:.1f}kL 降ろす便が積込済み）"
                    what = (
                        f"入るのは {take / 1000:.1f}kL" if take
                        else f"入る量が {DROP_STEP_BY_PRODUCT[p] / 1000:.0f}kL に満たないので車は出さない"
                    )
                    shortfalls.append(
                        f"{c.name} {PRODUCT_JA[p]}: 発注 {ordered_v / 1000:.1f}kL のうち{what}。"
                        f"{why}。**SSに確認が要る**"
                    )
                deliverable[(c.id, p)] = deliverable.get((c.id, p), 0) + take
                if take > 0:
                    merged[p] = merged.get(p, 0) + take
            c2 = replace(c2, orders=tuple(merged.items()))
        customers.append(c2)

    # --- 3. 車: 何時から空くか・今日の拘束・運転済み ------------------------
    base_day = {vd.vehicle_id: vd for vd in base.vehicle_days}
    vehicles: list[Vehicle] = []
    for veh in scenario.usable_vehicles():
        frozen = frozen_by_vehicle.get(veh.id, [])
        if frozen:
            vd = base_day[veh.id]
            vehicles.append(
                replace(
                    veh,
                    available_from_min=frozen[-1].return_min,
                    duty_origin_min=vd.duty_start_min,
                    driven_min=sum(t.drive_min for t in frozen),
                    # **朝の計画と同じ定義で数える**（運転＋荷卸し＋積込）。朝の計画は
                    # 手待ちを仕事に数えていない。手待ちまで数えると組み直しのほうが
                    # 厳しくなり、ぎりぎりで成り立っていた朝の計画が組み直すと成り立た
                    # なくなって、発注が1件も増えていないのに朝の発注が落ちた。
                    # 原則: **朝の計画は、組み直しの規則でも必ず成り立つ。**
                    worked_min=sum(
                        t.drive_min + t.service_min + scenario.depot.load_min for t in frozen
                    ),
                )
            )
        else:
            vehicles.append(replace(veh, available_from_min=now_min))

    drivers = list(scenario.drivers)
    for k in range(spot_standby):
        did = f"ds{k + 1}"
        drivers.append(Driver(id=did, name=f"非専属乗務員{k + 1}"))
        vehicles.append(
            Vehicle(
                id=f"s{k + 1}",
                name=f"非専属{k + 1}（単車 16kL）",
                vehicle_class=VehicleClass.RIGID,
                # 単車は2kL槽が多い（実務）。4号車と同じ区割り（仮置）
                compartments_l=(2_000, 2_000, 2_000, 2_000, 4_000, 4_000),
                driver_id=did,
                spot=True,
                available_from_min=now_min + SPOT_LEAD_MIN,
            )
        )

    # --- 4. 朝の担当を基準に解き直す --------------------------------------
    # --- 予備能力（20kLローリー何台ぶん残るか） -----------------------------
    #
    # **計画配送は在庫が切れない限り前後に動かせる**（実務）ので、余力があるうちは
    # ずらして吸収する。余力を割り込むなら、費用を払ってでも非専属を動かす。
    remaining_demand = sum(c.demand_l for c in customers if c.id not in served_frozen)
    # **数えるのは専属だけ。** 呼ぼうとしている非専属を余力に数えると、
    # 「非専属がいるから余力あり→だから非専属は呼ばない」という堂々巡りになる
    remaining_capacity = sum(
        veh.capacity_l * max(0, REPLAN_MAX_TRIPS - len(frozen_by_vehicle.get(veh.id, [])))
        for veh in vehicles if not veh.spot
    )
    reserve_units = (remaining_capacity - remaining_demand) // RESERVE_UNIT_L
    ample = reserve_units >= MIN_RESERVE_UNITS
    if ample:
        # **余力があるうちは非専属を呼ばない。** 計画配送は在庫が切れない限り
        # 前後に動かせるので、ずらして吸収するほうが安い。費用のほうを触ると
        # 「何も差し込んでいないのに勝手にずらす」計画が出る（2026-09-16 に修正）。
        vehicles = [v for v in vehicles if not v.spot]
        drivers = [d for d in drivers if d.id not in {f"ds{k + 1}" for k in range(spot_standby)}]

    baseline = {
        st.customer_id: vd.vehicle_id
        for vd in base.vehicle_days
        for t in vd.trips
        if t.load_start_min >= now_min
        for st in t.stops
    }
    sub = Scenario(
        date=scenario.date,
        season=scenario.season,
        seed=scenario.seed,
        depot=scenario.depot,
        customers=tuple(customers),
        vehicles=tuple(vehicles),
        drivers=tuple(drivers),
        notes=list(scenario.notes) + [f"当日の組み直し {minutes_to_hhmm(now_min)}"],
    )
    sub_plan = solve(
        sub,
        matrix=build_matrix(sub.depot, sub.customers),
        time_limit_s=time_limit_s,
        max_trips=REPLAN_MAX_TRIPS,
        mode=mode,
        baseline=baseline,
    )

    # --- 5. 合わせて1日にする ----------------------------------------------
    sub_day = {vd.vehicle_id: vd for vd in sub_plan.vehicle_days}
    merged_days: list[VehicleDay] = []
    for veh in vehicles:
        frozen = frozen_by_vehicle.get(veh.id, [])
        after = sub_day.get(veh.id)
        if not frozen and after is None:
            continue
        trips = list(frozen)
        if after is not None:
            trips += [
                replace(t, trip_index=len(frozen) + i) for i, t in enumerate(after.trips)
            ]
        first = base_day[veh.id] if frozen else after
        merged_days.append(
            VehicleDay(
                vehicle_id=veh.id,
                vehicle_name=veh.name,
                driver_id=first.driver_id,
                driver_name=first.driver_name,
                duty_start_min=first.duty_start_min,
                duty_end_min=after.duty_end_min if after else frozen[-1].return_min,
                drive_min=sum(t.drive_min for t in trips),
                break_min=after.break_min if after else base_day[veh.id].break_min,
                trips=tuple(trips),
            )
        )

    # --- 6. 朝の計画との違い ----------------------------------------------
    by_id = {c.id: c for c in customers}
    names = {v.id: v.name.split("（")[0] for v in vehicles}
    # 便番号は1日通しで数える（積込済みの便のあとに続く）。組み直した部分
    # だけの番号を出すと、朝4時の便を積んだ車の午後の便が「1便目」になる。
    new_assign: dict[str, tuple[str, int, int]] = {}
    for vd in sub_plan.vehicle_days:
        offset = len(frozen_by_vehicle.get(vd.vehicle_id, []))
        for t in vd.trips:
            for st in t.stops:
                new_assign[st.customer_id] = (vd.vehicle_id, offset + t.trip_index, st.arrive_min)

    # **量でも突き合わせる。** 大型店は1日に2〜3回に分けて届けるので（実務）、
    # 「どの車が行くか」だけを見ていると、**分けたうちの1回が落ちても気づかない**。
    # 実際、朝26.0kLの予定が組み直しで12.0kLになっても、担当は変わらないので
    # 何も報告されなかった（2026-09-20 に発見）。配車係が見て分かる形で出す。
    def volumes(days, *, only_after: int | None = None) -> dict[str, int]:
        out: dict[str, int] = {}
        for vd in days:
            for t in vd.trips:
                if only_after is not None and t.load_start_min < only_after:
                    continue
                for st in t.stops:
                    out[st.customer_id] = out.get(st.customer_id, 0) + st.delivered_l
        return out

    planned_l = volumes(base.vehicle_days, only_after=now_min)
    new_l = volumes(sub_plan.vehicle_days)

    changes: list[str] = []
    why = "余力の範囲で" if ample else "**余力を割り込むため**"
    missed_ids = {u.customer_id for u in sub_plan.mandatory_missed}
    for cid, old_vid in baseline.items():
        c = by_id[cid]
        if cid in new_assign:
            vid, trip, arrive = new_assign[cid]
            if vid != old_vid:
                changes.append(
                    f"{c.name}: {names.get(old_vid, old_vid)} → {names[vid]}"
                    f"の{trip + 1}便目（{minutes_to_hhmm(arrive)}着）に担当替え"
                )
        # **今日運ぶ量が減ったら、必ず言う。**
        #
        # **計画配送をずらすのは逃げ道ではない**（在庫が切れない限り前後に
        # 動かせる。実務・2026-09-16）。数えないのと、黙っているのは別である。
        # 必須で落ちたもの（発注・欠品の直前）は unfilled で別に出るので、
        # ここでは二重に言わない。
        short = planned_l.get(cid, 0) - new_l.get(cid, 0)
        if short <= 0:
            continue
        if cid in missed_ids and c.mandatory_basis != "planned":
            # 発注・欠品の直前で落ちたものは unfilled で別に出る。二重に言わない
            continue
        kind = "計画配送" if c.mandatory_basis == "planned" else "今日の予定"
        planned = planned_l.get(cid, 0)
        whole = short >= planned
        line = (
            f"{c.name}: {kind} {planned / 1000:.1f}kL を明日にずらす"
            if whole
            else f"{c.name}: {kind} {planned / 1000:.1f}kL のうち "
                 f"{short / 1000:.1f}kL を明日にずらす"
        )
        changes.append(f"{line}（{why}）")

    for o in orders:
        c = by_id[o.customer_id]
        take = deliverable.get((o.customer_id, o.product), 0)
        if take <= 0:
            continue      # 入る余地が無い。shortfalls で確認を求める
        amount = (
            f"{take / 1000:.1f}kL" if take == o.volume_l
            else f"{take / 1000:.1f}kL（発注 {o.volume_l / 1000:.1f}kL）"
        )
        via = {"system": "発注システム", "fax": "FAX", NOTICE_SOURCE: "予告・未確定"}[o.source]
        label = f"{c.name} {PRODUCT_JA[o.product]} {amount}（当日発注・{via} {o.evidence_ref}）"
        if o.customer_id in new_assign:
            vid, trip, arrive = new_assign[o.customer_id]
            changes.append(f"{label}: {names[vid]}の{trip + 1}便目で {minutes_to_hhmm(arrive)} 着")
        else:
            changes.append(f"{label}: **今日中に入りません**")
    for cid, plan_items in pulled.items():
        what = "・".join(
            f"{PRODUCT_JA[p]} {deliverable.get((cid, p), 0) / 1000:.0f}kL"
            for p, _ in plan_items if deliverable.get((cid, p), 0) > 0
        )
        if what:
            changes.append(
                f"{by_id[cid].name}: 小口の発注なので、明日の計画配送を前倒しして合わせて運ぶ（{what}）"
            )

    spot_ids = {v.id for v in vehicles if v.spot}
    spot_called = tuple(vd.vehicle_id for vd in sub_plan.vehicle_days if vd.vehicle_id in spot_ids)
    for sid in spot_called:
        changes.append(
            f"{names[sid]}を呼ぶ（{minutes_to_hhmm(now_min + SPOT_LEAD_MIN)} 以降に積込）"
        )

    unfilled = tuple(
        u for u in sub_plan.mandatory_missed
        if by_id[u.customer_id].mandatory_basis != "planned"
    )
    # **恒常化の監視が数えるのはここ。** 欠品と、発注に応えられなかった先。
    escapes = [
        Escape(by_id[u.customer_id].mandatory_basis or "shortage",
               u.customer_id, u.customer_name, u.demand_l)
        for u in unfilled
    ]

    return ReplanResult(
        now_min=now_min,
        vehicle_days=tuple(merged_days),
        frozen_trips=frozen_trips,
        sub_plan=sub_plan,
        changes=tuple(changes),
        escapes=tuple(escapes),
        unfilled=unfilled,
        spot_called=spot_called,
        orders=tuple(orders),
        shortfalls=tuple(shortfalls),
        dry_run=dry_run,
    )


def format_replan_ja(r: ReplanResult) -> str:
    """組み直しを配車係の言葉で書き出す。変わったところを先に出す。"""
    lines = [f"【当日の組み直し】{r.summary_ja()}", ""]
    if r.shortfalls:
        lines.append("■ 確認が要るもの")
        lines += [f"   {x}" for x in r.shortfalls]
        lines.append("")
    if r.changes:
        lines.append("■ 変わるところ")
        lines += [f"   {c}" for c in r.changes]
        lines.append("")
    lines.append(
        f"■ 動かさない便（{minutes_to_hhmm(r.now_min)} までに積込を始めた）: {len(r.frozen_trips)}便"
    )
    lines.append("")
    frozen_ids = {id(t) for t in r.frozen_trips}
    for vd in r.vehicle_days:
        lines.append(
            f"■ {vd.vehicle_name}　拘束 {minutes_to_hhmm(vd.duty_start_min)}〜"
            f"{minutes_to_hhmm(vd.duty_end_min)}（{vd.duty_min // 60}時間{vd.duty_min % 60}分）"
        )
        for t in vd.trips:
            mark = "（積込済み）" if id(t) in frozen_ids else ""
            stops = "、".join(
                f"{minutes_to_hhmm(s.arrive_min)} {s.customer_name} {s.delivered_l / 1000:.1f}kL"
                for s in t.stops
            )
            lines.append(
                f"   ‹{t.trip_index + 1}便›{mark} 積込 {minutes_to_hhmm(t.load_start_min)}　{stops}"
            )
    return "\n".join(lines)
