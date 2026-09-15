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

入らない発注があれば、次の順で手を打つ（ソルバーの目的関数がこの順に
なるよう罰を置いてある）:

    1. 専属の余り（午後の空き）で運ぶ
    2. 待機の非専属を呼ぶ（故障の穴埋めと同じく非専属が当たる。実務）
    3. 計画配送を明日にずらす ← **逃げ道。使ったら記録する**
    4. それでも入らない発注・欠品は「入らない」と正直に返す

逃げ道は一時的なら可、恒常的なら構造の欠陥（実務）。記録した逃げ道が
積み上がったら体制の見直しを提案する（エージェント層の仕事）。

**近似**:

- 午前にとった休憩の有無は追わない。1日ぶんの休憩を拘束の中に見込む
- 午前に降ろした先のタンクは、その量だけ上がったとする（午後の消費は見ない）
- 積み込みを始めた便は、まだ需要家に着いていなくても動かさない
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from core.domain import (
    Customer,
    Driver,
    Product,
    Scenario,
    Vehicle,
    VehicleClass,
    minutes_to_hhmm,
)
from core.events import SoldMore, sell_more, validate as validate_events
from core.solver import Plan, SolveMode, Trip, Unserved, VehicleDay, solve
from core.travel import build_matrix

#: 非専属を呼んでから、油槽所で積込を始められるまでの時間（分）。**仮置。**
SPOT_LEAD_MIN = 60

#: 組み直しで、午後に回れる便の上限（車1台あたり）。
REPLAN_MAX_TRIPS = 3

#: これより少ない量のためには車を出さない（いちばん小さい槽＝2kL）。
#: 発注量どおりに入らず、入る量がこれに満たないときは、SSに確認してからにする。
MIN_DROP_L = 2_000

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
    """使った逃げ道。**恒常化の監視はこれを数える。**"""

    kind: str               # "defer_planned"（計画配送を明日にずらす）
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
    #: 試算か。**試算の結果で車を動かしてはいけない**し、逃げ道も記録しない。
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
            parts.append(f"計画配送{len(self.escapes)}軒を明日にずらします（逃げ道）。")
        if self.unfilled:
            parts.append(f"**{len(self.unfilled)}軒は今日中に入りません。**")
        if self.shortfalls:
            parts.append(f"**発注量どおりに入らないものが{len(self.shortfalls)}件あります（SSに確認）。**")
        if not (self.spot_called or self.escapes or self.unfilled or self.shortfalls):
            parts.append("専属の余りで入ります。")
        return "".join(parts)


def _validate(scenario: Scenario, orders: list[NewOrder]) -> None:
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
    _validate(scenario, orders)
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
    shortfalls: list[str] = []
    deliverable: dict[tuple[str, Product], int] = {}
    for c in scenario.customers:
        c2 = c
        if c.id in served_frozen:
            delivered = c.demand_by_product()
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
            c2 = replace(c, tanks=tanks, planned=(), orders=())
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
            for p, v in new_by_customer[c.id].items():
                fits = max(0, room.get(p, 0) - merged.get(p, 0))
                take = min(v, fits)
                if take < v and take < MIN_DROP_L:
                    take = 0      # 2kL槽1つにも満たない量のために車は出さない
                if take < v:
                    why = f"タンクの空きが {fits / 1000:.1f}kL しかない"
                    if c.id in frozen_visit:
                        vname, at, vol = frozen_visit[c.id]
                        why += f"（{minutes_to_hhmm(at)} に{vname}が {vol / 1000:.1f}kL 降ろす便が積込済み）"
                    what = (
                        f"入るのは {take / 1000:.1f}kL" if take
                        else f"入る量が {MIN_DROP_L / 1000:.0f}kL に満たないので車は出さない"
                    )
                    shortfalls.append(
                        f"{c.name} {PRODUCT_JA[p]}: 発注 {v / 1000:.1f}kL のうち{what}。"
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
                compartments_l=(4_000, 4_000, 4_000, 4_000),
                driver_id=did,
                spot=True,
                available_from_min=now_min + SPOT_LEAD_MIN,
            )
        )

    # --- 4. 朝の担当を基準に解き直す --------------------------------------
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

    changes: list[str] = []
    escapes: list[Escape] = []
    for cid, old_vid in baseline.items():
        c = by_id[cid]
        if cid in new_assign:
            vid, trip, arrive = new_assign[cid]
            if vid != old_vid:
                changes.append(
                    f"{c.name}: {names.get(old_vid, old_vid)} → {names[vid]}"
                    f"の{trip + 1}便目（{minutes_to_hhmm(arrive)}着）に担当替え"
                )
        elif c.mandatory_basis == "planned":
            escapes.append(Escape("defer_planned", cid, c.name, c.demand_l))
            changes.append(f"{c.name}: 計画配送 {c.demand_l / 1000:.1f}kL を明日にずらす（逃げ道）")

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
