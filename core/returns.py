"""持ち戻り: 計画配送の先で、持って行った量を降ろせない。

**計画配送の先の在庫は元売が管理する。**（実務）荷卸しの前にSS等の人と乗務員が
計画どおり降ろせるかを確かめるが、在庫の管理責任は元売にある。見込みが外れると、
着いてからタンクに入らないことがまれに起きる。これを**持ち戻り**という。

対応には順番がある（実務）:

    1. 乗務員の連絡を車庫経由で受け、**近くで降ろせる先を探して配送する**（最優先）
    2. 見つからなければ、**大竹・岩国インター近くの大型SSがクッション**として受け入れる
    3. それでも無ければ**積場へ持ち戻り**、その油種・数量を積まないことで**買い戻し**

**費用は元売の管理責任**で、運送会社やSSには請求しない（空の荷を運んだのと同じ）。

**持ち戻り品の処分が最優先**（実務）
----------------------------------

降ろすのが先で、**次の配送の時刻がずれるのは受け入れる**。積んだままでは後の配送が
ずっとその持ち戻りの制約を受けるからで、先に身軽にするほうが一日として早く収まる。
**必要なら、次の配送先分は他のローリーが持つ。**

このため:

    - 寄り道は**降ろせなかった先のすぐ後**に置く。便の帰り道まで積んだまま走る案は
      取らない（以前はこちらを寄り道が短いという理由で選んでいた。優先順が逆だった）
    - 後の先が受入時間に間に合わないことは、**その手を捨てる理由にしない**。
      「他のローリーで持つ先」として結果に並べる
    - **拘束・運転の上限だけは崩さない**（法令）。超えるときは後の便から順に
      他のローリーへ渡して収める。渡しても収まらない手だけを、成り立たないとする

**持ち戻る量は槽の単位**（実務）
------------------------------

槽は2kLと4kLの組合せで、**その槽の全量が入らなければ、その槽は降ろさない**。途中まで
降ろして半端を持ち帰ることはしない。よって持ち戻りの量は必ず**2kLの倍数**で、3kLの
ような量は起こり得ない。降ろし先に「丸ごと入る空き」を求めているのは、この裏返し。

候補を探すときに見るもの:

    - その油種のタンクに**丸ごと入る空き**があるか。今日すでに降ろした先や、
      このあと降ろす予定の先は、そのぶん空きを減らして数える
    - 着く時刻に受け入れているか（早く着いたぶんは待つ。閉まった後は届かない）

**近似**:

    - 複数槽ぶんを持ち戻ったとき、槽ごとに分けて2軒へ降ろすことは見ない
      （丸ごと入る先だけを探す）
    - 引き継ぎの単位は「後の便まるごと」または「この便の後ろの先」。渡した先を
      他のどのローリーが持てるかまでは、ここでは決めない（当日の組み直しの仕事）
    - 休憩は入れ直さない。拘束は朝の計画の始まりから帰庫までで数える
"""

from __future__ import annotations

from dataclasses import dataclass

from core.domain import (
    COMPARTMENT_SIZES_L,
    Customer,
    CustomerType,
    Product,
    Scenario,
    Tank,
    minutes_to_hhmm,
    unload_minutes,
)
from core.solver import Plan, Trip, VehicleDay
from core.travel import leg_minutes

PRODUCT_JA = {
    Product.KEROSENE: "灯油",
    Product.GASOIL: "軽油",
    Product.REGULAR: "レギュラー",
    Product.PREMIUM: "ハイオク",
}

#: 乗務員から車庫へ連絡し、降ろし先を決めて動き出すまで（仮置）。
CONFIRM_MIN = 10

#: 持ち戻る量の刻み。槽は2kLと4kLなので、槽の単位＝2kLの倍数にしかならない。
COMPARTMENT_UNIT_L = min(COMPARTMENT_SIZES_L)

RESPONSIBILITY_JA = (
    "費用は元売の管理責任（計画配送の先の在庫は元売が管理する）。"
    "運送会社・SSへの請求はありません。"
)


def _cushion(cid: str, name: str, lat: float, lon: float) -> Customer:
    """クッションの大型SS。4油種とも大きな空きがあり、24時間受け入れる。"""
    tanks = tuple(
        Tank(product=p, capacity_l=40_000, level_l=10_000, reorder_l=8_000, daily_use_l=4_000,
             ullage_margin_l=2_000)
        for p in Product
    )
    return Customer(
        id=cid, name=name, type=CustomerType.URBAN_GS_24H, lat=lat, lon=lon, tanks=tanks,
        window_open_min=0, window_close_min=24 * 60, accepts_trailer=True, bore_inch=3,
    )


#: 大竹・岩国インター近くの大型SS。**持ち戻りのクッション**として受け入れる（実務）。
#: 店名は架空。座標は各インターの近くに置いた（仮置）。
CUSHION_STATIONS: tuple[Customer, ...] = (
    _cushion("k1", "大竹IC近くの大型SS", 34.2340, 132.2120),
    _cushion("k2", "岩国IC近くの大型SS", 34.1740, 132.1570),
)


@dataclass(frozen=True)
class ReturnEvent:
    """持ち戻りの連絡。"""

    vehicle_id: str
    customer_id: str          # 降ろせなかった先
    product: Product
    volume_l: int             # 降ろせなかった量


@dataclass(frozen=True)
class ReturnOption:
    """持ち戻りへの手の1つ。"""

    rank: int                 # 1=近くの降ろし先 2=クッション 3=積場へ持ち戻り（買い戻し）
    kind: str                 # "redeliver" / "cushion" / "buyback"
    destination: str
    extra_min: int            # この手を取ると1日の終わりが延びる時間（引き継げば縮むこともある）
    arrive_min: int | None
    feasible: bool
    reason: str               # 成り立つ／成り立たない理由
    handoff: tuple[str, ...] = ()   # 他のローリーで持つ先
    delay_min: int = 0              # 後の予定がずれる時間


@dataclass(frozen=True)
class _Sim:
    """寄り道を入れて、その便の残りと後の便をなぞった結果。"""

    arrive_min: int           # 降ろし先に着く時刻
    day_end_min: int          # 帰庫
    day_drive_min: int        # 1日の運転時間
    delay_min: int            # 後の予定がずれる時間
    handoff: tuple[str, ...]  # 他のローリーで持つ先


@dataclass
class ReturnPlan:
    event: ReturnEvent
    customer_name: str
    vehicle_name: str
    options: list[ReturnOption]

    @property
    def chosen(self) -> ReturnOption:
        """推奨: 近くの降ろし先 → クッション → 買い戻し。同じ順位なら引き継ぎの少ないほう。"""
        return min((o for o in self.options if o.feasible),
                   key=lambda o: (o.rank, len(o.handoff), o.extra_min))

    def summary_ja(self) -> str:
        e, c = self.event, self.chosen
        head = (
            f"{self.customer_name}で{PRODUCT_JA[e.product]} {e.volume_l / 1000:.1f}kL が"
            f"降ろせません（持ち戻り、{self.vehicle_name}）。"
        )
        if c.kind == "buyback":
            return head + "近くにもクッションにも降ろせないため、積場へ持ち戻り、買い戻しとします。"
        where = "近くの" if c.kind == "redeliver" else "クッションの"
        tail = ""
        if c.delay_min:
            tail += f"この後の予定は{c.delay_min}分ずれます。"
        if c.handoff:
            tail += f"{'、'.join(c.handoff)}は他のローリーで持ってください。"
        return (
            head + f"先に降ろすことを優先し、{where}{c.destination}へ回します"
            f"（{minutes_to_hhmm(c.arrive_min)}着、寄り道 +{c.extra_min}分）。" + tail
        )


def format_return_ja(r: ReturnPlan) -> str:
    lines = [f"【持ち戻り】{r.summary_ja()}", f"   {RESPONSIBILITY_JA}", "", "■ 検討した手（上から優先）"]
    labels = {1: "近くの降ろし先", 2: "クッション", 3: "積場へ持ち戻り"}
    for o in sorted(r.options, key=lambda o: (o.rank, not o.feasible, len(o.handoff), o.extra_min)):
        mark = "◎" if o is r.chosen else ("○" if o.feasible else "×")
        at = f" {minutes_to_hhmm(o.arrive_min)}着" if o.arrive_min is not None and o.extra_min else ""
        extra = f" +{o.extra_min}分" if o.kind != "buyback" and o.extra_min else ""
        lines.append(f"   {mark} {labels[o.rank]}: {o.destination}{at}{extra}　{o.reason}")
    return "\n".join(lines)


def _room(c: Customer, product: Product, delivered_today: dict[Product, int]) -> int:
    """今日このあとの予定まで含めた、その油種の空き（L）。"""
    for t in c.tanks:
        if t.product is product:
            return max(0, t.topup_l - delivered_today.get(product, 0))
    return 0


def handle_return(scenario: Scenario, plan: Plan, event: ReturnEvent) -> ReturnPlan:
    """持ち戻りへの手を、優先順に並べて返す。**計算はすべてここ（モデルは使わない）。**"""
    by_id = {c.id: c for c in scenario.customers}
    failed = by_id.get(event.customer_id)
    if failed is None:
        raise ValueError(f"名簿に無い需要家: {event.customer_id}")
    vd: VehicleDay | None = next((v for v in plan.vehicle_days if v.vehicle_id == event.vehicle_id), None)
    if vd is None:
        raise ValueError(f"今日の計画に {event.vehicle_id} は無い")
    trip: Trip | None = next(
        (t for t in vd.trips if any(s.customer_id == event.customer_id for s in t.stops)), None)
    if trip is None:
        raise ValueError(f"{vd.vehicle_name} は今日 {failed.name} に行く予定が無い")
    if event.volume_l <= 0:
        raise ValueError("持ち戻りの量は正の値")
    if event.volume_l % COMPARTMENT_UNIT_L:
        raise ValueError(
            f"持ち戻りの量は槽の単位（{COMPARTMENT_UNIT_L}L の倍数）: {event.volume_l}L。"
            "その槽の全量が入らなければその槽は降ろさないので、半端な量は残らない")

    veh = next(v for v in scenario.vehicles if v.id == event.vehicle_id)
    driver = scenario.driver(veh.driver_id)
    idx = next(i for i, s in enumerate(trip.stops) if s.customer_id == event.customer_id)
    here = trip.stops[idx]
    later = list(trip.stops[idx + 1:])
    later_trips = [t for t in vd.trips if t.trip_index > trip.trip_index]
    depot = scenario.depot

    # 後ろから順に他のローリーへ渡す単位。後の便まるごとが先、次にこの便の後ろの先。
    units: list[tuple[str, object]] = (
        [("trip", t) for t in reversed(later_trips)] + [("stop", s) for s in reversed(later)]
    )

    # 今日の計画で各先に降ろす量（油種ごと）。空きの計算に使う。
    delivered: dict[str, dict[Product, int]] = {}
    for v in plan.vehicle_days:
        for t in v.trips:
            for s in t.stops:
                if s.customer_id != event.customer_id:
                    delivered[s.customer_id] = by_id[s.customer_id].demand_by_product()

    def tail_drive_as_planned() -> int:
        """朝の計画での、この便の尻尾（降ろせなかった先 → 後の先 → 油槽所）の運転時間。"""
        lat, lon = failed.lat, failed.lon
        total = 0
        for s in later:
            c = by_id[s.customer_id]
            total += leg_minutes(lat, lon, c.lat, c.lon)
            lat, lon = c.lat, c.lon
        return total + leg_minutes(lat, lon, depot.lat, depot.lon)

    planned_tail = tail_drive_as_planned()

    def simulate(c: Customer, service: int, forced: int) -> _Sim | None:
        """寄り道を降ろせなかった先のすぐ後に入れて、その日の残りをなぞる。

        `forced` は、拘束を収めるために後ろから外す単位の数。受入時間に間に合わない
        先は、外さなくても引き継ぎになる。着けない（閉まった後）ときは None。
        """
        dropped = {id(u[1]) for u in units[:forced]}
        handoff: list[str] = []

        arrive = here.depart_min + CONFIRM_MIN + leg_minutes(failed.lat, failed.lon, c.lat, c.lon)
        if arrive > c.window_close_min:
            return None
        t = max(arrive, c.window_open_min) + service
        lat, lon = c.lat, c.lon
        drive = leg_minutes(failed.lat, failed.lon, c.lat, c.lon)

        for s in later:
            if id(s) in dropped:
                handoff.append(s.customer_name)
                continue
            cust = by_id[s.customer_id]
            d = leg_minutes(lat, lon, cust.lat, cust.lon)
            if t + d > s.window_close_min:
                handoff.append(s.customer_name)      # 間に合わない。他のローリーで持つ
                continue
            drive += d
            t = max(t + d, s.window_open_min) + (s.depart_min - s.arrive_min)
            lat, lon = cust.lat, cust.lon
        drive += leg_minutes(lat, lon, depot.lat, depot.lon)
        day_end = t + leg_minutes(lat, lon, depot.lat, depot.lon)

        delay = day_end - trip.return_min
        day_drive = vd.drive_min - planned_tail + drive
        for t2 in later_trips:
            missed = [s for s in t2.stops if s.arrive_min + delay > s.window_close_min]
            if id(t2) in dropped or len(missed) == len(t2.stops):
                handoff.extend(s.customer_name for s in t2.stops)
                day_drive -= t2.drive_min
                continue
            handoff.extend(s.customer_name for s in missed)
            day_end = t2.return_min + delay
        return _Sim(arrive, day_end, day_drive, delay, tuple(handoff))

    def evaluate(c: Customer, rank: int, kind: str) -> ReturnOption:
        if not any(t.product is event.product for t in c.tanks):
            return ReturnOption(rank, kind, c.name, 0, None, False, f"{PRODUCT_JA[event.product]}のタンクが無い")
        room = _room(c, event.product, delivered.get(c.id, {}))
        if room < event.volume_l:
            return ReturnOption(rank, kind, c.name, 0, None, False,
                                f"空きが {room / 1000:.1f}kL しかない（今日の予定を含む）")
        if not c.open_today or not c.accepts_delivery:
            return ReturnOption(rank, kind, c.name, 0, None, False, "今日は受け入れていない")
        service = unload_minutes({event.product: event.volume_l}, c.bore_inch)

        first: _Sim | None = None
        for forced in range(len(units) + 1):
            sim = simulate(c, service, forced)
            if sim is None:
                return ReturnOption(
                    rank, kind, c.name, 0, None, False,
                    f"着く頃には閉まっている（受入 〜{minutes_to_hhmm(c.window_close_min)}）")
            first = first or sim
            if (sim.day_end_min - vd.duty_start_min <= driver.max_duty_min
                    and sim.day_drive_min <= driver.max_drive_min):
                extra = sim.day_end_min - vd.duty_end_min
                if sim.handoff:
                    why = (f"空き {room / 1000:.1f}kL。先に降ろすことを優先し、"
                           f"{len(sim.handoff)}軒は他のローリーで持つ")
                elif sim.delay_min:
                    why = f"空き {room / 1000:.1f}kL。後の予定が{sim.delay_min}分ずれる"
                else:
                    why = f"空き {room / 1000:.1f}kL、後の予定に影響なし"
                return ReturnOption(rank, kind, c.name, extra, sim.arrive_min, True, why,
                                    sim.handoff, sim.delay_min)
        over = first.day_end_min - vd.duty_start_min if first else 0
        return ReturnOption(rank, kind, c.name, 0, first.arrive_min if first else None, False,
                            f"後を全部渡しても拘束が {over // 60}時間{over % 60}分 になり上限を超える")

    options: list[ReturnOption] = []
    for c in scenario.customers:
        if c.id == event.customer_id:
            continue
        o = evaluate(c, 1, "redeliver")
        # 関係の無い先（その油種のタンクが無い）は並べない
        if o.reason.endswith("のタンクが無い"):
            continue
        options.append(o)
    for c in CUSHION_STATIONS:
        options.append(evaluate(c, 2, "cushion"))
    options.append(ReturnOption(
        3, "buyback", "積場（麻里布）", 0, None, True,
        f"{PRODUCT_JA[event.product]} {event.volume_l / 1000:.1f}kL を持ち帰り、その量を積まない（買い戻し）",
    ))
    return ReturnPlan(event=event, customer_name=failed.name, vehicle_name=vd.vehicle_name.split("（")[0],
                      options=options)
