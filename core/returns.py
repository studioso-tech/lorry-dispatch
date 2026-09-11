"""持ち戻り: 計画配送の先で、持って行った量を降ろせない。

**計画配送の先の在庫は元売が管理する。**（実務）荷卸しの前にSS等の人と乗務員が
計画どおり降ろせるかを確かめるが、在庫の管理責任は元売にある。見込みが外れると、
着いてからタンクに入らないことがまれに起きる。これを**持ち戻り**という。

対応には順番がある（実務）:

    1. 乗務員の連絡を車庫経由で受け、**近くで降ろせる先を探して配送する**（最優先）
    2. 見つからなければ、**大竹・岩国インター近くの大型SSがクッション**として受け入れる
    3. それでも無ければ**積場へ持ち戻り**、その油種・数量を積まないことで**買い戻し**

**費用は元売の管理責任**で、運送会社やSSには請求しない（空の荷を運んだのと同じ）。

候補を探すときに見るもの:

    - その油種のタンクに**丸ごと入る空き**があるか。今日すでに降ろした先や、
      このあと降ろす予定の先は、そのぶん空きを減らして数える
    - 着く時刻に受け入れているか（営業時間・受入枠）
    - 寄り道で後ろにずれる予定が、それぞれの受入枠に間に合うか
    - 寄り道で延びる拘束・運転が、1日の上限に収まるか

寄り道の位置は2通り試して短いほうを取る:

    - 降ろせなかった先の**すぐ後**に寄る（近くに受け入れ先があるとき）
    - 便の**帰り道**（最後の先から油槽所へ戻る途中）に寄る。クッションの大型SSは
      インターの近く＝油槽所の近くにあるので、ふつうはこちら。最初はすぐ後しか
      見ておらず、広島市内から大竹へ行って温品へ戻る +137分 の大回りを出していた

**近似**:

    - 分けて降ろす（2軒に半分ずつ）は見ない
    - 寄り道のあとの予定は、寄り道の時間だけそのままずれるとする
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from core.domain import (
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
    extra_min: int            # 寄り道で延びる時間
    arrive_min: int | None
    feasible: bool
    reason: str               # 成り立つ／成り立たない理由


@dataclass
class ReturnPlan:
    event: ReturnEvent
    customer_name: str
    vehicle_name: str
    options: list[ReturnOption]

    @property
    def chosen(self) -> ReturnOption:
        """推奨: 成り立つもののうち、近くの降ろし先 → クッション → 買い戻し の順。"""
        return min((o for o in self.options if o.feasible), key=lambda o: (o.rank, o.extra_min))

    def summary_ja(self) -> str:
        e, c = self.event, self.chosen
        head = (
            f"{self.customer_name}で{PRODUCT_JA[e.product]} {e.volume_l / 1000:.1f}kL が"
            f"降ろせません（持ち戻り、{self.vehicle_name}）。"
        )
        if c.kind == "buyback":
            return head + "近くにもクッションにも降ろせないため、積場へ持ち戻り、買い戻しとします。"
        where = "近くの" if c.kind == "redeliver" else "クッションの"
        return (
            head + f"{where}{c.destination}に {minutes_to_hhmm(c.arrive_min)} に降ろします"
            f"（寄り道 +{c.extra_min}分）。"
        )


def format_return_ja(r: ReturnPlan) -> str:
    lines = [f"【持ち戻り】{r.summary_ja()}", f"   {RESPONSIBILITY_JA}", "", "■ 検討した手（上から優先）"]
    labels = {1: "近くの降ろし先", 2: "クッション", 3: "積場へ持ち戻り"}
    for o in sorted(r.options, key=lambda o: (o.rank, not o.feasible, o.extra_min)):
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

    veh = next(v for v in scenario.vehicles if v.id == event.vehicle_id)
    driver = scenario.driver(veh.driver_id)
    idx = next(i for i, s in enumerate(trip.stops) if s.customer_id == event.customer_id)
    here = trip.stops[idx]
    later = list(trip.stops[idx + 1:])
    later_trips = [t for t in vd.trips if t.trip_index > trip.trip_index]
    depot = scenario.depot
    nxt = by_id[later[0].customer_id] if later else None

    def pos(c: Customer | None) -> tuple[float, float]:
        return (c.lat, c.lon) if c is not None else (depot.lat, depot.lon)

    base_leg = leg_minutes(*pos(failed), *pos(nxt))

    # 今日の計画で各先に降ろす量（油種ごと）。空きの計算に使う。
    delivered: dict[str, dict[Product, int]] = {}
    for v in plan.vehicle_days:
        for t in v.trips:
            for s in t.stops:
                if s.customer_id != event.customer_id:
                    delivered[s.customer_id] = by_id[s.customer_id].demand_by_product()

    def evaluate(c: Customer, rank: int, kind: str) -> ReturnOption:
        room = _room(c, event.product, delivered.get(c.id, {}))
        if not any(t.product is event.product for t in c.tanks):
            return ReturnOption(rank, kind, c.name, 0, None, False, f"{PRODUCT_JA[event.product]}のタンクが無い")
        if room < event.volume_l:
            return ReturnOption(rank, kind, c.name, 0, None, False,
                                f"空きが {room / 1000:.1f}kL しかない（今日の予定を含む）")
        if not c.open_today or not c.accepts_delivery:
            return ReturnOption(rank, kind, c.name, 0, None, False, "今日は受け入れていない")
        service = unload_minutes({event.product: event.volume_l}, c.bore_inch)

        # (a) すぐ後に寄る
        go_a = leg_minutes(*pos(failed), c.lat, c.lon)
        arrive_a = here.arrive_min + 10 + go_a          # 確認と連絡に10分（仮置）
        back_a = leg_minutes(c.lat, c.lon, *pos(nxt))
        extra_a = 10 + go_a + service + back_a - base_leg
        drive_a = go_a + back_a - base_leg
        # (b) 帰り道に寄る（最後の先 → c → 油槽所）
        last = by_id[trip.stops[-1].customer_id]
        last_depart = trip.stops[-1].depart_min
        go_b = leg_minutes(last.lat, last.lon, c.lat, c.lon)
        arrive_b = last_depart + go_b
        back_b = leg_minutes(c.lat, c.lon, depot.lat, depot.lon)
        base_b = leg_minutes(last.lat, last.lon, depot.lat, depot.lon)
        extra_b = go_b + service + back_b - base_b
        drive_b = go_b + back_b - base_b

        def in_window(t: int) -> bool:
            return c.window_open_min <= t <= c.window_close_min

        tries = []
        if in_window(arrive_a):
            tries.append((extra_a, arrive_a, drive_a, True))     # すぐ後: 後ろの先もずれる
        if in_window(arrive_b):
            tries.append((extra_b, arrive_b, drive_b, False))    # 帰り道: 同じ便の先はずれない
        if not tries:
            return ReturnOption(rank, kind, c.name, 0, arrive_a, False,
                                f"{minutes_to_hhmm(arrive_a)} は受入時間外"
                                f"（{minutes_to_hhmm(c.window_open_min)}〜{minutes_to_hhmm(c.window_close_min)}）")
        extra, arrive, extra_drive, shifts_this_trip = min(tries)
        late = [
            s.customer_name for s in later
            if shifts_this_trip and s.arrive_min + extra > s.window_close_min
        ] + [
            s.customer_name for t in later_trips for s in t.stops if s.arrive_min + extra > s.window_close_min
        ]
        if late:
            return ReturnOption(rank, kind, c.name, extra, arrive, False,
                                f"後の予定が受入時間に間に合わない（{'、'.join(late)}）")
        if vd.duty_min + extra > driver.max_duty_min:
            return ReturnOption(rank, kind, c.name, extra, arrive, False,
                                f"拘束が {(vd.duty_min + extra) // 60}時間{(vd.duty_min + extra) % 60}分 になり上限を超える")
        if vd.drive_min + extra_drive > driver.max_drive_min:
            return ReturnOption(rank, kind, c.name, extra, arrive, False, "運転時間が上限を超える")
        return ReturnOption(rank, kind, c.name, extra, arrive, True,
                            f"空き {room / 1000:.1f}kL、後の予定に影響なし")

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
