"""当日発注の引き金になる事象: **見込みより売れた。**

当日発注が来るのは、そのSSが**計画より売れたから**（実務）。引き金は3つ:

    連休      高速道路のSS・行楽地のSSでガソリンが急に出る
    価格改定  値上げ前の駆け込み（値下げ前は逆に在庫を下げる＝受入見送り）
    天候      寒波で灯油が出る

生成器は元売の**見込みどおりに売れた世界**しか作らない。計画配送が3日周期で
タンクを満たすので、そのままでは当日発注を入れる空きがタンクに残らない
（2026-09-12 のデモの下見で、試した発注がすべて「空きが1.2〜5.5kLしかない」で
頭打ちになった）。売れていれば空きがあるはずで、筋が通らなかった。

**事象なので外から与える**（`declined_customers` と同じ扱い）。生成器が勝手に
作ると、事象を入れたから計画が変わったのか、たまたまそういう日だったのかが
切り分けられない。

量の単位は「**見込みの何日ぶん多く売れたか**」。「3倍売れた」だと前回の配送から
何日目かで量が変わり、配車係にも審査員にも読めない。「連休で2日ぶん多く出た」なら
そのSSの見込みの1日消費×2で、どの日に当てても同じ量になる。

使い道は2つ:

    朝の計画より前に分かっていた  → generate(sold_more=...)
    日中に分かった（FAXの但し書き等）→ replan(..., sold_more=...)

デモは後者。朝の計画は見込みで組まれていて、09:12 のFAXで初めて分かる。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from core.domain import Customer, Product, Tank

PRODUCT_JA = {
    Product.KEROSENE: "灯油",
    Product.GASOIL: "軽油",
    Product.REGULAR: "レギュラー",
    Product.PREMIUM: "ハイオク",
}

#: これより大きい日数は受け付けない。1週間ぶん多く売れたなら、それは事象ではなく
#: 見込み（需要予測）そのものの誤りで、半期の体制設計で扱う話になる。
MAX_EXTRA_DAYS = 7.0


@dataclass(frozen=True)
class SoldMore:
    """見込みより売れた。

    Attributes:
        customer_id: 需要家。
        extra_days: 見込みの何日ぶん多く売れたか（0より大きく MAX_EXTRA_DAYS 以下）。
        product: 油種。None なら全油種（連休で全体に出た等）。寒波の灯油のように
            油種が決まっているときは指定する。
    """

    customer_id: str
    extra_days: float
    product: Product | None = None

    def label_ja(self) -> str:
        what = PRODUCT_JA[self.product] if self.product else "全油種"
        return f"{self.customer_id} {what} 見込みより{self.extra_days:g}日ぶん多く売れた"


def parse_sold_more(text: str) -> SoldMore:
    """CLI の書き方を読む: `c02:2`（全油種）または `c02:kerosene:1.5`。"""
    parts = text.split(":")
    try:
        if len(parts) == 2:
            return SoldMore(parts[0], float(parts[1]))
        if len(parts) == 3:
            return SoldMore(parts[0], float(parts[2]), Product(parts[1]))
    except ValueError as exc:
        raise ValueError(f"売れた事象の書き方が違う: {text!r}（例 c02:2 / c02:kerosene:1.5）") from exc
    raise ValueError(f"売れた事象の書き方が違う: {text!r}（例 c02:2 / c02:kerosene:1.5）")


def validate(customers: tuple[Customer, ...] | list[Customer], events: tuple[SoldMore, ...]) -> None:
    """綴り違いや、タンクの無い油種を黙って通さない。

    黙って通すと、事象が効かないまま「事象を入れた計画」として読まれる。
    """
    by_id = {c.id: c for c in customers}
    for e in events:
        c = by_id.get(e.customer_id)
        if c is None:
            raise ValueError(f"名簿に無い需要家: {e.customer_id}")
        if not 0 < e.extra_days <= MAX_EXTRA_DAYS:
            raise ValueError(
                f"{c.name}: 多く売れた日数は 0 より大きく {MAX_EXTRA_DAYS:g} 日以下（{e.extra_days:g}）。"
                "それ以上は事象ではなく見込みの誤りで、半期の体制設計で扱う"
            )
        if e.product is not None and not any(t.product is e.product for t in c.tanks):
            raise ValueError(f"{c.name} に {PRODUCT_JA[e.product]} のタンクは無い")


def sell_more(tanks: tuple[Tank, ...], events: list[SoldMore] | tuple[SoldMore, ...]) -> tuple[Tank, ...]:
    """タンクの在庫を、多く売れたぶん下げる。在庫は0より下がらない。"""
    out = []
    for t in tanks:
        sold = sum(
            round(t.daily_use_l * e.extra_days)
            for e in events
            if e.product is None or e.product is t.product
        )
        out.append(replace(t, level_l=max(0, t.level_l - sold)) if sold else t)
    return tuple(out)


def apply_sold_more(customers: tuple[Customer, ...] | list[Customer],
                    events: tuple[SoldMore, ...]) -> tuple[Customer, ...]:
    """需要家の名簿に事象を当てる。事象の無い先はそのまま返す。"""
    validate(customers, events)
    by_customer: dict[str, list[SoldMore]] = {}
    for e in events:
        by_customer.setdefault(e.customer_id, []).append(e)
    return tuple(
        replace(c, tanks=sell_more(c.tanks, by_customer[c.id])) if c.id in by_customer else c
        for c in customers
    )
