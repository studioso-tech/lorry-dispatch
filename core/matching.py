"""FAXやメモに書かれた名前・油種・量を、名簿と型に照らす。**モデルは使わない。**

Gemini には「紙に何と書いてあるか」だけを読ませ、それが名簿のどの需要家か、
どの油種か、何リットルかはここで決める。

なぜコードでやるか:
    需要家の取り違えは配車の事故に直結する。「宇品港」と「宇品」、「祇園」と
    「祇園新橋」のような似た名前をモデルが気を利かせて寄せると、違うSSに
    油を運ぶことになる。**候補が1つに絞れなければ、寄せずに確認を求める。**
    量の単位（kL→L）の換算も計算なので、モデルに任せない
    （このプロジェクトの原則「計算はLLMに担わせない」）。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from core.domain import Customer, Product

#: 名前の照合で無視する語。表記ゆれの主な出どころ。
_NOISE = (
    "サービスステーション", "ガソリンスタンド", "給油所", "株式会社", "有限会社",
    "(株)", "(有)", "㈱", "㈲", "(24h)", "(日中)", "ss", "gs", "様", "御中",
)

#: 油種の書き方 → 型。実務の書き方を寄せてある。**灯油と軽油を取り違えない。**
_PRODUCT_WORDS: tuple[tuple[str, Product], ...] = (
    ("ハイオク", Product.PREMIUM),
    ("プレミアム", Product.PREMIUM),
    ("レギュラー", Product.REGULAR),
    ("ガソリン", Product.REGULAR),
    ("軽油", Product.GASOIL),
    ("ディーゼル", Product.GASOIL),
    ("灯油", Product.KEROSENE),
)


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"\s+", "", t)
    for w in _NOISE:
        t = t.replace(unicodedata.normalize("NFKC", w).lower(), "")
    return t


@dataclass(frozen=True)
class CustomerMatch:
    """名前の照合結果。customer が None なら確認が要る。"""

    customer: Customer | None
    candidates: tuple[Customer, ...]
    reason: str


def match_customer(text: str, customers: tuple[Customer, ...]) -> CustomerMatch:
    """書かれた名前を名簿の需要家に照らす。**1つに絞れたときだけ決める。**"""
    q = _normalize(text)
    if not q:
        return CustomerMatch(None, (), "名前が読めない")
    exact = [c for c in customers if _normalize(c.name) == q or c.id == q]
    if len(exact) == 1:
        return CustomerMatch(exact[0], tuple(exact), "一致")
    partial = [c for c in customers if q in _normalize(c.name) or _normalize(c.name) in q]
    if len(partial) == 1:
        return CustomerMatch(partial[0], tuple(partial), "部分一致")
    if not partial:
        return CustomerMatch(None, (), f"名簿に「{text}」に当たる先が無い")
    names = "、".join(c.name for c in partial)
    return CustomerMatch(None, tuple(partial), f"「{text}」に当たる先が複数ある（{names}）")


def parse_product(text: str) -> Product | None:
    """油種の書き方を型に直す。分からなければ None（寄せない）。"""
    t = unicodedata.normalize("NFKC", text)
    hits = {p for word, p in _PRODUCT_WORDS if word in t}
    # 「ハイオクガソリン」は ハイオク と ガソリン の両方に当たる。ハイオクを優先する。
    if Product.PREMIUM in hits:
        return Product.PREMIUM
    return hits.pop() if len(hits) == 1 else None


def to_liters(quantity: float, unit: str) -> int | None:
    """量と単位からリットルを出す。単位が分からなければ None（推測しない）。"""
    u = unicodedata.normalize("NFKC", unit).strip().lower()
    if quantity is None or quantity <= 0:
        return None
    if u in ("l", "ℓ", "リットル", "りっとる"):
        return int(round(quantity))
    if u in ("kl", "キロ", "キロリットル", "きろ"):
        return int(round(quantity * 1000))
    return None
