"""名前・油種・量の照合。**取り違えるくらいなら確認を求める。**"""

from __future__ import annotations

from core.domain import Product
from core.generator import generate
from core.matching import match_customer, parse_product, to_liters

CUSTOMERS = generate(seed=1).customers


def test_spelling_variants_resolve_to_one_customer():
    for text in ("宇品港SS", "宇品港ＳＳ", "宇品港 サービスステーション", "宇品港SS（24h）御中"):
        m = match_customer(text, CUSTOMERS)
        assert m.customer is not None and m.customer.name.startswith("宇品港"), text


def test_ambiguous_names_are_not_guessed():
    """候補が複数なら決めない。似た名前をモデルやコードが寄せると違うSSに運ぶ。"""
    m = match_customer("町SS", CUSTOMERS)        # 観音新町と錦町の両方に当たる
    assert m.customer is None and len(m.candidates) > 1
    assert "複数" in m.reason


def test_unknown_name_is_reported():
    m = match_customer("存在しない港SS", CUSTOMERS)
    assert m.customer is None and "無い" in m.reason


def test_products_do_not_confuse_kerosene_and_gasoil():
    """灯油と軽油を取り違えない。"""
    assert parse_product("灯油") is Product.KEROSENE
    assert parse_product("軽油") is Product.GASOIL
    assert parse_product("ハイオクガソリン") is Product.PREMIUM
    assert parse_product("レギュラーガソリン") is Product.REGULAR
    assert parse_product("重油") is None                # 扱っていない油種は寄せない


def test_units_are_converted_in_code_not_by_the_model():
    assert to_liters(8, "kL") == 8_000
    assert to_liters(8, "ＫＬ") == 8_000
    assert to_liters(4000, "L") == 4_000
    assert to_liters(8, "キロ") == 8_000
    assert to_liters(8, "") is None                   # 単位が無ければ推測しない
    assert to_liters(0, "kL") is None
