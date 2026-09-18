"""当日発注の受け口（FAX・メモ）と、逃げ道の監視。

Gemini は呼ばない。偽の応答を返す関数に差し替えて、**モデルが何を言っても
コードの側で守るべきこと**を確かめる:

- 名前が1つに絞れない・量が読めない・油種が分からない → 確定させずに確認を求める
- メモに「電話」とあれば、モデルが「発注」と分類しても予告に落とす
- 予告は試算しかしない。車を動かさず、逃げ道も記録しない
- 単位の換算はコード（モデルに「8」「kL」と写させ、8000L はコードが出す）
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from app import escapes, intake
from app.main import app
from core.generator import generate
from core.replan import Escape
from core.solver import SolveMode, solve
from core.travel import build_matrix

client = TestClient(app)
CUSTOMERS = generate(seed=20261016, date="2026-10-16").customers


def fake(payload: dict):
    """いつも同じ JSON を返す偽のモデル。"""
    def caller(contents, schema):
        return json.dumps(payload, ensure_ascii=False), 100, 50, 20
    return caller


def _free_customer():
    s = generate(seed=20261016, date="2026-10-16", season="winter")
    base = solve(s, matrix=build_matrix(s.depot, s.customers), mode=SolveMode.REPRODUCIBLE)
    served = {st.customer_id for t in base.trips for st in t.stops}
    # 4kL の発注が丸ごと入る先を選ぶ（タンクの空きが小さい先だと、正しく
    # 「SSに確認」で止まってしまい、組み直しの確認にならない）
    for c in s.customers:
        if c.id in served or not c.open_today:
            continue
        t = max(c.tanks, key=lambda t: t.topup_l)
        if t.topup_l >= 4_000:
            return c, t
    raise AssertionError("4kL入る先が無い")


def _line(customer, product_text, qty, unit):
    return {"customer_text": customer, "product_text": product_text, "quantity": qty, "unit": unit, "note": ""}


# ---------------------------------------------------------------------------
# FAX
# ---------------------------------------------------------------------------


def test_clean_fax_becomes_an_order_with_evidence():
    c, t = _free_customer()
    word = {"regular": "レギュラー", "premium": "ハイオク", "gasoil": "軽油", "kerosene": "灯油"}[t.product.value]
    got = intake.read_fax(b"fake-image-bytes", "image/png", CUSTOMERS, caller=fake({
        "is_order_form": True, "sender_text": c.name, "date_text": "10/16",
        "lines": [_line(c.name, word, 4, "kL")], "illegible": [],
    }))
    assert got.ready
    o = got.orders[0]
    assert (o.customer_id, o.product, o.volume_l, o.source) == (c.id, t.product, 4_000, "fax")
    assert o.evidence_ref.startswith("FAX:")      # 画像のハッシュ＝証拠


def test_fax_with_ambiguous_name_asks_for_confirmation():
    """名前が1つに絞れなければ、モデルがどう読んでも確定させない。"""
    got = intake.read_fax(b"x" * 20, "image/png", CUSTOMERS, caller=fake({
        "is_order_form": True, "lines": [_line("町SS", "レギュラー", 4, "kL")], "illegible": [],
    }))
    assert not got.ready and "複数" in got.issues[0]


def test_fax_without_unit_is_not_guessed():
    got = intake.read_fax(b"x" * 20, "image/png", CUSTOMERS, caller=fake({
        "is_order_form": True, "lines": [_line("宇品港SS", "レギュラー", 8, "")], "illegible": [],
    }))
    assert not got.ready and "量" in got.issues[0]


def test_illegible_parts_block_the_order():
    """読めない箇所があれば、読めた行があっても確定させない。"""
    got = intake.read_fax(b"x" * 20, "image/png", CUSTOMERS, caller=fake({
        "is_order_form": True, "lines": [_line("宇品港SS", "レギュラー", 8, "kL")],
        "illegible": ["2行目の量"],
    }))
    assert got.orders and not got.ready


def test_not_an_order_form():
    got = intake.read_fax(b"x" * 20, "image/png", CUSTOMERS, caller=fake({
        "is_order_form": False, "lines": [], "illegible": [],
    }))
    assert not got.ready and "発注書ではない" in got.issues[0]


# ---------------------------------------------------------------------------
# メモ
# ---------------------------------------------------------------------------


def test_phone_note_is_a_notice_even_if_the_model_says_order():
    """「電話で」とあれば予告。**経路の判定はモデルに任せきらない。**"""
    got = intake.read_note("宇品港から電話でレギュラー8kL", CUSTOMERS, caller=fake({
        "kind": "order", "channel": "system", "order_ref": "ORD-9",
        "lines": [_line("宇品港", "レギュラー", 8, "kL")],
    }))
    assert got.notice
    assert all(o.source == "notice" for o in got.orders)


def test_system_order_note_becomes_an_order():
    got = intake.read_note("発注システム ORD-1234 で宇品港SSからレギュラー8kL", CUSTOMERS, caller=fake({
        "kind": "order", "channel": "system", "order_ref": "ORD-1234",
        "lines": [_line("宇品港SS", "レギュラー", 8, "kL")],
    }))
    assert not got.notice and got.ready
    assert got.orders[0].source == "system" and got.orders[0].evidence_ref == "ORD-1234"


def test_order_without_reference_falls_back_to_notice():
    """伝票番号の無い「発注」は予告。証拠の所在が無い。"""
    got = intake.read_note("宇品港SSのレギュラー8kL追加", CUSTOMERS, caller=fake({
        "kind": "order", "channel": "system", "order_ref": "",
        "lines": [_line("宇品港SS", "レギュラー", 8, "kL")],
    }))
    assert got.notice


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_escapes():
    escapes.reset_for_tests()
    yield
    escapes.reset_for_tests()


# 朝の計画を再現モードで2回組む（_free_customer で1回、API の中で1回）。
# 再現モードは時間でなく探索量で切るので、既定の180秒には収まらない。
@pytest.mark.timeout(420)
def test_api_phone_note_returns_a_dry_run_and_records_nothing(monkeypatch):
    c, t = _free_customer()
    word = {"regular": "レギュラー", "premium": "ハイオク", "gasoil": "軽油", "kerosene": "灯油"}[t.product.value]
    monkeypatch.setattr(intake, "gemini_caller", fake({
        "kind": "notice", "channel": "phone", "order_ref": "",
        "lines": [_line(c.name, word, 4, "kL")],
    }))
    r = client.post("/api/intake/note", json={
        "seed": 20261016, "date": "2026-10-16", "now": "09:00",
        "note": f"{c.name}から電話で{word}4kL欲しいとのこと",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "dry_run" and body["notice"]
    assert body["replan"]["dry_run"] is True
    assert "試算" in body["summary_ja"]
    assert escapes.summary("2026-10-16")["escape_count"] == 0


# 朝の計画を再現モードで2回組む（_free_customer で1回、API の中で1回）。
# 再現モードは時間でなく探索量で切るので、既定の180秒には収まらない。
@pytest.mark.timeout(420)
def test_api_fax_replans_when_everything_is_readable(monkeypatch):
    c, t = _free_customer()
    word = {"regular": "レギュラー", "premium": "ハイオク", "gasoil": "軽油", "kerosene": "灯油"}[t.product.value]
    monkeypatch.setattr(intake, "gemini_caller", fake({
        "is_order_form": True, "sender_text": c.name, "lines": [_line(c.name, word, 4, "kL")],
        "illegible": [],
    }))
    r = client.post("/api/intake/fax", json={
        "seed": 20261016, "date": "2026-10-16", "now": "09:00",
        "image_base64": base64.b64encode(b"fake fax image bytes").decode(),
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "replanned"
    assert any("FAX:" in ch for ch in body["replan"]["changes"])


def test_api_fax_with_issues_does_not_replan(monkeypatch):
    monkeypatch.setattr(intake, "gemini_caller", fake({
        "is_order_form": True, "lines": [_line("町SS", "レギュラー", 4, "kL")], "illegible": [],
    }))
    r = client.post("/api/intake/fax", json={
        "now": "09:00", "image_base64": base64.b64encode(b"fake fax image bytes").decode(),
    })
    body = r.json()
    assert body["action"] == "confirm" and "replan" not in body


# ---------------------------------------------------------------------------
# 恒常化の監視
# ---------------------------------------------------------------------------


def test_occasional_escapes_are_temporary_and_frequent_ones_are_chronic():
    """一時的なら可、恒常的なら構造の欠陥（実務）。日をまたいで数えて初めて見える。"""
    e = Escape("defer_planned", "c01", "広島中央SS（24h）", 10_000)
    escapes.record("2026-10-01", [e])
    s = escapes.summary("2026-10-15")
    assert not s["chronic"] and s["escape_days"] == ["2026-10-01"]

    for day in ("2026-10-03", "2026-10-07", "2026-10-12"):
        escapes.record(day, [e])
    s = escapes.summary("2026-10-15")
    assert s["chronic"]
    assert "恒常化の恐れ" in s["message_ja"]
    assert s["repeated_customers"][0]["count"] == 4
