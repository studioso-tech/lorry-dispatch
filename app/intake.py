"""当日発注の受け口。**Gemini には読み取りと分類だけをさせる。**

    FAX の画像   → Gemini が「紙に何と書いてあるか」を読む（先・油種・量・単位）
                 → 名簿との照合・油種の型・単位の換算はコード（core/matching.py）
                 → 1つに絞れないもの、読めないものがあれば、確定させずに確認を求める
                 → 全部そろえば発注（source="fax"、証拠＝画像のハッシュ）

    配車係のメモ → Gemini が「発注か、予告か、それ以外か」と中身を読む
                 → 発注システムの伝票番号があれば発注（source="system"）
                 → 電話の連絡などは**予告**。確定させず、入る余地の試算だけ返す

**判断と計算はコード。** 何リットルか、どの先か、どの車で何時に着くかを
モデルに決めさせると、散文と数字が食い違う事故が起きる。モデルの出力は
「読み取った文字列」として扱い、型に直すところで検める。

**発注は証拠の残る形でしか受けない**（実務）。メモに「電話で」とあれば、
モデルが「発注」と分類しても予告に落とす（コードの側で強制する）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import settings
from app.meter import CostMeter
from core.domain import Customer
from core.matching import match_customer, parse_product, to_liters
from core.replan import NOTICE_SOURCE, PRODUCT_JA, NewOrder

logger = logging.getLogger(__name__)

#: 読み取りの思考の深さ。紙に書いてあることを写すだけなので浅くてよい。
THINKING_LEVEL = "LOW"

_LINE = {
    "type": "object",
    "properties": {
        "customer_text": {"type": "string", "description": "届け先（SS名など）の記載そのまま"},
        "product_text": {"type": "string", "description": "油種の記載そのまま（例: レギュラー、灯油）"},
        "quantity": {"type": "number", "description": "量の数字。書かれていなければ 0"},
        "unit": {"type": "string", "description": "量の単位の記載そのまま（L、kL、キロ等）。無ければ空"},
        "note": {"type": "string", "description": "納入希望時刻などの添え書き"},
    },
    "required": ["customer_text", "product_text", "quantity", "unit"],
}

FAX_SCHEMA = {
    "type": "object",
    "properties": {
        "is_order_form": {"type": "boolean", "description": "燃料の発注書か"},
        "sender_text": {"type": "string", "description": "発注元の記載そのまま"},
        "date_text": {"type": "string", "description": "日付の記載そのまま"},
        "lines": {"type": "array", "items": _LINE},
        "illegible": {"type": "array", "items": {"type": "string"},
                      "description": "読めなかった箇所。推測で埋めずにここへ書く"},
    },
    "required": ["is_order_form", "lines", "illegible"],
}

NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["order", "notice", "other"],
                 "description": "order=発注システムやFAXの発注を指している、notice=電話などの連絡、other=それ以外"},
        "channel": {"type": "string", "enum": ["system", "fax", "phone", "unknown"],
                    "description": "連絡の経路"},
        "order_ref": {"type": "string", "description": "発注システムの伝票番号など。無ければ空"},
        "lines": {"type": "array", "items": _LINE},
    },
    "required": ["kind", "channel", "lines"],
}

FAX_PROMPT = (
    "これは燃料（ガソリン・軽油・灯油）の発注書のFAX画像です。"
    "書かれていることを**そのまま写してください**。推測で補わないこと。"
    "読めない箇所は illegible に書き、その項目は空か 0 にしてください。"
    "量の単位の換算はしないこと（書かれた数字と単位をそのまま返す）。"
)

NOTE_PROMPT = (
    "配車係が書いたメモです。燃料の発注に関する連絡かどうかを分類し、中身を写してください。"
    "発注システムの伝票番号やFAXの発注書に触れていれば kind=order、"
    "電話や口頭の連絡なら kind=notice、それ以外は other。"
    "量の単位の換算はしないこと。推測で補わないこと。\n\nメモ:\n"
)


@dataclass
class Intake:
    """受け口の結果。orders が空で issues があれば、確認が要る。"""

    orders: list[NewOrder] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    notice: bool = False               # 予告（試算しかしない）
    raw: dict[str, Any] = field(default_factory=dict)
    usd: float = 0.0

    @property
    def ready(self) -> bool:
        """確認なしで組み直しに進めるか。"""
        return bool(self.orders) and not self.issues


#: モデルを呼ぶ関数の型。テストでは偽物に差し替える。
#: (contents, schema) -> (JSON文字列, 入力トークン, 出力トークン, 思考トークン)
Caller = Callable[[list[Any], dict], tuple[str, int, int, int]]


def gemini_caller(contents: list[Any], schema: dict) -> tuple[str, int, int, int]:
    """Gemini を1回呼ぶ。**global で呼ぶ**（us-central1 では404。2026-09-03 確認）。"""
    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=settings.project_id, location=settings.genai_location)
    r = client.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        ),
    )
    meta = r.usage_metadata
    return (
        r.text or "{}",
        getattr(meta, "prompt_token_count", 0) or 0,
        getattr(meta, "candidates_token_count", 0) or 0,
        getattr(meta, "thoughts_token_count", 0) or 0,
    )


def _lines_to_orders(
    lines: list[dict], customers: tuple[Customer, ...], *, source: str, evidence_ref: str,
    fallback_customer_text: str = "",
) -> tuple[list[NewOrder], list[str]]:
    """読み取った行を発注に直す。**1つでも決められない項目があれば確認を求める。**"""
    orders: list[NewOrder] = []
    issues: list[str] = []
    for i, ln in enumerate(lines, 1):
        who = (ln.get("customer_text") or "").strip() or fallback_customer_text
        m = match_customer(who, customers)
        product = parse_product(ln.get("product_text") or "")
        liters = to_liters(ln.get("quantity") or 0, ln.get("unit") or "")
        where = f"{i}行目（{who or '届け先不明'}）"
        if m.customer is None:
            issues.append(f"{where}: {m.reason}")
            continue
        if product is None:
            issues.append(f"{where}: 油種「{ln.get('product_text', '')}」が分からない")
            continue
        if liters is None:
            issues.append(f"{where}: 量「{ln.get('quantity')}{ln.get('unit', '')}」が読めない（単位が無い等）")
            continue
        if not any(t.product is product for t in m.customer.tanks):
            issues.append(f"{where}: {m.customer.name} に {PRODUCT_JA[product]} のタンクは無い")
            continue
        orders.append(NewOrder(m.customer.id, product, liters, source, evidence_ref))
    return orders, issues


def read_fax(
    image: bytes, mime: str, customers: tuple[Customer, ...], *,
    caller: Caller = gemini_caller, meter: CostMeter | None = None,
) -> Intake:
    """FAX の画像から発注を読み取る。"""
    from google.genai import types

    evidence = "FAX:" + hashlib.sha256(image).hexdigest()[:12]
    text, tin, tout, tthink = caller([types.Part.from_bytes(data=image, mime_type=mime), FAX_PROMPT], FAX_SCHEMA)
    usd = meter.record(settings.gemini_model, tin, tout, tthink) if meter else 0.0
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Intake(issues=["FAXを読み取れませんでした（応答が壊れていた）。手で確認してください"], usd=usd)

    if not data.get("is_order_form"):
        return Intake(issues=["発注書ではないようです。手で確認してください"], raw=data, usd=usd)
    orders, issues = _lines_to_orders(
        data.get("lines") or [], customers, source="fax", evidence_ref=evidence,
        fallback_customer_text=data.get("sender_text") or "",
    )
    issues += [f"読めない箇所: {x}" for x in data.get("illegible") or []]
    if not orders and not issues:
        issues.append("発注の行が見つかりませんでした")
    return Intake(orders=orders, issues=issues, raw=data, usd=usd)


def read_note(
    note: str, customers: tuple[Customer, ...], *,
    caller: Caller = gemini_caller, meter: CostMeter | None = None,
) -> Intake:
    """配車係のメモを読む。**電話の連絡は予告に落とす**（コードで強制する）。"""
    text, tin, tout, tthink = caller([NOTE_PROMPT + note], NOTE_SCHEMA)
    usd = meter.record(settings.gemini_model, tin, tout, tthink) if meter else 0.0
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Intake(issues=["メモを読み取れませんでした"], usd=usd)

    kind = data.get("kind")
    channel = data.get("channel")
    ref = (data.get("order_ref") or "").strip()
    # **経路の判定はモデルに任せきらない。** 「電話」と書いてあれば、モデルが
    # 発注と分類しても予告にする。伝票番号が無い「発注」も予告にする。
    phone_in_text = bool(re.search(r"電話|口頭|tel|ＴＥＬ|でんわ", note, re.IGNORECASE))
    is_order = kind == "order" and channel in ("system", "fax") and ref and not phone_in_text
    if kind == "other":
        return Intake(issues=["発注に関する連絡ではないようです"], raw=data, usd=usd)

    if is_order:
        orders, issues = _lines_to_orders(
            data.get("lines") or [], customers, source=channel, evidence_ref=ref)
        return Intake(orders=orders, issues=issues, raw=data, usd=usd)

    orders, issues = _lines_to_orders(
        data.get("lines") or [], customers, source=NOTICE_SOURCE,
        evidence_ref="予告（" + note.strip()[:40] + "）")
    return Intake(orders=orders, issues=issues, notice=True, raw=data, usd=usd)
