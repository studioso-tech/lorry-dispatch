"""API の受け入れテスト。

画面はまだ無いので、審査員が触る面に一番近いのがここ。
「動く URL を出す」という提出要件は、この層が壊れていないことに直結する。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import ledger
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean_ledger():
    """テストごとに台帳を空にする。上限に引っかかって落ちるのを防ぐ。"""
    ledger.reset_for_tests()
    yield
    ledger.reset_for_tests()


def test_healthz():
    """生存確認。台帳のバックエンドも返る。"""
    r = client.get("/api/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ledger_backend"] in ("memory", "firestore")


def test_plan_returns_routes_and_readable_text():
    """計画が返り、機械が読む形と人が読む形の両方が入っている。"""
    r = client.post("/api/plan", json={"seed": 20261015, "season": "winter", "mode": "fast"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("OPTIMAL", "FEASIBLE")
    assert body["vehicle_days"]
    assert body["vehicle_days"][0]["trips"]
    assert "配車" in body["text_ja"]
    assert body["plan_id"]

    # 時刻は分と HH:MM の両方。片方だけだとどちらかで変換ミスが起きる。
    stop = body["vehicle_days"][0]["trips"][0]["stops"][0]
    hh, mm = stop["arrive"].split(":")
    assert int(hh) * 60 + int(mm) == stop["arrive_min"]


def test_plan_reports_whether_it_is_reproducible():
    """再現できる計画かどうかを、計画自身が申告する。

    **API の既定は再現モード**（審査員が手元で計画を再現する入口なので）。
    fast を指定したときは、両段が最適性を証明できた場合だけ再現できると
    申告する。**言えないことを言わない**ために、根拠も一緒に出す。
    """
    fast = client.post(
        "/api/plan", json={"seed": 4242, "season": "winter", "mode": "fast"}
    ).json()
    assert isinstance(fast["diagnostics"]["reproducible"], bool)
    assert fast["diagnostics"]["solve_mode"] == "fast"
    assert fast["diagnostics"]["reproducible_basis"] in ("proven_optimal", None)
    assert fast["diagnostics"]["reproducible"] == (
        fast["diagnostics"]["reproducible_basis"] == "proven_optimal"
    )


def test_plan_defaults_to_reproducible_mode():
    """mode を指定しなければ再現モードで解く。"""
    from app.main import PlanRequest
    from core.solver import SolveMode

    assert PlanRequest().mode is SolveMode.REPRODUCIBLE


def test_unknown_season_is_rejected():
    """季節の綴り違いは 422（pydantic のパターン）で弾く。黙って通さない。"""
    r = client.post("/api/plan", json={"season": "Winter"})
    assert r.status_code == 422


def test_absent_drivers_change_the_plan():
    """欠勤を入れると計画が変わり、その乗務員は出てこない（デモ4の骨格）。"""
    base = client.post("/api/plan", json={"seed": 99, "mode": "fast"}).json()
    short = client.post(
        "/api/plan", json={"seed": 99, "absent_drivers": ["d1"], "mode": "fast"}
    ).json()
    assert short["status"] in ("OPTIMAL", "FEASIBLE")
    assert "d1" not in [vd["driver_id"] for vd in short["vehicle_days"]]
    assert short["scenario"]["usable_vehicles"] == base["scenario"]["usable_vehicles"] - 1


def test_scenario_view_exposes_tank_state():
    """需要側だけを見る面。W4 の再計画トリガーが最初に読む。"""
    r = client.get("/api/scenario", params={"seed": 1, "season": "winter"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["customers"]) == 15
    assert body["depot"]["name"] == "麻里布油槽所"
    tanks = body["customers"][0]["tanks"]
    assert tanks and "needs_topup" in tanks[0]


def test_usage_reports_caps():
    """上限が実際に読める。ここが空なら暴走対策が入っていない。"""
    r = client.get("/api/usage")
    assert r.status_code == 200
    caps = r.json()["caps"]
    assert caps["per_request_usd"] > 0
    assert caps["global_daily_usd"] > 0


def test_over_request_cap_is_refused():
    """1リクエストの上限を超える要求は実行前に弾く。"""
    with pytest.raises(ledger.BudgetExceeded) as exc:
        ledger.reserve("s", "explain", usd=ledger.MAX_USD_PER_REQUEST + 0.01)
    assert exc.value.scope == "request"


def test_global_daily_cap_is_the_last_line():
    """全体の1日上限は、識別子を変えても越えられない。

    利用者ごとの上限だけだと、subject を変えるだけで素通りする。

    1回あたりは request 上限の範囲に収める。ここを超える金額で試すと
    request 上限のほうが先に発火して、global を確かめたことにならない
    （最初に書いたテストがこの取り違えで通らなかった）。
    """
    step = ledger.MAX_USD_PER_REQUEST
    per_subject = int(ledger.DAILY_USD_PER_SUBJECT / step)
    n_subjects = int(ledger.DAILY_USD_GLOBAL / ledger.DAILY_USD_PER_SUBJECT) + 2

    hit_global = False
    for i in range(n_subjects):
        for _ in range(per_subject):
            try:
                ledger.reserve(f"user{i}", "x", usd=step)
            except ledger.BudgetExceeded as exc:
                assert exc.scope in ("subject", "global")
                if exc.scope == "global":
                    hit_global = True
                break
        if hit_global:
            break
    assert hit_global


def test_release_returns_the_reservation():
    """失敗した処理で枠を食わせない。"""
    ledger.reserve("bob", "explain")
    before = ledger.usage()["total_usd"]
    amount = ledger.reserve("bob", "explain")
    ledger.release("bob", amount)
    assert ledger.usage()["total_usd"] == pytest.approx(before)


def test_estimates_and_thinking_levels_cover_the_same_operations():
    """コストの見積りと思考レベルが、同じ操作の一覧を持っている。

    Gemini 3.x flash 系は思考トークンを出力として課金し、思考レベルを
    上げると実費が2倍以上になる（2026-09-03 の実測: low 498トークン →
    high 1220トークン）。見積りだけあって思考レベルの指定が漏れた操作は、
    既定の medium で走って見積りを超える。

    solve は決定論コアなのでモデルを呼ばない。ここだけ例外。
    """
    assert set(ledger.ESTIMATES) - {"solve"} == set(ledger.THINKING_LEVEL)
    assert ledger.ESTIMATES["solve"] == 0.0
    for kind, level in ledger.THINKING_LEVEL.items():
        # 3.8-flash は minimal を受け付けない
        assert level in ("low", "medium", "high"), (kind, level)


def test_every_estimate_fits_under_the_per_request_cap():
    """どの操作も、1リクエストの上限に収まっている。

    見積りが上限を超えると、その操作は**必ず** 429 で弾かれる。
    モデルを上げて見積りを引き上げたときに、静かにこうなり得る。
    """
    for kind, usd in ledger.ESTIMATES.items():
        assert usd <= ledger.MAX_USD_PER_REQUEST, (kind, usd)



def test_declined_customers_are_accepted_and_validated():
    """受入見送りを API から渡せる。名簿に無い ID は 400 で落とす。"""
    ok = client.post(
        "/api/plan",
        json={"seed": 20261015, "declined_customers": ["c03"], "mode": "fast"},
    )
    assert ok.status_code == 200
    body = ok.json()
    assert [d["customer_id"] for d in body["diagnostics"]["declined"]] == ["c03"]
    visited = {
        st["customer_id"]
        for vd in body["vehicle_days"] for t in vd["trips"] for st in t["stops"]
    }
    assert "c03" not in visited

    bad = client.post("/api/plan", json={"declined_customers": ["c99"], "mode": "fast"})
    assert bad.status_code == 400


# ---------------------------------------------------------------------------
# 当日の組み直し
# ---------------------------------------------------------------------------


def _free_customer(seed=20261016, date="2026-10-16"):
    """朝の計画に無く、今日受けられる先（当日発注を入れる相手）。"""
    from core.generator import generate
    from core.solver import SolveMode, solve
    from core.travel import build_matrix

    s = generate(seed=seed, date=date, season="winter")
    base = solve(s, matrix=build_matrix(s.depot, s.customers), mode=SolveMode.REPRODUCIBLE)
    served = {st.customer_id for t in base.trips for st in t.stops}
    # 発注は槽の単位（2kL・4kL）。空きに丸ごと入る量で頼む先を選ぶ
    for c in s.customers:
        if c.id in served or not c.open_today:
            continue
        t = max(c.tanks, key=lambda t: t.topup_l)
        vol = min(4_000, t.topup_l) // 2_000 * 2_000
        if vol:
            return c.id, t.product.value, vol
    raise AssertionError("2kL入る先が無い")


def test_replan_inserts_an_order_with_its_evidence():
    """FAXの発注を差し込むと、どの車で何時に着くかが証拠つきで返る。"""
    cid, product, vol = _free_customer()
    r = client.post("/api/replan", json={
        "seed": 20261016, "date": "2026-10-16", "now": "09:00",
        "orders": [{"customer_id": cid, "product": product, "volume_l": vol,
                    "source": "fax", "evidence_ref": "FAX-API-001"}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"]
    assert any("FAX-API-001" in c for c in body["changes"])
    assert "組み直し" in body["text_ja"]
    # 朝の計画は再現モードで組み直している（/api/plan と同じ土台）
    assert body["diagnostics"]["base_reproducible"] is True


def test_replan_refuses_phone_orders():
    """電話は発注ではない。証拠の残る形（発注システムかFAX）でしか受けない。"""
    r = client.post("/api/replan", json={
        "now": "09:00",
        "orders": [{"customer_id": "c01", "product": "regular", "volume_l": 4000,
                    "source": "phone", "evidence_ref": "電話 9:00"}],
    })
    assert r.status_code == 422


def test_replan_rejects_unknown_customer():
    r = client.post("/api/replan", json={
        "now": "09:00",
        "orders": [{"customer_id": "c99", "product": "regular", "volume_l": 4000,
                    "source": "system", "evidence_ref": "ORD-1"}],
    })
    assert r.status_code == 400



def test_return_endpoint_proposes_options_in_priority_order():
    """持ち戻りの連絡に、優先順の手と責任の所在を返す。"""
    from tests.test_returns import _day

    _s, _plan, e = _day()
    r = client.post("/api/return", json={
        "seed": 20261015, "date": "2026-10-15", "vehicle_id": e.vehicle_id,
        "customer_id": e.customer_id, "product": e.product.value, "volume_l": e.volume_l,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chosen"]["kind"] in ("redeliver", "cushion", "buyback")
    assert "元売の管理責任" in body["responsibility_ja"]
    assert any(o["kind"] == "buyback" for o in body["options"])
