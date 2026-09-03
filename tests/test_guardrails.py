"""暴走対策の検算。

一つ前のプロジェクトで、判定条件の誤りにより成果物が毎回3回作り直されて
いた不具合があった。費用が静かに3倍になっていたが、トークン使用量が
どこにも記録されておらず、請求書を見るまで分からなかった。

ここで確かめるのは「上限があること」ではなく、**上限が実際に効くこと**。
"""

from __future__ import annotations

import pytest

from app import ledger
from app.guardrails import Limits, RunawayGuard
from app.meter import PRICES, UNKNOWN_PRICE, BudgetExceeded, CostMeter


def _meter(ceiling: float = 0.10) -> CostMeter:
    return CostMeter(request_id="t", ceiling_usd=ceiling)


# ---------------------------------------------------------------------------
# 実測（CostMeter）
# ---------------------------------------------------------------------------


def test_thought_tokens_are_billed_as_output():
    """思考トークンを出力として課金する。

    **ここを別会計にすると実費の大半を見落とす。** 実測では出力17に対し
    思考が498出た。思考を数えないと、実費の96%が見えないことになる。
    """
    m = _meter(ceiling=100.0)   # 上限に引っかかる前に計上させる
    m.record("gemini-3.8-flash", input_tokens=0, output_tokens=0, thought_tokens=1_000_000)
    assert m.usd == pytest.approx(PRICES["gemini-3.8-flash"].output_per_m)
    assert m.thought_tokens == 1_000_000


def test_unknown_model_is_priced_pessimistically():
    """単価表に無いモデルは高く見積もる。

    安く見積もると上限が素通りする。モデルを差し替えて単価表の更新を
    忘れたときに、静かにそうなる。
    """
    m = _meter(ceiling=100.0)
    m.record("gemini-99-imaginary", input_tokens=1_000_000)
    assert m.usd == pytest.approx(UNKNOWN_PRICE.input_per_m)
    assert UNKNOWN_PRICE.input_per_m > max(p.input_per_m for p in PRICES.values())


def test_meter_stops_a_runaway_before_it_spends():
    """上限を超えたらその場で止める。**次の呼び出しを待たない。**"""
    m = _meter(ceiling=0.01)
    with pytest.raises(BudgetExceeded):
        for _ in range(50):
            m.record("gemini-3.8-flash", input_tokens=1000, thought_tokens=5000)
    assert m.usd > 0.01
    assert m.calls < 50           # 50回まわる前に止まっている


def test_meter_summary_hides_nothing():
    """使った量を隠さない。裏側ビューがこれをそのまま出す。"""
    m = _meter()
    m.record("gemini-3.8-flash", input_tokens=34, output_tokens=23, thought_tokens=522)
    s = m.summary()
    assert s["thought_tokens"] == 522
    assert s["model_calls"] == 1
    # 2026-09-03 の実測とおおむね一致すること
    assert s["estimated_cost_usd"] == pytest.approx(0.00207, abs=5e-5)


# ---------------------------------------------------------------------------
# 回数と時間（RunawayGuard）
# ---------------------------------------------------------------------------


def test_identical_tool_calls_are_stopped():
    """同じ道具を同じ引数で呼び続けるのが、輪のいちばん分かりやすい徴候。"""
    g = RunawayGuard("t", _meter(), Limits(max_identical_calls=2))
    args = {"customer": "c06"}
    assert g.check_tool("lookup", args) is None
    assert g.check_tool("lookup", args) is None
    reason = g.check_tool("lookup", args)
    assert reason is not None and "同じ引数" in reason


def test_a_different_argument_is_not_a_loop():
    """引数が違えば輪ではない。ここを取り違えると正常な調査を殺す。"""
    g = RunawayGuard("t", _meter(), Limits(max_identical_calls=1))
    assert g.check_tool("lookup", {"customer": "c06"}) is None
    assert g.check_tool("lookup", {"customer": "c07"}) is None


def test_solver_calls_are_capped_separately():
    """求解はモデル呼び出しと別枠で数える。

    このシステムでは求解のほうが高い（1回 約$0.006 対 約$0.002）。しかも
    1回60秒なので、5回まわすと Cloud Run のリクエスト上限に達して落ちる。
    """
    g = RunawayGuard("t", _meter(), Limits(max_solver_calls=2))
    assert g.check_solver() is None
    assert g.check_solver() is None
    reason = g.check_solver()
    assert reason is not None and "組み直し" in reason
    # モデル側の枠は減っていない
    assert g.model_calls == 0


def test_explain_reserve_keeps_budget_for_the_explanation():
    """調査で使い切って、計画は出たが理由を説明できない、を防ぐ。

    調査側は上限より explain_reserve だけ手前で止まる。説明役はそこから
    先も動ける。
    """
    limits = Limits(max_model_calls=5, explain_reserve=3)
    g = RunawayGuard("t", _meter(), limits)

    for _ in range(2):
        assert g.check_model(agent="researcher") is None
    assert g.check_model(agent="researcher") is not None   # 5-3=2 で止まる

    # 説明役はまだ動ける
    assert g.check_model(agent="explainer") is None


def test_cost_ceiling_stops_both_models_and_tools():
    """費用の上限は、モデルにも道具にも効く。片方だけだと抜け道になる。"""
    m = _meter(ceiling=0.001)
    m.record("gemini-3.8-flash", input_tokens=1000, thought_tokens=0)  # 上限内
    g = RunawayGuard("t", m, Limits(max_cost_usd=0.0005))
    assert g.check_model() is not None
    assert g.check_tool("anything") is not None
    assert g.check_solver() is not None


def test_trips_are_recorded_once_and_visible():
    """打ち切ったことを黙って隠さない。裏側ビューに出す。"""
    g = RunawayGuard("t", _meter(), Limits(max_solver_calls=0))
    g.check_solver()
    g.check_solver()
    assert len(g.trips) == 1              # 同じ理由は1度だけ数える
    assert g.status()["tripped"] == g.trips


def test_stop_message_tells_the_model_to_admit_it_was_cut_short():
    """打ち切られたことを、答えの中で述べさせる。

    黙って短くなった答えは、短いことに気づかれないまま信用される。
    """
    msg = RunawayGuard.stop_message("時間切れです。")
    assert "打ち切った" in msg
    assert "手元にある材料" in msg


def test_interactive_limits_are_tighter():
    """画面から叩く経路は、配車係を待たせられないので厳しくする。"""
    strict, normal = Limits.interactive(), Limits()
    assert strict.deadline_seconds < normal.deadline_seconds
    assert strict.max_model_calls < normal.max_model_calls
    assert strict.max_solver_calls < normal.max_solver_calls


# ---------------------------------------------------------------------------
# 月の上限（台帳）
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_ledger():
    ledger.reset_for_tests()
    yield
    ledger.reset_for_tests()


def test_monthly_cap_is_the_last_line(monkeypatch):
    """月の上限は、**日をまたいでも**識別子を変えても越えられない。

    予算アラートは月次で、しかも通知するだけで支出を止めない。
    日次上限だけだと、毎日上限まで使って月末に30倍になる。

    1日の中では日次上限が先に効くので、日を進めないとここは確かめられない
    （最初に書いたテストがこの取り違えで通らなかった）。同じ月の中で
    日付だけを進める。
    """
    step = ledger.MAX_USD_PER_REQUEST
    hit = None
    for day in range(1, 29):
        monkeypatch.setattr(ledger, "_today", lambda d=day: f"2026-09-{d:02d}")
        for i in range(50):
            try:
                ledger.reserve(f"d{day}u{i}", "x", usd=step)
            except ledger.BudgetExceeded as exc:
                if exc.scope == "month":
                    hit = "month"
                break
        if hit:
            break

    assert hit == "month", "日をまたいでも月の上限が効かなかった"
    assert ledger.usage()["month_total_usd"] <= ledger.MONTHLY_USD_GLOBAL


def test_daily_cap_fires_first_within_one_day():
    """1日の中では、月ではなく日次上限が先に効く。段の順序の確認。"""
    hit = None
    for i in range(500):
        try:
            ledger.reserve(f"user{i}", "x", usd=ledger.MAX_USD_PER_REQUEST)
        except ledger.BudgetExceeded as exc:
            hit = exc.scope
            break
    assert hit == "global"


def test_monthly_cap_sits_above_the_daily_cap():
    """段の順序が逆転していないこと。

    月 < 日 になっていると、日次上限が一度も効かない飾りになる。
    """
    assert ledger.MONTHLY_USD_GLOBAL > ledger.DAILY_USD_GLOBAL
    assert ledger.DAILY_USD_GLOBAL > ledger.DAILY_USD_PER_SUBJECT
    assert ledger.DAILY_USD_PER_SUBJECT >= ledger.MAX_USD_PER_REQUEST


def test_usage_reports_the_month_and_says_alerts_do_not_stop_spending():
    """月の使用状況が読めること。予算アラートの限界も明記されること。"""
    ledger.reserve("bob", "explain")
    u = ledger.usage()
    assert u["month_total_usd"] > 0
    assert u["caps"]["global_monthly_usd"] == ledger.MONTHLY_USD_GLOBAL
    assert "止めるのはこの上限" in u["note"]


def test_release_returns_the_monthly_reservation_too():
    """失敗した処理で月の枠を食わせない。日次だけ戻して月を戻し忘れると、
    月末に向かって静かに枠が痩せていく。"""
    before = ledger.usage()["month_total_usd"]
    amount = ledger.reserve("bob", "explain")
    ledger.release("bob", amount)
    assert ledger.usage()["month_total_usd"] == pytest.approx(before)


# ---------------------------------------------------------------------------
# ADK への差し込み
# ---------------------------------------------------------------------------


class _FakeAgent:
    """ADK のエージェントの代わり。attach が置く属性だけを持つ。"""

    before_model_callback = None
    before_tool_callback = None


def test_attach_wires_every_path():
    """attach 1行で、モデルもツールもガードを通るようになる。

    過去の事故は ADK の暴走だった。「ガードを通す経路」と「通さない経路」が
    併存すると、必ず通さないほうから漏れる。
    """
    g = RunawayGuard("t", _meter())
    agent = _FakeAgent()
    g.attach(agent)
    assert agent.before_model_callback == g.before_model_callback
    assert agent.before_tool_callback == g.before_tool_callback


def test_adk_tool_callback_replaces_the_result_instead_of_raising():
    """上限に達したら、例外ではなく道具の結果を差し替える。

    ADK の中で例外を投げると、そこまでに集めた材料ごと失われる。
    """
    g = RunawayGuard("t", _meter(), Limits(max_tool_calls=0))
    result = g.before_tool_callback(tool=_named("lookup"), args={})
    assert isinstance(result, dict)
    assert "停止" in result["error"]
    assert "もう一度呼ばない" in result["instruction"]


def test_adk_tool_callback_counts_the_solver_separately():
    """求解は別枠。名前に solve を含む道具は求解として数える。"""
    g = RunawayGuard("t", _meter(), Limits(max_solver_calls=1, max_tool_calls=99))
    assert g.before_tool_callback(tool=_named("solve_plan"), args={}) is None
    assert g.solver_calls == 1
    assert g.tool_calls == 0          # ツール枠は減っていない
    assert g.before_tool_callback(tool=_named("solve_plan"), args={}) is not None


def _named(name: str):
    class _T:
        pass
    t = _T()
    t.name = name
    return t
