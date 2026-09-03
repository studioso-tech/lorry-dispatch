"""エージェントの暴走対策。

道具を持った LLM は輪に入る。道具を呼び、答えが気に入らず、ほとんど同じ
引数でもう一度呼び、それを延々と続ける。放っておくと金を燃やし、
リクエストが返らなくなる。

**これは仮定ではない。** 過去に ADK の暴走で、無料クレジットの半分
（2万数千円分）を一度に失っている。残高がある限り、止めるものが無ければ
走り続ける。

このモジュールは、モデル呼び出しとツール呼び出しの手前に**独立した上限を
7本**置く。

設計上の判断（一つ前のプロジェクトから引き継ぐ）:
    **上限に達しても例外を投げない。** モデルに「そこで止めて、いま持って
    いる材料で答えよ」という平文を渡す。打ち切られた答えでも、実際の根拠の
    上に立っていれば、スタックトレースより役に立つ。そして打ち切ったことは
    記録に残して**見えるようにする**。黙って切らない。

    唯一の強制終了は実時間の締切。止まらないリクエストは、どこかで
    終わらせるしかない。

本PJ固有の事情:
    **求解の回数が、モデル呼び出しより高くつく。**

        Gemini 3.8 Flash 1回              約 $0.002
        求解1回（Cloud Run 4vCPU × 60秒） 約 $0.006

    エージェントが「条件を少し変えて解き直す」を繰り返すのが、この
    システムで最も高くつく暴走の形になる。しかも1回60秒なので、
    5回まわせば Cloud Run のリクエスト上限（300秒）に達して落ちる。
    よってモデルとは別に、求解の回数にも上限を置く。
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.meter import CostMeter

logger = logging.getLogger("lorry.guardrails")


@dataclass
class Limits:
    """ここにある数字はすべて天井であって、目標ではない。"""

    #: リクエスト全体でのモデル呼び出し回数。
    #: ツール呼び出しは1回ごとにモデルの手番を消費するので、
    #: モデルの天井はツールの天井より上に置く。そうしないと
    #: ガードが自分の道具を絞め殺す。
    max_model_calls: int = 20

    #: リクエスト全体でのツール呼び出し回数。
    max_tool_calls: int = 12

    #: 同じ道具を何度も使う場合の上限。
    max_calls_per_tool: int = 5

    #: **同じ道具を同じ引数で呼ぶ回数。これが輪の徴候。**
    max_identical_calls: int = 2

    #: **求解の回数。** このシステムで最も高くつく操作（下記）。
    max_solver_calls: int = 3

    #: 実時間の締切（秒）。Cloud Run のリクエスト上限より内側に置く。
    deadline_seconds: int = 240

    #: 実費の上限（USD）。CostMeter と同じ値を渡す。
    max_cost_usd: float = 0.10

    #: 最後の説明用に取っておくモデル呼び出し。
    #:
    #: 調べるだけ調べて上限に達し、**計画は出たが理由を説明できない**、
    #: という終わり方を防ぐ。調査側の実効上限は
    #: max_model_calls - explain_reserve になる。
    explain_reserve: int = 3

    @classmethod
    def interactive(cls) -> "Limits":
        """配車係が画面から叩く経路。待たせられないので厳しくする。"""
        return cls(
            max_model_calls=6,
            max_tool_calls=4,
            max_calls_per_tool=2,
            max_identical_calls=1,
            max_solver_calls=1,
            deadline_seconds=90,
            explain_reserve=1,
        )


class RunawayGuard:
    """1リクエストにつき1個。

    スレッド安全にしていないのは意図的で、1リクエスト1ループだから。
    並行に走らせるなら、それぞれに別の Guard を持たせること。
    """

    def __init__(
        self,
        request_id: str,
        meter: CostMeter,
        limits: Limits | None = None,
    ) -> None:
        self.request_id = request_id
        self.meter = meter
        self.limits = limits or Limits()
        self.started = time.monotonic()
        self.current_agent = ""
        self.model_calls = 0
        self.tool_calls = 0
        self.solver_calls = 0
        self.per_tool: Counter[str] = Counter()
        self.signatures: Counter[str] = Counter()
        self.trips: list[str] = []

    # -- 補助 ---------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def _trip(self, reason: str) -> str:
        """上限に達したことを記録する。**同じ理由は1度だけ数える。**"""
        if reason not in self.trips:
            self.trips.append(reason)
            logger.warning(
                "GUARDRAIL request=%s 発動: %s", self.request_id, reason
            )
        return reason

    @staticmethod
    def _signature(tool_name: str, args: dict[str, Any]) -> str:
        """同じ道具・同じ引数を見分ける鍵。輪の検知はこれで行う。"""
        try:
            payload = json.dumps(args, sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            payload = repr(sorted(args))
        return f"{tool_name}::{payload}"

    def status(self) -> dict:
        """裏側ビューに出す形。**何が効いたかを隠さない。**"""
        return {
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "solver_calls": self.solver_calls,
            "elapsed_seconds": round(self.elapsed, 2),
            "limits": {
                "max_model_calls": self.limits.max_model_calls,
                "max_tool_calls": self.limits.max_tool_calls,
                "max_calls_per_tool": self.limits.max_calls_per_tool,
                "max_identical_calls": self.limits.max_identical_calls,
                "max_solver_calls": self.limits.max_solver_calls,
                "deadline_seconds": self.limits.deadline_seconds,
                "max_cost_usd": self.limits.max_cost_usd,
                "explain_reserve": self.limits.explain_reserve,
            },
            "tripped": list(self.trips),
        }

    # -- 判定 ---------------------------------------------------------------

    def _common_reason(self) -> str | None:
        """モデルにもツールにも共通で効く上限。"""
        if self.elapsed > self.limits.deadline_seconds:
            return f"時間切れです（{self.limits.deadline_seconds}秒）。"
        if self.meter.usd >= self.limits.max_cost_usd:
            return f"費用の上限に達しました（${self.limits.max_cost_usd:.2f}）。"
        return None

    def check_model(self, agent: str = "") -> str | None:
        """モデルを呼ぶ前に。止めるべきなら理由を返す（例外は投げない）。"""
        self.current_agent = agent or self.current_agent

        # 説明役のぶんを取っておく。調査側だけが早めに止まる。
        ceiling = self.limits.max_model_calls
        if self.limits.explain_reserve and agent and agent != "explainer":
            ceiling = max(1, ceiling - self.limits.explain_reserve)

        reason = self._common_reason()
        if reason is None and self.model_calls >= ceiling:
            reason = f"モデル呼び出しの上限に達しました（{ceiling}回）。"

        if reason:
            return self._trip(reason)
        self.model_calls += 1
        return None

    def check_tool(self, tool_name: str, args: dict[str, Any] | None = None) -> str | None:
        """ツールを呼ぶ前に。止めるべきなら理由を返す。"""
        args = args or {}
        signature = self._signature(tool_name, args)

        reason = self._common_reason()
        if reason is None:
            if self.tool_calls >= self.limits.max_tool_calls:
                reason = f"道具の呼び出し上限に達しました（{self.limits.max_tool_calls}回）。"
            elif self.per_tool[tool_name] >= self.limits.max_calls_per_tool:
                reason = (
                    f"『{tool_name}』はすでに{self.per_tool[tool_name]}回"
                    f"呼んでいます。"
                )
            elif self.signatures[signature] >= self.limits.max_identical_calls:
                reason = (
                    f"『{tool_name}』を**まったく同じ引数で**"
                    f"{self.signatures[signature]}回呼んでいます。"
                    f"もう一度呼んでも同じ答えが返ります。"
                )

        if reason:
            return self._trip(reason)
        self.tool_calls += 1
        self.per_tool[tool_name] += 1
        self.signatures[signature] += 1
        return None

    def check_solver(self) -> str | None:
        """求解の前に。

        **モデル呼び出しより高くつく操作なので、別枠で数える。**
        1回60秒・約$0.006。5回まわすと Cloud Run のリクエスト上限に達する。
        """
        reason = self._common_reason()
        if reason is None and self.solver_calls >= self.limits.max_solver_calls:
            reason = (
                f"配車の組み直しは{self.limits.max_solver_calls}回までです。"
                f"これ以上は条件のほうを見直してください。"
            )
        if reason:
            return self._trip(reason)
        self.solver_calls += 1
        return None

    # -- ADK への差し込み ----------------------------------------------------
    #
    # 過去の事故は ADK の暴走だった。W4 で ADK を使うときに「ガードを通す
    # 経路」と「通さない経路」が併存すると、必ず通さないほうから漏れる。
    # ADK 側のフックにそのまま挿せる形をここに置いて、**agent に attach
    # するだけで全経路が通る**ようにしておく。

    def before_model_callback(self, callback_context=None, llm_request=None):
        """ADK がモデルを呼ぶ直前に走る。応答を返すと呼び出しが取り消される。

        ADK はコールバックをキーワード引数で呼ぶので、**引数名は飾りでは
        なく契約**である。変えると呼ばれない。
        """
        agent = getattr(callback_context, "agent_name", "") or self.current_agent
        reason = self.check_model(agent=agent)
        if reason is None:
            return None

        # 例外を投げず、モデルに「止めて、いまある材料で答えよ」と伝える。
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types as genai_types

        return LlmResponse(
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=self.stop_message(reason))],
            )
        )

    def before_tool_callback(self, tool=None, args=None, tool_context=None):
        """ADK が道具を呼ぶ直前に走る。dict を返すと道具の結果に差し替わる。"""
        name = getattr(tool, "name", str(tool))
        # 求解だけは別枠で数える（このシステムで最も高くつく操作）。
        reason = (
            self.check_solver() if "solve" in name or "配車" in name
            else self.check_tool(name, args or {})
        )
        if reason is None:
            return None
        return {
            "error": self.stop_message(reason),
            "instruction": "この道具をもう一度呼ばないこと。いま持っている材料で答えること。",
        }

    def attach(self, agent) -> None:
        """ADK のエージェントに配線する。**これ1行で全経路が通る。**"""
        agent.before_model_callback = self.before_model_callback
        agent.before_tool_callback = self.before_tool_callback

    # -- 上限に達したときにモデルへ渡す文 ------------------------------------

    @staticmethod
    def stop_message(reason: str) -> str:
        """**例外ではなくこれを返す。**

        打ち切られた答えでも、実際の根拠の上に立っていれば役に立つ。
        そして打ち切ったこと自体を答えの中で述べさせる。黙って短く
        なった答えは、短いことに気づかれないまま信用される。
        """
        return (
            f"[ガードレールにより停止] {reason} "
            f"いま手元にある材料だけで答えてください。"
            f"そして、検討を途中で打ち切ったことを明記してください。"
        )


class DeadlineExceeded(RuntimeError):
    """柔らかい上限が止められず、実時間の締切に達した。"""


async def with_deadline(coro, seconds: int, request_id: str):
    """外側の最後の砦。柔らかい上限が先に効くのが正常。

    ここが効いたということは、上のどれかが機能していないということなので、
    エラーとして記録する。
    """
    import asyncio

    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError as exc:
        logger.error(
            "GUARDRAIL request=%s 実時間の締切 %s秒 を超過し、強制終了しました",
            request_id, seconds,
        )
        raise DeadlineExceeded(
            f"リクエスト {request_id} が締切 {seconds}秒 を超えたため打ち切りました"
        ) from exc


__all__ = [
    "DeadlineExceeded",
    "Limits",
    "RunawayGuard",
    "with_deadline",
]
