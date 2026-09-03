"""1リクエストの中でのトークン計測と、実費の上限。

台帳（app/ledger.py）との役割分担:
    台帳    リクエストを**またいだ**上限。1日・1か月で何人来ても青天井に
            しない最後の砦。事前に見積りを予約する。
    ここ    1リクエストの**中**の上限。実際に使ったトークンを数え、
            使いすぎたらその場で止める。

    **台帳だけでは足りない。** 台帳が知っているのは「1回予約した」ことだけで、
    その1リクエストの中でエージェントが50回ループしても素通りする。
    見積りと実費が乖離するのはこの経路。

なぜ実測が要るか:
    一つ前のプロジェクトで、判定条件の誤りで成果物が毎回3回作り直されていた
    不具合があった。費用が静かに3倍になっていたが、**トークン使用量が
    どこにも記録されておらず、請求書を見るまで分からなかった**。
    見えないものは止められない。

COST 行:
    呼び出しのたびに1行を標準出力へ出す。Cloud Logging で `COST` を検索
    すれば集計できる。Firestore を開かなくても当日の傾向が読める。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger("lorry.meter")


@dataclass(frozen=True)
class Price:
    """100万トークンあたりの単価（USD）。"""

    input_per_m: float
    output_per_m: float


#: Vertex AI の公開料金（global）。2026-09-03 に確認。
#:
#: **出力単価には思考トークンが含まれる。** Gemini 3.x flash 系は思考を
#: 出力として課金し、しかも思考が出力本体の20倍以上出ることがある
#: （実測: 出力17に対し思考498）。
#:
#: **導入価格は 2026-12-31 で終了し、2027-01-01 に倍額になる。**
#: 年明けにここを書き換えること。忘れると COST 行が実費の半分に見える。
#:
#: モデルを変えたら必ずここも更新する。単価を据え置いたままモデルだけ
#: 上げると、上限が実質的に外れる。
PRICES: dict[str, Price] = {
    "gemini-3.8-flash": Price(input_per_m=0.75, output_per_m=3.75),
    "gemini-3.7-flash": Price(input_per_m=0.75, output_per_m=3.75),
    "gemini-3.6-flash": Price(input_per_m=1.50, output_per_m=7.50),
}

#: 表に無いモデルの単価。**わざと高く置いてある。**
#: 安く見積もると上限が素通りする。警告を出したうえでこれを使う。
UNKNOWN_PRICE = Price(input_per_m=3.00, output_per_m=15.00)


class BudgetExceeded(RuntimeError):
    """1リクエストの実費が上限を超えた。"""


@dataclass
class CostMeter:
    """1リクエストにつき1個。入ってくる API 呼び出しごとに作る。

    スレッド安全にしてあるのは、エージェントが複数の下請けを並行に
    走らせる場合があるため。
    """

    request_id: str
    ceiling_usd: float
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    usd: float = 0.0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
        thought_tokens: int = 0,
    ) -> float:
        """1回ぶんを計上する。上限を超えたら BudgetExceeded。

        Args:
            model: 呼んだモデル名。
            input_tokens: 入力トークン。
            output_tokens: 出力トークン（思考を含まない本体）。
            thought_tokens: 思考トークン。**出力として課金される**ので
                出力側に足し込む。ここを別会計にすると、実費の大半を
                見落とすことになる。

        Returns:
            この1回の実費（USD）。
        """
        price = PRICES.get(model)
        if price is None:
            price = UNKNOWN_PRICE
            logger.warning(
                "COST 単価表に無いモデル=%s → 高めの既定単価で計上する", model
            )

        billed_out = output_tokens + thought_tokens
        delta = (
            input_tokens / 1_000_000 * price.input_per_m
            + billed_out / 1_000_000 * price.output_per_m
        )

        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.thought_tokens += thought_tokens
            self.usd += delta
            self.calls += 1
            total = self.usd
            calls = self.calls

        logger.info(
            "COST request=%s model=%s in=%d out=%d think=%d "
            "call_usd=%.6f total_usd=%.6f ceiling=%.4f",
            self.request_id, model, input_tokens, output_tokens, thought_tokens,
            delta, total, self.ceiling_usd,
        )

        if total > self.ceiling_usd:
            logger.error(
                "COST 上限超過 request=%s total_usd=%.6f ceiling=%.4f calls=%d",
                self.request_id, total, self.ceiling_usd, calls,
            )
            raise BudgetExceeded(
                f"リクエスト {self.request_id} が上限 ${self.ceiling_usd:.2f} を"
                f"超えました（{calls}回の呼び出しで ${total:.4f}）"
            )
        return delta

    def check(self) -> None:
        """高くつく処理の前に呼ぶ。すでに上限なら即座に落とす。"""
        if self.usd > self.ceiling_usd:
            raise BudgetExceeded(
                f"リクエスト {self.request_id} はすでに ${self.usd:.4f} 使っています"
            )

    def summary(self) -> dict:
        """応答に載せる形。**使った量を利用者から隠さない。**"""
        return {
            "model_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thought_tokens": self.thought_tokens,
            "estimated_cost_usd": round(self.usd, 6),
            "ceiling_usd": self.ceiling_usd,
        }


def estimate_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """単発の見積り。台帳の ESTIMATES を作り直すときに使う。"""
    price = PRICES.get(model, UNKNOWN_PRICE)
    return (
        input_tokens / 1_000_000 * price.input_per_m
        + output_tokens / 1_000_000 * price.output_per_m
    )


__all__ = [
    "PRICES",
    "UNKNOWN_PRICE",
    "BudgetExceeded",
    "CostMeter",
    "Price",
    "estimate_usd",
]
