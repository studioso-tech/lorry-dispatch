"""需要の鎖: 地区の需要から、1社の運送会社が担う需要を出す。

**因果の向きがこの順であることが要点**（実務）:

    地区全体の需要
      × ENEOS のシェア
      × その運送会社への引き当て割合
      ＝ その運送会社が担う需要
      → 必要なローリー台数（専属＋非専属）
      → 必要な乗務員数

台数を入力として受け取るのは、この鎖の一番下だけを見ていることになる。
元売が半期ごとにシェアや引き当てを見直せば、運送会社の必要台数が動く。
このモジュールはその「動いたら何台か」を出すための入口。

配送エリア:
    当日の配車は名簿（15軒）を1つの配送エリアとして組む。運送会社1社は
    このエリアの何個ぶんかを担う、とみなす（docs/業務前提.md §5.1.1b）。
    名簿を1社ぶん（150〜200軒）に広げて1日を丸ごと解くのは、ソルバーの
    規模と日程の両方で見合わないため。

**水準は逆算した仮置。** 地区全体の需要は公開統計（資源・エネルギー統計の
都道府県別販売量）から見積もるべきだが、まだ取得していない。いまはモデルの
運送会社（専属20台前後）が冬に20台前後になる水準に置いてある。水準は逆算
でも、シェアや引き当てが動いたときに台数がどう動くか（因果の向き）は正しく出る。
"""

from __future__ import annotations

from dataclasses import dataclass

from core.generator import CLOSED_ON_SUNDAY, ROSTER, _planned_daily

#: 半期の月と、その月の需要を代表させる季節。冬の半期（10〜3月）の中でも、
#: 灯油が跳ねるのは12〜2月で、10〜11月と3月は穏やか。専属を冬の山に
#: 合わせて取ると、10月や3月は車が遊ぶ。その差を非専属で埋めるのが
#: 専属と非専属の配分の問題になる。
HALF_YEAR_MONTHS: dict[str, tuple[tuple[int, str], ...]] = {
    "winter_half": ((10, "autumn"), (11, "autumn"), (12, "winter"),
                    (1, "winter"), (2, "winter"), (3, "spring")),
    "summer_half": ((4, "spring"), (5, "spring"), (6, "summer"),
                    (7, "summer"), (8, "summer"), (9, "autumn")),
}


@dataclass(frozen=True)
class DemandChain:
    """地区の需要から運送会社の担う需要を出す3つの係数。

    Attributes:
        district_kl_per_day: 地区全体の需要（冬の平日、kL/日）。**逆算した仮置。**
        eneos_share: ENEOS のシェア。**仮置**（全国の燃料油で約5割とされる水準）。
        allocation: ENEOS の需要のうち、この運送会社に引き当てる割合。
            **仮置。** 麻里布の旧日石系は2社が担い、専属は20台前後と30台以下
            （実務・記憶ベース）。台数の比から 20 / (20 + 30) = 0.4 と置いた。
    """

    district_kl_per_day: float = 2_350.0
    eneos_share: float = 0.5
    allocation: float = 0.4

    def __post_init__(self) -> None:
        for name in ("eneos_share", "allocation"):
            v = getattr(self, name)
            if not 0.0 < v <= 1.0:
                raise ValueError(f"{name} は 0〜1 の割合（{v}）")
        if self.district_kl_per_day <= 0:
            raise ValueError("地区の需要は正の値")

    @property
    def carrier_kl_per_day(self) -> float:
        """この運送会社が担う需要（冬の平日、kL/日）。"""
        return self.district_kl_per_day * self.eneos_share * self.allocation


def zone_kl_per_day(season: str = "winter") -> float:
    """名簿1つぶん（配送エリア1つ）が平日1日に売る量（kL/日）。"""
    return sum(sum(_planned_daily(spec, season).values()) for spec in ROSTER) / 1000


def zones_for(chain: DemandChain) -> int:
    """運送会社の需要が、配送エリアの何個ぶんにあたるか（冬の平日で数える）。

    端数は切り上げる。0.3エリアぶんの需要も、誰かが運ばなければならない。
    """
    ratio = chain.carrier_kl_per_day / zone_kl_per_day("winter")
    return max(1, -(-int(round(ratio * 1000)) // 1000))


def sunday_share_of_zone(season: str = "winter") -> float:
    """配送エリアの日曜の売上が平日の何割か（休業の先は売れない）。"""
    open_on_sunday = sum(
        sum(_planned_daily(spec, season).values())
        for spec in ROSTER
        if not CLOSED_ON_SUNDAY[spec.type]
    )
    return open_on_sunday / (zone_kl_per_day(season) * 1000)
