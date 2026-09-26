"""合成データ生成器 v0。

なぜ合成データか:
    実在の需要家のタンク在庫・消費量・受入時間枠は営業情報であり、外に出せない。
    一方で審査員には**手元で全デモを再現してもらう**必要がある（提出要件）。
    そこで、地理と制約の形は実際の二次配送に合わせ、数字は合成する。

    地理を合成しないのは理由がある。「岩国から浜田まで灯油を運ぶと拘束時間が
    厳しい」という制約の効き方は、実際の距離でないと成立しない。作り物の
    座標で作った問題は、作り物の制約しか生まない。

何を固定し、何を振るか:
    - 固定: 需要家の顔ぶれ・座標・類型・受入時間枠・タンク容量
      → 地図が日によって変わらない。等時線マップ（W5）の見え方が安定する。
    - 周期で決まる: 計画配送の日と量、在庫水準（前回の配送からの経過日数）
      → 計画配送は月〜土でほぼ一定量になる（実務）。日付を変えると周期が進む。
    - 振る: 消費の日々の揺れ、発注配送の先が頼む日
      → 発注が「計画配送の上に乗る、振れる部分」になる。

    seed を変えれば別の日が出る。seed が同じなら完全に同じ日が出る。
    これは回帰テスト（W6）の前提であり、シードを固定しない生成は禁止。

季節:
    冬は灯油が跳ねる（度日法の乗数は W4 のシグナル層で扱う）。ここでは
    平年の季節差だけを入れる。寒波という**事象**は生成器ではなくシグナルで
    与える。生成器に事象を混ぜると、「寒波を注入したから再計画された」のか
    「たまたまそういう日が生成された」のかが切り分けられなくなる。
"""

from __future__ import annotations

import datetime
import functools
import random
from dataclasses import dataclass, replace

from core.domain import (
    DROP_STEP_BY_PRODUCT,
    Customer,
    CustomerType,
    Depot,
    Driver,
    Product,
    Scenario,
    Tank,
    VEHICLE_CLASS_JA,
    Vehicle,
    VehicleClass,
    hhmm_to_minutes,
    unload_minutes,
    whole_drops_l,
)
from core.events import SoldMore, sell_more, validate as validate_events
from core.travel import leg_minutes

# ---------------------------------------------------------------------------
# 油槽所
# ---------------------------------------------------------------------------

#: 麻里布（岩国・和木）。ENEOS 麻里布製油所に併設の油槽所という設定。
#: 04:00 開所。24時間稼働（二交代）の車はこの時刻から積める。日勤の車は
#: 通勤の混雑を避けて 5:00 に車庫を出る（domain.DAY_SHIFT_DEPART_MIN）。
MARIFU = Depot(
    id="depot_marifu",
    name="麻里布油槽所",
    lat=34.2028,
    lon=132.2244,
    open_min=hhmm_to_minutes("04:00"),
    close_min=hhmm_to_minutes("22:00"),
    # 積込は平均20分（実務）。積場は8レーンで8台まで同時に積める。
    load_min=20,
)

#: 二交代の日の油槽所。**夜勤が積めるように閉所を翌05:00まで延ばす。**
#: **仮置**——麻里布の積場が夜間も開いているかは未確認。夜に積めないなら
#: 夜勤は1便しか持てず、二交代そのものが成り立たない（§9の未確認に追記）。
MARIFU_NIGHT = replace(MARIFU, close_min=hhmm_to_minutes("29:00"))


# ---------------------------------------------------------------------------
# 需要家の名簿（固定）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CustomerSpec:
    """名簿1行。座標と類型と時間枠は固定、在庫はここから seed で振る。"""

    id: str
    name: str
    type: CustomerType
    lat: float
    lon: float
    #: 地下タンク側の配管の口径（インチ）。**系列の出自で決まる**（実務）。
    #:
    #:     旧三菱系のSS   4インチ
    #:     それ以外       3インチ（特に旧日石系は設備が制約になる）
    #:
    #: 立地や規模では決まらない。ENEOS は日石と三菱の合流でできているので、
    #: 同じ看板の下に配管の違う2系統のSSが混在している。
    piping_inch: int = 3
    #: 自分で発注する先か（発注配送）。False なら計画配送（元売・運送会社が
    #: 需要見込みで運ぶ）。
    ordering: bool = False
    #: 規模（月間の燃料油販売量）。**需要はここから割り出す。**
    size: str = "medium"
    #: 立地（油種の構成を決める）。city / trunk / sanin / suburb / depot。
    mix: str = "suburb"


#: 重要家15軒。引き継ぎ文書 §4.2 の4類型を「1回のデモで全制約系が見える」よう配分。
#: 座標は実在の市町の位置に合わせてあるが、事業所そのものは架空である。
#:
#: 配管4インチ（旧三菱系）はGS12軒のうち4軒に置いた。**どの先がそうかは架空**
#: で、比率（約1/3）も合流時の系列の規模感から置いた仮置。
#:
#: 発注配送（ordering=True）は運送会社営業所の3軒と、GSのうち2軒に置いた。
#: 営業所は自社のトラックに入れる軽油を自分で手配する。**これも仮置**で、
#: どの先が発注で動くかは確認が取れていない。
ROSTER: tuple[CustomerSpec, ...] = (
    # --- 広島市中心部・24時間（朝は到達に時間、夜は短時間 → 時間帯係数が効く）
    CustomerSpec("c01", "広島中央SS（24h）", CustomerType.URBAN_GS_24H, 34.3925, 132.4553,
                 piping_inch=4, size="large", mix="city"),
    CustomerSpec("c02", "宇品港SS（24h）", CustomerType.URBAN_GS_24H, 34.3620, 132.4620,
                 size="large", mix="trunk"),      # 港湾でトラックが多い
    CustomerSpec("c03", "観音新町SS（24h）", CustomerType.URBAN_GS_24H, 34.3800, 132.4200,
                 ordering=True, size="medium", mix="trunk"),
    # --- 広島市中心部・日中のみ（時間枠×ラッシュの二重縛り）
    CustomerSpec("c04", "温品SS（日中）", CustomerType.URBAN_GS_DAY, 34.4160, 132.5150,
                 size="medium", mix="city"),
    CustomerSpec("c05", "祇園SS（日中）", CustomerType.URBAN_GS_DAY, 34.4450, 132.4560,
                 piping_inch=4, size="medium", mix="city"),
    # --- 山陰側・長距離（冬の灯油ピーク＋拘束時間）
    CustomerSpec("c06", "浜田SS", CustomerType.SANIN_GS, 34.8990, 132.0800,
                 piping_inch=4, size="medium", mix="sanin"),
    CustomerSpec("c07", "益田SS", CustomerType.SANIN_GS, 34.6760, 131.8430,
                 size="medium", mix="sanin"),
    # --- 運送会社営業所（早朝枠＋軽油だけ → 製品と槽の割当が効く）
    CustomerSpec("c08", "岩国装束運輸", CustomerType.CARRIER_DEPOT, 34.1720, 132.2200,
                 ordering=True, size="depot", mix="depot"),
    CustomerSpec("c09", "大竹陸運", CustomerType.CARRIER_DEPOT, 34.2380, 132.2210,
                 ordering=True, size="depot", mix="depot"),
    CustomerSpec("c10", "廿日市物流センター", CustomerType.CARRIER_DEPOT, 34.3480, 132.3320,
                 ordering=True, size="depot", mix="depot"),
    # --- 郊外GS（小規模・灯油比率が高い）
    CustomerSpec("c11", "周東SS", CustomerType.SUBURBAN_GS, 34.1020, 132.1360,
                 size="small", mix="suburb"),
    CustomerSpec("c12", "錦町SS", CustomerType.SUBURBAN_GS, 34.3300, 132.0000,
                 ordering=True, size="small", mix="suburb"),
    CustomerSpec("c13", "玖珂SS", CustomerType.SUBURBAN_GS, 34.1380, 132.1240,
                 piping_inch=4, size="small", mix="suburb"),
    CustomerSpec("c14", "湯来SS", CustomerType.SUBURBAN_GS, 34.4720, 132.2560,
                 size="small", mix="suburb"),
    CustomerSpec("c15", "吉和SS", CustomerType.SUBURBAN_GS, 34.4560, 132.1200,
                 size="small", mix="suburb"),
)


# ---------------------------------------------------------------------------
# 類型ごとの性格
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeProfile:
    """類型ごとのタンク構成・時間枠・荷卸しの口径。

    window は「受入を開始できる時刻の範囲」。close は締切であって、
    その時刻に着けばよい。運送会社営業所の 08:30 締切は、朝の出庫が
    終わったあとの受入を想定していない現場の運用そのもの。

    荷卸時間はここでは持たない。**降ろす量から決まる**（domain.unload_minutes）。
    以前は類型ごとの定数（30〜45分）を置いていて、2kL置くのも16kL置くのも
    同じ時間になっていた。配管の口径も類型では決まらない（系列の出自で決まる
    ので、名簿の1軒ずつに持たせてある）。
    """

    window_open: str
    window_close: str

    #: 油種 -> (タンク容量L, 平年の1日消費L)
    tanks: dict[Product, tuple[int, int]]


#: 類型ごとの受入時間枠。**タンクと消費量はここでは持たない**（規模×立地から割り出す）。
PROFILES: dict[CustomerType, tuple[str, str]] = {
    CustomerType.URBAN_GS_24H: ("05:00", "21:00"),
    CustomerType.URBAN_GS_DAY: ("08:00", "17:00"),
    CustomerType.SANIN_GS: ("07:00", "18:00"),
    CustomerType.CARRIER_DEPOT: ("05:00", "08:30"),
    CustomerType.SUBURBAN_GS: ("07:00", "19:00"),
}

# --- 二交代（docs/業務前提.md §5.2） ----------------------------------------
#
# **車と人は別の資源。** 1台を2人で回して24時間動かす。1人あたりの拘束は
# 13時間（引継ぎを考慮した値）なので、勤務帯は次のように置く。24時を跨ぐ
# 時刻は 25:00・29:00 のように1440を超える分で書く（domain.minutes_to_hhmm
# がそのまま「翌日の時刻」として出す）。
#
#: 勤務帯。**仮置**——実務の交代時刻は未確認。13時間ずつで24時間を覆い、
#: 引継ぎに1時間重ねた形にしてある。
SHIFT_WINDOWS: dict[str, tuple[str, str]] = {
    "day": ("04:00", "17:00"),     # 13時間
    "night": ("16:00", "29:00"),   # 16:00〜翌05:00。13時間
}

#: 夜に降ろせる先の類型。**ここが二交代の成否を決める**（設備 → 体制の順。実務）。
#: 元売は24時間営業SS向けに無人荷卸しを展開済み（§8）なので、24h SSだけを夜の
#: 行き先とする。営業所（05:00〜08:30）や日中のSSは夜間に受けない。
NIGHT_DELIVERABLE: frozenset[CustomerType] = frozenset({CustomerType.URBAN_GS_24H})

#: 夜も受ける先の受入枠。**仮置**——無人荷卸しの消防法要件は未確認（§9）。
NIGHT_PROFILE: tuple[str, str] = ("04:00", "29:00")

#: **需要は「月間いくら売るSSか」から割り出す**（実務・2026-09-16）。
#:
#: 月間70kL程度のSSは淘汰されている側で、運送会社が回る先の姿ではない
#: （粗利15円/Lとしても月105万円で、人件費・設備費を回収できない）。
#: 下限は月150kL（粗利225万円）に置いた。中250kLは全国平均どころ
#: （ガソリン55.6百万kL/年 ÷ 27,414SS ÷ 12 ≒ 月169kL、これに軽油・灯油の
#: SS扱い分を足すと月250kL前後）。**仮置。**
SIZE_MONTHLY_L: dict[str, int] = {
    "large": 450_000,     # 幹線沿い・市内の主力店
    "medium": 250_000,    # 全国平均どころ
    "small": 150_000,     # これでも人件費＋設備で手一杯
    "depot": 42_500,      # 運送会社営業所（自社トラック用の軽油だけ）
}

#: 稼働日数（月）。日曜休業の先も、月間販売量はこの日数で割って1日あたりにする。
OPERATING_DAYS_PER_MONTH = 25

#: 立地ごとの油種構成（年平均）。**軽油は輸送需要に比例する**（実務）ので、
#: 大型トラックが入りやすい幹線沿いほど厚い。灯油は寒冷地ほど厚い。
#:
#: 全国の年計（ガソリン49%・軽油36%・灯油15%）と、SS過疎地等実態調査の
#: 1SSあたり（ガソリン43%・軽油32%・灯油25%）の間に収まるように置いた。**仮置。**
MIX: dict[str, dict[str, float]] = {
    "city":   {"gasoline": 0.60, "gasoil": 0.20, "kerosene": 0.20},
    "trunk":  {"gasoline": 0.40, "gasoil": 0.45, "kerosene": 0.15},
    "sanin":  {"gasoline": 0.35, "gasoil": 0.30, "kerosene": 0.35},
    "suburb": {"gasoline": 0.43, "gasoil": 0.27, "kerosene": 0.30},
    "depot":  {"gasoline": 0.0, "gasoil": 1.0, "kerosene": 0.0},
}

#: ガソリンに占めるハイオクの比率。**生産量の統計から**（生産14.5%・販売部門出荷14.7%、
#: 経済産業省 生産動態統計「自動車用ガソリン 高級／並級」）。docs/統計の裏取り.md §2。
PREMIUM_SHARE_OF_GASOLINE = 0.145

#: 実在する地下タンクの大きさ（L）。**30kLが多く、古い先で10kL。分割は2kL単位**（実務）。
TANK_SIZES_L: tuple[int, ...] = (4_000, 6_000, 10_000, 20_000, 30_000)


#: ローリーに積んでいる荷卸しホースの口径（インチ）。**実務。**
#:
#: 4インチと3インチの2種類があり、**3インチが多い**（2026-09-11 運送会社への
#: 聞き取り）。乗務員の年齢が上がってきて、重い4インチのホースを避けるため。
#:
#: 荷卸しは自然落下で、経路の一番細いところが律速する。ホースが3インチなら
#: 需要家の配管が4インチでも流量は3インチ相当になり、**流下はおよそ倍かかる**
#: （4kLで5分 → 8分）。
#:
#: つまり乗務員の労働環境が設備の選択を決め、それが荷卸時間を通じて拘束時間と
#: 必要台数に返ってくる。4インチに替えれば荷卸しは速くなるが、乗務員の負担が
#: 増える。**体制の選択肢の1つ**として、ここを動かして比べられる。
#:
#: ただし4インチに替えて速くなるのは**配管も4インチの先（旧三菱系）だけ**。
#: 旧日石系は配管が律速するので、ホースを替えても変わらない。替える価値が
#: あるかは、配送先に旧三菱系がどれだけあるかで決まる。
HOSE_INCH = 3


#: 計画配送の周期（営業日）。**ほぼ3日に1回**（実務）。
#:
#: SSの月間売上はある程度一定で、たとえば月100kL売る先は稼働25日で
#: 1日4kL、それを3日に1回まとめて運ぶ（1回12kL前後）。名簿の冬の消費量
#: では、どの先のどのタンクも3日ぶんが空き容量に収まる（3.1日以上もつ）。
#:
#: **槽の単位で運ぶので、3日ぶん＋2kL（槽1本ぶんの持ち越し）が収まる必要がある。**
#: 収まらないタンクは周期のうちに空になる（test_every_tank_holds_a_cycle_and_a_compartment）。
PLANNED_INTERVAL_DAYS = 3

#: 日々の消費の揺れ（計画量に対する標準偏差の比）。**仮置。**
#:
#: 月の売上がほぼ一定なので、揺れは小さい。冬の灯油は気温との相関が高いが、
#: 寒波という**事象**はシグナルで与えるので、ここには平時の揺れだけを置く。
CONSUMPTION_NOISE = 0.12

#: 日曜に休業する類型。休業日は売れないので消費もしない。**仮置。**
#:
#: 日曜の配送量は平日の6割程度（実務）。運送会社営業所と郊外GSを休業に
#: すると、計算上の日曜の計画配送は平日の5〜6割になる。
CLOSED_ON_SUNDAY: dict[CustomerType, bool] = {
    CustomerType.URBAN_GS_24H: False,
    CustomerType.URBAN_GS_DAY: False,
    CustomerType.SANIN_GS: False,
    CustomerType.CARRIER_DEPOT: True,
    CustomerType.SUBURBAN_GS: True,
}

#: 周期を数える起点（日曜）。どの日付でも同じ周期の位置が出るように固定する。
EPOCH = datetime.date(2026, 1, 4)


#: 季節ごとの消費乗数（秋＝1.00）。平年の季節差であって、寒波などの事象ではない。
#:
#: **全国の月別販売量（石油統計 年報・2023年）から計算した**（2026-09-16）。
#: 出どころと計算は docs/統計の裏取り.md §2.1。
#:
#:     灯油     冬2.58 / 春1.05 / 夏0.46
#:     ガソリン 冬0.98 / 春0.99 / 夏1.07
#:     軽油     冬0.93 / 春0.96 / 夏0.97
#:
#: 以前は灯油の夏を0.15・春を0.8と置いていたが、統計より低すぎた（夏でも給湯・
#: 農業用・業務用が残る）。**冬の2.6倍は統計の2.58とほぼ一致していた。**
SEASON_MULTIPLIER: dict[str, dict[Product, float]] = {
    "winter": {Product.KEROSENE: 2.58, Product.REGULAR: 0.98, Product.PREMIUM: 0.98, Product.GASOIL: 0.93},
    "spring": {Product.KEROSENE: 1.05, Product.REGULAR: 0.99, Product.PREMIUM: 0.99, Product.GASOIL: 0.96},
    "summer": {Product.KEROSENE: 0.46, Product.REGULAR: 1.07, Product.PREMIUM: 1.07, Product.GASOIL: 0.97},
    "autumn": {Product.KEROSENE: 1.0, Product.REGULAR: 1.0, Product.PREMIUM: 1.0, Product.GASOIL: 1.0},
}

#: 発注点を容量の何割に置くか。運送会社営業所は切らすと車が出ないので高め。
REORDER_RATIO: dict[CustomerType, float] = {
    CustomerType.URBAN_GS_24H: 0.25,
    CustomerType.URBAN_GS_DAY: 0.25,
    CustomerType.SANIN_GS: 0.30,   # 遠いので早めに手を打たないと間に合わない
    CustomerType.CARRIER_DEPOT: 0.35,
    CustomerType.SUBURBAN_GS: 0.28,
}

#: 満タンまで入れずに残す安全代（容量比）。受入側の実務上の余裕。
ULLAGE_RATIO = 0.05


# ---------------------------------------------------------------------------
# 車両・乗務員
# ---------------------------------------------------------------------------

#: 保有するローリー。**容量は実在する型式だけを使う。**
#:
#: 単車       14kL / 16kL
#: トレーラー 18kL / 20kL / 24kL / 26kL
#:
#: 最初の実装では 10kL の車を置いていたが、そんなローリーは無い。
#: 作り物の車で組んだ計画は、作り物の制約しか検証しない。
#:
#: 単車を混ぜてあるのは容量の都合ではなく、**トレーラーが物理的に入れない
#: 需要家があるから**。狭い道・小さい構内・切り返しの余地の無さで決まる。
#: 単車だけ、あるいはトレーラーだけの編成にすると、この制約が消えてしまう。
#:
#: 台数と構成そのものは、本来この生成器が決めることではない。
#: 元売りが半期（4-9月／10-3月）ごとに専属契約の台数を提示し、運送会社が
#: それに非専属を足す。**何台要るかは体制設計層が掃引で出す答え**であって、
#: ここはその掃引の出発点にすぎない。
#:
#: **槽は 2kL と 4kL の組合せだけ。** 法令（危険物の規制に関する政令 第15条）は
#: 1室4,000L以下と定め、実際の車はこの2種類を組み合わせて作られている。
#: 以前は 5,000L や 3,000L の槽を置いていたが、どちらも存在しない。
#: 2号車の 2/4/4/2/4/4 は、実在する20kL車の6槽タイプそのもの。
FLEET: tuple[tuple[str, str, VehicleClass, tuple[int, ...]], ...] = (
    ("v1", "1号車", VehicleClass.TRAILER, (2_000, 2_000, 4_000, 2_000, 4_000, 2_000, 4_000, 4_000)),  # 24kL 8槽
    ("v2", "2号車", VehicleClass.TRAILER, (2_000, 4_000, 2_000, 4_000, 4_000, 2_000, 2_000)),  # 20kL 7槽
    ("v3", "3号車", VehicleClass.TRAILER, (2_000, 4_000, 2_000, 4_000, 2_000, 4_000)),         # 18kL 6槽
    # **単車は2kL槽が多い**（需要に対応しやすいように）。2kL槽が4kL槽より少ない
    # 区割りはあり得ない（実務・2026-09-16）。全部同じ大きさの槽の車も存在しない。
    # 以前は 4/4/4/4 と 2/4/4/4 を置いていた（どちらもあり得ない）
    ("v4", "4号車", VehicleClass.RIGID, (2_000, 2_000, 2_000, 2_000, 4_000, 4_000)),           # 16kL 6槽
    ("v5", "5号車", VehicleClass.RIGID, (2_000, 2_000, 2_000, 4_000, 4_000)),                  # 14kL 5槽
    # **6台目は掃引の答え**（2026-09-16）。SSの規模を実務に寄せたら、冬の1週間で
    # 5台では木曜に浜田SSが落ちた。6台で7日とも落ちゼロ。台数は需要から出る。
    ("v6", "6号車", VehicleClass.TRAILER, (2_000, 2_000, 4_000, 2_000, 4_000, 2_000, 4_000, 4_000)),  # 24kL 8槽
)

DRIVER_NAMES: tuple[str, ...] = ("乗務員A", "乗務員B", "乗務員C", "乗務員D", "乗務員E")


def _driver_name(i: int) -> str:
    """i番目の乗務員の名前。A〜Zで足りなくなったらAA・ABと続ける。

    二交代では人数が台数の2倍になるので、5人ぶんの名簿では足りない。
    """
    if i < len(DRIVER_NAMES):
        return DRIVER_NAMES[i]
    letters = ""
    k = i
    while True:
        letters = chr(ord("A") + k % 26) + letters
        k = k // 26 - 1
        if k < 0:
            break
    return f"乗務員{letters}"

#: 類型ごとに、トレーラーが入れるか。**荷役条件のマスターに載っている情報**。
#:
#: 郊外GSはタンクが小さく構内も狭い。運送会社の営業所も、車庫の配置次第で
#: 大型が回せないところがある。単車はそういう先へ優先的に配車される。
ACCEPTS_TRAILER: dict[CustomerType, bool] = {
    CustomerType.URBAN_GS_24H: True,
    CustomerType.URBAN_GS_DAY: True,
    CustomerType.SANIN_GS: True,
    CustomerType.CARRIER_DEPOT: True,
    CustomerType.SUBURBAN_GS: False,   # 郊外GSは単車のみ
}

#: 類型に関わらず単車しか入れない需要家。個別の事情（前面道路が狭い等）。
#: 実務では荷役条件マスターに1軒ずつ持っている情報を、ここで代用している。
RIGID_ONLY_CUSTOMERS: frozenset[str] = frozenset({"c08"})   # 岩国装束運輸


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------


#: 季節をならした平均の乗数（油種ごと）。月間販売量は年平均なので、平年（秋=1.00）の
#: 基準値に直すときにこれで割る。こうしないと、季節を掛けた合計が年間販売量とずれる。
_SEASON_AVG: dict[Product, float] = {
    p: sum(m[p] for m in SEASON_MULTIPLIER.values()) / len(SEASON_MULTIPLIER)
    for p in Product
}


@functools.lru_cache(maxsize=None)
def _base_daily(spec: CustomerSpec) -> dict[Product, int]:
    """平年（秋）の1日消費（油種ごと・L）。**規模×立地から割り出す。**

        月間販売量 × 立地の油種構成 ÷ 稼働25日 ÷ 季節の年平均

    ガソリンはさらにハイオク（14.5%・生産量の統計）とレギュラーに分ける。
    """
    monthly = SIZE_MONTHLY_L[spec.size]
    mix = MIX[spec.mix]
    out: dict[Product, int] = {}
    for key, share in mix.items():
        if share <= 0:
            continue
        daily = monthly * share / OPERATING_DAYS_PER_MONTH
        if key == "gasoline":
            pairs = ((Product.PREMIUM, PREMIUM_SHARE_OF_GASOLINE),
                     (Product.REGULAR, 1 - PREMIUM_SHARE_OF_GASOLINE))
        else:
            pairs = ((Product.GASOIL if key == "gasoil" else Product.KEROSENE, 1.0),)
        for product, part in pairs:
            out[product] = int(round(daily * part / _SEASON_AVG[product]))
    return out


@functools.lru_cache(maxsize=None)
def _tanks_and_interval(spec: CustomerSpec) -> tuple[dict[Product, int], int]:
    """タンク容量（油種ごと）と、配送周期（営業日）。**規模から決まる。**

    タンクは「一番売れる季節の3日ぶん ＋ 刻み1つ」が空き容量に収まる最小の
    大きさを選ぶ。**大型店はそれが30kLでも足りない**ので、その場合は2日ぶん、
    さらに1日ぶんと落として選ぶ。**大型店ほど配送が頻繁になる**（実務）。

    周期は、選んだタンクに何日ぶん入るかで決まる（最長3日）。油種ごとに違う
    ので、いちばん短いものに合わせる（1回の訪問で全油種を降ろすため）。
    """
    peak = {p: max(v * SEASON_MULTIPLIER[s][p] for s in SEASON_MULTIPLIER)
            for p, v in _base_daily(spec).items()}
    caps: dict[Product, int] = {}
    for product, daily in peak.items():
        step = DROP_STEP_BY_PRODUCT[product]
        for want_days in (3, 2, 1):
            need = daily * want_days + step
            fit = [c for c in TANK_SIZES_L if c - round(c * ULLAGE_RATIO) >= need]
            if fit:
                caps[product] = min(fit)
                break
        else:
            caps[product] = max(TANK_SIZES_L)
    interval = PLANNED_INTERVAL_DAYS
    for product, cap in caps.items():
        usable = cap - round(cap * ULLAGE_RATIO)
        days = int((usable - DROP_STEP_BY_PRODUCT[product]) // max(1, peak[product]))
        interval = min(interval, max(1, days))
    # **1回の配送は1台に載らなければならない。** 積み分け（1軒を2台で運ぶ）は
    # モデルが持っていないし、実務でも大型店は台数ではなく頻度で吸収する。
    # トレーラーが入れない先は単車の容量で見る。
    biggest = max(
        sum(comps) for _vid, _name, vclass, comps in FLEET
        if ACCEPTS_TRAILER[spec.type] and spec.id not in RIGID_ONLY_CUSTOMERS
        or vclass is VehicleClass.RIGID
    )
    per_day = sum(peak.values())
    if per_day > 0:
        interval = min(interval, max(1, int(biggest // per_day)))
    return caps, interval


def _interval_days(spec: CustomerSpec) -> int:
    """この先の配送周期（営業日）。大型店ほど短い。"""
    return _tanks_and_interval(spec)[1]


def _planned_daily(spec: CustomerSpec, season: str) -> dict[Product, int]:
    """計画配送で見込む1日の消費（油種ごと）。元売の需要見込みにあたる。"""
    mult = SEASON_MULTIPLIER[season]
    return {p: int(round(base * mult[p])) for p, base in _base_daily(spec).items()}


def _planned_drop(spec: CustomerSpec, season: str) -> tuple[tuple[Product, int], ...]:
    """計画配送1回で運ぶ量の**平均的な姿**（油種ごと）。曜日を均す重さにだけ使う。

    実際の日ごとの量は端数の持ち越しで揺れる（_planned_drop_on）。ここでは
    3日ぶんの見込みを、その油種の刻み（主要3油種4kL・ハイオク2kL）に丸める。
    """
    out = []
    for p, v in _planned_daily(spec, season).items():
        step = DROP_STEP_BY_PRODUCT[p]
        q = int(v * _interval_days(spec) / step + 0.5) * step
        if q > 0:
            out.append((p, q))
    return tuple(out)


def _work_minutes(spec: CustomerSpec, season: str) -> int:
    """計画配送1回が車の時間をどれだけ食うか（分）。

    油槽所との往復 ＋ 荷卸し ＋ 積込。**曜日を均すときの重さはこれ。**
    """
    drop = dict(_planned_drop(spec, season))
    one_way = leg_minutes(MARIFU.lat, MARIFU.lon, spec.lat, spec.lon)
    unload = unload_minutes(drop, min(HOSE_INCH, spec.piping_inch))
    return 2 * one_way + unload + MARIFU.load_min


def _phases(season: str) -> dict[str, int]:
    """計画配送の先ごとに、周期の起点をどこに置くか。

    **日々の車の時間がならされるように割り付ける。** 3日周期の先を全部同じ
    日に置くと、その日だけ3倍になる。重い先から順に、いちばん軽い起点へ
    入れていく（大きいものから詰める均し方）。

    **重さは量ではなく車の時間**（往復＋荷卸し＋積込、_work_minutes）。
    最初は量で均していたが、秋に灯油が減って量の順番が入れ替わり、浜田と
    益田（どちらも片道2時間半の山陰）が同じ日に割り付けられた。1台で1日に
    両方は回れないので、需要の少ない秋の日曜に冬の平日と同じ台数が要る、
    という答えが出た。縛っているのは量ではなく時間なので、時間で均す。

    日曜休業の先と毎日営業の先は、営業日の数え方が違う（6日と7日）ので、
    別々に均す。休業の先は営業日6日÷3日＝週2回で、曜日が固定される
    （月木・火金・水土のいずれか）。
    """
    phases: dict[str, int] = {}
    for closed in (False, True):
        for interval in sorted({_interval_days(spec) for spec in ROSTER}):
            group = [
                spec for spec in ROSTER
                if not spec.ordering and CLOSED_ON_SUNDAY[spec.type] == closed
                and _interval_days(spec) == interval
            ]
            group.sort(key=lambda spec: (-_work_minutes(spec, season), spec.id))
            load = [0] * interval
            for spec in group:
                k = min(range(interval), key=lambda i: (load[i], i))
                phases[spec.id] = k
                load[k] += _work_minutes(spec, season)
    return phases


def _operating_day(n: int, closed_sunday: bool) -> tuple[int, bool]:
    """起点からの日数 n を、その需要家の営業日の通し番号に直す。

    Returns:
        (通し番号, 今日営業しているか)。休業日は「次の営業日」の番号を返す。
        休業日は売れないので、次の営業日の朝と在庫が同じになるため。
    """
    if not closed_sunday:
        return n, True
    sundays = n // 7 + 1                 # 起点（日曜）から今日までの日曜の数
    if n % 7 == 0:
        return n - sundays + 1, False
    return n - sundays, True


def _carried(product: Product, daily_l: int, deliveries: int,
             interval: int = PLANNED_INTERVAL_DAYS) -> int:
    """deliveries 回の計画配送を終えた時点で、**刻みに満たず運びきれていない量**（L）。

    積込は槽の単位で、需要の多い油種は4kL槽の単位（実務）なので、3日ぶん5.7kLは
    毎回そのまま運べない。毎回切り捨てると端数が消え、運ぶ量が売れる量の55%まで
    落ちた（2026-09-16）。実際は端数ぶんタンクが低いまま残り、次の配送でまとめて
    運ぶ。周期の番号から決まるので乱数は使わず、同じ日付なら同じ量になる。
    """
    return daily_l * interval * deliveries % DROP_STEP_BY_PRODUCT[product]


def _planned_drop_on(spec: CustomerSpec, season: str, k: int) -> tuple[tuple[Product, int], ...]:
    """k 回目の計画配送で運ぶ量。**3日ぶんに前回までの端数を足し、刻みに切り捨てる。**

    長い目で見れば運ぶ量は売れる量と等しい（毎回の量は4kL・8kL・4kL…と揺れる）。
    刻みに満たない油種はその回は運ばず、次の回にまとめる（ハイオクなど）。
    """
    interval = _interval_days(spec)
    out = []
    for p, d in _planned_daily(spec, season).items():
        q = whole_drops_l(p, d * interval + _carried(p, d, k - 1, interval))
        if q > 0:
            out.append((p, q))
    return tuple(out)


def _cycle_tanks(
    rng: random.Random, spec: CustomerSpec, season: str, days_since: int,
    carried: dict[Product, int] | None = None,
) -> tuple[Tank, ...]:
    """計画配送の先のタンク。**在庫は周期から決まる。**

    前回の配送で満タン近く（空き容量の安全代を残す）まで入れ、そこから
    days_since 営業日ぶん売れた、とする。以前は在庫を三角分布の乱数で
    振っていたが、それだと計画配送の量まで日によってばらつき、「月〜土は
    ほぼ一定量」という実態から外れていた。揺れは消費の日々の揺れだけ。

    carried は前回までに**槽の単位に満たず運びきれなかった量**。そのぶん
    タンクは満タンより低い（_carried）。
    """
    caps, _iv = _tanks_and_interval(spec)
    daily = _planned_daily(spec, season)
    reorder_ratio = REORDER_RATIO[spec.type]
    carried = carried or {}
    tanks = []
    for product, capacity in caps.items():
        ullage = int(round(capacity * ULLAGE_RATIO))
        factor = max(0.0, rng.gauss(1.0, CONSUMPTION_NOISE))
        sold = int(round(daily[product] * days_since * factor))
        tanks.append(
            Tank(
                product=product,
                capacity_l=capacity,
                level_l=max(0, capacity - ullage - carried.get(product, 0) - sold),
                reorder_l=int(round(capacity * reorder_ratio)),
                daily_use_l=daily[product],
                ullage_margin_l=ullage,
            )
        )
    return tuple(tanks)


def generate(
    *,
    seed: int = 20261015,
    date: str = "2026-10-15",
    season: str = "winter",
    absent_drivers: tuple[str, ...] = (),
    declined_customers: tuple[str, ...] = (),
    sold_more: tuple[SoldMore, ...] = (),
    fleet_size: int | None = None,
    shifts: int = 1,
) -> Scenario:
    """1日ぶんの配車問題を作る。

    Args:
        seed: 乱数種。同じ seed は完全に同じ日を返す。
        date: 対象日（表示と記録のため。計算には使わない）。
        season: winter / spring / summer / autumn。
        absent_drivers: 欠勤する乗務員のID。デモ4（乗務員欠勤）で使う。
        declined_customers: 今日の受入を見送る需要家のID。値下げ前に在庫を
            下げたい需要家など。**事象なので外から与える**（生成器が勝手に
            作ると、見送りを注入したから計画が変わったのか、たまたまそういう
            日だったのかが切り分けられなくなる）。
        fleet_size: 出せるローリーの台数。**台数は需要から出る答え**なので、
            体制設計（scripts/size_fleet.py）がここを振って「欠品ゼロで回る
            最小の台数」を探す。既定は保有の5台。
        sold_more: 見込みより売れた先（連休・値上げ前・寒波）。**朝の計画より前に
            分かっていた**ときに使う。日中に分かったなら replan に渡す。在庫を下げる
            のは乱数を引いたあとなので、事象の無い先は事象なしの日と完全に同じになる。
            在庫が発注点を割れば、計画配送の日でない先には発注が入る。

    Returns:
        Scenario。ソルバーにもエージェント層にもこれを渡す。

    Raises:
        ValueError: 未知の季節。黙って autumn にフォールバックしない。
            季節の綴り違いで灯油需要が2.6倍から1.0倍に化けるのは、
            気づかないまま計画が変わる最悪の壊れ方なので、必ず落とす。
    """
    if season not in SEASON_MULTIPLIER:
        raise ValueError(
            f"未知の季節: {season!r}（{'/'.join(SEASON_MULTIPLIER)} のいずれか）"
        )

    known = {spec.id for spec in ROSTER}
    unknown = [cid for cid in declined_customers if cid not in known]
    if unknown:
        # 綴り違いを黙って通すと、見送りが効かないまま「見送りを入れた計画」
        # として読まれる。
        raise ValueError(f"名簿に無い需要家: {unknown}")
    unknown = sorted({e.customer_id for e in sold_more} - known)
    if unknown:
        raise ValueError(f"名簿に無い需要家: {unknown}")
    events_by_customer: dict[str, list[SoldMore]] = {}
    for e in sold_more:
        events_by_customer.setdefault(e.customer_id, []).append(e)

    if shifts not in (1, 2):
        raise ValueError(f"交代は1か2（{shifts}）")
    rng = random.Random(seed)
    day = datetime.date.fromisoformat(date)
    n = (day - EPOCH).days
    phases = _phases(season)

    # **日曜は二交代を解除する**（実務・§5.2）。SSに休むところが多く配送量が
    # 減るので、前もって分かっている周期として体制を組める。逼迫した日だけ
    # 二交代にする、という当日の切り替えはできない（設備と人事が先に要る）。
    two_shift = shifts == 2 and day.weekday() != 6

    customers = []
    for spec in ROSTER:
        window_open, window_close = PROFILES[spec.type]
        if two_shift and spec.type in NIGHT_DELIVERABLE:
            # 夜も受ける先。二交代の夜勤が行けるのはここだけ（設備 → 体制の順）
            window_open, window_close = NIGHT_PROFILE
        op, open_today = _operating_day(n, CLOSED_ON_SUNDAY[spec.type])
        planned: tuple[tuple[Product, int], ...] = ()
        tomorrow_planned: tuple[tuple[Product, int], ...] = ()
        if spec.ordering:
            # 発注配送の先: 自分の在庫を見て不定期に頼む。**計画配送の一定量の
            # 上に乗る、振れる部分**にあたる。
            #
            # 在庫の減り方は計画配送の先と同じ（消費に見合う）で、違うのは
            # **頼む日が決まっていない**こと。周期のどこにいるかを日ごとに振り、
            # 周期の終わりに当たった日に3日ぶんを頼む。平均すれば消費量どおり、
            # 日ごとには振れる。
            #
            # 最初は在庫を乱数で振り「2日先に発注点を割るなら発注」としたが、
            # 1日2.2kLしか使わない営業所が毎日15kLずつ頼み、発注が計画配送より
            # 大きくなった。量が消費と釣り合わない在庫の作り方は使えない。
            interval = _interval_days(spec)
            days_since = rng.randint(1, interval)
            # 前回までの端数は周期の番号を持たないので、刻み未満を別の乱数で振る
            # （主の乱数の流れは変えない）。平均すれば頼む量は売れる量と釣り合う。
            side = random.Random(f"{seed}|{date}|{spec.id}")
            carried = {p: side.randrange(DROP_STEP_BY_PRODUCT[p]) for p in _base_daily(spec)}
            tanks = sell_more(_cycle_tanks(rng, spec, season, days_since, carried),
                              events_by_customer.get(spec.id, ()))
            # 頼む量も刻みの単位（主要3油種は4kL槽）
            orders = (
                tuple((t.product, whole_drops_l(t.product, t.topup_l)) for t in tanks
                      if whole_drops_l(t.product, t.topup_l) > 0)
                if open_today and days_since == interval else ()
            )
        else:
            # 計画配送の先: 周期（ほぼ3日に1回）で運ぶ。今日が何日目かで在庫と
            # 計画量が決まる。周期の途中で発注点を割ったら、需要家から発注が入る
            # （計画配送の上に発注を乗せる、の実際の形）。
            phase = phases[spec.id]
            interval = _interval_days(spec)
            days_since = ((op - phase - 1) % interval) + 1
            due = open_today and (op - phase) % interval == 0
            # 直近に済んだ配送の番号。計画配送の日はその朝はまだ運んでいない
            k = (op - phase) // interval
            done = k - 1 if (op - phase) % interval == 0 else k
            daily = _planned_daily(spec, season)
            carried = {p: _carried(p, d, done, interval) for p, d in daily.items()}
            tanks = sell_more(_cycle_tanks(rng, spec, season, days_since, carried),
                              events_by_customer.get(spec.id, ()))
            if due:
                planned = _planned_drop_on(spec, season, k)
            orders = (
                tuple((t.product, whole_drops_l(t.product, t.topup_l)) for t in tanks
                      if t.is_below_reorder and whole_drops_l(t.product, t.topup_l) > 0)
                if open_today and not due else ()
            )
            # 明日が計画配送の日なら、その量。**小口の発注は翌日の計画配送を前倒しして
            # 合わせ、当日1回で運ぶ**（実務）ので、組み直しがこれを使う。
            op1, open1 = _operating_day(n + 1, CLOSED_ON_SUNDAY[spec.type])
            if open1 and op1 != op and (op1 - phase) % interval == 0:
                tomorrow_planned = _planned_drop_on(spec, season, (op1 - phase) // interval)
        customers.append(
            Customer(
                id=spec.id,
                name=spec.name,
                type=spec.type,
                lat=spec.lat,
                lon=spec.lon,
                tanks=tanks,
                window_open_min=hhmm_to_minutes(window_open),
                window_close_min=hhmm_to_minutes(window_close),
                accepts_trailer=(
                    ACCEPTS_TRAILER[spec.type]
                    and spec.id not in RIGID_ONLY_CUSTOMERS
                ),
                bore_inch=min(HOSE_INCH, spec.piping_inch),
                ordering=spec.ordering,
                orders=orders,
                planned=planned,
                tomorrow_planned=tomorrow_planned,
                open_today=open_today,
                accepts_delivery=spec.id not in declined_customers,
            )
        )
    # 日数の範囲・タンクの無い油種を黙って通さない（名簿の綴りは上で確かめた）
    validate_events(customers, tuple(sold_more))

    # **台数は需要から出る答え**なので、外から振れるようにしてある（size_fleet が掃引する）。
    # 保有の5台を超える分は、同じ車格を順に足す（車格の構成比は変えない）。
    count = len(FLEET) if fleet_size is None else fleet_size
    if count < 1:
        raise ValueError(f"台数は1台以上（{fleet_size}）")

    # **台数（車）と人数（乗務員）は別に数える。** 二交代では1台に昼勤と夜勤の
    # 2人が付くので、乗務員は台数の2倍になる（実務・§5.2）。
    per_vehicle = 2 if two_shift else 1
    drivers = tuple(
        Driver(
            id=f"d{i + 1}",
            name=_driver_name(i),
            available=f"d{i + 1}" not in absent_drivers,
        )
        for i in range(count * per_vehicle)
    )

    # **同型の車が隣り合うように並べる。** ソルバーの対称性つぶしは隣り合う
    # 同型の組にしか効かないので、並びが崩れていると探索が無駄に広がる。
    patterns = [FLEET[i % len(FLEET)] for i in range(count)]
    patterns.sort(key=lambda t: (t[2].value, sum(t[3]), t[3]))
    vehicles = []
    for i in range(count):
        _vid, _vname, vclass, comps = patterns[i]
        base = f"{i + 1}号車（{VEHICLE_CLASS_JA[vclass]} {sum(comps) // 1000}kL）"
        if not two_shift:
            vehicles.append(
                Vehicle(
                    id=f"v{i + 1}",
                    name=base,
                    vehicle_class=vclass,
                    compartments_l=comps,
                    driver_id=f"d{i + 1}",
                )
            )
            continue
        # 二交代。**1台を勤務帯ごとの2行で表す**（同じ車・別の乗務員・重ならない
        # 時間帯）。台数は Vehicle.physical で数える。
        for k, shift in enumerate(("day", "night")):
            lo, hi = SHIFT_WINDOWS[shift]
            vehicles.append(
                Vehicle(
                    id=f"v{i + 1}{'ab'[k]}",
                    name=f"{base}{'昼勤' if shift == 'day' else '夜勤'}",
                    vehicle_class=vclass,
                    compartments_l=comps,
                    driver_id=f"d{i * 2 + k + 1}",
                    round_the_clock=True,
                    physical_id=f"v{i + 1}",
                    shift=shift,
                    duty_window_min=(hhmm_to_minutes(lo), hhmm_to_minutes(hi)),
                )
            )
    vehicles = tuple(vehicles)

    return Scenario(
        date=date,
        season=season,
        seed=seed,
        depot=MARIFU_NIGHT if two_shift else MARIFU,
        customers=tuple(customers),
        vehicles=vehicles,
        drivers=drivers,
        notes=[
            "合成データ。地理と制約の形は実際の二次配送に合わせ、在庫と消費は生成値。",
            f"season={season} / seed={seed}",
        ],
    )
