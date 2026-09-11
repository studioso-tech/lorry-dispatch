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
    - 振る: 当日の在庫水準・消費量のばらつき
      → 「今日どこが切迫しているか」が日替わりになり、再計画の意味が出る。

    seed を変えれば別の日が出る。seed が同じなら完全に同じ日が出る。
    これは回帰テスト（W6）の前提であり、シードを固定しない生成は禁止。

季節:
    冬は灯油が跳ねる（度日法の乗数は W4 のシグナル層で扱う）。ここでは
    平年の季節差だけを入れる。寒波という**事象**は生成器ではなくシグナルで
    与える。生成器に事象を混ぜると、「寒波を注入したから再計画された」のか
    「たまたまそういう日が生成された」のかが切り分けられなくなる。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from core.domain import (
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
)

# ---------------------------------------------------------------------------
# 油槽所
# ---------------------------------------------------------------------------

#: 麻里布（岩国・和木）。ENEOS 麻里布製油所に併設の油槽所という設定。
#: 04:00 開所は早朝便（運送会社営業所の 05:00 枠）に間に合わせるため。
MARIFU = Depot(
    id="depot_marifu",
    name="麻里布油槽所",
    lat=34.2028,
    lon=132.2244,
    open_min=hhmm_to_minutes("04:00"),
    close_min=hhmm_to_minutes("22:00"),
    load_min=40,
)


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


#: 重要家15軒。引き継ぎ文書 §4.2 の4類型を「1回のデモで全制約系が見える」よう配分。
#: 座標は実在の市町の位置に合わせてあるが、事業所そのものは架空である。
#:
#: 配管4インチ（旧三菱系）はGS12軒のうち4軒に置いた。**どの先がそうかは架空**
#: で、比率（約1/3）も合流時の系列の規模感から置いた仮置。
ROSTER: tuple[CustomerSpec, ...] = (
    # --- 広島市中心部・24時間（朝は到達に時間、夜は短時間 → 時間帯係数が効く）
    CustomerSpec("c01", "広島中央SS（24h）", CustomerType.URBAN_GS_24H, 34.3925, 132.4553, piping_inch=4),
    CustomerSpec("c02", "宇品港SS（24h）", CustomerType.URBAN_GS_24H, 34.3620, 132.4620),
    CustomerSpec("c03", "観音新町SS（24h）", CustomerType.URBAN_GS_24H, 34.3800, 132.4200),
    # --- 広島市中心部・日中のみ（時間枠×ラッシュの二重縛り）
    CustomerSpec("c04", "温品SS（日中）", CustomerType.URBAN_GS_DAY, 34.4160, 132.5150),
    CustomerSpec("c05", "祇園SS（日中）", CustomerType.URBAN_GS_DAY, 34.4450, 132.4560, piping_inch=4),
    # --- 山陰側・長距離（冬の灯油ピーク＋拘束時間）
    CustomerSpec("c06", "浜田SS", CustomerType.SANIN_GS, 34.8990, 132.0800, piping_inch=4),
    CustomerSpec("c07", "益田SS", CustomerType.SANIN_GS, 34.6760, 131.8430),
    # --- 運送会社営業所（早朝枠＋軽油中心 → 製品と槽の割当が効く）
    CustomerSpec("c08", "岩国装束運輸", CustomerType.CARRIER_DEPOT, 34.1720, 132.2200),
    CustomerSpec("c09", "大竹陸運", CustomerType.CARRIER_DEPOT, 34.2380, 132.2210),
    CustomerSpec("c10", "廿日市物流センター", CustomerType.CARRIER_DEPOT, 34.3480, 132.3320),
    # --- 郊外GS（灯油比率が高くタンクが小さい → 在庫トリガーが頻繁に発火）
    CustomerSpec("c11", "周東SS", CustomerType.SUBURBAN_GS, 34.1020, 132.1360),
    CustomerSpec("c12", "錦町SS", CustomerType.SUBURBAN_GS, 34.3300, 132.0000),
    CustomerSpec("c13", "玖珂SS", CustomerType.SUBURBAN_GS, 34.1380, 132.1240, piping_inch=4),
    CustomerSpec("c14", "湯来SS", CustomerType.SUBURBAN_GS, 34.4720, 132.2560),
    CustomerSpec("c15", "吉和SS", CustomerType.SUBURBAN_GS, 34.4560, 132.1200),
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


#: 類型ごとのタンク構成。数字は国内のGS地下タンクの実勢に寄せてある。
#:
#: **1サイト合計 20〜40kL** が目安。ここを大きく取りすぎると、1軒の補給で
#: ローリー1台を使い切る計画になり、車が何台あっても足りなくなる。
#: 消費量は「満タンから約10日で使い切る」水準に置いてあり、発注点（容量の
#: 25〜35%）まで約7日。2日先読み（DELIVERY_LOOKAHEAD_DAYS）で拾うと、
#: 15軒のうち日に5〜8軒が対象になる。これが二次配送の実際の頻度に近い。
PROFILES: dict[CustomerType, TypeProfile] = {
    CustomerType.URBAN_GS_24H: TypeProfile(
        window_open="05:00",
        window_close="21:00",
        tanks={
            Product.REGULAR: (15_000, 1_700),
            Product.PREMIUM: (5_000, 420),
            Product.GASOIL: (8_000, 800),
            Product.KEROSENE: (4_000, 350),
        },
    ),
    CustomerType.URBAN_GS_DAY: TypeProfile(
        window_open="08:00",
        window_close="17:00",
        tanks={
            Product.REGULAR: (12_000, 1_300),
            Product.PREMIUM: (4_000, 320),
            Product.GASOIL: (6_000, 600),
            Product.KEROSENE: (4_000, 350),
        },
    ),
    CustomerType.SANIN_GS: TypeProfile(
        window_open="07:00",
        window_close="18:00",
        tanks={
            Product.REGULAR: (12_000, 1_200),
            Product.GASOIL: (10_000, 1_000),
            # 山陰側は冬の灯油が主役。タンクも大きい。
            Product.KEROSENE: (20_000, 900),
        },
    ),
    CustomerType.CARRIER_DEPOT: TypeProfile(
        window_open="05:00",
        window_close="08:30",
        tanks={
            # 営業所は軽油だけ。まとまった量を、朝の出庫前に入れる。
            Product.GASOIL: (20_000, 2_200),
        },
    ),
    CustomerType.SUBURBAN_GS: TypeProfile(
        window_open="07:00",
        window_close="19:00",
        tanks={
            Product.REGULAR: (8_000, 750),
            Product.GASOIL: (6_000, 550),
            # 郊外は灯油比率が高いのにタンクが小さい → 発注点割れが頻発する
            Product.KEROSENE: (8_000, 620),
        },
    ),
}


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


#: 季節ごとの消費乗数。平年の季節差であって、寒波などの事象ではない。
#: 灯油が冬に跳ね、夏はほぼ止まる。ガソリンは夏の行楽で少し伸びる。
SEASON_MULTIPLIER: dict[str, dict[Product, float]] = {
    "winter": {Product.KEROSENE: 2.6, Product.REGULAR: 0.95, Product.PREMIUM: 0.95, Product.GASOIL: 1.0},
    "spring": {Product.KEROSENE: 0.8, Product.REGULAR: 1.0, Product.PREMIUM: 1.0, Product.GASOIL: 1.0},
    "summer": {Product.KEROSENE: 0.15, Product.REGULAR: 1.1, Product.PREMIUM: 1.15, Product.GASOIL: 1.05},
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
    ("v1", "1号車", VehicleClass.TRAILER, (2_000, 4_000, 4_000, 2_000, 4_000, 4_000, 4_000)),  # 24kL 7槽
    ("v2", "2号車", VehicleClass.TRAILER, (2_000, 4_000, 4_000, 2_000, 4_000, 4_000)),         # 20kL 6槽
    ("v3", "3号車", VehicleClass.TRAILER, (2_000, 4_000, 4_000, 2_000, 4_000, 2_000)),         # 18kL 6槽
    ("v4", "4号車", VehicleClass.RIGID, (4_000, 4_000, 4_000, 4_000)),                         # 16kL 4槽
    ("v5", "5号車", VehicleClass.RIGID, (2_000, 4_000, 4_000, 4_000)),                         # 14kL 4槽
)

DRIVER_NAMES: tuple[str, ...] = ("乗務員A", "乗務員B", "乗務員C", "乗務員D", "乗務員E")

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


def _make_tanks(
    rng: random.Random, ctype: CustomerType, season: str
) -> tuple[Tank, ...]:
    """1軒ぶんのタンクを作る。

    在庫水準の振り方に意図がある。一様乱数で振ると、必須訪問（発注点割れ）の
    数が日によって 0 軒から 12 軒まで暴れ、デモにならない日が混ざる。
    ここでは「大半は余裕、一部が切迫」という現場の姿になるよう、
    残量比を三角分布（下側に裾）で振っている。
    """
    profile = PROFILES[ctype]
    mult = SEASON_MULTIPLIER[season]
    reorder_ratio = REORDER_RATIO[ctype]

    tanks = []
    for product, (capacity, base_use) in profile.tanks.items():
        daily_use = int(round(base_use * mult[product]))
        # 残量比: 最頻値0.42、下限0.08、上限0.85。下側に裾があるので
        # 「今日切迫している軒」が毎日いくらか出る。
        ratio = rng.triangular(0.08, 0.85, 0.42)
        level = int(round(capacity * ratio))
        tanks.append(
            Tank(
                product=product,
                capacity_l=capacity,
                level_l=level,
                reorder_l=int(round(capacity * reorder_ratio)),
                daily_use_l=daily_use,
                ullage_margin_l=int(round(capacity * ULLAGE_RATIO)),
            )
        )
    return tuple(tanks)


def generate(
    *,
    seed: int = 20261015,
    date: str = "2026-10-15",
    season: str = "winter",
    absent_drivers: tuple[str, ...] = (),
) -> Scenario:
    """1日ぶんの配車問題を作る。

    Args:
        seed: 乱数種。同じ seed は完全に同じ日を返す。
        date: 対象日（表示と記録のため。計算には使わない）。
        season: winter / spring / summer / autumn。
        absent_drivers: 欠勤する乗務員のID。デモ4（乗務員欠勤）で使う。

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

    rng = random.Random(seed)

    customers = []
    for spec in ROSTER:
        profile = PROFILES[spec.type]
        customers.append(
            Customer(
                id=spec.id,
                name=spec.name,
                type=spec.type,
                lat=spec.lat,
                lon=spec.lon,
                tanks=_make_tanks(rng, spec.type, season),
                window_open_min=hhmm_to_minutes(profile.window_open),
                window_close_min=hhmm_to_minutes(profile.window_close),
                accepts_trailer=(
                    ACCEPTS_TRAILER[spec.type]
                    and spec.id not in RIGID_ONLY_CUSTOMERS
                ),
                bore_inch=min(HOSE_INCH, spec.piping_inch),
            )
        )

    drivers = tuple(
        Driver(
            id=f"d{i + 1}",
            name=name,
            available=f"d{i + 1}" not in absent_drivers,
        )
        for i, name in enumerate(DRIVER_NAMES)
    )

    vehicles = tuple(
        Vehicle(
            id=vid,
            name=f"{vname}（{VEHICLE_CLASS_JA[vclass]} {sum(comps) // 1000}kL）",
            vehicle_class=vclass,
            compartments_l=comps,
            driver_id=f"d{i + 1}",
        )
        for i, (vid, vname, vclass, comps) in enumerate(FLEET)
    )

    return Scenario(
        date=date,
        season=season,
        seed=seed,
        depot=MARIFU,
        customers=tuple(customers),
        vehicles=vehicles,
        drivers=drivers,
        notes=[
            "合成データ。地理と制約の形は実際の二次配送に合わせ、在庫と消費は生成値。",
            f"season={season} / seed={seed}",
        ],
    )
