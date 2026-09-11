"""石油ローリー二次配送のドメインモデル。

なぜ dataclass で持つか:
    このプロジェクトの原則は「AIは何を計算させるか決める、Python が計算する」。
    数字を出す側（決定論コア）が扱う対象は、モデルの出力ではなく**型のある値**で
    なければならない。辞書のまま回すと、キーの綴り違いが実行時まで分からず、
    「計画は出たが単位がリットルでなくキロリットルだった」類の事故が起きる。

単位の約束（ここでしか決めない）:
    - 量        : リットル（L）。整数。CP-SAT に渡す都合で小数を持たない。
    - 時刻・時間: 0時からの経過分（int）。08:30 → 510。
    - 距離      : キロメートル（float）。表示にしか使わない。
    - 座標      : WGS84 の十進度。

    キロリットル（kL）は現場語なので画面表示のときだけ使う。内部は必ず L。

このデータは合成である:
    需要家の座標・タンク容量・消費量はすべて合成データで、実在の事業所の
    実績値ではない（core/generator.py が生成する）。配送圏の地理と制約の
    形は実際の二次配送に合わせてあるが、数字そのものは審査用の作り物である。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


#: 何日先まで見て「今日入れておく」と判断するか。
#:
#: 二次配送は「空になってから運ぶ」のではなく「次に来られるまでに割らない
#: ように入れておく」仕事なので、当日の在庫だけを見ても需要は決まらない。
#: ここを大きくすると毎日ほぼ全軒に入れることになって車が足りず、
#: 小さくすると翌日に発注点割れが大量発生する。2日は「明日また来られる」
#: を前提にした配車間隔。W4 の寒波シグナルは daily_use_l を押し上げるので、
#: この定数を触らなくても対象タンクが自然に増える。
DELIVERY_LOOKAHEAD_DAYS = 2.0

#: 「今日行かなければ欠品する」の判定に使う日数。
#:
#: 二次配送は「明日また来られる」を前提に組む。したがって**今日行かないと
#: 明日の配送より先に空になる**タンクが、本当の欠品にあたる。発注点
#: （reorder_l）はそれより手前に置いた計画上の目安で、割っただけでは
#: まだ空にならない。
STOCKOUT_HORIZON_DAYS = 1.0


#: 労働基準法34条の休憩。**基準は拘束時間ではなく労働時間**（＝拘束−休憩）。
#:
#:     労働 6時間以下      → 不要
#:     労働 6時間超〜8時間 → 45分以上
#:     労働 8時間超        → 1時間以上
#:
#: **上の段は無い。** 12時間で1時間30分、16時間で2時間といった比例付与は
#: 義務づけられていない（厚労省の解釈も二段のみ）。上乗せがあるとすれば
#: それは就業規則か労働協約の側の話であって、法令の要求ではない。
#:
#: 判定に要るのは労働時間だが、計画側が持っているのは拘束時間なので、
#: 休憩を差し引く前の拘束時間で判定できるよう逆算した閾値で持つ:
#:
#:     拘束 ≤ 360分（6時間）        → 労働 ≤ 6時間  → 0分
#:     拘束 ≤ 525分（480 + 45）     → 労働 ≤ 8時間  → 45分
#:     拘束 > 525分                 → 労働 8時間超  → 60分
LABOR_BREAK_45_MIN = 45
LABOR_BREAK_60_MIN = 60

#: 段の境目を**労働時間**で持つ。条文が語っているのはこちら。
LABOR_WORK_45_MIN = 360    # 労働6時間
LABOR_WORK_60_MIN = 480    # 労働8時間

#: 同じ段を**拘束時間**で見たときの境目（上の逆算）。
LABOR_BREAK_DUTY_45_MIN = LABOR_WORK_45_MIN
LABOR_BREAK_DUTY_60_MIN = LABOR_WORK_60_MIN + LABOR_BREAK_45_MIN


def labor_break_min(duty_min: int) -> int:
    """その拘束時間で労基法34条が要求する休憩（分）。

    改善基準告示の430休憩（連続運転4時間ごとに30分）とは**足し合わせない**。
    2024年4月の改正で、430の「運転の中断」は原則として休憩とされ、荷積み・
    荷卸しを算入できなくなった。つまり430の30分は労基法の休憩の内数であって、
    別枠ではない。要る休憩は両者の**大きいほう**になる。
    """
    if duty_min <= LABOR_BREAK_DUTY_45_MIN:
        return 0
    if duty_min <= LABOR_BREAK_DUTY_60_MIN:
        return LABOR_BREAK_45_MIN
    return LABOR_BREAK_60_MIN


#: 移動貯蔵タンクの1室の上限（危険物の規制に関する政令 第15条）。
#:
#: 「4,000L以下ごとに完全な間仕切を設ける」。**槽はこれより大きくできない。**
#: 最初の実装には 5,000L の槽を持つ車がいたが、法令上そんな槽は無い。
COMPARTMENT_MAX_L = 4_000

#: 実在する槽の大きさ。**2kL と 4kL の組合せだけ。**（実務）
#:
#: 法令は上限（4kL）しか定めていないが、実際のローリーはこの2種類を
#: 組み合わせて作られている。たとえば20kL車なら
#:
#:     2/4/2/4/4/2/2（7槽）   2/4/4/2/4/4（6槽）
#:
#: 以前は 3kL の槽を置いていたが、そんな槽は無い。
COMPARTMENT_SIZES_L: tuple[int, ...] = (2_000, 4_000)

#: 荷卸しの準備と手仕舞（着車・アース・ホース接続・検尺・伝票・収納）。
#:
#: 流下そのものとは別にかかる時間。流量の実測と、運送会社が1か所あたりに
#: 見込んでいる時間（30分／60分）の両方を、この1つの数字で説明できる:
#:
#:     4インチで8kL（2槽）   流下10分 ＋ 20分 ＝ 30分   （見込み 30分）
#:     3インチで20kL（1台分） 流下40分 ＋ 20分 ＝ 60分   （見込み 60分）
UNLOAD_SETUP_MIN = 20

#: 荷卸しの流下。**自然落下なので、経路で一番細いところが律速する。**
#:
#: ホースは4インチと3インチの2種類。条件のよい先（地下タンク側の配管も
#: 4インチ）なら4インチ、そうでなければ3インチ相当になる。流量は断面積に
#: 比例するので、3インチは4インチのおよそ半分（断面積比 (3/4)² ≒ 0.56）。
#:
#: 運送会社への聞き取り（2026-09-11）:
#:
#:     4インチ  4kL で5分 / 2kL で3分
#:     3インチ  4kL で8分 / 2kL で4分
#:
#: 2点を通る直線にすると、1槽あたり
#:
#:     4インチ  1分 ＋ 1.0分/kL   （1分は槽ごとの弁の開閉などの切替）
#:     3インチ  0分 ＋ 2.0分/kL
#:
#: 口径 → (1槽あたりの切替分, 1kLあたりの分)
UNLOAD_FLOW: dict[int, tuple[float, float]] = {
    4: (1.0, 1.0),
    3: (0.0, 2.0),
}


def unload_minutes(volume_by_product: dict, bore_inch: int) -> int:
    """1か所の荷卸しに要する時間（着車〜出発）。

    **降ろす量に比例する。** 以前は需要家の類型ごとの定数（30〜45分）で、
    2kL置くのも16kL置くのも同じ時間になっていた。

    槽は油種ごとに分かれ、1槽は4kLまで（COMPARTMENT_MAX_L）なので、
    油種ごとに「何槽ぶんか」を数えて切替分を積む。槽は1本ずつ順に降ろす
    前提（ホース1本で同時に2槽は降ろさない）。

    槽の数は「4kL槽で詰めた場合」の最少数で数えている。実際は2kL槽も
    あるので、どの槽に積むか（多コンパートメント割当、未実装）次第で
    1〜2槽ぶん増えうる。4インチなら1槽1分なので影響は小さい。
    """
    if bore_inch not in UNLOAD_FLOW:
        raise ValueError(f"荷卸しの口径は {sorted(UNLOAD_FLOW)} インチのいずれか: {bore_inch}")
    per_compartment, per_kl = UNLOAD_FLOW[bore_inch]
    compartments = sum(-(-v // COMPARTMENT_MAX_L) for v in volume_by_product.values() if v > 0)
    total_kl = sum(v for v in volume_by_product.values() if v > 0) / 1000
    flow = compartments * per_compartment + total_kl * per_kl
    return UNLOAD_SETUP_MIN + int(round(flow))


class Product(str, Enum):
    """積む油種。

    槽（コンパートメント）は1つにつき1油種しか積めない。混ぜると製品事故に
    なるため、これは最適化の都合ではなく物理と法令の制約である。
    """

    KEROSENE = "kerosene"    # 灯油
    GASOIL = "gasoil"        # 軽油
    REGULAR = "regular"      # レギュラーガソリン
    PREMIUM = "premium"      # ハイオクガソリン


#: 画面と説明文で使う現場語。OR の用語も英語も配車係には出さない。
PRODUCT_JA: dict[Product, str] = {
    Product.KEROSENE: "灯油",
    Product.GASOIL: "軽油",
    Product.REGULAR: "レギュラー",
    Product.PREMIUM: "ハイオク",
}


class CustomerType(str, Enum):
    """需要家の類型。

    網羅ではなく「1回のデモで全制約系が見えるように散らす」ための4類型
    （引き継ぎ文書 §4.2）。類型ごとに効く制約が違う。
    """

    URBAN_GS_24H = "urban_gs_24h"      # 広島市中心部・24時間営業 → 時間帯係数が効く
    URBAN_GS_DAY = "urban_gs_day"      # 広島市中心部・日中のみ   → 時間枠×ラッシュ
    SANIN_GS = "sanin_gs"              # 山陰側・長距離           → 季節需要＋拘束時間
    CARRIER_DEPOT = "carrier_depot"    # 運送会社営業所           → 早朝枠＋軽油中心
    SUBURBAN_GS = "suburban_gs"        # 郊外GS                   → 灯油比率高・タンク小


class VehicleClass(str, Enum):
    """ローリーの車格。**積載量だけの違いではない。**

    単車は小回りが利くので、狭い道・小さい構内・切り返しの余地がない
    需要家へ入れる。トレーラーは同じ場所へ物理的に入れない。

    容量は実在する型式に合わせる（それ以外の数字を置かない）:
        単車       14kL / 16kL
        トレーラー 18kL / 20kL / 24kL / 26kL
    """

    RIGID = "rigid"        # 単車
    TRAILER = "trailer"    # トレーラー


#: 車格ごとに実在する積載量（L）。ここに無い容量の車を作らない。
#:
#: 最初の実装では 10kL の車を置いていたが、**そんなローリーは無い**。
#: 作り物の車で組んだ計画は、作り物の制約しか検証しない。
CAPACITIES_L: dict[VehicleClass, tuple[int, ...]] = {
    VehicleClass.RIGID: (14_000, 16_000),
    VehicleClass.TRAILER: (18_000, 20_000, 24_000, 26_000),
}

VEHICLE_CLASS_JA: dict[VehicleClass, str] = {
    VehicleClass.RIGID: "単車",
    VehicleClass.TRAILER: "トレーラー",
}


class LegClass(str, Enum):
    """区間の幹線区分。

    W3 で時間帯係数 factor(leg_class, dow, time_bucket, season) を持たせる
    ときの次元になる。v0 では所要時間の平均速度を決めるためだけに使う。
    """

    URBAN = "urban"            # 市街地（広島市内など）
    ARTERIAL = "arterial"      # 幹線一般道（国道2号など）
    EXPRESSWAY = "expressway"  # 高速（山陽道・広島呉道路など）
    MOUNTAIN = "mountain"      # 中国山地越え（山陰方面）


@dataclass(frozen=True)
class Tank:
    """需要家の地下タンク1本。油種ごとに1本持つ前提。

    Attributes:
        product: 油種。
        capacity_l: 満タン容量。
        level_l: 当日朝の在庫。
        reorder_l: 発注点。これを割ったら「必須訪問」になる（制約4）。
        daily_use_l: 1日あたりの平年消費量。寒波・値上げのシグナルはこれに
            乗数をかけて効かせる（W4）。
        ullage_margin_l: 満タンまで入れずに残す余裕。受入側の安全代であり、
            これを無視すると「入れ切れずに持ち帰る」計画が出る。
    """

    product: Product
    capacity_l: int
    level_l: int
    reorder_l: int
    daily_use_l: int
    ullage_margin_l: int = 0

    @property
    def is_below_reorder(self) -> bool:
        """発注点を割っているか。割っていれば当日の必須訪問になる。"""
        return self.level_l <= self.reorder_l

    @property
    def topup_l(self) -> int:
        """今日入れられる量（受入可能量）。

        満タン容量から現在庫と安全代を引いた分。負にはならない。
        """
        return max(0, self.capacity_l - self.level_l - self.ullage_margin_l)

    @property
    def needs_topup(self) -> bool:
        """このタンクを今日の積荷に入れるか。

        判断は「DELIVERY_LOOKAHEAD_DAYS 日後に発注点を割るか」。
        すでに割っていれば当然入れるし、まだ余裕があっても、次に来られる
        までに割るなら今日のうちに入れる。

        **全タンクを毎日満タンにしない**のがここの要点。実務でも、余裕の
        あるタンクは触らず、低いタンクだけを狙って積んでくる。全部満タンに
        する前提で需要を積むと、1軒で1台ぶんの積載を食い、車が何台あっても
        足りない計画になる（v0 の最初の実装がこれで壊れた）。
        """
        if self.topup_l <= 0:
            return False
        projected = self.level_l - self.daily_use_l * DELIVERY_LOOKAHEAD_DAYS
        return projected <= self.reorder_l

    @property
    def stockout_before_next_delivery(self) -> bool:
        """今日行かなければ、明日の配送より先に空になるか。**これが本当の欠品。**

        発注点割れ（is_below_reorder）とは別物。発注点は「そろそろ手配する」
        目安であって、割っても今日明日で空になるとは限らない。
        """
        return self.days_to_empty() < STOCKOUT_HORIZON_DAYS

    def days_to_empty(self) -> float:
        """このまま補給しなければ何日で空になるか。

        「次の訪問前に切れる需要家」を検知するための素の指標。
        消費がゼロのタンクは無限大を返す（ゼロ除算を呼び出し側に投げない）。
        """
        if self.daily_use_l <= 0:
            return float("inf")
        return self.level_l / self.daily_use_l


@dataclass(frozen=True)
class Customer:
    """需要家（GS または運送会社営業所）。

    受入時間枠は「開いている時間」であって「いつ来てもよい時間」ではない。
    運送会社営業所は早朝しか受けないなど、枠が狭いほど計画を縛る。
    """

    id: str
    name: str
    type: CustomerType
    lat: float
    lon: float
    tanks: tuple[Tank, ...]
    window_open_min: int      # 受入開始（0時からの分）
    window_close_min: int     # 受入終了。この時刻までに**荷卸を始められる**こと

    #: トレーラーが入れるか。**荷役条件のマスターに載っている情報**。
    #:
    #: 小さいSSや運送会社の構内は、狭い道・切り返しの余地の無さで
    #: トレーラーが物理的に入れない。そこには単車が優先的に配車される。
    #: 積載量の問題ではないので、容量制約とは別に持つ。
    accepts_trailer: bool = True

    #: 荷卸しの実効口径（インチ）。**ホースと地下タンク側の配管のうち細いほう。**
    #:
    #: 配管は荷役条件のマスターに載っている需要家側の情報、ホースはローリー側。
    #: 実務ではホースが3インチのことが多い（乗務員の負担を軽くするため）ので、
    #: 配管が4インチでも3インチになる先がほとんど。
    bore_inch: int = 3

    #: 需要家が自分で発注する先か（発注配送）。False なら、元売・運送会社が
    #: 需要見込みで運ぶ先（計画配送）。**どちらも前日に決まり、合わさって
    #: 翌日の配送計画になる。**
    ordering: bool = False

    #: 前日までに受けた発注（油種, L）。**需要家の約束**なので必須訪問になる。
    orders: tuple[tuple["Product", int], ...] = ()

    #: 今日の計画配送（油種, L）。**運送会社側の約束。** 計画配送の先は
    #: 決まった周期（ほぼ3日に1回）で、消費量に見合った量を運ぶ。今日が
    #: その日でなければ空。
    planned: tuple[tuple["Product", int], ...] = ()

    #: 今日営業しているか。日曜休業のSSは日曜に受けられない（無人荷卸しが
    #: 無い前提）。見送り（accepts_delivery）とは別物で、警告の対象にしない。
    open_today: bool = True

    #: 今日の受入を受けるか。**需要家の意思。**
    #:
    #: 値下げ前には在庫を下げたがって受入を見送る（安い油が来る前に高い
    #: 在庫を抱えたくない）。**発注点を割っていても「要らない」と言われる。**
    #: 在庫は需要家の意思を代理できないので、見送りの先には配車しない。
    #: 見送りは乗務員の欠勤と同じく、外から与える事象として扱う。
    accepts_delivery: bool = True

    def __post_init__(self) -> None:
        if self.bore_inch not in UNLOAD_FLOW:
            raise ValueError(
                f"{self.id}: 荷卸しの口径は {sorted(UNLOAD_FLOW)} インチのいずれか"
                f"（{self.bore_inch}）"
            )

    @property
    def service_min(self) -> int:
        """荷卸に要する時間（着車〜出発）。**今日降ろす量から決まる。**

        配送量は計画時点で需要どおりに固定されている（ソルバーは量を
        動かさない）ので、ここで前もって計算できる。
        """
        return unload_minutes(self.demand_by_product(), self.bore_inch)

    @property
    def demand_l(self) -> int:
        """今日この需要家に降ろす量の合計（demand_by_product の合計）。"""
        return sum(self.demand_by_product().values())

    @property
    def is_mandatory(self) -> bool:
        """必須訪問か。**誰の約束かで決める。** 理由は mandatory_basis。

            受入見送り・休業  → 必須にしない（行っても降ろせない）
            計画配送の日      → 必須（運送会社側の約束）
            発注がある        → 必須（需要家の約束）
            欠品の直前        → 必須（今日行かないと明日の配送より先に空になる）

        以前は「1本でも発注点を割っていれば必須」で、需要家の意思を見て
        いなかった。値下げ前に在庫を意図的に下げて受入を見送る需要家へ、
        **断られても配車する**計画が出ていた。

        発注点割れを一律に任意訪問へ落とす案は試して撤回した（2026-09-11）。
        罰が軽くなり、在庫あと1.5日の先を「車を1台増やすより明日に回すほうが
        安い」と判断して落とした。いまは計画配送が周期で決まっているので、
        発注点割れを待たずに運ぶ日が先に決まる。
        """
        return self.mandatory_basis is not None

    @property
    def mandatory_basis(self) -> str | None:
        """必須にする理由。強い順に stockout / order / planned。無ければ None。

        この順は、車が足りずにどれかを落とすときの優先順位でもある
        （solver.PENALTY_BY_BASIS）。欠品は元売の信用、発注は需要家の約束、
        計画配送は運送会社側の都合で1日ずらせる。
        """
        if not self.accepts_delivery or not self.open_today:
            return None
        if self.demand_l <= 0:
            return None
        if any(t.stockout_before_next_delivery and t.topup_l > 0 for t in self.tanks):
            return "stockout"
        if self.orders:
            return "order"
        if self.planned:
            return "planned"
        return None

    @property
    def declined_at_risk(self) -> bool:
        """受入を見送っているのに、在庫が欠品に向かっているか。

        **計画では救えない**（行っても降ろせない）ので、配車せずに知らせる。
        誰にどう伝えるかはエージェント層の仕事。
        """
        return not self.accepts_delivery and any(
            t.is_below_reorder or t.stockout_before_next_delivery for t in self.tanks
        )

    def demand_by_product(self) -> dict[Product, int]:
        """今日この需要家に降ろす量（油種ごと）。

            受入見送り・休業  → 何も降ろさない
            計画配送の日      → 計画量（入りきらない分は降ろせないので空き容量まで）
            発注の油種        → 発注量（同上）
            欠品の直前        → 空き容量まで（発注・計画の有無によらない）

        同じ油種に計画と発注が重なったら大きいほう（タンクは1本なので、
        足しても空き容量を超えては入らない）。
        """
        if not self.accepts_delivery or not self.open_today:
            return {}
        ordered = dict(self.orders)
        planned = dict(self.planned)
        out: dict[Product, int] = {}
        for t in self.tanks:
            q = 0
            if t.product in ordered:
                q = max(q, min(ordered[t.product], t.topup_l))
            if t.product in planned:
                q = max(q, min(planned[t.product], t.topup_l))
            if t.stockout_before_next_delivery:
                q = t.topup_l
            if q > 0:
                out[t.product] = out.get(t.product, 0) + q
        return out

    def min_days_to_empty(self) -> float:
        """一番早く切れるタンクが何日もつか。切迫度の並べ替えに使う。"""
        return min((t.days_to_empty() for t in self.tanks), default=float("inf"))


@dataclass(frozen=True)
class Depot:
    """油槽所（麻里布）。出発地であり帰着地。

    load_min は積込に要する時間。1日1便の v0 では出発前に1回だけ発生するが、
    W2 で複数便（積込回数・回転の制約5）を入れるときに便ごとに効いてくる。
    """

    id: str
    name: str
    lat: float
    lon: float
    open_min: int
    close_min: int
    load_min: int


@dataclass(frozen=True)
class Vehicle:
    """ローリー1台。

    compartments_l は槽ごとの容量。v0 では合計しか使わないが、W2 で
    「槽ごとに単一油種」の制約（制約1）を入れるために形を保っておく。
    合計を先に潰すと、W2 で生成器から作り直すことになる。
    """

    id: str
    name: str
    vehicle_class: VehicleClass
    compartments_l: tuple[int, ...]
    driver_id: str
    requires_hazmat: bool = True   # 危険物取扱者が要る。乗務員資格と突き合わせる

    def __post_init__(self) -> None:
        """実在しない容量の車を作らせない。

        黙って通すと、作り物の車で組んだ計画を現実だと思って読むことになる。
        """
        oversized = [c for c in self.compartments_l if c > COMPARTMENT_MAX_L]
        if oversized:
            raise ValueError(
                f"{self.id}: 1室は {COMPARTMENT_MAX_L}L 以下（危険物の規制に関する政令"
                f" 第15条）。{oversized} は法令上作れない"
            )
        odd = [c for c in self.compartments_l if c not in COMPARTMENT_SIZES_L]
        if odd:
            raise ValueError(
                f"{self.id}: 槽は {'/'.join(f'{c // 1000}kL' for c in COMPARTMENT_SIZES_L)}"
                f" の組合せだけ。{odd} の槽は無い"
            )
        allowed = CAPACITIES_L[self.vehicle_class]
        total = sum(self.compartments_l)
        if total not in allowed:
            raise ValueError(
                f"{self.id}: {VEHICLE_CLASS_JA[self.vehicle_class]}に "
                f"{total / 1000:.0f}kL 積は無い"
                f"（{'/'.join(f'{c // 1000}kL' for c in allowed)} のいずれか）"
            )

    @property
    def capacity_l(self) -> int:
        """総積載量。単槽 CVRPTW ではこれを容量制約に使う。"""
        return sum(self.compartments_l)

    @property
    def is_trailer(self) -> bool:
        """トレーラーか。入れない需要家があるので、経路の可否に効く。"""
        return self.vehicle_class is VehicleClass.TRAILER


@dataclass(frozen=True)
class Driver:
    """乗務員。

    拘束時間・運転時間の上限は改善基準告示（自動車運転者の労働時間等の
    改善のための基準）に合わせた既定値を持つ。ここを緩めると「法令上
    走れない計画」が平然と出るので、既定値は現実側に寄せてある。

    Attributes:
        max_duty_min: 1日の拘束時間の上限。原則13時間＝780分（改善基準告示）。
        max_drive_min: 1日の運転時間の上限。2日平均で9時間＝540分（同）。
        continuous_drive_limit_min: 連続運転の上限。4時間＝240分（同）。
        break_min: 連続運転の上限に達したときに要する休憩。30分（430休憩）。
            **これとは別に労基法34条の休憩がある**（labor_break_min）。
            両者は足し算ではなく、大きいほうを取る。
        has_hazmat: 危険物取扱者を持つか。持たない乗務員はローリーに乗れない。
        available: その日出勤するか。欠勤シナリオ（デモ4）でここを倒す。
    """

    id: str
    name: str
    max_duty_min: int = 780
    max_drive_min: int = 540
    continuous_drive_limit_min: int = 240
    break_min: int = 30
    has_large_license: bool = True
    has_hazmat: bool = True
    available: bool = True


@dataclass
class Scenario:
    """1日ぶんの配車問題の入力一式。

    これが決定論コアへの唯一の入口。エージェント層（W4）が外部シグナルを
    読んで書き換えるのも、回帰テスト（W6）が固定して読み込むのも、この形。
    seed を持たせてあるのは、同じ seed なら完全に同じ問題が再現できるように
    するため。再現できない問題では、計画品質の回帰テストが成立しない。
    """

    date: str                  # YYYY-MM-DD
    season: str                # winter / spring / summer / autumn
    seed: int
    depot: Depot
    customers: tuple[Customer, ...]
    vehicles: tuple[Vehicle, ...]
    drivers: tuple[Driver, ...]
    notes: list[str] = field(default_factory=list)

    def driver(self, driver_id: str) -> Driver:
        """乗務員を引く。見つからないのは生成器かシグナル適用のバグなので落とす。"""
        for d in self.drivers:
            if d.id == driver_id:
                return d
        raise KeyError(f"乗務員が見つからない: {driver_id}")

    def usable_vehicles(self) -> tuple[Vehicle, ...]:
        """今日実際に出せるローリー。

        乗務員が欠勤、または危険物資格を持たない場合、その車は使えない。
        「車はあるが人がいない」を計画側で握りつぶさないための関門。
        """
        out = []
        for v in self.vehicles:
            d = self.driver(v.driver_id)
            if not d.available:
                continue
            if v.requires_hazmat and not d.has_hazmat:
                continue
            if not d.has_large_license:
                continue
            out.append(v)
        return tuple(out)

    @property
    def fleet_capacity_l(self) -> int:
        """今日出せる車の総積載量。需要と突き合わせて実行可能性を先に見る。"""
        return sum(v.capacity_l for v in self.usable_vehicles())

    @property
    def mandatory_demand_l(self) -> int:
        """必須訪問ぶんの合計必要量。fleet_capacity_l を超えたらその日は解けない。"""
        return sum(c.demand_l for c in self.customers if c.is_mandatory)


def minutes_to_hhmm(minutes: int) -> str:
    """0時からの経過分を HH:MM にする。翌日にまたがる時刻も 24:30 のように出す。

    24時を跨いだ値を 00:30 に丸めないのは意図的。深夜便が「翌日の00:30」なのか
    「当日の00:30」なのかを、画面を見た配車係が取り違えないようにするため。
    """
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def hhmm_to_minutes(hhmm: str) -> int:
    """HH:MM を0時からの経過分にする。生成器と設定ファイルの読み込み用。"""
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)
