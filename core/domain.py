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
    service_min: int          # 荷卸に要する時間（着車〜出発）

    @property
    def demand_l(self) -> int:
        """今日この需要家に降ろす量の合計。

        対象は「2日先までに発注点を割るタンク」だけ（Tank.needs_topup）。
        余裕のあるタンクは触らない。

        v0（単槽モデル）では油種を区別せず合計だけを見る。W2 で
        「どの槽にどの油種を積むか」に分解する。
        """
        return sum(t.topup_l for t in self.tanks if t.needs_topup)

    @property
    def is_mandatory(self) -> bool:
        """必須訪問か。1本でも発注点を割っていれば必須（制約4）。"""
        return any(t.is_below_reorder for t in self.tanks)

    def demand_by_product(self) -> dict[Product, int]:
        """油種ごとの必要量。W2 の多コンパートメント割当が使う。"""
        out: dict[Product, int] = {}
        for t in self.tanks:
            if t.needs_topup:
                out[t.product] = out.get(t.product, 0) + t.topup_l
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
    compartments_l: tuple[int, ...]
    driver_id: str
    requires_hazmat: bool = True   # 危険物取扱者が要る。乗務員資格と突き合わせる

    @property
    def capacity_l(self) -> int:
        """総積載量。v0 の単槽 CVRPTW ではこれを容量制約に使う。"""
        return sum(self.compartments_l)


@dataclass(frozen=True)
class Driver:
    """乗務員。

    拘束時間・運転時間の上限は改善基準告示（自動車運転者の労働時間等の
    改善のための基準）に合わせた既定値を持つ。ここを緩めると「法令上
    走れない計画」が平然と出るので、既定値は現実側に寄せてある。

    Attributes:
        max_duty_min: 1日の拘束時間の上限。原則13時間＝780分。
        max_drive_min: 1日の運転時間の上限。2日平均で9時間＝540分。
        continuous_drive_limit_min: 連続運転の上限。4時間＝240分。
        break_min: 上限に達したときに要する休憩。30分。
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
