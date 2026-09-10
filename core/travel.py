"""所要時間行列。v0 は地図APIを呼ばない。

なぜ地図APIを呼ばないか:
    2025年3月の課金体系変更で、交通考慮つきの Routes API（Pro ティア）は
    毎月無料枠が5,000要素、1リクエスト100要素上限になった。ラッシュを
    表現しようとすると複数の出発時刻で取り直すことになり、要素数が数倍に
    膨らむ。加えて**ライブ値は実行のたびに変わるので、計画の再現性が壊れる**。
    回帰テスト（W6）で「同じ入力なら同じ計画」が言えなくなるのは致命的。

    よって方針は引き継ぎ文書 §5 のとおり:
      1. ベース（自由流）の所要時間行列は Essentials で**1回だけ**取り、
         スナップショットして使い回す
      2. ラッシュは時間帯係数カーブという**透明な式**で乗せる（W3）
      3. 交通考慮 API はフラグ付きの stretch モードに留める

v0 の位置づけ:
    ベース行列すらまだ取っていない段階なので、緯度経度から幾何的に見積もる。
    直線距離に道路の迂回率をかけ、区間の性格ごとの平均速度で割る。
    **これは W2 で実測のベース行列に差し替える前提の仮の値**であり、
    差し替えたときに solver 側を書き換えずに済むよう、外に出す形
    （TravelMatrix）だけを先に固定してある。

W3 で入るもの（ここにはまだ無い）:
    - 有向の時間帯係数（朝は油槽所→広島方面が重く、逆は軽い）
    - 高速 vs 一般道の経路オプションと toll
    - 工事・事故・積雪のイベント層
    - p50 / p80 / p95 の分布（ソルバーは p80 で解く）
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from core.domain import Customer, Depot, LegClass

#: 直線距離に対する実道路距離の比。**区分ごとに変える。**
#:
#: 当初は全区分を一律 1.35 にしていたが、実際の道路距離と突き合わせたら
#: 系統的に短すぎた（2026-09-10 に検算）。
#:
#:     需要家              直線   モデル   実際   実際の迂回率
#:     廿日市物流センター    19      26     25     1.32
#:     広島中央SS          30      40     48     1.60
#:     宇品港SS            28      38     50     1.79
#:     浜田SS              79     106    140     1.77
#:     益田SS              63      85    130     2.06
#:     玖珂SS              12      16     22     1.83
#:
#: 実際は 1.32〜2.06、平均 1.73。**山越えは 1.8〜2.1**。中国山地を越える道が
#: 直線からどれだけ回り込むかを甘く見ていた。
#:
#: 距離が短いと所要時間も短くなるので、楽観側に倒れた計画が出る。
#: 「実際より楽に回れる」前提の計画は、時間枠を割ってから気づくことになる。
#: **迂回率は安全側（大きめ）に置く。**
WINDING_FACTOR: dict[LegClass, float] = {
    LegClass.URBAN: 1.75,        # 市街地。一方通行と河川の橋で回り込む
    LegClass.ARTERIAL: 1.70,     # 幹線一般道
    LegClass.EXPRESSWAY: 1.55,   # 高速は比較的まっすぐ。IC への出入りで増える
    LegClass.MOUNTAIN: 1.85,     # 山越え。谷筋に沿って大きく迂回する
}

#: 区間区分ごとの平均速度（km/h）。ローリーは大型で、法定と積載の両方で遅い。
#: 高速の 75 は制限80に対する実勢、市街地の 24 は信号待ち込みの実効値。
SPEED_KMH: dict[LegClass, float] = {
    LegClass.URBAN: 24.0,
    LegClass.ARTERIAL: 38.0,
    LegClass.EXPRESSWAY: 75.0,
    LegClass.MOUNTAIN: 42.0,
}

#: 高速を使う区間で、IC への出入りに食われる固定時間（分）。
#: 短距離で高速が常に有利になってしまうのを防ぐ。
EXPRESSWAY_ACCESS_MIN = 12.0

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """2点間の大円距離（km）。"""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def classify_leg(
    lat1: float, lon1: float, lat2: float, lon2: float, straight_km: float
) -> LegClass:
    """区間の性格を座標と距離から決める。

    v0 の割り切り。W3 では実在の幹線（国道2号、山陽道、広島呉道路）に
    紐付けた leg_class を持たせるので、この関数は係数プロファイルを引く
    ためのキー生成に置き換わる。

    判定順に意味がある:
        1. 両端が広島市中心部に近ければ市街地
        2. 山陰側（北緯34.6度以北かつ内陸）に跨るなら山越え
        3. 長距離なら高速が現実的
        4. それ以外は幹線一般道
    """
    hiroshima = (34.39, 132.46)

    def near_hiroshima(lat: float, lon: float) -> bool:
        return haversine_km(lat, lon, *hiroshima) < 12.0

    if near_hiroshima(lat1, lon1) and near_hiroshima(lat2, lon2):
        return LegClass.URBAN
    if max(lat1, lat2) > 34.60 and min(lon1, lon2) < 132.30:
        return LegClass.MOUNTAIN
    if straight_km > 35.0:
        return LegClass.EXPRESSWAY
    return LegClass.ARTERIAL


@dataclass(frozen=True)
class TravelMatrix:
    """節点間の所要時間と距離。

    節点の並びは「0 = 油槽所、1..n = 需要家（Scenario の順）」で固定する。
    ソルバーはこの添字でしか世界を見ない。W2 で中身を実測のベース行列に
    差し替えても、ソルバー側は一行も変わらない。

    Attributes:
        labels: 節点の表示名。計画の説明文と地図描画が使う。
        minutes: minutes[i][j] = i から j への所要時間（分・整数）。
        km: km[i][j] = 実道路距離の見積り。表示と将来の toll 計算用。
        leg_class: 区間区分。W3 の係数プロファイルのキーになる。
    """

    labels: tuple[str, ...]
    minutes: tuple[tuple[int, ...], ...]
    km: tuple[tuple[float, ...], ...]
    leg_class: tuple[tuple[LegClass, ...], ...]

    @property
    def size(self) -> int:
        return len(self.labels)

    def to_json(self) -> str:
        """スナップショット用。W2 で実測行列を保存するときも同じ形にする。"""
        return json.dumps(
            {
                "labels": list(self.labels),
                "minutes": [list(r) for r in self.minutes],
                "km": [[round(v, 3) for v in r] for r in self.km],
                "leg_class": [[c.value for c in r] for r in self.leg_class],
            },
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "TravelMatrix":
        d = json.loads(text)
        return cls(
            labels=tuple(d["labels"]),
            minutes=tuple(tuple(int(v) for v in r) for r in d["minutes"]),
            km=tuple(tuple(float(v) for v in r) for r in d["km"]),
            leg_class=tuple(tuple(LegClass(v) for v in r) for r in d["leg_class"]),
        )

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TravelMatrix":
        return cls.from_json(path.read_text(encoding="utf-8"))


def build_matrix(depot: Depot, customers: Sequence[Customer]) -> TravelMatrix:
    """幾何モデルで所要時間行列を組む（v0）。

    対称行列になる。実際の道路は方向で所要時間が違う（朝の上り／下り）が、
    それは W3 の**有向の時間帯係数**で入れる。ベースの自由流所要時間自体は
    向きでほぼ変わらないので、ここで非対称にすると二重計上になる。
    """
    lats = [depot.lat] + [c.lat for c in customers]
    lons = [depot.lon] + [c.lon for c in customers]
    labels = tuple([depot.name] + [c.name for c in customers])

    n = len(labels)
    minutes: list[list[int]] = [[0] * n for _ in range(n)]
    kms: list[list[float]] = [[0.0] * n for _ in range(n)]
    classes: list[list[LegClass]] = [[LegClass.URBAN] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            straight = haversine_km(lats[i], lons[i], lats[j], lons[j])
            leg = classify_leg(lats[i], lons[i], lats[j], lons[j], straight)
            road_km = straight * WINDING_FACTOR[leg]
            base_min = road_km / SPEED_KMH[leg] * 60.0
            if leg is LegClass.EXPRESSWAY:
                base_min += EXPRESSWAY_ACCESS_MIN
            minutes[i][j] = max(1, int(round(base_min)))
            kms[i][j] = road_km
            classes[i][j] = leg

    return TravelMatrix(
        labels=labels,
        minutes=tuple(tuple(r) for r in minutes),
        km=tuple(tuple(r) for r in kms),
        leg_class=tuple(tuple(r) for r in classes),
    )
