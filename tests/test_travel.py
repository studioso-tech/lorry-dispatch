"""所要時間行列の性質テスト。

v0 は幾何モデルなので値そのものに正解は無い。W2 で実測のスナップショットに
差し替えたあとも成り立っていなければならない性質だけを見る。
"""

from __future__ import annotations

from core.generator import MARIFU, generate
from core.travel import LegClass, TravelMatrix, build_matrix, haversine_km


def _matrix():
    s = generate(seed=1)
    return s, build_matrix(s.depot, s.customers)


def test_shape_and_diagonal():
    """節点は「0=油槽所、1..n=需要家」。対角は0。"""
    s, m = _matrix()
    assert m.size == len(s.customers) + 1
    assert m.labels[0] == MARIFU.name
    for i in range(m.size):
        assert m.minutes[i][i] == 0


def test_all_off_diagonal_times_are_positive():
    """どこへ行くにも時間はかかる。0分の区間があると経路が縮退する。"""
    _, m = _matrix()
    for i in range(m.size):
        for j in range(m.size):
            if i != j:
                assert m.minutes[i][j] >= 1


def test_base_matrix_is_symmetric():
    """自由流の所要時間は向きでほぼ変わらない。

    方向差（朝の上り／下り）は W3 の**有向の時間帯係数**で入れる。
    ここで非対称にすると二重計上になる。
    """
    _, m = _matrix()
    for i in range(m.size):
        for j in range(m.size):
            assert m.minutes[i][j] == m.minutes[j][i]


def test_longer_distance_takes_longer_within_same_class():
    """同じ区間区分なら、遠いほうが時間がかかる。"""
    s, m = _matrix()
    pts = [(s.depot.lat, s.depot.lon)] + [(c.lat, c.lon) for c in s.customers]
    same_class = [
        (haversine_km(*pts[0], *pts[j]), m.minutes[0][j])
        for j in range(1, m.size)
        if m.leg_class[0][j] is LegClass.ARTERIAL
    ]
    same_class.sort()
    for (d1, t1), (d2, t2) in zip(same_class, same_class[1:]):
        assert t1 <= t2, f"{d1}km={t1}分 > {d2}km={t2}分"


def test_sanin_legs_are_the_long_ones():
    """山陰側（浜田・益田）が配送圏で一番遠い。

    地理が壊れていないことの確認。ここが崩れると「拘束時間が厳しい」という
    制約の効き方そのものが作り物になる。
    """
    s, m = _matrix()
    idx = {c.id: i + 1 for i, c in enumerate(s.customers)}
    far = max(range(1, m.size), key=lambda j: m.minutes[0][j])
    assert far in (idx["c06"], idx["c07"])
    assert m.minutes[0][idx["c06"]] > 60


def test_json_round_trip():
    """スナップショットして読み戻しても同じ。W2 で実測行列を保存する経路。"""
    _, m = _matrix()
    back = TravelMatrix.from_json(m.to_json())
    assert back.labels == m.labels
    assert back.minutes == m.minutes
    assert back.leg_class == m.leg_class
