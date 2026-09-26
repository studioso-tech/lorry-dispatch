"""二交代（1台を2人で回す）のテスト。

確かめたいのは**車と人が別の資源として扱えているか**。1台に昼勤・夜勤の
2行が立ち、行の時間帯は重ならず、台数は行数ではなく物理の台数で数える。

実務の根拠は docs/業務前提.md §5.2。体制は曜日の周期で固定され、日曜だけ
1交代に戻る（SSに休むところが多く配送量が減るため）。
"""

from __future__ import annotations

import pytest

from core.domain import CustomerType, Vehicle, VehicleClass, minutes_to_hhmm
from core.generator import NIGHT_DELIVERABLE, SHIFT_WINDOWS, generate
from core.solver import SolveMode, solve
from core.travel import build_matrix

# 2026-10-16 は金曜、2026-10-18 は日曜
FRIDAY = "2026-10-16"
SUNDAY = "2026-10-18"


# ---------------------------------------------------------------------------
# 形
# ---------------------------------------------------------------------------


def test_two_shift_puts_two_rows_and_two_drivers_on_one_vehicle():
    """1台につき昼勤・夜勤の2行と、2人の乗務員が立つ。"""
    one = generate(seed=20261015, date=FRIDAY, season="winter", shifts=1)
    two = generate(seed=20261015, date=FRIDAY, season="winter", shifts=2)

    assert len(two.vehicles) == 2 * len(one.vehicles)
    assert len(two.drivers) == 2 * len(one.drivers)
    # **台数は物理で数える。** 行数で数えると倍に見える
    assert len({v.physical for v in two.vehicles}) == len(one.vehicles)
    assert {v.shift for v in two.vehicles} == {"day", "night"}
    # 同じ車の2行は、同じ車（槽の構成が同じ）で別の乗務員
    by_physical: dict[str, list[Vehicle]] = {}
    for v in two.vehicles:
        by_physical.setdefault(v.physical, []).append(v)
    for rows in by_physical.values():
        assert len(rows) == 2
        assert rows[0].compartments_l == rows[1].compartments_l
        assert rows[0].driver_id != rows[1].driver_id


def test_shift_windows_do_not_overlap_for_the_same_vehicle():
    """昼勤と夜勤の勤務帯は重ならない。

    重なると、同じ1台が同時に2か所を走る計画が作れてしまう。
    """
    s = generate(seed=20261015, date=FRIDAY, season="winter", shifts=2)
    rows = {v.shift: v for v in s.vehicles if v.physical == "v1"}
    day, night = rows["day"].duty_window_min, rows["night"].duty_window_min
    assert day[1] <= night[0] + 60, "引継ぎの重なりは1時間まで"
    assert night[0] >= day[0]
    # 夜勤は24時を跨ぐ。**1440を超える分で書く**（翌05:00 = 1740）
    assert night[1] > 24 * 60
    assert minutes_to_hhmm(night[1]) == "29:00"


def test_each_shift_fits_one_driver_duty_limit():
    """1つの勤務帯は、1人の拘束の上限（13時間）に収まる。"""
    s = generate(seed=20261015, date=FRIDAY, season="winter", shifts=2)
    for v in s.vehicles:
        lo, hi = v.duty_window_min
        assert hi - lo <= s.driver(v.driver_id).max_duty_min


def test_sunday_falls_back_to_one_shift():
    """**日曜は二交代を解除する**（実務）。当日の判断ではなく曜日の周期。"""
    sunday = generate(seed=20261015, date=SUNDAY, season="winter", shifts=2)
    assert {v.shift for v in sunday.vehicles} == {"day"}
    assert len(sunday.vehicles) == len(sunday.drivers)
    assert all(v.duty_window_min is None for v in sunday.vehicles)


def test_only_night_deliverable_customers_get_a_night_window():
    """夜の枠が付くのは、夜に降ろせる先だけ。

    **設備が先で体制が後**（実務）。夜に降ろせる先が無ければ夜勤は走れない。
    """
    s = generate(seed=20261015, date=FRIDAY, season="winter", shifts=2)
    for c in s.customers:
        if c.type in NIGHT_DELIVERABLE:
            assert c.window_close_min > 24 * 60
        else:
            assert c.window_close_min <= 24 * 60


def test_one_shift_is_unchanged():
    """1交代の日は、二交代を入れる前と同じ姿のまま。"""
    s = generate(seed=20261015, date=FRIDAY, season="winter")
    assert all(v.duty_window_min is None and v.shift == "day" for v in s.vehicles)
    assert all(v.physical == v.id for v in s.vehicles)
    assert s.depot.close_min == 22 * 60


def test_rejects_unknown_shift_count():
    with pytest.raises(ValueError, match="交代"):
        generate(seed=1, shifts=3)


def test_vehicle_rejects_a_broken_shift():
    with pytest.raises(ValueError, match="勤務帯"):
        Vehicle(
            id="vx", name="x", vehicle_class=VehicleClass.RIGID,
            compartments_l=(2_000, 2_000, 2_000, 4_000, 4_000), driver_id="d1",
            shift="evening",
        )
    with pytest.raises(ValueError, match="勤務帯"):
        Vehicle(
            id="vx", name="x", vehicle_class=VehicleClass.RIGID,
            compartments_l=(2_000, 2_000, 2_000, 4_000, 4_000), driver_id="d1",
            duty_window_min=(1020, 240),
        )


# ---------------------------------------------------------------------------
# 解いた結果
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_night_shift_stays_inside_its_window():
    """夜勤は勤務帯の中だけを走り、昼勤は17時までに帰る。

    ここが崩れると「法令上走れない計画」が二交代の顔をして出てくる。
    """
    s = generate(seed=20261015, date=FRIDAY, season="winter", shifts=2)
    plan = solve(s, matrix=build_matrix(s.depot, s.customers),
                 mode=SolveMode.FAST, time_limit_s=45)
    assert plan.vehicle_days
    by_id = {v.id: v for v in s.vehicles}
    for vd in plan.vehicle_days:
        lo, hi = by_id[vd.vehicle_id].duty_window_min
        assert vd.duty_start_min >= lo
        assert vd.duty_end_min <= hi
        assert vd.duty_min <= s.driver(vd.driver_id).max_duty_min


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_two_shifts_cover_the_same_day_with_fewer_vehicles():
    """同じ日を、二交代なら少ない台数で回せる。

    **これが二交代を入れる理由。** 車は遊ばせず、人を交代させる。
    """
    one = generate(seed=20261015, date=FRIDAY, season="winter", shifts=1)
    two = generate(seed=20261015, date=FRIDAY, season="winter", shifts=2)
    kw = dict(mode=SolveMode.FAST, time_limit_s=45)
    p1 = solve(one, matrix=build_matrix(one.depot, one.customers), **kw)
    p2 = solve(two, matrix=build_matrix(two.depot, two.customers), **kw)

    by_id = {v.id: v for v in two.vehicles}
    used_one = {vd.vehicle_id for vd in p1.vehicle_days}
    used_two = {by_id[vd.vehicle_id].physical for vd in p2.vehicle_days}
    assert not p1.unserved and not p2.unserved
    assert len(used_two) < len(used_one)
