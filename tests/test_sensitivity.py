import pytest
from crrt_revision.sensitivity import evalue, robustness_value, point_bounds


def test_evalue_symmetry_and_domain():
    assert evalue(3.9) == pytest.approx(7.263034344)
    assert evalue(.8) == pytest.approx(1.809016994)
    assert evalue(.8) == evalue(1 / .8)
    assert evalue(1) == 1
    with pytest.raises(ValueError):
        evalue(-.1)


def test_rv_reaches_the_point_null():
    rv = robustness_value(-.12, .2, 5)
    lower, upper = point_bounds(-.12, .2, 5, rv, rv)
    assert upper == pytest.approx(0, abs=1e-12)
    assert lower < -.12
    assert robustness_value(0, .2, 5) == 0


