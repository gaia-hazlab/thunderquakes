from thunderquakes.evaluation import two_proportion_ztest


def test_identical_rates_give_zero_z():
    z, p = two_proportion_ztest(10, 20, 10, 20)
    assert abs(z) < 1e-9
    assert abs(p - 0.5) < 1e-9


def test_higher_candidate_rate_gives_positive_z_and_small_p():
    z, p = two_proportion_ztest(80, 100, 40, 100)
    assert z > 5
    assert p < 0.001


def test_zero_denominator_returns_nan():
    z, p = two_proportion_ztest(0, 0, 5, 10)
    assert z != z  # NaN
    assert p != p
