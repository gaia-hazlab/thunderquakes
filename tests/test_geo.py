
from thunderquakes.geo import haversine_km


def test_zero_distance():
    assert haversine_km(47.6, -122.3, 47.6, -122.3) == 0.0


def test_known_distance_seattle_portland():
    # ~233 km great-circle
    d = haversine_km(47.61, -122.33, 45.52, -122.68)
    assert 220 < d < 245
