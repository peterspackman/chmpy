from chmpy.crystal.point_group import POINT_GROUP_FROM_NUMBER


def test_d3_settings_have_no_mirrors():
    symbols = [pg.symbol for pg in POINT_GROUP_FROM_NUMBER[18]]
    assert not any("m" in s for s in symbols)
