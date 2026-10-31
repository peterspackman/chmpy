from chmpy.crystal.point_group import POINT_GROUP_DATA, POINT_GROUP_FROM_NUMBER


def test_d3_settings_have_no_mirrors():
    symbols = [pg.symbol for pg in POINT_GROUP_FROM_NUMBER[18]]
    assert not any("m" in s for s in symbols)


ORDERS = {
    1: 1, 2: 2, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8, 9: 4, 10: 4, 11: 8,
    12: 8, 13: 8, 14: 8, 15: 16, 16: 3, 17: 6, 18: 6, 19: 6, 20: 12,
    21: 6, 22: 6, 23: 12, 24: 12, 25: 12, 26: 12, 27: 24, 28: 12, 29: 24,
    30: 24, 31: 24, 32: 48,
}  # fmt: skip


def test_every_point_group_has_its_order():
    for pg in POINT_GROUP_DATA:
        assert len(set(pg.symmetry_operations)) == ORDERS[pg.number], pg


def test_every_point_group_is_closed():
    for pg in POINT_GROUP_DATA:
        ops = set(pg.symmetry_operations)
        assert {a.compose(b) for a in ops for b in ops} == ops, pg


def test_operations_have_no_translation():
    for pg in POINT_GROUP_DATA:
        assert all(not s.translation.any() for s in pg.symmetry_operations), pg
