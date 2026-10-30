from chmpy.util.util import cache_defaults


class Counter:
    def __init__(self):
        self.calls = 0

    @cache_defaults
    def value(self, scale=1, **kwargs):
        self.calls += 1
        return [scale * self.calls]


def test_default_call_is_cached():
    c = Counter()
    assert c.value() is c.value()


def test_explicit_default_uses_cache():
    c = Counter()
    first = c.value()
    assert c.value(scale=1) is first


def test_non_default_call_is_not_cached():
    c = Counter()
    c.value(scale=2)
    c.value(scale=2)
    assert c.calls == 2


def test_non_default_call_does_not_replace_cache():
    c = Counter()
    first = c.value()
    c.value(scale=2)
    assert c.value() is first


def test_extra_kwargs_bypass_cache():
    c = Counter()
    first = c.value()
    assert c.value(extra=True) is not first


def test_cache_is_per_instance():
    a, b = Counter(), Counter()
    assert a.value() is not b.value()
