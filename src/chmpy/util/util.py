import inspect
import logging
from functools import wraps

LOG = logging.getLogger(__name__)


def cache_defaults(method):
    """
    Class method decorator to cache the results when calling
    with default arguments.
    """
    sig = inspect.signature(method)
    m_name = method.__name__
    cache_attr = f"_cached_{m_name}"

    # Extract default values for all parameters except self
    default_bound = sig.bind_partial()
    default_bound.apply_defaults()
    defaults = default_bound.arguments

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind_partial(self, *args, **kwargs)
        bound.apply_defaults()

        # Check if all arguments match the method's defaults
        called_args = dict(bound.arguments)
        called_args.pop(next(iter(sig.parameters)), None)
        if called_args != defaults:
            # If arguments do not match defaults, run the method
            LOG.debug("Running %s() with arguments: %s", m_name, called_args)
            return method(self, *args, **kwargs)

        if not hasattr(self, cache_attr):
            LOG.debug(
                "Running %s() and caching with arguments: %s", m_name, called_args
            )
            result = method(self, *args, **kwargs)
            setattr(self, cache_attr, result)
            return result

        LOG.debug("Using cached result for %s()", m_name)
        return getattr(self, cache_attr)

    return wrapper
