from functools import wraps


def cached_property(method):
    """Decorates like a @property but also caches the result on the instance."""
    name = "_%s_cached_property" % method.__name__  # assumed to be unique

    @property
    @wraps(method)
    def wrapper(self):
        try:
            return getattr(self, name)
        except AttributeError:
            pass
        result = method(self)
        setattr(self, name, result)
        return result

    return wrapper
