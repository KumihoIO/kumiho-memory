"""Request-local, content-free recall timing. Nested stages may overlap."""
from contextvars import ContextVar
from functools import wraps
from time import perf_counter
import inspect

_current = ContextVar('recall_timings', default=None)


def timed(name):
    def decorate(fn):
        def record(start):
            timings = _current.get()
            if timings is not None:
                timings[name] = timings.get(name, 0.0) + (perf_counter() - start) * 1000
        if inspect.iscoroutinefunction(fn):
            @wraps(fn)
            async def async_wrapped(*args, **kwargs):
                start = perf_counter()
                try:
                    return await fn(*args, **kwargs)
                finally:
                    record(start)
            return async_wrapped
        @wraps(fn)
        def wrapped(*args, **kwargs):
            start = perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                record(start)
        return wrapped
    return decorate


def engage_timing(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        timings = {}
        token = _current.set(timings)
        start = perf_counter()
        try:
            result = fn(*args, **kwargs)
            timings['total'] = (perf_counter() - start) * 1000
            result['timing_ms'] = {key: round(value, 3) for key, value in timings.items()}
            # Account for diagnostics in the whole-envelope token estimate.
            from .context_compose import approx_tokens
            import json
            result.pop('approx_payload_tokens', None)
            result['approx_payload_tokens'] = approx_tokens(json.dumps(result, ensure_ascii=False, default=str))
            return result
        finally:
            _current.reset(token)
    return wrapped
