import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def track(name: str | None = None, phase: str | None = None):
    """
    Lightweight tracing decorator 
    We can replace this with Opik/OpenTelemetry.
    """
    def decorator(func: Callable[..., Any]):
        trace_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info("trace_start", extra={"event": trace_name, "phase": phase})

            try:
                result = await func(*args, **kwargs)
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.info(
                    "trace_end",
                    extra={"event": trace_name, "phase": phase, "latency_ms": latency_ms},
                )
                return result
            except Exception:
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.exception(
                    "trace_error",
                    extra={"event": trace_name, "phase": phase, "latency_ms": latency_ms},
                )
                raise

        return async_wrapper

    return decorator
