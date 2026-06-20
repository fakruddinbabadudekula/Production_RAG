"""Basic Logging setup"""
import logging
from datetime import datetime,timezone
from app.core.contextvar import request_id_ctx,request_route_ctx
import json
import sys
class TraceIDFilter(logging.Filter):
            """
            Injects ContextVar values into every log record automatically.
            Attached to the handler, so it runs for every logger in the app.
            """
            def filter(self, record: logging.LogRecord) -> bool:
                record.request_id=request_id_ctx.get()
                record.request_route=request_route_ctx.get()
                return True 
            
            
class JSONFormatter(logging.Formatter):
            """
            Emits one JSON object per line — the standard for
            log aggregators like Datadog, CloudWatch, Loki, ELK.
            """
            def format(self, record: logging.LogRecord) -> str:
                # Base structure every log line will have
                log_record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level":     record.levelname,
                    "logger":    record.name,
                    "message":   record.getMessage(),  #why we use getMessage() instead of var .message because of when we log we pass like this "no.of %s",5 where code %s replace it with 5 but when we use .message it gives as it is and 5 goes record.args so we use getMessage() where it performs messages+record.args
                    # Injected by TraceIDFilter
                    "request_id":  getattr(record, "request_id", "-"),
                    "request_route":   getattr(record, "request_route", "-"),
                    # Code location — invaluable for debugging
                    "file":      f"{record.filename}:{record.lineno}",
                    "function":  record.funcName,
                }

                # If the log call included extra={} fields, merge them in
                # e.g. logger.info("msg", extra={"duration_ms": 42})
                if hasattr(record, "__dict__"):
                    standard_keys = {
                        "timestamp", "level", "logger", "message",
                        "trace_id", "route", "file", "function",
                        # internal Python logging fields to exclude
                        "name", "msg", "args", "levelname", "levelno",
                        "pathname", "filename", "module", "funcName",
                        "lineno", "created", "msecs", "relativeCreated",
                        "thread", "threadName", "processName", "process",
                        "exc_info", "exc_text", "stack_info",
                    }
                    #this is for now only later optimize them,
                    #what __dict__.items() returns it gives all the keys
                    for key, val in record.__dict__.items():
                        if key not in standard_keys:
                            if isinstance(val, float):
                                log_record[key] = round(val, 3)
                            else:
                                log_record[key] = val
            #Note why we just do this log_record['extra']=record.extra
            # why not: when we pass extra internally python flattens this like record.first_key_in_extra=value.... for all values in extra fields so we don't know what extra fields are that's why we create a standard keys value list where already handled them or log inot log_record remaining all are considered as extra valeus
            
                if hasattr(record, 'duration'):
                    log_record['duration']=round(log_record['duration'],3)
                # Attach exception traceback if present
                if record.exc_info:
                    log_record["exception"] = self.formatException(record.exc_info)
                #why we use default as str bcz it automatically type conversion to str if it can't handle data formate like uuid, timezones and other formates or data types it simply do str(uuid)
                return json.dumps(log_record, default=str)


def setup_logging(level: str = "INFO") -> None:
    """
    Call once at app startup in main.py.
    Configures root logger so every logger in the app
    (including third-party libraries) inherits the setup.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    handler.addFilter(TraceIDFilter())   # ← attaches to handler, not per-logger

    root_logger = logging.getLogger()    # root = parent of all loggers
    root_logger.setLevel(level)
    root_logger.handlers.clear()        # remove default handlers
    root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    