from functools import wraps
from .clip_gate import track_clip_score


def clip_guard(th=.30):
    def deco(fn):
        @wraps(fn)
        def wrap(track_candidates, *a, **kw):
            track_uuid = track_candidates
            log_dir = next(x for x in a if hasattr(x, "exists"))  # 第一个 Path 视为 log_dir
            if track_clip_score(track_uuid, log_dir) < th:
                return [] if fn.__name__ != "is_category" else []  # 返回“空掩码”
            return fn(track_candidates, *a, **kw)

        return wrap

    return deco
