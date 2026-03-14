import hashlib
import os
import pickle
import time

import yfinance as yf


CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

YF_TZ_CACHE_DIR = os.path.join(CACHE_DIR, "yfinance_tz_cache")
os.makedirs(YF_TZ_CACHE_DIR, exist_ok=True)

try:
    yf.set_tz_cache_location(YF_TZ_CACHE_DIR)
except Exception:
    pass


_cache_loads = 0
_cache_misses = 0


def get_cached_data(key_name, fetch_func, *args, **kwargs):
    """Load cached data when fresh, otherwise fetch and persist it."""
    safe_key = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in key_name)
    if len(safe_key) > 200:
        safe_key = safe_key[:150] + "_" + hashlib.md5(key_name.encode()).hexdigest()

    filepath = os.path.join(CACHE_DIR, safe_key)

    global _cache_loads, _cache_misses
    if os.path.exists(filepath):
        mtime = os.path.getmtime(filepath)
        if (time.time() - mtime) < 86400:
            try:
                with open(filepath, "rb") as f:
                    _cache_loads += 1
                    return pickle.load(f)
            except Exception:
                pass

    _cache_misses += 1
    data = fetch_func(*args, **kwargs)

    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[CACHE] Error saving {filepath}: {e}")

    return data


def get_cache_stats():
    return {"loads": _cache_loads, "misses": _cache_misses}
