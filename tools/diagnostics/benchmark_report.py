"""Measure complete fresh redraws; never reuse a completed report or chart."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')

    started = time.perf_counter()
    import portfolio.notebook as notebook
    import portfolio.cache as cache
    import_seconds = time.perf_counter() - started
    sources = sorted((ROOT / 'data/transactions').glob('*.csv'))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    original_reload = notebook.importlib.reload
    writes = []

    def instrument_reload(module):
        result = original_reload(module)
        if module.__name__ == 'portfolio.svg_charts':
            original_save = result.Figure.savefig

            @functools.wraps(original_save)
            def save(figure, path, **kwargs):
                start = time.perf_counter()
                original_save(figure, path, **kwargs)
                writes.append({'file': Path(path).name, 'seconds': time.perf_counter() - start})

            result.Figure.savefig = save
        return result

    runs = []
    with patch.object(notebook.importlib, 'reload', instrument_reload):
        for _ in range(args.runs):
            writes.clear()
            before = {path.name: path.stat().st_mtime_ns for path in (ROOT / 'output/charts').glob('*.svg')}
            start = time.perf_counter()
            notebook.generate_report()
            elapsed = time.perf_counter() - start
            after = {path.name: path.stat().st_mtime_ns for path in (ROOT / 'output/charts').glob('*.svg')}
            runs.append({'full_seconds': elapsed, 'drawing_seconds': sum(row['seconds'] for row in writes),
                         'draw_calls': len(writes), 'rewritten_files': sum(before.get(k) != v for k, v in after.items()),
                         'cache': cache.get_cache_stats()})
    assert hashes == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    result = {'initial_import_seconds': import_seconds, 'runs': runs, 'source_csv_unchanged': True}
    target = ROOT / 'output/diagnostics/svg-report-timing.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
