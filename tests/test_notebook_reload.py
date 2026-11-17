from pathlib import Path
import json
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import portfolio.app
import portfolio.cache as cache
import portfolio.market_data as market_data
import portfolio.notebook as notebook
import portfolio.reporting as reporting
import portfolio.transactions as transactions
import portfolio.benchmarking as benchmarking
import portfolio.us_portfolio as us_portfolio
from datetime import datetime


def test_report_refreshes_stale_cash_and_benchmark_modules(monkeypatch, tmp_path):
    monkeypatch.delattr(transactions, 'build_combined_cash_ledger')
    monkeypatch.setattr(benchmarking, 'simulate_stock_full', None)
    real_reload = notebook.importlib.reload
    calls = []

    def reload_with_stubbed_report(module):
        calls.append(module.__name__)
        result = real_reload(module)
        if module.__name__ == 'portfolio.app':
            def fake_main():
                assert callable(result.build_combined_cash_ledger)
                assert callable(result.simulate_stock_full)
                monkeypatch.chdir(tmp_path)
                (tmp_path / 'output').mkdir()
                (tmp_path / 'output/report.md').write_text('# Refreshed report', encoding='utf-8')
            monkeypatch.setattr(result, 'main', fake_main)
        return result

    monkeypatch.setattr(notebook.importlib, 'reload', reload_with_stubbed_report)
    path, text, generated_at = notebook.generate_report()
    assert text == '# Refreshed report'
    assert path.exists()
    assert calls.index('portfolio.transactions') < calls.index('portfolio.app')
    assert calls.index('portfolio.benchmarking') < calls.index('portfolio')


def test_report_refreshes_stale_cache_before_all_consumers(monkeypatch, tmp_path):
    def stale_cache(key, fetch, *args, **kwargs):
        # The old cache passed unknown keywords through to a fetch callback.
        return fetch(*args, **kwargs)

    consumers = (portfolio.app, reporting, us_portfolio, benchmarking, market_data)
    monkeypatch.setattr(cache, 'get_cached_data', stale_cache)
    for module in consumers:
        monkeypatch.setattr(module, 'get_cached_data', stale_cache)
    real_reload = notebook.importlib.reload
    calls = []

    def reload_with_stubbed_report(module):
        calls.append(module.__name__)
        result = real_reload(module)
        if module.__name__ == 'portfolio.app':
            def fake_main():
                monkeypatch.chdir(tmp_path)
                (tmp_path / 'cache').mkdir()
                (tmp_path / 'output').mkdir()
                refreshed_cache = cache.get_cached_data
                assert refreshed_cache is not stale_cache
                assert refreshed_cache('notebook-ttl.pkl', lambda: 42, ttl_seconds=900) == 42
                for consumer in consumers:
                    assert consumer.get_cached_data is refreshed_cache
                (tmp_path / 'output/report.md').write_text('# Refreshed cache', encoding='utf-8')
            monkeypatch.setattr(result, 'main', fake_main)
        return result

    monkeypatch.setattr(notebook.importlib, 'reload', reload_with_stubbed_report)
    path, text, generated_at = notebook.generate_report()

    assert text == '# Refreshed cache'
    assert path.exists()
    assert calls[0] == 'portfolio.cache'
    assert all(calls.index('portfolio.cache') < calls.index(module.__name__) for module in consumers)


def test_notebook_links_resolve_from_report_directory(monkeypatch):
    import IPython.display as ipython_display

    source = ('[details](report_details.md) [cash](combined_cash_ledger.csv) '
              '![chart](asset_pie_chart.png) [web](https://example.com) [section](#summary)')
    monkeypatch.setattr(notebook, 'generate_report', lambda: (
        Path('output/report.md'), source, datetime(2026, 10, 9)))
    displayed = []
    monkeypatch.setattr(ipython_display, 'clear_output', lambda **kwargs: None)
    monkeypatch.setattr(ipython_display, 'display', displayed.append)

    notebook.show_report()

    text = displayed[0].data
    assert '[details](output/report_details.md)' in text
    assert '[cash](output/combined_cash_ledger.csv)' in text
    assert '![chart](output/asset_pie_chart.png?v=20261009000000000000)' in text
    assert '[web](https://example.com)' in text
    assert '[section](#summary)' in text


def test_notebook_versions_local_images_without_changing_other_links(monkeypatch):
    import IPython.display as ipython_display

    source = ('![chart](charts/asset.png?size=large#preview) '
              '[details](report_details.md?section=put#summary) '
              '![web](https://example.com/chart.png) '
              '![cdn](//example.com/chart.png) [section](#summary)')
    generated_at = [datetime(2026, 11, 18, 8, 1, 2, 123456)]
    monkeypatch.setattr(notebook, 'generate_report', lambda: (
        Path('output/report.md'), source, generated_at[0]))
    displayed = []
    monkeypatch.setattr(ipython_display, 'clear_output', lambda **kwargs: None)
    monkeypatch.setattr(ipython_display, 'display', displayed.append)

    notebook.show_report()
    generated_at[0] = datetime(2026, 11, 18, 8, 1, 3, 654321)
    notebook.show_report()

    first, second = [item.data for item in displayed]
    assert '![chart](output/charts/asset.png?size=large&v=20261118080102123456#preview)' in first
    assert '![chart](output/charts/asset.png?size=large&v=20261118080103654321#preview)' in second
    for text in (first, second):
        assert '[details](output/report_details.md?section=put#summary)' in text
        assert '![web](https://example.com/chart.png)' in text
        assert '![cdn](//example.com/chart.png)' in text
        assert '[section](#summary)' in text


def test_startup_cell_reloads_notebook_before_generating_report(monkeypatch):
    import portfolio

    source = ''.join(json.loads((Path(__file__).resolve().parents[1] / 'startup.ipynb')
                                .read_text(encoding='utf-8'))['cells'][0]['source'])
    stale_notebook = types.ModuleType('portfolio.notebook')
    calls = []
    stale_notebook.show_report = lambda: calls.append('stale')
    monkeypatch.setitem(sys.modules, 'portfolio.notebook', stale_notebook)
    monkeypatch.setattr(portfolio, 'notebook', stale_notebook)

    def reload(module):
        assert module is stale_notebook
        calls.append('reload')
        module.show_report = lambda: calls.append('fresh')
        return module

    monkeypatch.setattr(notebook.importlib, 'reload', reload)
    exec(compile(source, 'startup.ipynb', 'exec'), {})

    assert calls == ['reload', 'fresh']


def test_report_recalculates_and_redraws_on_every_call(monkeypatch, tmp_path):
    real_reload = notebook.importlib.reload
    builds = []

    def reload_with_renderer(module):
        result = real_reload(module)
        if module.__name__ == 'portfolio.app':
            def main():
                monkeypatch.chdir(tmp_path)
                builds.append(len(builds) + 1)
                chart_path = tmp_path / 'output/charts/redraw.svg'
                result.charts.figure()
                result.charts.plot([0, 1], [0, builds[-1]], label='Recomputed')
                result.save_report_chart(chart_path)
                result.charts.close()
                (tmp_path / 'output/report.md').write_text('# Current report', encoding='utf-8')
            monkeypatch.setattr(result, 'main', main)
        return result

    monkeypatch.setattr(notebook.importlib, 'reload', reload_with_renderer)
    notebook.generate_report()
    first = (tmp_path / 'output/charts/redraw.svg').read_bytes()
    notebook.generate_report()
    second = (tmp_path / 'output/charts/redraw.svg').read_bytes()

    assert builds == [1, 2]
    # Shape is proportional for two points; distinct domains still change tick labels.
    assert first != second
