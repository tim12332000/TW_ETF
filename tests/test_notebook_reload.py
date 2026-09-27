from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import portfolio.app
import portfolio.notebook as notebook
import portfolio.transactions as transactions
import portfolio.benchmarking as benchmarking


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
    # The production notebook suppresses plot windows; restore shared pyplot
    # state when this test exits.
    monkeypatch.setattr(portfolio.app.plt, 'show', portfolio.app.plt.show)
    path, text, generated_at = notebook.generate_report()
    assert text == '# Refreshed report'
    assert path.exists()
    assert calls.index('portfolio.transactions') < calls.index('portfolio.app')
    assert calls.index('portfolio.benchmarking') < calls.index('portfolio')
