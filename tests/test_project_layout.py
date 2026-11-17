from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import portfolio.app as app


def test_chart_gallery_uses_current_charts_directory(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    chart = tmp_path / 'output/charts/example.svg'
    chart.parent.mkdir(parents=True)
    chart.write_bytes(b'chart')
    monkeypatch.setattr(app, 'REPORT_CHARTS', [('Current chart', 'example.svg')])

    app.print_chart_gallery()

    assert '![Current chart](charts/example.svg)' in capsys.readouterr().out
