import contextlib
import importlib
import io
from datetime import datetime
from pathlib import Path


def generate_report():
    """Generate the portfolio report from a live notebook kernel."""
    import portfolio
    import portfolio.transactions
    import portfolio.performance
    import portfolio.benchmarking
    import portfolio.positions
    import portfolio.reporting
    import portfolio.tw_portfolio
    import portfolio.us_portfolio

    # Refresh dependencies before importing/reloading app: a live kernel may
    # still have transactions from before a newly added function existed.
    importlib.reload(portfolio.transactions)
    importlib.reload(portfolio.performance)
    importlib.reload(portfolio.benchmarking)
    importlib.reload(portfolio.positions)
    importlib.reload(portfolio.reporting)
    importlib.reload(portfolio.tw_portfolio)
    importlib.reload(portfolio.us_portfolio)
    importlib.reload(portfolio)
    import portfolio.app
    importlib.reload(portfolio.app)

    portfolio.app.plt.show = lambda *args, **kwargs: None
    portfolio.reporting.plt.show = lambda *args, **kwargs: None

    with contextlib.redirect_stdout(io.StringIO()):
        portfolio.app.main()

    report_path = Path("output/report.md")
    report_text = report_path.read_text(encoding="utf-8")
    _assert_report_sanity(report_text)
    return report_path, report_text, datetime.now()


def show_report():
    from IPython.display import Markdown, clear_output, display

    clear_output(wait=True)
    report_path, report_text, generated_at = generate_report()
    timestamp = generated_at.strftime("%Y-%m-%d %H:%M:%S")
    display(Markdown(f"> Fresh report generated from `{report_path}` at {timestamp}.\n\n" + report_text))


def _assert_report_sanity(report_text):
    spcx_line = next((line for line in report_text.splitlines() if line.startswith("| SPCX")), "")
    if "|     0" in spcx_line or "| 0.00      |        555    |     0" in spcx_line:
        raise RuntimeError(f"SPCX sanity check failed after report generation: {spcx_line}")
