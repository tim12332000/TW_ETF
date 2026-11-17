from pathlib import Path
from datetime import datetime, timedelta
import math
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from portfolio import svg_charts as charts


NS = {'s': 'http://www.w3.org/2000/svg'}


def document(figure):
    return ET.fromstring(figure.to_svg())


def matching(root, tag, css_class):
    return [node for node in root.findall(f'.//s:{tag}', NS) if node.get('class') == css_class]


@pytest.fixture(autouse=True)
def reset_chart():
    charts.close()
    yield
    charts.close()


def test_lines_preserve_all_points_and_missing_value_gaps():
    fig, ax = charts.subplots()
    values = np.array([1., 2., np.nan, 4., 5., 6.])
    before = values.copy()
    ax.plot(np.arange(6), values, label='Portfolio')
    path = matching(document(fig), 'path', 'series-line')[0].get('d')

    assert path.count('M') == 2
    assert path.count('L') == 3
    assert 'nan' not in path
    np.testing.assert_array_equal(values, before)


def test_dense_line_does_not_downsample():
    fig = charts.figure()
    charts.plot(np.arange(1500), np.sin(np.arange(1500)))
    path = matching(document(fig), 'path', 'series-line')[0].get('d')

    assert path.count('M') == 1
    assert path.count('L') == 1499


def test_fill_where_keeps_disjoint_regions_separate():
    fig, ax = charts.subplots()
    ax.fill_between(np.arange(7), np.arange(7) + 1, 0,
                    where=[True, True, False, True, True, False, False])
    areas = matching(document(fig), 'path', 'series-area')

    assert len(areas) == 2
    assert all(node.get('d').endswith(' Z') for node in areas)


def test_dates_and_twenty_day_bars_use_date_units():
    fig, ax = charts.subplots()
    dates = pd.to_datetime(['2026-01-01', '2026-01-31'])
    bars = ax.bar(dates, [100, 200], width=20)
    ax.set_xlabel('月份')
    root = document(fig)
    plot_width = float(root.find('.//s:clipPath/s:rect', NS).get('width'))
    bar_width = float(matching(root, 'rect', 'series-bar')[0].get('width'))

    assert bars[0].get_width() == 20
    assert 0.38 < bar_width / plot_width < 0.41
    assert '01-' in fig.to_svg()


def test_categorical_bars_have_labels_and_correct_handles():
    fig, ax = charts.subplots()
    bars = ax.bar(['-30%', '-50%', '-70%', '-90%'], [20, 50, 100, 150])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{bar.get_height():.0f}%', ha='center', va='bottom')
    svg = fig.to_svg()

    assert len(matching(document(fig), 'rect', 'series-bar')) == 4
    assert bars[2].get_x() + bars[2].get_width() / 2 == 2
    assert all(label in svg for label in ('-30%', '-90%', '150%'))


def test_stackplot_preserves_cash_and_exact_percent_limits():
    fig, ax = charts.subplots(figsize=(14, 7))
    ax.stackplot([0, 1, 2], [[60, 55, 50], [36, 41, 46], [4, 4, 4]],
                 labels=['股票', 'ETF', 'Cash (inferred)'], colors=['red', 'blue', '#969696'])
    ax.set_ylim(0, 100)
    ax.legend()
    root = document(fig)

    np.testing.assert_array_equal(ax.artists[-1].y, [100, 100, 100])
    np.testing.assert_array_equal(ax.artists[-1].y2, [96, 96, 96])
    assert len(matching(root, 'path', 'series-area')) == 3
    assert 'Cash (inferred)' in fig.to_svg()
    assert ax._limits('y') == (0, 100)


def test_pie_keeps_every_slice_and_escapes_long_legend():
    fig = charts.figure(figsize=(12, 7))
    values = np.arange(1, 11)
    wedges, labels = charts.pie(values, colors=charts.get_cmap('tab20').colors, startangle=140)
    names = [f'股票 & ETF <{index}> {value / values.sum():.2%}' for index, value in enumerate(values)]
    charts.legend(wedges, names)
    root = document(fig)

    assert len(wedges) == len(matching(root, 'path', 'pie-slice')) == 10
    assert wedges[-1].x[-1] - wedges[0].x[0] == pytest.approx(360)
    assert len(labels) == 0
    visible_text = ''.join(root.itertext())
    assert all(name in visible_text for name in names)
    assert '&amp;' in fig.to_svg() and '&lt;' in fig.to_svg()


@pytest.mark.parametrize('startangle', [0, 140, 359])
def test_single_slice_pie_is_a_circle(startangle):
    fig = charts.figure()
    charts.pie([100], startangle=startangle)

    assert len(document(fig).findall('.//s:circle', NS)) == 1


def test_twin_axes_use_independent_scales_and_combined_legend():
    fig, left = charts.subplots()
    left.plot([0, 1], [0, 100], label='資產', color='blue')
    right = left.twinx()
    right.plot([0, 1], [0, 1000], label='Put', color='purple')
    right.set_ylabel('Put 價值')
    handles, labels = left.get_legend_handles_labels()
    other_handles, other_labels = right.get_legend_handles_labels()
    left.legend(handles + other_handles, labels + other_labels)
    paths = matching(document(fig), 'path', 'series-line')

    assert paths[0].get('d') == paths[1].get('d')
    assert 'Put 價值' in fig.to_svg()
    assert len(left._legend_entries()) == 2


def test_two_panel_ratios_and_offset_annotation_are_valid_svg():
    fig, (left, right) = charts.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.7, 1]})
    left.plot([0, 1], [0, 1])
    left.scatter([1], [1], marker='X')
    left.annotate('目前實價', xy=(1, 1), xytext=(-120, 20), arrowprops={'color': 'blue'})
    right.bar(['A', 'B'], [1, 2])
    fig.suptitle('損益比較')
    fig.text(.5, .01, '模型說明', ha='center')
    root = document(fig)
    rects = root.findall('.//s:clipPath/s:rect', NS)

    assert len(rects) == 2
    assert float(rects[0].get('width')) > float(rects[1].get('width'))
    assert '目前實價' in ''.join(root.itertext())
    assert '模型說明' in ''.join(root.itertext())


def test_repeated_save_serializes_new_data_without_mutating_previous_file(tmp_path):
    fig = charts.figure()
    charts.plot([0, 1], [1, 2], label='My Portfolio')
    charts.legend()
    first = tmp_path / 'portfolio.svg'
    second = tmp_path / 'comparison.svg'
    fig.savefig(first)
    original = first.read_bytes()
    charts.plot([0, 1], [2, 3], label='QQQ')
    fig.savefig(second)
    fig.savefig(first)

    assert original != first.read_bytes()
    assert first.read_bytes() == second.read_bytes()
    assert len(matching(ET.fromstring(original), 'path', 'series-line')) == 1
    assert len(matching(ET.parse(second).getroot(), 'path', 'series-line')) == 2


def test_xml_escaping_applies_to_titles_annotations_and_labels():
    fig, ax = charts.subplots()
    ax.plot([0, 1], [0, 1], label='A&B <C>')
    ax.set_title('<script>alert("x")</script>')
    ax.annotate('A&B', xy=(.5, .5))
    ax.legend()
    root = document(fig)

    assert root.findall('.//s:script', NS) == []
    assert '<script>alert("x")</script>' in ''.join(root.itertext())


def test_png_extension_is_not_silently_mislabeled(tmp_path):
    fig = charts.figure()
    with pytest.raises(ValueError, match='SVG'):
        fig.savefig(tmp_path / 'chart.png')


def test_series_plot_uses_its_date_index():
    fig = charts.figure()
    series = pd.Series([10, 20], index=pd.to_datetime(['2026-01-01', '2026-02-01']))
    charts.plot(series)

    assert fig.axes[0].xkind == 'date'
    assert fig.axes[0].artists[0].x[1] - fig.axes[0].artists[0].x[0] == 31


def test_shared_date_panels_keep_identical_horizontal_scale():
    fig, (top, bottom) = charts.subplots(2, 1, sharex=True)
    top.plot([0, 1], [2, 3])
    bottom.plot([0, 1, 2], [2, 3, 4])

    assert top._limits('x') == bottom._limits('x')
    assert len(matching(document(fig), 'path', 'series-line')) == 2


def test_nonfinite_bars_do_not_emit_invalid_svg_coordinates():
    fig, ax = charts.subplots()
    ax.bar(['A', 'B'], [np.nan, 10])
    svg = fig.to_svg()

    assert len(matching(ET.fromstring(svg), 'rect', 'series-bar')) == 1
    assert not re.search(r'(?:x|y|width|height)="(?:nan|inf)', svg)


def test_explicit_colors_and_fills_do_not_advance_default_line_colors():
    fig, ax = charts.subplots()
    ax.plot([0, 1], [1, 2], color='blue', label='My Portfolio')
    ax.fill_between([0, 1], [1, 2], color='red')
    for name in ('SPY', 'QQQ', 'EWT'):
        ax.plot([0, 1], [2, 3], label=name)

    lines = [artist for artist in ax.artists if artist.kind == 'line']
    assert [line.style['color'] for line in lines] == ['blue', *charts.TAB10[:3]]


def test_monthly_bars_share_one_default_color():
    fig, ax = charts.subplots()
    bars = ax.bar(pd.date_range('2026-01-01', periods=4, freq='MS'), [1, 2, 3, 4], width=20)

    assert {bar.style['color'] for bar in bars} == {charts.TAB10[0]}


def test_nice_money_ticks_have_simple_steps_and_compact_labels():
    ticks = charts._nice_ticks((0, 3_588_990))

    np.testing.assert_array_equal(ticks, np.arange(0, 3_500_001, 500_000))
    assert charts._tick_label(2_843_850, 'number', 3_000_000) == '2.84M'
    assert charts._tick_label(25_000, 'number', 30_000) == '25k'
    assert charts._tick_label(0, 'number', 1) == '0'


def test_long_date_ticks_are_calendar_years_not_december_offsets():
    fig, ax = charts.subplots()
    ax.plot(pd.to_datetime(['2021-01-11', '2026-11-18']), [1, 2])
    values, labels = charts._date_ticks(ax._limits('x'))

    assert labels == ['2021', '2022', '2023', '2024', '2025', '2026']
    dates = [datetime(1970, 1, 1) + timedelta(days=float(value)) for value in values]
    assert all(date.month == date.day == 1 for date in dates)


@pytest.mark.parametrize('values, expected_zero', [([0, 10, 100], 0), ([-100, -10, 0], 1)])
def test_padding_does_not_cross_zero_for_one_sided_values(values, expected_zero):
    fig, ax = charts.subplots()
    ax.plot(np.arange(len(values)), values)

    assert ax._limits('y')[expected_zero] == 0
    assert all(math.isfinite(value) for value in ax._limits('y'))


def test_explicit_percent_limits_and_flat_zero_remain_valid():
    fig, ax = charts.subplots()
    ax.plot([0, 1], [0, 0])
    assert ax._limits('y') == (0, 1)
    ax.set_ylim(0, 100)
    assert ax._limits('y') == (0, 100)


def test_single_date_uses_one_day_padding_instead_of_epoch_magnitude():
    fig, ax = charts.subplots()
    ax.plot(pd.to_datetime(['2026-11-18']), [100])
    day = ax.artists[0].x[0]

    assert ax._limits('x') == (day - 1, day + 1)
    assert len(charts._date_ticks(ax._limits('x'))[0]) == 3
