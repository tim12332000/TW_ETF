"""Small, fresh SVG renderer for the portfolio report's chart vocabulary.

Coordinates stay in their original units until serialization.  Every save
walks the current data; there is no image or result cache.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np


TAB10 = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
TAB20 = ('#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
         '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
         '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
         '#17becf', '#9edae5')
_current = None


def _color(value):
    if isinstance(value, (tuple, list, np.ndarray)):
        return '#%02x%02x%02x' % tuple(round(float(c) * 255) for c in value[:3])
    return str(value)


def _f(value):
    return f'{float(value):.3f}'.rstrip('0').rstrip('.')


def _text(x, y, text, *, size=12, color='#334155', anchor='start', weight='normal', rotate=None):
    transform = f' transform="rotate({_f(rotate)} {_f(x)} {_f(y)})"' if rotate else ''
    return (f'<text x="{_f(x)}" y="{_f(y)}" font-size="{_f(size)}" '
            f'fill="{escape(_color(color), quote=True)}" text-anchor="{anchor}" '
            f'font-weight="{escape(str(weight), quote=True)}"{transform}>'
            f'{escape(str(text))}</text>')


def _line(x1, y1, x2, y2, color='#cbd5e1', width=1, opacity=1, dash=''):
    return (f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" '
            f'stroke="{escape(_color(color), quote=True)}" stroke-width="{_f(width)}" '
            f'opacity="{_f(opacity)}"{dash}/>')


def _runs(mask):
    """Yield contiguous true regions without connecting across missing data."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _path(x, y, mask=None):
    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= mask
    return ' '.join('M' + ' L'.join(f'{_f(a)},{_f(b)}' for a, b in zip(x[start:end], y[start:end]))
                    for start, end in _runs(valid))


def _numbers(values):
    array = np.atleast_1d(np.asarray(values))
    if array.dtype.kind == 'M':
        return array.astype('datetime64[ms]').astype(float) / 86400000.0, 'date', []
    if array.dtype.kind in 'OUS' and len(array):
        sample = next((value for value in array if value is not None), None)
        if isinstance(sample, (datetime, np.datetime64)):
            return array.astype('datetime64[ms]').astype(float) / 86400000.0, 'date', []
        if isinstance(sample, str):
            return np.arange(len(array), dtype=float), 'category', list(map(str, array))
    return array.astype(float), 'number', []


def _label_width(text, size=11):
    return sum(1.0 if ord(char) > 255 else 0.56 for char in str(text)) * size


def _tick_label(value, kind, span):
    if kind == 'date':
        date = datetime(1970, 1, 1) + timedelta(days=float(value))
        return date.strftime('%Y-%m' if span > 180 else '%m-%d')
    value = 0.0 if abs(value) < 1e-12 else value
    if abs(value) >= 1_000_000:
        return f'{value / 1_000_000:.3g}M'
    if abs(value) >= 1000:
        return f'{value / 1000:.3g}k'
    return f'{value:.3g}'


def _nice_ticks(limits, count=6):
    low, high = limits
    raw_step = (high - low) / max(count - 1, 1)
    scale = 10 ** math.floor(math.log10(raw_step))
    step = min((scale * value for value in (1, 2, 2.5, 5, 10)),
               key=lambda value: abs(value - raw_step))
    start = math.ceil(low / step - 1e-10)
    stop = math.floor(high / step + 1e-10)
    return np.arange(start, stop + 1, dtype=float) * step


def _date_ticks(limits, count=7):
    low, high = limits
    span = high - low
    epoch = datetime(1970, 1, 1)
    if span >= 60:
        raw_months = span / 30.44 / max(count - 1, 1)
        step = min((1, 2, 3, 6, 12, 24, 36, 60, 120), key=lambda value: abs(value - raw_months))
        first = epoch + timedelta(days=low)
        month = first.year * 12 + first.month - 1
        month = math.ceil(month / step) * step
        values = []
        while True:
            date = datetime(month // 12, month % 12 + 1, 1)
            value = (date - epoch).total_seconds() / 86400
            if value > high:
                break
            if value >= low:
                values.append(value)
            month += step
        pattern = '%Y' if step >= 12 else '%Y-%m'
    else:
        raw_days = span / max(count - 1, 1)
        step = min((1, 2, 7, 14, 28), key=lambda value: abs(value - raw_days))
        anchor = 4 if step >= 7 else 0  # Monday, 1970-01-05.
        start = math.ceil((low - anchor) / step)
        stop = math.floor((high - anchor) / step)
        values = np.arange(start, stop + 1) * step + anchor
        pattern = '%m-%d'
    return np.asarray(values, dtype=float), [(epoch + timedelta(days=float(value))).strftime(pattern) for value in values]


@dataclass
class Artist:
    kind: str
    x: np.ndarray
    y: np.ndarray
    style: dict = field(default_factory=dict)
    label: str = ''
    y2: np.ndarray | None = None
    where: np.ndarray | None = None
    width: float = 0.8

    def get_x(self):
        return float(self.x[0] - self.width / 2)

    def get_width(self):
        return self.width

    def get_height(self):
        return float(self.y[0])


class Axes:
    def __init__(self, figure, row=0, col=0, *, twin_of=None):
        self.figure = figure
        self.row, self.col = row, col
        self.twin_of = twin_of
        self.twin = None
        self.artists = []
        self.texts = []
        self.annotations = []
        self.title = self.xlabel = self.ylabel = ''
        self.title_style = {}
        self.ylabel_style = {}
        self.ylim = None
        self.xkind = 'number'
        self.categories = []
        self.grid_options = None
        self.legend_options = None
        self.legend_handles = None
        self.rotation = 0
        self.ycolor = '#475569'
        self._line_color_index = 0

    def _x(self, values):
        x, kind, labels = _numbers(values)
        if kind != 'number':
            self.xkind, self.categories = kind, labels
        return x

    def _artist(self, kind, x, y, **kwargs):
        style = dict(kwargs)
        label = str(style.pop('label', ''))
        style.setdefault('color', TAB10[0])
        artist = Artist(kind, np.asarray(x, dtype=float), np.asarray(y, dtype=float), style, label)
        self.artists.append(artist)
        return artist

    def plot(self, x, y=None, **kwargs):
        if y is None:
            y, x = x, getattr(x, 'index', np.arange(len(x)))
        if kwargs.get('color') is None:
            kwargs['color'] = TAB10[self._line_color_index % len(TAB10)]
            self._line_color_index += 1
        return [self._artist('line', self._x(x), np.atleast_1d(y), **kwargs)]

    def fill_between(self, x, y1, y2=0, where=None, **kwargs):
        x = self._x(x)
        y1, y2 = np.broadcast_arrays(np.asarray(y1, dtype=float), np.asarray(y2, dtype=float), x)[:2]
        artist = self._artist('area', x, y1, **kwargs)
        artist.y2 = y2.copy()
        artist.where = None if where is None else np.broadcast_to(np.asarray(where, dtype=bool), x.shape).copy()
        return artist

    def stackplot(self, x, values, labels=None, colors=None, **kwargs):
        x = self._x(x)
        labels = list(labels) if labels is not None else [''] * len(values)
        colors = list(colors) if colors is not None else TAB20
        lower = np.zeros(len(x))
        result = []
        for index, values_y in enumerate(values):
            upper = lower + np.asarray(values_y, dtype=float)
            artist = self._artist('area', x, upper.copy(), label=labels[index],
                                  color=colors[index % len(colors)], **kwargs)
            artist.y2 = lower.copy()
            result.append(artist)
            lower = upper
        return result

    def scatter(self, x, y, **kwargs):
        return self._artist('scatter', self._x(x), np.atleast_1d(y), **kwargs)

    def bar(self, x, y, width=0.8, **kwargs):
        x = self._x(x)
        y = np.atleast_1d(np.asarray(y, dtype=float))
        artists = []
        for index, (value_x, value_y) in enumerate(zip(x, y)):
            style = dict(kwargs)
            if index:
                style.pop('label', None)
            artist = self._artist('bar', [value_x], [value_y], **style)
            artist.width = float(width)
            artists.append(artist)
        return artists

    def pie(self, values, colors=None, startangle=0, **kwargs):
        values = np.asarray(values, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
            raise ValueError('Pie values must be finite, non-negative and have a positive total')
        colors = list(colors) if colors is not None else TAB20
        angle = float(startangle)
        result = []
        for index, value in enumerate(values):
            end = angle + float(value / values.sum()) * 360
            result.append(self._artist('pie', [angle, end], [value], color=colors[index % len(colors)]))
            angle = end
        return result, []

    def axhline(self, y=0, **kwargs):
        return self._artist('hline', [], [float(y)], **kwargs)

    def axvline(self, x=0, **kwargs):
        return self._artist('vline', self._x([x]), [], **kwargs)

    def set_title(self, title, **kwargs):
        self.title, self.title_style = str(title), kwargs

    def set_xlabel(self, label, **kwargs):
        self.xlabel = str(label)

    def set_ylabel(self, label, **kwargs):
        self.ylabel, self.ylabel_style = str(label), kwargs
        self.ycolor = kwargs.get('color', self.ycolor)

    def set_ylim(self, bottom=None, top=None):
        if top is None and np.ndim(bottom):
            bottom, top = bottom
        self.ylim = (bottom, top)

    def tick_params(self, axis='both', rotation=0, labelcolor=None, **kwargs):
        if axis in ('x', 'both'):
            self.rotation = rotation
        if labelcolor is not None and axis in ('y', 'both'):
            self.ycolor = labelcolor

    def grid(self, visible=True, **kwargs):
        self.grid_options = kwargs if visible else None

    def legend(self, handles=None, labels=None, **kwargs):
        self.legend_options = kwargs
        self.legend_handles = None if handles is None else list(zip(handles, map(str, labels)))

    def get_legend_handles_labels(self):
        artists = [artist for artist in self.artists if artist.label and not artist.label.startswith('_')]
        return artists, [artist.label for artist in artists]

    def text(self, x, y, text, **kwargs):
        self.texts.append((float(x), float(y), str(text), kwargs))

    def annotate(self, text, xy, xytext=(0, 0), **kwargs):
        self.annotations.append((self._x([xy[0]])[0], float(xy[1]), str(text), xytext, kwargs))

    def twinx(self):
        if self.twin is None:
            self.twin = Axes(self.figure, self.row, self.col, twin_of=self)
        return self.twin

    def _legend_entries(self):
        if self.legend_options is None:
            return []
        if self.legend_handles is not None:
            return self.legend_handles
        artists, labels = self.get_legend_handles_labels()
        return list(zip(artists, labels))

    def _limits(self, coordinate):
        values = []
        for artist in self.artists:
            data = artist.x if coordinate == 'x' else artist.y
            if artist.kind == 'pie':
                continue
            if data.size:
                values.extend(data[np.isfinite(data)].tolist())
            if artist.kind == 'area' and coordinate == 'y':
                values.extend(artist.y2[np.isfinite(artist.y2)].tolist())
            if artist.kind == 'bar':
                if coordinate == 'x':
                    values.extend([artist.x[0] - artist.width / 2, artist.x[0] + artist.width / 2])
                else:
                    values.append(0.0)
        if coordinate == 'x' and self.twin is not None:
            for artist in self.twin.artists:
                values.extend(artist.x[np.isfinite(artist.x)].tolist())
        if coordinate == 'x' and self.figure.sharex:
            for other in self.figure.axes:
                if other is self:
                    continue
                for artist in other.artists:
                    if artist.kind != 'pie':
                        values.extend(artist.x[np.isfinite(artist.x)].tolist())
        low, high = (min(values), max(values)) if values else (0.0, 1.0)
        if coordinate == 'y' and self.ylim is not None:
            low = low if self.ylim[0] is None else float(self.ylim[0])
            high = high if self.ylim[1] is None else float(self.ylim[1])
            return low, high if high != low else low + 1
        if low == high == 0 and not (coordinate == 'x' and self.xkind == 'date'):
            return 0.0, 1.0
        if high == low:
            padding = 1.0 if coordinate == 'x' and self.xkind == 'date' else abs(high) * 0.05 or 1.0
        else:
            padding = (high - low) * (0.045 if coordinate == 'y' else 0.01)
        padded_low, padded_high = low - padding, high + padding
        if coordinate == 'y':
            if low >= 0:
                padded_low = max(0.0, padded_low)
            if high <= 0:
                padded_high = min(0.0, padded_high)
        return padded_low, padded_high

    def _render(self, panel, panel_index):
        left, top, width, height = panel
        entries = self._legend_entries()
        font_size = 11
        legend_rows, row_width = 0, 0
        legend_positions = []
        for artist, label in entries:
            item_width = min(width - 20, _label_width(label, font_size) + 36)
            if row_width and row_width + item_width > width - 15:
                legend_rows += 1
                row_width = 0
            legend_positions.append((row_width, legend_rows, artist, label))
            row_width += item_width + 12
        legend_height = (legend_rows + 1) * 20 if entries else 0
        is_pie = any(artist.kind == 'pie' for artist in self.artists)
        plot_left = left + (25 if is_pie else 83)
        plot_top = top + (42 if self.title else 23)
        plot_width = width - (50 if is_pie else (163 if self.twin else 108))
        rotation_space = 22 if self.rotation and not is_pie else 0
        plot_height = max(100, height - (90 if self.xlabel else 68) - legend_height - rotation_space)
        bottom = plot_top + plot_height
        pieces = []
        if self.title:
            pieces.append(_text(left + width / 2, top + 19, self.title, size=self.title_style.get('fontsize', 15),
                                anchor='middle', weight=self.title_style.get('fontweight', '600')))
        if is_pie:
            cx, cy = plot_left + plot_width / 2, plot_top + plot_height / 2
            radius = min(plot_width / 2 - 12, plot_height / 2 - 10)
            for artist in self.artists:
                if artist.kind != 'pie':
                    continue
                start, end = np.deg2rad(artist.x)
                color = escape(_color(artist.style['color']), quote=True)
                if end - start >= 2 * math.pi - 1e-10:
                    pieces.append(f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(radius)}" fill="{color}"/>')
                else:
                    x1, y1 = cx + radius * math.cos(start), cy - radius * math.sin(start)
                    x2, y2 = cx + radius * math.cos(end), cy - radius * math.sin(end)
                    d = f'M{_f(cx)},{_f(cy)} L{_f(x1)},{_f(y1)} A{_f(radius)},{_f(radius)} 0 {int(end-start > math.pi)} 0 {_f(x2)},{_f(y2)} Z'
                    pieces.append(f'<path class="pie-slice" d="{d}" fill="{color}" stroke="white" stroke-width="1"/>')
        else:
            xlim, ylim = self._limits('x'), self._limits('y')
            sx = lambda values: plot_left + (np.asarray(values) - xlim[0]) / (xlim[1] - xlim[0]) * plot_width
            sy = lambda values: bottom - (np.asarray(values) - ylim[0]) / (ylim[1] - ylim[0]) * plot_height
            clip = f'plot-{panel_index}'
            pieces.append(f'<defs><clipPath id="{clip}"><rect x="{_f(plot_left)}" y="{_f(plot_top)}" width="{_f(plot_width)}" height="{_f(plot_height)}"/></clipPath></defs>')
            for value in _nice_ticks(ylim):
                y = float(sy(value))
                if self.grid_options is not None:
                    pieces.append(_line(plot_left, y, plot_left + plot_width, y, '#e2e8f0'))
                pieces.append(_text(plot_left - 12, y + 4, _tick_label(value, 'number', ylim[1] - ylim[0]), anchor='end', size=11, color=self.ycolor))
            if self.xkind == 'category':
                tick_values = np.arange(len(self.categories))
                tick_labels = self.categories
            elif self.xkind == 'date':
                tick_values, tick_labels = _date_ticks(xlim)
            else:
                tick_values = _nice_ticks(xlim, 7)
                tick_labels = [_tick_label(value, self.xkind, xlim[1] - xlim[0]) for value in tick_values]
            for value, label in zip(tick_values, tick_labels):
                x = float(sx(value))
                pieces.append(_text(x, bottom + 21, label, anchor='end' if self.rotation else 'middle',
                                    size=11, rotate=-self.rotation))
            pieces.append(_line(plot_left, bottom, plot_left + plot_width, bottom, '#94a3b8'))
            pieces.append(_line(plot_left, plot_top, plot_left, bottom, '#94a3b8'))
            pieces.append(f'<g clip-path="url(#{clip})">')
            pieces.extend(self._render_artists(sx, sy, plot_left, plot_top, plot_width, plot_height))
            if self.twin:
                twin_limits = self.twin._limits('y')
                twin_sy = lambda values: bottom - (np.asarray(values) - twin_limits[0]) / (twin_limits[1] - twin_limits[0]) * plot_height
                pieces.extend(self.twin._render_artists(sx, twin_sy, plot_left, plot_top, plot_width, plot_height))
            pieces.append('</g>')
            pieces.extend(self._render_texts(sx, sy, plot_left, plot_top, plot_width, plot_height))
            if self.twin:
                for value in _nice_ticks(twin_limits):
                    pieces.append(_text(plot_left + plot_width + 10, float(twin_sy(value)) + 4,
                                        _tick_label(value, 'number', twin_limits[1] - twin_limits[0]),
                                        size=11, color=self.twin.ycolor))
                pieces.append(_text(left + width - 12, plot_top + plot_height / 2, self.twin.ylabel,
                                    anchor='middle', rotate=90, color=self.twin.ycolor))
            if self.ylabel:
                pieces.append(_text(left + 16, plot_top + plot_height / 2, self.ylabel, anchor='middle', rotate=-90, color=self.ycolor))
            if self.xlabel:
                pieces.append(_text(plot_left + plot_width / 2, bottom + 49 + rotation_space, self.xlabel, anchor='middle'))
        legend_top = bottom + (67 if self.xlabel else 43) + rotation_space
        for x_offset, row, artist, label in legend_positions:
            x, y = left + 18 + x_offset, legend_top + row * 20
            color = artist.style.get('color', '#64748b')
            pieces.append(_line(x, y - 4, x + 20, y - 4, color, 3))
            pieces.append(_text(x + 27, y, label, size=font_size))
        return ''.join(pieces)

    def _render_artists(self, sx, sy, left, top, width, height):
        pieces = []
        for artist in self.artists:
            style = artist.style
            color = escape(_color(style.get('color', TAB10[0])), quote=True)
            opacity = float(style.get('alpha', 1.0))
            line_width = float(style.get('linewidth', 1.7))
            dash_value = {'--': '7 4', ':': '2 4', '-.': '7 3 2 3'}.get(style.get('linestyle'), '')
            dash = f' stroke-dasharray="{dash_value}"' if dash_value else ''
            common = f'stroke="{color}" stroke-width="{_f(line_width)}" opacity="{_f(opacity)}"{dash}'
            if artist.kind == 'line':
                d = _path(sx(artist.x), sy(artist.y))
                pieces.append(f'<path class="series-line" d="{d}" fill="none" {common}/>')
            elif artist.kind == 'area':
                valid = np.isfinite(artist.x) & np.isfinite(artist.y) & np.isfinite(artist.y2)
                if artist.where is not None:
                    valid &= artist.where
                for start, end in _runs(valid):
                    if end - start < 2:
                        continue
                    x = sx(artist.x[start:end])
                    upper, lower = sy(artist.y[start:end]), sy(artist.y2[start:end])
                    d = ('M' + ' L'.join(f'{_f(a)},{_f(b)}' for a, b in zip(x, upper)) +
                         ' L' + ' L'.join(f'{_f(a)},{_f(b)}' for a, b in zip(x[::-1], lower[::-1])) + ' Z')
                    pieces.append(f'<path class="series-area" d="{d}" fill="{color}" opacity="{_f(opacity)}"/>')
            elif artist.kind == 'hline':
                y = float(sy(artist.y[0]))
                pieces.append(_line(left, y, left + width, y, color, line_width, opacity, dash))
            elif artist.kind == 'vline':
                x = float(sx(artist.x[0]))
                pieces.append(_line(x, top, x, top + height, color, line_width, opacity, dash))
            elif artist.kind == 'bar':
                if not np.isfinite(artist.x[0] + artist.y[0]):
                    continue
                x = float(sx(artist.x[0] - artist.width / 2))
                right = float(sx(artist.x[0] + artist.width / 2))
                y, zero = float(sy(artist.y[0])), float(sy(0))
                pieces.append(f'<rect class="series-bar" x="{_f(x)}" y="{_f(min(y, zero))}" width="{_f(right-x)}" height="{_f(abs(y-zero))}" fill="{color}" opacity="{_f(opacity)}"/>')
            elif artist.kind == 'scatter':
                radius = max(3.0, math.sqrt(float(style.get('s', 45))) / 2)
                for value_x, value_y in zip(artist.x, artist.y):
                    if not np.isfinite(value_x + value_y):
                        continue
                    x, y = float(sx(value_x)), float(sy(value_y))
                    if str(style.get('marker', 'o')).lower() == 'x':
                        pieces.append(_line(x-radius, y-radius, x+radius, y+radius, color, 2.8))
                        pieces.append(_line(x-radius, y+radius, x+radius, y-radius, color, 2.8))
                    else:
                        pieces.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(radius)}" fill="{color}" stroke="white" stroke-width="1"/>')
        return pieces

    def _render_texts(self, sx, sy, left, top, width, height):
        pieces = []
        for x, y, text, style in self.texts:
            px, py = float(sx(x)), float(sy(y))
            size = style.get('fontsize', 11)
            if style.get('va') == 'bottom':
                py -= 6
            pieces.append(_text(px, py, text, size=size, color=style.get('color', '#334155'),
                                weight=style.get('fontweight', 'normal'),
                                anchor={'center': 'middle', 'right': 'end'}.get(style.get('ha'), 'start'),
                                rotate=-style.get('rotation', 0)))
        for x, y, text, offset, style in self.annotations:
            px, py = float(sx(x)), float(sy(y))
            size = style.get('fontsize', 11)
            tx = min(max(px + offset[0], left + 8), max(left + 8, left + width - _label_width(text, size) - 8))
            ty = min(max(py - offset[1], top + size + 5), top + height - 7)
            color = style.get('color', '#334155')
            pieces.append(_line(px, py, tx, ty + 3, style.get('arrowprops', {}).get('color', color), 1))
            pieces.append(_text(tx, ty, text, size=size, color=color, weight=style.get('fontweight', 'normal')))
        return pieces


class Figure:
    def __init__(self, figsize=(10, 6), rows=1, cols=1, width_ratios=None, sharex=False):
        self.width, self.height = (float(size) * 100 for size in figsize)
        self.rows, self.cols = rows, cols
        self.sharex = sharex
        self.width_ratios = np.asarray(width_ratios if width_ratios is not None else np.ones(cols), dtype=float)
        self.axes = [Axes(self, row, col) for row in range(rows) for col in range(cols)]
        self.title = ''
        self.title_style = {}
        self.texts = []

    def suptitle(self, title, **kwargs):
        self.title, self.title_style = str(title), kwargs

    def text(self, x, y, text, **kwargs):
        self.texts.append((float(x), float(y), str(text), kwargs))

    def to_svg(self):
        title = self.title or (self.axes[0].title if self.axes else '')
        pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{_f(self.width)}" height="{_f(self.height)}" viewBox="0 0 {_f(self.width)} {_f(self.height)}" role="img">',
                  f'<title>{escape(title)}</title>',
                  '<style>text{font-family:"Microsoft YaHei","PingFang TC","Noto Sans CJK TC",sans-serif}path,line,rect{vector-effect:non-scaling-stroke}</style>',
                  f'<rect width="{_f(self.width)}" height="{_f(self.height)}" fill="white"/>']
        top = 40 if self.title else 8
        footer = 35 if self.texts else 10
        if self.title:
            pieces.append(_text(self.width / 2, 25, self.title, size=self.title_style.get('fontsize', 17), anchor='middle', weight=self.title_style.get('fontweight', '600')))
        panel_height = (self.height - top - footer) / self.rows
        widths = self.width_ratios / self.width_ratios.sum() * self.width
        starts = np.r_[0.0, widths.cumsum()]
        for index, axes in enumerate(self.axes):
            panel = (starts[axes.col], top + axes.row * panel_height, widths[axes.col], panel_height)
            pieces.append(axes._render(panel, index))
        for x, y, text, style in self.texts:
            pieces.append(_text(x * self.width, (1-y) * self.height - 3, text,
                                size=style.get('fontsize', 10), color=style.get('color', '#64748b'),
                                anchor={'center': 'middle', 'right': 'end'}.get(style.get('ha'), 'start')))
        pieces.append('</svg>')
        return ''.join(pieces)

    def savefig(self, path, **kwargs):
        path = Path(path)
        if path.suffix.lower() != '.svg':
            raise ValueError('The report renderer writes SVG files only')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_svg(), encoding='utf-8')


def figure(figsize=(10, 6), **kwargs):
    global _current
    _current = Figure(figsize)
    return _current


def subplots(nrows=1, ncols=1, figsize=(10, 6), sharex=False, gridspec_kw=None, **kwargs):
    global _current
    _current = Figure(figsize, nrows, ncols, (gridspec_kw or {}).get('width_ratios'), sharex)
    axes = _current.axes[0] if nrows * ncols == 1 else np.asarray(_current.axes, dtype=object).reshape(nrows, ncols).squeeze()
    return _current, axes


def gcf():
    return _current if _current is not None else figure()


def close(figure=None):
    global _current
    _current = None


def get_cmap(name):
    if name not in ('tab10', 'tab20'):
        raise ValueError(f'Unsupported report palette: {name}')
    return SimpleNamespace(colors=TAB10 if name == 'tab10' else TAB20)


def _axes():
    return gcf().axes[0]


def plot(*args, **kwargs): return _axes().plot(*args, **kwargs)
def fill_between(*args, **kwargs): return _axes().fill_between(*args, **kwargs)
def bar(*args, **kwargs): return _axes().bar(*args, **kwargs)
def pie(*args, **kwargs): return _axes().pie(*args, **kwargs)
def axhline(*args, **kwargs): return _axes().axhline(*args, **kwargs)
def axvline(*args, **kwargs): return _axes().axvline(*args, **kwargs)
def title(*args, **kwargs): return _axes().set_title(*args, **kwargs)
def xlabel(*args, **kwargs): return _axes().set_xlabel(*args, **kwargs)
def ylabel(*args, **kwargs): return _axes().set_ylabel(*args, **kwargs)
def legend(*args, **kwargs): return _axes().legend(*args, **kwargs)
def grid(*args, **kwargs): return _axes().grid(*args, **kwargs)


def axis(value):
    if value != 'equal':
        raise ValueError('Only pie-chart equal aspect is used by the report')
