import os
import sys
import csv
import math
from datetime import datetime, timedelta

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QFileDialog, QFrame, QDoubleSpinBox, QButtonGroup,
    QToolButton, QHeaderView, QAbstractItemView, QMessageBox,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dapm.data import load_market_data
from dapm.backtest import rolling_dapm_betas, asymmetry_screen

_BENCHMARK_MAP = {
    "S&P 500": "^GSPC",
    "NIFTY 50": "^NSEI",
    "NIFTY 500": "^CRSLDX",
}
COLORS = {
    "paper": "#FAFAF7", "panel": "#FFFFFF", "ink": "#171A1F",
    "muted": "#5B6472", "faint": "#8A8F98", "hairline": "#DEDACF",
    "up": "#1E7A5F", "up_soft": "#E4F1EC",
    "down": "#B23A2E", "down_soft": "#F7E9E7",
    "amber": "#B8862E", "amber_soft": "#F5EBD9",
}
SANS = "Segoe UI"
MONO = "Consolas"

def normal_cdf(z: float) -> float:
    t = 1 / (1 + 0.2316419 * abs(z))
    d = 0.3989423 * math.exp(-z * z / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return 1 - p if z > 0 else p


def linreg(points):
    n = len(points)
    if n < 5:
        return {"slope": 0.0, "intercept": 0.0, "se": 0.0, "tstat": 0.0, "pvalue": 1.0, "r2": 0.0, "n": n}
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    syy = sum((p[1] - my) ** 2 for p in points)
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    ssr = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in points)
    r2 = 1 - ssr / syy if syy else 0.0
    dof = n - 2
    sigma2 = ssr / dof if dof > 0 else 0.0
    se = math.sqrt(sigma2 / sxx) if sxx and sigma2 >= 0 else 0.0
    tstat = slope / se if se else 0.0
    pvalue = 2 * (1 - normal_cdf(abs(tstat)))
    return {"slope": slope, "intercept": intercept, "se": se, "tstat": tstat, "pvalue": pvalue, "r2": r2, "n": n}


def generate_data(ticker: str, n_days: int, benchmark: str = "S&P 500"):
    bench = _BENCHMARK_MAP.get(benchmark, "^GSPC")
    start = (datetime.now() - timedelta(days=int(n_days * 1.5) + 120)).strftime("%Y-%m-%d")

    assets, market, rf = load_market_data([ticker], benchmark=bench, start=start)

    xm = (market - rf).dropna()
    xi = (assets[ticker] - rf).dropna()
    common = xm.index.intersection(xi.index)
    if len(common) == 0:
        raise ValueError(f"No overlapping data for {ticker} against {bench}")
    xm, xi = xm.loc[common], xi.loc[common]

    rows = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "mkt": float(m) * 100,
            "stock": float(s) * 100,
            "regime": "up" if m >= 0 else "down",
        }
        for d, m, s in zip(common, xm.to_numpy(), xi.to_numpy())
    ]
    return rows[-n_days:]

class StatBlock(QFrame):
    def __init__(self, label, accent, info=None, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"border-bottom: 3px solid {accent}; background: transparent;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 10)
        layout.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(4)
        lbl = QLabel(label)
        lbl.setFont(QFont(SANS, 9))
        lbl.setStyleSheet(f"color: {COLORS['muted']}; border: none;")
        top.addWidget(lbl)
        if info:
            info_lbl = QLabel("ⓘ")
            info_lbl.setToolTip(info)
            info_lbl.setStyleSheet(f"color: {COLORS['faint']}; border: none;")
            top.addWidget(info_lbl)
        top.addStretch()
        layout.addLayout(top)

        self.value_lbl = QLabel("--")
        self.value_lbl.setFont(QFont(MONO, 20, QFont.DemiBold))
        self.value_lbl.setStyleSheet(f"color: {COLORS['ink']}; border: none;")
        layout.addWidget(self.value_lbl)

        self.sub_lbl = QLabel("")
        self.sub_lbl.setFont(QFont(MONO, 9))
        self.sub_lbl.setStyleSheet(f"color: {COLORS['muted']}; border: none;")
        layout.addWidget(self.sub_lbl)

    def set_value(self, value_text, sub_text=""):
        self.value_lbl.setText(value_text)
        self.sub_lbl.setText(sub_text)


class MplCanvas(FigureCanvas):
    def __init__(self, height=3.6):
        fig = Figure(figsize=(6, height), dpi=100)
        fig.patch.set_facecolor(COLORS["panel"])
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.style_axes()

    def style_axes(self):
        ax = self.ax
        ax.set_facecolor(COLORS["panel"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["hairline"])
        ax.tick_params(colors=COLORS["muted"], labelsize=8)
        ax.grid(True, color=COLORS["hairline"], linewidth=0.6, linestyle=(0, (2, 3)))
        ax.set_axisbelow(True)


class DapmDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAPM — Asymmetric Beta Analytics")
        self.resize(1180, 860)
        self.setStyleSheet(f"background: {COLORS['paper']};")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(18)

        root.addLayout(self._build_header())
        root.addWidget(self._build_input_panel())
        root.addLayout(self._build_stats_row())
        root.addWidget(self._build_scatter_panel())
        root.addWidget(self._build_tabs())
        root.addWidget(self._build_footer())

        self.run_analysis()

    def _build_header(self):
        row = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Asymmetric Beta Analytics")
        title.setFont(QFont(SANS, 16, QFont.Bold))
        title.setStyleSheet(f"color: {COLORS['ink']};")
        subtitle = QLabel("Dichotomous Asset Pricing Model — split-regime regression vs single-factor CAPM")
        subtitle.setFont(QFont(SANS, 9))
        subtitle.setStyleSheet(f"color: {COLORS['muted']};")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        row.addLayout(title_box)
        row.addStretch()
        self.stale_lbl = QLabel("no estimate yet")
        stale = self.stale_lbl
        stale.setFont(QFont(MONO, 9))
        stale.setStyleSheet(f"color: {COLORS['faint']};")
        row.addWidget(stale, alignment=Qt.AlignBottom)
        return row

    def _build_input_panel(self):
        panel = QFrame()
        panel.setStyleSheet(f"background: {COLORS['panel']}; border: 1px solid {COLORS['hairline']};")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(16)

        def field(label_text, widget):
            box = QVBoxLayout()
            box.setSpacing(4)
            lbl = QLabel(label_text)
            lbl.setFont(QFont(SANS, 8))
            lbl.setStyleSheet(f"color: {COLORS['muted']}; border: none;")
            box.addWidget(lbl)
            box.addWidget(widget)
            return box

        self.ticker_input = QLineEdit("AAPL")
        self.ticker_input.setFont(QFont(MONO, 10))
        self.ticker_input.setFixedWidth(110)
        self.ticker_input.setStyleSheet(self._input_style())

        self.benchmark_input = QComboBox()
        self.benchmark_input.addItems(["S&P 500", "NIFTY 50", "NIFTY 500"])
        self.benchmark_input.setFont(QFont(MONO, 10))
        self.benchmark_input.setFixedWidth(120)
        self.benchmark_input.setStyleSheet(self._input_style())

        range_box = QHBoxLayout()
        range_box.setSpacing(0)
        self.range_group = QButtonGroup(self)
        self.range_group.setExclusive(True)
        for i, yrs in enumerate([1, 3, 5]):
            btn = QToolButton()
            btn.setText(f"{yrs}Y")
            btn.setCheckable(True)
            btn.setChecked(yrs == 3)
            btn.setFont(QFont(MONO, 9))
            btn.setStyleSheet(self._toggle_style())
            self.range_group.addButton(btn, yrs)
            range_box.addWidget(btn)
        range_widget = QWidget()
        range_widget.setLayout(range_box)

        self.rf_input = QDoubleSpinBox()
        self.rf_input.setRange(0, 20)
        self.rf_input.setSingleStep(0.1)
        self.rf_input.setValue(6.5)
        self.rf_input.setSuffix(" %")
        self.rf_input.setFont(QFont(MONO, 10))
        self.rf_input.setFixedWidth(90)
        self.rf_input.setStyleSheet(self._input_style())

        self.erp_input = QDoubleSpinBox()
        self.erp_input.setRange(0, 20)
        self.erp_input.setSingleStep(0.1)
        self.erp_input.setValue(7.0)
        self.erp_input.setSuffix(" %")
        self.erp_input.setFont(QFont(MONO, 10))
        self.erp_input.setFixedWidth(90)
        self.erp_input.setStyleSheet(self._input_style())

        layout.addLayout(field("Ticker", self.ticker_input))
        layout.addLayout(field("Benchmark", self.benchmark_input))
        layout.addLayout(field("Lookback window", range_widget))
        layout.addLayout(field("Risk-free rate", self.rf_input))
        layout.addLayout(field("Market premium", self.erp_input))
        layout.addStretch()

        self.run_btn = QPushButton("Run analysis  →")
        self.run_btn.setFont(QFont(SANS, 10, QFont.DemiBold))
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setStyleSheet(
            f"QPushButton {{ background: {COLORS['ink']}; color: {COLORS['paper']}; "
            f"border: none; padding: 9px 18px; }}"
            f"QPushButton:hover {{ background: #2A2F38; }}"
        )
        self.run_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_btn, alignment=Qt.AlignBottom)
        return panel

    def _input_style(self):
        return (
            f"QWidget {{ background: {COLORS['panel']}; color: {COLORS['ink']}; "
            f"border: 1px solid {COLORS['hairline']}; padding: 5px 7px; }}"
        )

    def _toggle_style(self):
        return (
            f"QToolButton {{ background: {COLORS['panel']}; color: {COLORS['muted']}; "
            f"border: 1px solid {COLORS['hairline']}; padding: 6px 10px; }}"
            f"QToolButton:checked {{ background: {COLORS['ink']}; color: {COLORS['paper']}; }}"
        )

    def _build_stats_row(self):
        grid = QGridLayout()
        grid.setSpacing(20)
        self.stat_capm = StatBlock("CAPM beta", COLORS["muted"])
        self.stat_plus = StatBlock("Upside beta (β+)", COLORS["up"],
                                    info="Beta estimated only on days the market posted a positive excess return.")
        self.stat_minus = StatBlock("Downside beta (β−)", COLORS["down"],
                                     info="Beta estimated only on days the market posted a negative excess return.")
        self.stat_ratio = StatBlock("Asymmetry ratio β−/β+", COLORS["amber"])
        for i, w in enumerate([self.stat_capm, self.stat_plus, self.stat_minus, self.stat_ratio]):
            grid.addWidget(w, 0, i)
        return grid

    def _build_scatter_panel(self):
        panel = QFrame()
        panel.setStyleSheet(f"background: {COLORS['panel']}; border: 1px solid {COLORS['hairline']};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 6)
        header = QLabel("Stock vs. market excess returns")
        header.setFont(QFont(SANS, 10, QFont.DemiBold))
        header.setStyleSheet(f"color: {COLORS['ink']}; border: none;")
        layout.addWidget(header)
        self.scatter_canvas = MplCanvas(height=4.0)
        layout.addWidget(self.scatter_canvas)
        return panel

    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid {COLORS['hairline']}; background: {COLORS['panel']}; }}"
            f"QTabBar::tab {{ background: {COLORS['panel']}; color: {COLORS['muted']}; "
            f"padding: 8px 14px; border: none; border-bottom: 2px solid transparent; }}"
            f"QTabBar::tab:selected {{ color: {COLORS['ink']}; border-bottom: 2px solid {COLORS['ink']}; }}"
        )
        self.tabs.setFont(QFont(SANS, 9))

        self.stats_table = QTableWidget(3, 7)
        self.stats_table.setHorizontalHeaderLabels(
            ["Model", "β", "Std. error", "t-stat", "p-value (approx.)", "R²", "N"]
        )
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setFont(QFont(MONO, 9))
        self.tabs.addTab(self._wrap(self.stats_table), "Regression stats")

        self.rolling_canvas = MplCanvas(height=3.2)
        self.tabs.addTab(self._wrap(self.rolling_canvas), "Rolling beta")

        self.dist_canvas = MplCanvas(height=3.2)
        self.tabs.addTab(self._wrap(self.dist_canvas), "Distribution")

        raw_widget = QWidget()
        raw_layout = QVBoxLayout(raw_widget)
        top_row = QHBoxLayout()
        self.raw_note = QLabel("")
        self.raw_note.setFont(QFont(SANS, 9))
        self.raw_note.setStyleSheet(f"color: {COLORS['muted']};")
        top_row.addWidget(self.raw_note)
        top_row.addStretch()
        export_btn = QPushButton("⬇ download full CSV")
        export_btn.setFont(QFont(SANS, 9))
        export_btn.setStyleSheet(
            f"QPushButton {{ background: {COLORS['panel']}; color: {COLORS['ink']}; "
            f"border: 1px solid {COLORS['hairline']}; padding: 6px 10px; }}"
        )
        export_btn.clicked.connect(self.export_csv)
        top_row.addWidget(export_btn)
        raw_layout.addLayout(top_row)

        self.raw_table = QTableWidget(0, 4)
        self.raw_table.setHorizontalHeaderLabels(["Date", "Market excess", "Stock excess", "Regime"])
        self.raw_table.verticalHeader().setVisible(False)
        self.raw_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.raw_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.raw_table.setFont(QFont(MONO, 9))
        raw_layout.addWidget(self.raw_table)
        self.tabs.addTab(raw_widget, "Raw data")

        self.tabs.addTab(self._build_screen_tab(), "Cross-sectional screen")

        return self.tabs

    def _build_screen_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        universe_lbl = QLabel("Universe")
        universe_lbl.setFont(QFont(SANS, 8))
        universe_lbl.setStyleSheet(f"color: {COLORS['muted']};")
        top_row.addWidget(universe_lbl)

        self.universe_input = QLineEdit(
            "AAPL,MSFT,GOOGL,AMZN,META,NVDA,JPM,BAC,GS,JNJ,UNH,XOM,CVX,KO,PEP,WMT,BA,CAT,DIS,NFLX"
        )
        self.universe_input.setFont(QFont(MONO, 9))
        self.universe_input.setStyleSheet(self._input_style())
        top_row.addWidget(self.universe_input, stretch=1)

        self.screen_btn = QPushButton("Run screen")
        self.screen_btn.setFont(QFont(SANS, 9, QFont.DemiBold))
        self.screen_btn.setCursor(Qt.PointingHandCursor)
        self.screen_btn.setStyleSheet(
            f"QPushButton {{ background: {COLORS['ink']}; color: {COLORS['paper']}; "
            f"border: none; padding: 7px 14px; }}"
            f"QPushButton:hover {{ background: #2A2F38; }}"
        )
        self.screen_btn.clicked.connect(self.run_screen)
        top_row.addWidget(self.screen_btn)
        layout.addLayout(top_row)

        self.screen_note = QLabel(
            "Ranked by β+/β−. Top of the list participates in rallies more than "
            "it follows selloffs; the bottom is the fragile end."
        )
        self.screen_note.setWordWrap(True)
        self.screen_note.setFont(QFont(SANS, 8))
        self.screen_note.setStyleSheet(f"color: {COLORS['faint']};")
        layout.addWidget(self.screen_note)

        self.screen_table = QTableWidget(0, 6)
        self.screen_table.setHorizontalHeaderLabels(
            ["#", "Asset", "β+", "β−", "CAPM β", "β+/β−"]
        )
        self.screen_table.verticalHeader().setVisible(False)
        self.screen_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.screen_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.screen_table.setFont(QFont(MONO, 9))
        layout.addWidget(self.screen_table)

        return widget

    def _wrap(self, widget):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(widget)
        return w

    def _build_footer(self):
        note = QLabel(
            "Daily excess returns from yfinance — benchmark less the ^IRX risk-free rate. "
            "Betas are estimated on the {date, mkt, stock, regime} row shape produced by "
            "src/dapm.data.load_market_data(). Regressions here are OLS with an intercept; "
            "the src/dapm engine uses Zou's through-origin moment estimator, so headline "
            "betas can differ slightly from the /beta API."
        )
        note.setWordWrap(True)
        note.setFont(QFont(SANS, 8))
        note.setStyleSheet(f"color: {COLORS['faint']};")
        return note

    def run_analysis(self):
        self.run_btn.setText("Running…")
        self.run_btn.setEnabled(False)
        QTimer.singleShot(150, self._do_run)

    def _do_run(self):
        try:
            self._run_inner()
        except Exception as e:
            QMessageBox.warning(self, "Analysis failed", str(e))
        finally:
            self.run_btn.setText("Run analysis  →")
            self.run_btn.setEnabled(True)

    def run_screen(self):
        self.screen_btn.setText("Screening…")
        self.screen_btn.setEnabled(False)
        QTimer.singleShot(150, self._do_screen)

    def _do_screen(self):
        try:
            self._screen_inner()
        except Exception as e:
            QMessageBox.warning(self, "Screen failed", str(e))
        finally:
            self.screen_btn.setText("Run screen")
            self.screen_btn.setEnabled(True)

    def _screen_inner(self):
        symbols = [s.strip().upper() for s in self.universe_input.text().split(",") if s.strip()]
        if len(symbols) < 2:
            raise ValueError("Enter at least two comma-separated tickers")

        years = self.range_group.checkedId()
        n_days = years * 250
        bench = _BENCHMARK_MAP.get(self.benchmark_input.currentText(), "^GSPC")
        start = (datetime.now() - timedelta(days=int(n_days * 1.5) + 120)).strftime("%Y-%m-%d")

        assets, market, rf = load_market_data(symbols, benchmark=bench, start=start)
        panel = rolling_dapm_betas(assets, market, rf, window=126, step=21)
        if panel.empty:
            raise ValueError("No beta estimates — widen the lookback window")

        snap = asymmetry_screen(panel)
        as_of = str(panel["date"].max().date())
        self.screen_note.setText(
            f"{len(snap)} assets ranked by β+/β− as of {as_of}. Top of the list "
            f"participates in rallies more than it follows selloffs; the bottom "
            f"is the fragile end. Estimated on a 126-day trailing window."
        )

        self.screen_table.setRowCount(len(snap))
        for i, r in enumerate(snap.to_dict(orient="records")):
            ratio = float(r["asymmetry_ratio"])
            cells = [
                str(i + 1),
                str(r["asset"]),
                f"{float(r['beta_plus']):.3f}",
                f"{float(r['beta_minus']):.3f}",
                f"{float(r['capm_beta']):.3f}",
                f"{ratio:.3f}",
            ]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter if j != 1 else Qt.AlignLeft | Qt.AlignVCenter)
                if j == 5:
                    item.setForeground(QColor(COLORS["up"] if ratio >= 1 else COLORS["down"]))
                self.screen_table.setItem(i, j, item)

    def _run_inner(self):
        ticker = self.ticker_input.text().strip().upper() or "AAPL"
        years = self.range_group.checkedId()
        n_days = years * 250
        rf = self.rf_input.value() / 100
        erp = self.erp_input.value() / 100

        rows = generate_data(ticker, n_days, self.benchmark_input.currentText())

        capm = linreg([(r["mkt"], r["stock"]) for r in rows])
        up_pts = [(r["mkt"], r["stock"]) for r in rows if r["regime"] == "up"]
        down_pts = [(r["mkt"], r["stock"]) for r in rows if r["regime"] == "down"]
        beta_plus = linreg(up_pts)
        beta_minus = linreg(down_pts)

        ratio = (beta_minus["slope"] / beta_plus["slope"]) if beta_plus["slope"] else 0.0
        verdict = "Defensive" if ratio < 0.85 else "Aggressive" if ratio > 1.15 else "Symmetric"
        verdict_color = COLORS["up"] if verdict == "Defensive" else COLORS["down"] if verdict == "Aggressive" else COLORS["amber"]

        er_capm = (rf + capm["slope"] * erp) * 100
        er_plus = (rf + beta_plus["slope"] * erp) * 100
        er_minus = (rf + beta_minus["slope"] * erp) * 100

        self.stat_capm.set_value(f"{capm['slope']:.2f}", f"E[r] {er_capm:.1f}%")
        self.stat_plus.set_value(f"{beta_plus['slope']:.2f}", f"E[r] {er_plus:.1f}% in up markets")
        self.stat_minus.set_value(f"{beta_minus['slope']:.2f}", f"E[r] {er_minus:.1f}% in down markets")
        self.stat_ratio.set_value(f"{ratio:.2f}", verdict)
        self.stat_ratio.setStyleSheet(f"border-bottom: 3px solid {verdict_color};")
        self.stat_ratio.sub_lbl.setStyleSheet(f"color: {verdict_color}; border: none; font-weight: 600;")

        self._draw_scatter(rows, capm, beta_plus, beta_minus)
        self._draw_stats_table(capm, beta_plus, beta_minus)
        self._draw_rolling(rows)
        self._draw_distribution(rows)
        self._fill_raw_table(rows)

        self.stale_lbl.setText(f"{ticker} · estimated through {rows[-1]['date']}")

        self._rows_cache = rows
        self._ticker_cache = ticker

    def _draw_scatter(self, rows, capm, beta_plus, beta_minus):
        ax = self.scatter_canvas.ax
        ax.clear()
        self.scatter_canvas.style_axes()

        up = [r for r in rows if r["regime"] == "up"]
        down = [r for r in rows if r["regime"] == "down"]
        ax.scatter([r["mkt"] for r in up], [r["stock"] for r in up], color=COLORS["up"], alpha=0.5, s=10, label="up-market day")
        ax.scatter([r["mkt"] for r in down], [r["stock"] for r in down], color=COLORS["down"], alpha=0.5, s=10, label="down-market day")

        min_x = min(r["mkt"] for r in rows)
        max_x = max(r["mkt"] for r in rows)
        ax.plot([min_x, max_x],
                [capm["intercept"] + capm["slope"] * min_x, capm["intercept"] + capm["slope"] * max_x],
                color=COLORS["faint"], linestyle="--", linewidth=1.4, label="CAPM")
        ax.plot([0, max_x], [beta_plus["intercept"], beta_plus["intercept"] + beta_plus["slope"] * max_x],
                color=COLORS["up"], linewidth=2.0, label="β+")
        ax.plot([min_x, 0], [beta_minus["intercept"] + beta_minus["slope"] * min_x, beta_minus["intercept"]],
                color=COLORS["down"], linewidth=2.0, label="β−")

        ax.set_xlabel("Market excess return (%)", fontsize=8, color=COLORS["muted"])
        ax.set_ylabel("Stock excess return (%)", fontsize=8, color=COLORS["muted"])
        ax.legend(loc="upper left", fontsize=7, frameon=False)
        self.scatter_canvas.draw()

    def _draw_stats_table(self, capm, beta_plus, beta_minus):
        rows_data = [
            ("CAPM (full sample)", capm),
            ("DAPM β+ (up days)", beta_plus),
            ("DAPM β− (down days)", beta_minus),
        ]
        for i, (name, m) in enumerate(rows_data):
            pval = "<0.001" if m["pvalue"] < 0.001 else f"{m['pvalue']:.3f}"
            values = [name, f"{m['slope']:.3f}", f"{m['se']:.3f}", f"{m['tstat']:.2f}", pval, f"{m['r2']:.3f}", str(m["n"])]
            for j, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter if j > 0 else Qt.AlignLeft | Qt.AlignVCenter)
                self.stats_table.setItem(i, j, item)

    def _draw_rolling(self, rows):
        ax = self.rolling_canvas.ax
        ax.clear()
        self.rolling_canvas.style_axes()

        window, step = 60, 5
        xs, bp_series, bm_series = [], [], []
        for i in range(window, len(rows) + 1, step):
            win = rows[i - window:i]
            up = [(r["mkt"], r["stock"]) for r in win if r["regime"] == "up"]
            down = [(r["mkt"], r["stock"]) for r in win if r["regime"] == "down"]
            bp = linreg(up)["slope"] if len(up) >= 8 else None
            bm = linreg(down)["slope"] if len(down) >= 8 else None
            xs.append(i)
            bp_series.append(bp)
            bm_series.append(bm)

        ax.plot(xs, bp_series, color=COLORS["up"], linewidth=1.8, label="β+ (60d window)")
        ax.plot(xs, bm_series, color=COLORS["down"], linewidth=1.8, label="β− (60d window)")
        ax.set_xlabel("trading day index", fontsize=8, color=COLORS["muted"])
        ax.legend(loc="upper left", fontsize=7, frameon=False)
        self.rolling_canvas.draw()

    def _draw_distribution(self, rows):
        ax = self.dist_canvas.ax
        ax.clear()
        self.dist_canvas.style_axes()

        vals = [r["stock"] for r in rows]
        lo, hi = min(vals), max(vals)
        bins = 12
        width = (hi - lo) / bins if hi > lo else 1
        up_counts = [0] * bins
        down_counts = [0] * bins
        for r in rows:
            idx = min(bins - 1, max(0, int((r["stock"] - lo) / width)))
            if r["regime"] == "up":
                up_counts[idx] += 1
            else:
                down_counts[idx] += 1
        positions = list(range(bins))
        ax.bar([p - 0.2 for p in positions], up_counts, width=0.4, color=COLORS["up"], alpha=0.75, label="up-market days")
        ax.bar([p + 0.2 for p in positions], down_counts, width=0.4, color=COLORS["down"], alpha=0.75, label="down-market days")
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{lo + i * width:.1f}" for i in positions], fontsize=6, rotation=45)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
        self.dist_canvas.draw()

    def _fill_raw_table(self, rows):
        last60 = list(reversed(rows[-60:]))
        self.raw_note.setText(f"showing last 60 of {len(rows)} observations")
        self.raw_table.setRowCount(len(last60))
        for i, r in enumerate(last60):
            self.raw_table.setItem(i, 0, QTableWidgetItem(r["date"]))
            mkt_item = QTableWidgetItem(f"{r['mkt']:+.2f}%")
            mkt_item.setForeground(QColor(COLORS["up"] if r["mkt"] >= 0 else COLORS["down"]))
            self.raw_table.setItem(i, 1, mkt_item)
            self.raw_table.setItem(i, 2, QTableWidgetItem(f"{r['stock']:+.2f}%"))
            regime_item = QTableWidgetItem(r["regime"])
            regime_item.setForeground(QColor(COLORS["up"] if r["regime"] == "up" else COLORS["down"]))
            self.raw_table.setItem(i, 3, regime_item)

    def export_csv(self):
        if not hasattr(self, "_rows_cache"):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export daily returns", f"{self._ticker_cache.lower()}_daily_returns.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "market_excess_return_pct", "stock_excess_return_pct", "regime"])
            for r in self._rows_cache:
                writer.writerow([r["date"], f"{r['mkt']:.4f}", f"{r['stock']:.4f}", r["regime"]])


def main():
    app = QApplication(sys.argv)
    window = DapmDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
