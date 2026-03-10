import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from scipy.signal import argrelextrema
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StockEdge Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL DARK-THEME CSS  (XM-style)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:    #080c18;
    --bg-secondary:  #0e1526;
    --bg-card:       #111a2e;
    --bg-card2:      #162040;
    --border:        #1e2d4a;
    --border-bright: #2a3f6a;
    --accent-green:  #00e676;
    --accent-red:    #ff4757;
    --accent-blue:   #3d8eff;
    --accent-gold:   #ffd700;
    --accent-purple: #a855f7;
    --text-primary:  #e8eaf6;
    --text-muted:    #8892b0;
    --sidebar-bg:    #080c18;
    --glow-blue:     0 0 20px rgba(61,142,255,0.25);
    --glow-green:    0 0 20px rgba(0,230,118,0.25);
    --glow-red:      0 0 20px rgba(255,71,87,0.25);
    --glow-gold:     0 0 20px rgba(255,215,0,0.2);
}

/* ── Keyframes ── */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(400%); }
}
@keyframes ticker {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
@keyframes cardGlow {
    0%, 100% { box-shadow: 0 4px 24px rgba(61,142,255,0.08); }
    50%       { box-shadow: 0 4px 24px rgba(61,142,255,0.22); }
}

/* ── Full-page background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stMain"] { background-color: var(--bg-primary) !important; }

/* ── Block container ── */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
    animation: fadeInUp 0.45s ease both;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c18 0%, #0d1526 60%, #0a1120 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 32px rgba(0,0,0,0.6) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Inputs ── */
[data-testid="stSidebar"] .stSelectbox > div > div,
.stSelectbox > div > div {
    background: #121c34 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    transition: border-color 0.2s;
}
.stSelectbox > div > div:hover { border-color: var(--accent-blue) !important; }
.stSelectbox svg { fill: var(--text-muted) !important; }
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #121c34 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(61,142,255,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3d8eff 0%, #1a5fd4 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.03em !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg,transparent,rgba(255,255,255,0.15),transparent);
    animation: shimmer 2.5s infinite;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(61,142,255,0.45) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
}
.stDataFrame thead tr th {
    background: linear-gradient(135deg,#162040,#1c2a50) !important;
    color: var(--text-muted) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.stDataFrame tbody tr td {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border) !important;
    font-size: 0.85rem !important;
}
.stDataFrame tbody tr:hover td {
    background-color: #1a2846 !important;
    transition: background-color 0.15s;
}

/* ── Alerts ── */
.stAlert { border-radius: 12px !important; border: 1px solid var(--border) !important; }
[data-testid="stInfo"]    { background: rgba(14,32,64,0.9) !important; }
[data-testid="stWarning"] { background: rgba(31,21,0,0.9) !important; }
[data-testid="stError"]   { background: rgba(30,0,16,0.9) !important; }

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #3d8eff, #00e676) !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: #1a2540 !important;
    border-radius: 4px !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent-blue) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg,#3d8eff44,#00e67644);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: #3d8eff88; }

/* ── Canvas / charts ── */
canvas { border-radius: 14px; }

/* ── Hamburger ── */
[data-testid="collapsedControl"] { position: relative !important; }
[data-testid="collapsedControl"] span,
[data-testid="collapsedControl"] svg { display: none !important; }
[data-testid="collapsedControl"]::after {
    content: "☰";
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB DARK STYLE  (global)
# ─────────────────────────────────────────────
CHART_BG   = "#111827"
CHART_FG   = "#8892b0"
GRID_COLOR = "#1e2d4a"
GREEN      = "#00e676"
RED        = "#ff4757"
BLUE       = "#3d8eff"
GOLD       = "#ffd700"
PURPLE     = "#a855f7"

# ─────────────────────────────────────────────
#  HELPER: detect currency from ticker suffix
# ─────────────────────────────────────────────
def get_currency(ticker: str) -> tuple:
    """Return (symbol, name) for a ticker based on its exchange suffix."""
    t = ticker.upper().strip()
    suffix = t.rsplit(".", 1)[-1] if "." in t else ""
    _MAP = {
        # ── Indian exchanges ──
        "NS":  ("₹", "INR"),
        "BO":  ("₹", "INR"),
        # ── UK ──
        "L":   ("£", "GBP"),
        # ── European ──
        "DE":  ("€", "EUR"),
        "PA":  ("€", "EUR"),
        "AS":  ("€", "EUR"),
        "MI":  ("€", "EUR"),
        "MC":  ("€", "EUR"),
        "BR":  ("€", "EUR"),
        "F":   ("€", "EUR"),
        "ST":  ("kr", "SEK"),
        "CO":  ("kr", "DKK"),
        "OL":  ("kr", "NOK"),
        "HE":  ("€", "EUR"),
        # ── Japan ──
        "T":   ("¥", "JPY"),
        # ── Hong Kong ──
        "HK":  ("HK$", "HKD"),
        # ── China ──
        "SS":  ("¥", "CNY"),
        "SZ":  ("¥", "CNY"),
        # ── Taiwan ──
        "TW":  ("NT$", "TWD"),
        # ── Australia ──
        "AX":  ("A$", "AUD"),
        # ── Canada ──
        "TO":  ("C$", "CAD"),
        "V":   ("C$", "CAD"),
        # ── Singapore ──
        "SI":  ("S$", "SGD"),
        # ── South Korea ──
        "KS":  ("₩", "KRW"),
        "KQ":  ("₩", "KRW"),
    }
    return _MAP.get(suffix, ("$", "USD"))


def fmt_price(ticker: str, value: float) -> str:
    """Format a price with the correct currency symbol for the ticker."""
    sym, _ = get_currency(ticker)
    # For high-value currencies (JPY, KRW) skip decimals
    if sym in ("¥", "₩"):
        return f"{sym}{value:,.0f}"
    return f"{sym}{value:,.2f}"


def capital_label(tickers_list: list) -> str:
    """Return a capital label string: ₹ if any Indian ticker, else $."""
    for t in tickers_list:
        sym, _ = get_currency(t)
        if sym != "$":
            return sym
    return "$"


def apply_dark_style(fig, ax_list=None):
    fig.patch.set_facecolor(CHART_BG)
    axes = ax_list if ax_list else [fig.axes[i] for i in range(len(fig.axes))]
    for ax in axes:
        ax.set_facecolor(CHART_BG)
        ax.tick_params(colors=CHART_FG, labelsize=9)
        ax.xaxis.label.set_color(CHART_FG)
        ax.yaxis.label.set_color(CHART_FG)
        ax.title.set_color("#e8eaf6")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)
    return fig

# ─────────────────────────────────────────────
#  HELPER: render a styled metric card
# ─────────────────────────────────────────────
def metric_card(title, value, delta=None, color="#3d8eff", icon="📊"):
    delta_html = ""
    if delta is not None:
        d_color = GREEN if str(delta).startswith("+") or (isinstance(delta, (int, float)) and delta >= 0) else RED
        arrow = "▲" if d_color == GREEN else "▼"
        delta_html = f'<div style="font-size:0.78rem;color:{d_color};margin-top:4px;font-weight:500;">{arrow} {delta}</div>'
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg,#111a2e 0%,#162040 100%);
        border: 1px solid #1e2d4a;
        border-top: 2px solid {color};
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.6rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    " onmouseover="this.style.boxShadow='0 8px 32px {color}44';this.style.transform='translateY(-2px)'"
       onmouseout="this.style.boxShadow='0 4px 20px rgba(0,0,0,0.3)';this.style.transform='translateY(0)'"
    >
        <div style="position:absolute;top:0;right:0;width:60px;height:60px;
            background:radial-gradient(circle at top right,{color}18,transparent 70%);"></div>
        <div style="font-size:0.68rem;color:#8892b0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">{icon}&nbsp;{title}</div>
        <div style="font-size:1.6rem;font-weight:800;color:#e8eaf6;font-variant-numeric:tabular-nums;letter-spacing:-0.02em;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def section_header(title, subtitle=""):
    sub_html = f'<div style="color:#8892b0;font-size:0.82rem;margin-top:3px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="margin: 1.4rem 0 1.2rem 0;">
        <div style="font-size:1.3rem;font-weight:700;color:#e8eaf6;letter-spacing:-0.01em;">{title}</div>
        {sub_html}
        <div style="height:2px;background:linear-gradient(90deg,#3d8eff,#00e67640,transparent);
            margin-top:10px;border-radius:2px;"></div>
    </div>
    """, unsafe_allow_html=True)

def badge(label, color):
    return f'<span style="background:{color}22;color:{color};border:1px solid {color}55;padding:3px 12px;border-radius:20px;font-size:0.72rem;font-weight:700;letter-spacing:0.05em;">{label}</span>'

def chart_container_open(title=""):
    header = f'<div style="font-size:0.78rem;color:#8892b0;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.7rem;">{title}</div>' if title else ''
    st.markdown(f"""
    <div style="
        background:#0e1526;
        border:1px solid #1e2d4a;
        border-radius:16px;
        padding:1.2rem;
        margin:0.8rem 0;
        box-shadow:0 8px 32px rgba(0,0,0,0.4);
    ">{header}""", unsafe_allow_html=True)

def chart_container_close():
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.4rem 0 1.2rem 0;border-bottom:1px solid #1e2d4a;margin-bottom:1.2rem;">
        <div style="
            font-size:1.6rem;font-weight:800;
            background:linear-gradient(135deg,#3d8eff,#00e676,#3d8eff);
            background-size:200% 200%;
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            animation:gradientShift 4s ease infinite;
            letter-spacing:-0.02em;
        ">⚡ StockEdge Pro</div>
        <div style="font-size:0.65rem;color:#8892b0;letter-spacing:0.14em;margin-top:4px;">TRADING INTELLIGENCE PLATFORM</div>
        <div style="display:inline-block;margin-top:8px;background:#0b2a1a;border:1px solid #00e67655;
            border-radius:20px;padding:2px 12px;font-size:0.65rem;color:#00e676;font-weight:700;">
            <span style="display:inline-block;width:6px;height:6px;background:#00e676;
                border-radius:50%;margin-right:5px;animation:pulse 1.5s ease-in-out infinite;
                vertical-align:middle;"></span>LIVE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.65rem;color:#8892b0;letter-spacing:0.1em;margin-bottom:8px;">◈ SELECT MODULE</div>', unsafe_allow_html=True)
    MODE = st.selectbox(
        "MODE",
        ["📊 Breakout Strategy", "💹 Portfolio Optimization", "🕒 Intraday Prediction"],
        label_visibility="collapsed"
    )

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#1e2d4a,transparent);margin:1.2rem 0;"></div>', unsafe_allow_html=True)

    _nav_items = [
        ("📊", "Breakout Strategy", "Support/Resistance signals", "#3d8eff"),
        ("💹", "Portfolio Optimizer", "XGBoost + Efficient Frontier", "#00e676"),
        ("🕒", "Intraday Predictor", "Random Forest 30-min model", "#a855f7"),
    ]
    for _ic, _nm, _ds, _cl in _nav_items:
        _active = _nm.split()[0] in MODE
        _bg = f"background:{_cl}18;border-left:3px solid {_cl};" if _active else "background:#111a2e;border-left:3px solid #1e2d4a;"
        st.markdown(f"""
        <div style="{_bg}border-radius:10px;padding:0.6rem 0.8rem;margin-bottom:0.5rem;">
            <div style="font-size:0.82rem;font-weight:600;color:#e8eaf6;">{_ic} {_nm}</div>
            <div style="font-size:0.68rem;color:#8892b0;margin-top:2px;">{_ds}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#1e2d4a,transparent);margin:1.2rem 0;"></div>', unsafe_allow_html=True)
    now = datetime.now()
    # Market open: weekdays 9:15–15:30 IST
    _hour = now.hour + now.minute / 60
    _is_open = now.weekday() < 5 and 9.25 <= _hour <= 15.5
    _mkt_color = "#00e676" if _is_open else "#ff4757"
    _mkt_label = "NSE OPEN" if _is_open else "NSE CLOSED"
    st.markdown(f"""
    <div style="text-align:center;">
        <div style="font-size:0.68rem;color:#8892b0;margin-bottom:6px;">� {now.strftime('%H:%M')} IST &nbsp;·&nbsp; {now.strftime('%d %b %Y')}</div>
        <div style="display:inline-block;background:{_mkt_color}18;border:1px solid {_mkt_color}55;
            border-radius:20px;padding:3px 14px;font-size:0.65rem;color:{_mkt_color};font-weight:700;">
            {_mkt_label}
        </div>
    </div>
    <div style="text-align:center;margin-top:1rem;font-size:0.58rem;color:#4a5568;">v2.0 · Educational Use Only</div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TICKER MARQUEE BANNER
# ─────────────────────────────────────────────
_ticker_items = [
    ("NIFTY 50","22,500.40","+0.62%",True),
    ("SENSEX","74,119.00","+0.58%",True),
    ("BANK NIFTY","48,320.10","-0.21%",False),
    ("S&amp;P 500","5,762.48","+0.11%",True),
    ("NASDAQ","18,290.11","+0.35%",True),
    ("DOW JONES","42,981.20","-0.08%",False),
    ("GOLD","$2,648.30","+0.40%",True),
    ("USD/INR","84.22","-0.05%",False),
    ("CRUDE OIL","$77.40","+1.12%",True),
    ("BTC/USD","$67,450","+2.30%",True),
]
_ticker_html = "".join([
    f'<span style="margin:0 2rem;color:#8892b0;"><span style="color:#e8eaf6;font-weight:600;">{n}</span>'
    f'&nbsp;<span style="font-family:\'JetBrains Mono\',monospace;">{v}</span>'
    f'&nbsp;<span style="color:{"#00e676" if up else "#ff4757"}">{ch}</span></span>'
    for n, v, ch, up in _ticker_items
])
st.markdown(f"""
<div style="
    background:#0a1020;
    border:1px solid #1e2d4a;
    border-radius:12px;
    padding:0.5rem 0;
    margin-bottom:1rem;
    overflow:hidden;
    white-space:nowrap;
">
  <div style="display:inline-block;animation:ticker 30s linear infinite;font-size:0.78rem;">
    {_ticker_html}{_ticker_html}
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TOP HEADER BAR
# ─────────────────────────────────────────────
mode_label = MODE.split(" ", 1)[1]
_mode_descs = {
    "Breakout Strategy":      ("Support / Resistance breakout signals with buy & sell detection", "#3d8eff"),
    "Portfolio Optimization": ("XGBoost return forecasting + Efficient Frontier optimisation",   "#00e676"),
    "Intraday Prediction":    ("Random Forest next-bar price prediction with feature analysis",    "#a855f7"),
}
_desc, _hdr_color = _mode_descs.get(mode_label, ("AI-powered market intelligence", "#3d8eff"))
st.markdown(f"""
<div style="
    background:linear-gradient(135deg,#0d1526 0%,#111a2e 60%,#0d1526 100%);
    border:1px solid #1e2d4a;
    border-top:2px solid {_hdr_color};
    border-radius:16px;
    padding:1.2rem 1.6rem;
    margin-bottom:1.5rem;
    display:flex; align-items:center; justify-content:space-between;
    box-shadow:0 4px 32px rgba(0,0,0,0.5), 0 0 0 1px {_hdr_color}18;
    position:relative; overflow:hidden;
">
    <div style="position:absolute;top:0;left:0;right:0;height:100%;
        background:radial-gradient(ellipse at top left,{_hdr_color}10 0%,transparent 60%);
        pointer-events:none;"></div>
    <div>
        <div style="font-size:1.5rem;font-weight:800;color:#e8eaf6;letter-spacing:-0.02em;">
            {MODE.split()[0]}&nbsp;{mode_label}
        </div>
        <div style="font-size:0.8rem;color:#8892b0;margin-top:4px;">{_desc}</div>
    </div>
    <div style="text-align:right;display:flex;flex-direction:column;gap:6px;align-items:flex-end;">
        <div style="background:#0b2a1a;border:1px solid #00e67655;
             border-radius:20px;padding:4px 14px;font-size:0.7rem;color:#00e676;font-weight:700;">
            <span style="display:inline-block;width:6px;height:6px;background:#00e676;
                border-radius:50%;margin-right:5px;animation:pulse 1.5s ease-in-out infinite;
                vertical-align:middle;"></span>LIVE DATA
        </div>
        <div style="font-size:0.68rem;color:#4a5568;">yFinance &nbsp;·&nbsp; Real-time</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  MODULE 1 — BREAKOUT STRATEGY
# ═══════════════════════════════════════════════════════════
if "Breakout" in MODE:
    section_header("Support / Resistance Breakout Strategy",
                   "Identifies key price levels and generates buy/sell signals automatically")

    col_inp, col_n = st.columns([3, 1])
    with col_inp:
        stocks = st.text_input("🔍  Ticker Symbols (comma-separated)", "AAPL, RELIANCE.NS, HAL.NS",
                               help="e.g.  AAPL, TCS.NS, INFY.NS")
    with col_n:
        n = st.slider("Extrema Order (n)", 5, 20, 10,
                      help="Higher = fewer but stronger signals")

    # ── Timeframe controls ──
    st.markdown('<div style="margin:0.6rem 0 0.2rem 0;font-size:0.7rem;color:#8892b0;letter-spacing:0.08em;">⏱ TIMEFRAME</div>', unsafe_allow_html=True)
    tf_c1, tf_c2 = st.columns(2)
    with tf_c1:
        bo_period = st.selectbox(
            "Lookback Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="How far back to fetch historical data",
        )
    with tf_c2:
        bo_interval = st.selectbox(
            "Candle Interval",
            options=["1d", "1wk", "1mo"],
            index=0,
            format_func=lambda x: {"1d": "Daily (1D)", "1wk": "Weekly (1W)", "1mo": "Monthly (1M)"}[x],
            help="Candlestick resolution",
        )

    stock_list = [s.strip().upper() for s in stocks.split(",") if s.strip()]

    if stock_list:
        run_btn = st.button("🚀  Run Breakout Analysis", use_container_width=True)
        if run_btn:
            for symbol in stock_list:
                st.markdown(f"""
                <div style="background:#161d2f;border:1px solid #1e2d4a;border-radius:12px;
                     padding:1rem 1.4rem 0.5rem 1.4rem;margin-top:1.2rem;">
                    <div style="font-size:1rem;font-weight:700;color:#e8eaf6;margin-bottom:0.8rem;">
                        📊 &nbsp;{symbol}
                    </div>
                """, unsafe_allow_html=True)

                with st.spinner(f"Fetching data for {symbol}…"):
                    try:
                        df = yf.download(symbol, period=bo_period, interval=bo_interval,
                                         auto_adjust=True, progress=False)
                        if df.empty:
                            st.error(f"No data found for {symbol}")
                            st.markdown("</div>", unsafe_allow_html=True)
                            continue

                        # Flatten MultiIndex if present
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)

                        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                        df.dropna(inplace=True)

                        # Support / Resistance
                        res_idx = argrelextrema(df['High'].values, np.greater_equal, order=n)[0]
                        sup_idx = argrelextrema(df['Low'].values, np.less_equal,    order=n)[0]
                        df['resistance'] = np.nan
                        df['support']    = np.nan
                        df.iloc[res_idx, df.columns.get_loc('resistance')] = df['High'].iloc[res_idx]
                        df.iloc[sup_idx, df.columns.get_loc('support')]    = df['Low'].iloc[sup_idx]
                        df['resistance'] = df['resistance'].ffill()
                        df['support']    = df['support'].ffill()

                        def breakout_signal(row):
                            if pd.isna(row['resistance']) or pd.isna(row['support']):
                                return 0
                            if row['Close'] > row['resistance']: return 1
                            if row['Close'] < row['support']:    return -1
                            return 0
                        df['Signal'] = df.apply(breakout_signal, axis=1)

                        buys  = (df['Signal'] ==  1).sum()
                        sells = (df['Signal'] == -1).sum()
                        holds = (df['Signal'] ==  0).sum()
                        cur_price = df['Close'].iloc[-1]
                        cur_res   = df['resistance'].iloc[-1]
                        cur_sup   = df['support'].iloc[-1]

                        # Currency for this symbol
                        _sym, _cur = get_currency(symbol)

                        # Metric row
                        m1, m2, m3, m4 = st.columns(4)
                        with m1: metric_card("Current Price", fmt_price(symbol, cur_price), icon="💵", color=BLUE)
                        with m2: metric_card("Resistance",    fmt_price(symbol, cur_res),   icon="🔴", color=RED)
                        with m3: metric_card("Support",       fmt_price(symbol, cur_sup),   icon="🟢", color=GREEN)
                        with m4:
                            sig_now = df['Signal'].iloc[-1]
                            sig_label = "BUY 🟢" if sig_now == 1 else ("SELL 🔴" if sig_now == -1 else "HOLD ⚪")
                            sig_color = GREEN if sig_now == 1 else (RED if sig_now == -1 else "#8892b0")
                            metric_card("Current Signal", sig_label, icon="⚡", color=sig_color)

                        # Chart
                        chart_container_open("📈 Price Chart with Support / Resistance")
                        fig, ax = plt.subplots(figsize=(13, 4))
                        ax.plot(df.index, df['Close'], color=BLUE, linewidth=1.6, label='Close Price', zorder=3)
                        ax.plot(df.index, df['resistance'], color=RED,   linewidth=1.2, linestyle='--', alpha=0.8, label='Resistance')
                        ax.plot(df.index, df['support'],    color=GREEN, linewidth=1.2, linestyle='--', alpha=0.8, label='Support')
                        ax.fill_between(df.index, df['support'], df['resistance'], alpha=0.06, color=BLUE)
                        ax.scatter(df.index[df['Signal'] ==  1], df['Close'][df['Signal'] ==  1],
                                   marker='^', color=GREEN, s=80, zorder=5, label=f'Buy ({buys})')
                        ax.scatter(df.index[df['Signal'] == -1], df['Close'][df['Signal'] == -1],
                                   marker='v', color=RED,   s=80, zorder=5, label=f'Sell ({sells})')
                        ax.set_title(f"{symbol} — Support/Resistance Breakout", fontsize=13, fontweight='bold', pad=12)
                        ax.set_xlabel('Date');  ax.set_ylabel(f'Price ({_cur})')
                        leg = ax.legend(facecolor='#1c2540', edgecolor='#1e2d4a', labelcolor='#e8eaf6', fontsize=8)
                        apply_dark_style(fig)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        chart_container_close()

                        # Signal summary badges — styled trade card
                        sig_now = df['Signal'].iloc[-1]
                        sig_label = "BUY" if sig_now == 1 else ("SELL" if sig_now == -1 else "HOLD")
                        sig_color = GREEN if sig_now == 1 else (RED if sig_now == -1 else "#8892b0")
                        total_signals = max(buys + sells + holds, 1)
                        buy_pct  = buys  / total_signals * 100
                        sell_pct = sells / total_signals * 100
                        hold_pct = holds / total_signals * 100
                        st.markdown(f"""
                        <div style="background:#0e1526;border:1px solid #1e2d4a;border-radius:14px;
                            padding:1rem 1.2rem;margin-top:0.8rem;">
                          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
                            <div style="font-size:0.78rem;color:#8892b0;text-transform:uppercase;letter-spacing:0.07em;">Signal Distribution</div>
                            <div style="background:{sig_color}22;color:{sig_color};border:1px solid {sig_color}55;
                                padding:3px 14px;border-radius:20px;font-size:0.78rem;font-weight:700;">
                                Current: {sig_label}
                            </div>
                          </div>
                          <div style="display:flex;gap:4px;height:8px;border-radius:4px;overflow:hidden;">
                            <div style="flex:{buy_pct};background:#00e676;"></div>
                            <div style="flex:{hold_pct};background:#374151;"></div>
                            <div style="flex:{sell_pct};background:#ff4757;"></div>
                          </div>
                          <div style="display:flex;gap:1rem;margin-top:0.6rem;font-size:0.75rem;">
                            {badge(f"BUY {buys}", GREEN)}&nbsp;
                            {badge(f"SELL {sells}", RED)}&nbsp;
                            {badge(f"HOLD {holds}", "#374151")}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error processing {symbol}: {e}")

                st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  MODULE 2 — PORTFOLIO OPTIMIZATION
# ═══════════════════════════════════════════════════════════
elif "Portfolio" in MODE:
    section_header("Portfolio Optimization — Efficient Frontier",
                   "XGBoost return forecasting combined with mean-variance optimization")

    col_t, col_c = st.columns([3, 1])
    with col_t:
        tickers_input = st.text_input("🔍  Stock Tickers (comma-separated)",
                                      "TCS.NS, INFY.NS, HDFCBANK.NS",
                                      help="Use .NS suffix for NSE stocks")
    with col_c:
        capital = st.number_input("💰  Investment Capital (₹ / $)",
                                  min_value=100.0, value=100000.0, step=1000.0)

    # ── Timeframe controls ──
    st.markdown('<div style="margin:0.6rem 0 0.2rem 0;font-size:0.7rem;color:#8892b0;letter-spacing:0.08em;">⏱ DATE RANGE FOR TRAINING DATA</div>', unsafe_allow_html=True)
    pf_c1, pf_c2, pf_c3 = st.columns(3)
    with pf_c1:
        pf_start = st.date_input(
            "Start Date",
            value=datetime(2020, 1, 1).date(),
            min_value=datetime(2010, 1, 1).date(),
            max_value=datetime(2025, 12, 31).date(),
        )
    with pf_c2:
        pf_end = st.date_input(
            "End Date",
            value=datetime(2025, 6, 30).date(),
            min_value=datetime(2010, 1, 1).date(),
            max_value=datetime(2026, 12, 31).date(),
        )
    with pf_c3:
        pf_interval = st.selectbox(
            "Price Interval",
            options=["1d", "1wk", "1mo"],
            index=0,
            format_func=lambda x: {"1d": "Daily (1D)", "1wk": "Weekly (1W)", "1mo": "Monthly (1M)"}[x],
        )

    run_btn = st.button("🚀  Optimize Portfolio", use_container_width=True)

    if run_btn and tickers_input:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        with st.spinner("Fetching historical price data…"):
            adj_close_df = pd.DataFrame()
            for ticker in tickers:
                data = yf.download(ticker, start=str(pf_start), end=str(pf_end),
                                   interval=pf_interval, auto_adjust=True, progress=False)
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    adj_close_df[ticker] = data['Close']

        if adj_close_df.empty or adj_close_df.shape[0] < 30:
            st.error("Not enough historical data. Please check the tickers and try again.")
            st.stop()

        adj_close_df.ffill(inplace=True)
        adj_close_df.bfill(inplace=True)

        # ── ML feature engineering & XGBoost prediction ──
        def calculate_rsi(data, periods=14):
            delta = data.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=periods).mean()
            avg_loss = loss.rolling(window=periods).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        def prepare_features(df, ticker):
            d = df[[ticker]].copy()
            d['LogReturn'] = np.log(d[ticker] / d[ticker].shift(1))
            d['Lag1']  = d['LogReturn'].shift(1)
            d['Lag5']  = d['LogReturn'].shift(5)
            d['MA5']   = d[ticker].rolling(5).mean()
            d['MA10']  = d[ticker].rolling(10).mean()
            d['Vol10'] = d['LogReturn'].rolling(10).std()
            d['RSI']   = calculate_rsi(d[ticker])
            d.dropna(inplace=True)
            X = d[['Lag1', 'Lag5', 'MA5', 'MA10', 'Vol10', 'RSI']]
            y = d['LogReturn']
            return X, y

        section_header("🤖 XGBoost Model Performance", "Training on 80% | Testing on 20%")

        predicted_returns = {}
        valid_tickers     = []
        perf_rows         = []

        progress_bar = st.progress(0)
        for i, ticker in enumerate(tickers):
            try:
                X, y = prepare_features(adj_close_df, ticker)
                if len(X) < 30:
                    st.warning(f"Not enough data for {ticker} — skipping.")
                    continue

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                param_grid = {
                    'n_estimators':  [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth':     [3, 5],
                    'reg_alpha':     [0, 0.1],
                    'reg_lambda':    [1],
                }
                xgb = XGBRegressor(random_state=42, verbosity=0)
                gs  = GridSearchCV(xgb, param_grid, cv=3,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
                gs.fit(X_train, y_train)
                best = gs.best_estimator_

                y_pred = best.predict(X_test)
                mae  = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2   = r2_score(y_test, y_pred)

                perf_rows.append({
                    "Ticker": ticker,
                    "MAE":       round(mae,  6),
                    "RMSE":      round(rmse, 6),
                    "R² Score":  round(r2,   4),
                })

                pred_return = best.predict(X.iloc[-1:].values)[0] * 252
                predicted_returns[ticker] = pred_return
                valid_tickers.append(ticker)

            except Exception as e:
                st.warning(f"Error with {ticker}: {e}")

            progress_bar.progress(int((i + 1) / len(tickers) * 100))

        progress_bar.empty()

        if not valid_tickers:
            st.error("No valid stocks for ML prediction.")
            st.stop()

        # Model performance table
        perf_df = pd.DataFrame(perf_rows)
        st.dataframe(perf_df.style.background_gradient(cmap='Blues', subset=['R² Score'])
                     .format({"MAE": "{:.6f}", "RMSE": "{:.6f}", "R² Score": "{:.4f}"}),
                     use_container_width=True, hide_index=True)

        # ── Portfolio Optimization ──
        section_header("⚖️ Portfolio Optimization Results", "Mean-variance optimisation with Ledoit-Wolf covariance shrinkage")

        with st.spinner("Running Efficient Frontier optimisation…"):
            mu = expected_returns.mean_historical_return(adj_close_df[valid_tickers]) * 252
            S  = risk_models.CovarianceShrinkage(adj_close_df[valid_tickers]).ledoit_wolf() * 252
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.max_sharpe(risk_free_rate=0.06)
            optimal_weights = ef.clean_weights()
            weights_dict    = dict(optimal_weights)
            port_return, port_volatility, sharpe = ef.portfolio_performance(risk_free_rate=0.06)

        # Metric cards
        mc1, mc2, mc3 = st.columns(3)
        with mc1: metric_card("Expected Annual Return", f"{port_return:.2%}",
                               delta=f"+{port_return:.2%}", color=GREEN, icon="📈")
        with mc2: metric_card("Expected Volatility",    f"{port_volatility:.2%}", color=GOLD,  icon="📉")
        with mc3: metric_card("Sharpe Ratio",           f"{sharpe:.2f}",          color=BLUE,  icon="⚡")

        # Weights with progress bars
        st.markdown('<div style="margin-top:1rem;font-size:0.78rem;color:#8892b0;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">Optimal Portfolio Weights</div>', unsafe_allow_html=True)
        _w_colors = [GREEN, BLUE, GOLD, PURPLE, RED, "#64b5f6", "#ff7043"]
        for _wi, (k, v) in enumerate([(_k, _v) for _k, _v in weights_dict.items() if _v > 0.001]):
            _wc = _w_colors[_wi % len(_w_colors)]
            st.markdown(f"""
            <div style="margin-bottom:0.6rem;">
              <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:0.82rem;color:#e8eaf6;font-weight:600;">{k}</span>
                <span style="font-size:0.82rem;color:{_wc};font-family:'JetBrains Mono',monospace;font-weight:600;">{v:.1%}</span>
              </div>
              <div style="background:#1a2540;border-radius:4px;height:6px;">
                <div style="width:{v*100:.1f}%;height:6px;background:{_wc};border-radius:4px;
                    box-shadow:0 0 8px {_wc}66;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Discrete Allocation ──
        _cap_sym = capital_label(valid_tickers)
        section_header("🧾 Discrete Allocation", f"Number of shares to buy with capital = {_cap_sym}{capital:,.0f}")

        latest_prices = get_latest_prices(adj_close_df[valid_tickers])
        da = DiscreteAllocation(weights_dict, latest_prices, total_portfolio_value=capital)
        allocation, leftover = da.greedy_portfolio()

        alloc_df = pd.DataFrame([
            {
                "Ticker":        k,
                "Shares":        v,
                "Currency":      get_currency(k)[1],
                "Price/Share":   f"{fmt_price(k, float(latest_prices[k]))}",
                "Total Value":   f"{fmt_price(k, v * float(latest_prices[k]))}",
            }
            for k, v in allocation.items()
        ])
        total_invested = sum(v * float(latest_prices[k]) for k, v in allocation.items())

        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

        da1, da2 = st.columns(2)
        with da1: metric_card("Total Invested",    f"{_cap_sym}{total_invested:,.2f}", color=GREEN, icon="💸")
        with da2: metric_card("Remaining Capital", f"{_cap_sym}{leftover:,.2f}",       color=GOLD,  icon="🏦")

        # ── Visualizations ──
        section_header("📊 Portfolio Visualizations")

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Pie chart
        colors_pie = [GREEN, BLUE, GOLD, PURPLE, RED, "#64b5f6", "#ff7043"]
        non_zero_tickers = [t for t in valid_tickers if weights_dict.get(t, 0) > 0.001]
        non_zero_weights = [weights_dict[t] for t in non_zero_tickers]
        axs[0, 0].pie(non_zero_weights, labels=non_zero_tickers,
                      autopct='%1.1f%%', startangle=140,
                      colors=colors_pie[:len(non_zero_tickers)],
                      textprops={'color': '#e8eaf6', 'fontsize': 9},
                      wedgeprops={'edgecolor': CHART_BG, 'linewidth': 2})
        axs[0, 0].set_title("Optimal Portfolio Allocation", color='#e8eaf6', fontweight='bold')

        # Efficient frontier
        ef_plot = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        plotting.plot_efficient_frontier(ef_plot, ax=axs[0, 1], show_assets=False,
                                         ef_param_range=np.linspace(0.01, 0.4, 100))
        clrs = [GREEN, BLUE, GOLD, PURPLE, RED]
        for idx, t in enumerate(valid_tickers):
            axs[0, 1].scatter(np.sqrt(S.loc[t, t]), mu[t],
                              s=80, color=clrs[idx % len(clrs)], label=t, zorder=5)
        axs[0, 1].scatter(port_volatility, port_return,
                          marker='*', color=RED, s=300, label='Optimal', zorder=6)
        axs[0, 1].legend(facecolor='#1c2540', edgecolor='#1e2d4a', labelcolor='#e8eaf6', fontsize=8)
        axs[0, 1].set_title("Efficient Frontier", color='#e8eaf6', fontweight='bold')

        # Historical portfolio performance
        log_ret = np.log(adj_close_df[valid_tickers] / adj_close_df[valid_tickers].shift(1)).dropna()
        port_val = (log_ret.mul([weights_dict[t] for t in valid_tickers], axis=1)
                    .sum(axis=1)).cumsum().apply(np.exp)
        axs[1, 0].plot(port_val.index, port_val.values, color=BLUE, linewidth=2, label='Portfolio')
        axs[1, 0].fill_between(port_val.index, 1, port_val.values, alpha=0.15, color=BLUE)
        axs[1, 0].axhline(1, color=CHART_FG, linewidth=0.8, linestyle='--', alpha=0.4)
        axs[1, 0].set_title("Historical Portfolio Performance (Cumulative Return)",
                             color='#e8eaf6', fontweight='bold')
        axs[1, 0].legend(facecolor='#1c2540', edgecolor='#1e2d4a', labelcolor='#e8eaf6', fontsize=8)

        # Weight bar chart
        bar_colors = [GREEN if w >= max(non_zero_weights) * 0.5 else BLUE for w in non_zero_weights]
        axs[1, 1].barh(non_zero_tickers, non_zero_weights, color=bar_colors, edgecolor=CHART_BG)
        axs[1, 1].set_title("Portfolio Weight Distribution", color='#e8eaf6', fontweight='bold')
        for i, w in enumerate(non_zero_weights):
            axs[1, 1].text(w + 0.002, i, f"{w:.1%}", va='center', color='#e8eaf6', fontsize=8)

        apply_dark_style(fig, list(axs.flat))
        fig.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close(fig)

# ═══════════════════════════════════════════════════════════
#  MODULE 3 — INTRADAY PREDICTION
# ═══════════════════════════════════════════════════════════
elif "Intraday" in MODE:
    # dynamic subtitle filled after widget render
    section_header("Intraday Price Prediction — Random Forest",
                   "Select your preferred lookback period and bar interval below")

    col_t, col_b = st.columns([3, 1])
    with col_t:
        ticker = st.text_input("🔍  Stock Ticker", "HAL.NS",
                               help="e.g. HAL.NS, RELIANCE.NS, AAPL")
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀  Run Prediction", use_container_width=True)

    # ── Timeframe controls ──
    st.markdown('<div style="margin:0.6rem 0 0.2rem 0;font-size:0.7rem;color:#8892b0;letter-spacing:0.08em;">⏱ TIMEFRAME</div>', unsafe_allow_html=True)
    id_c1, id_c2 = st.columns(2)

    # yfinance intraday interval → max allowed period mapping
    _INTRA_PERIODS = {
        "1m":  ["1d", "5d"],
        "2m":  ["1d", "5d", "60d"],
        "5m":  ["1d", "5d", "60d"],
        "15m": ["1d", "5d", "60d"],
        "30m": ["1d", "5d", "10d", "30d", "60d"],
        "60m": ["1d", "5d", "10d", "30d", "60d"],
        "90m": ["1d", "5d", "10d", "30d", "60d"],
        "1h":  ["1d", "5d", "10d", "30d", "60d"],
    }
    with id_c1:
        id_interval = st.selectbox(
            "Bar Interval",
            options=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"],
            index=4,
            format_func=lambda x: {
                "1m": "1 Minute", "2m": "2 Minutes", "5m": "5 Minutes",
                "15m": "15 Minutes", "30m": "30 Minutes",
                "60m": "60 Minutes", "90m": "90 Minutes", "1h": "1 Hour",
            }[x],
        )
    with id_c2:
        allowed_periods = _INTRA_PERIODS.get(id_interval, ["5d", "10d"])
        id_period = st.selectbox(
            "Lookback Period",
            options=allowed_periods,
            index=min(1, len(allowed_periods) - 1),
            format_func=lambda x: {
                "1d": "1 Day", "5d": "5 Days", "10d": "10 Days",
                "30d": "30 Days", "60d": "60 Days",
            }.get(x, x),
            help="Max lookback depends on the chosen interval (yFinance limit)",
        )

    if run_btn:
        with st.spinner(f"Downloading {id_period} of {id_interval} data for {ticker}…"):
            df = yf.download(ticker, period=id_period, interval=id_interval,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        if df.empty:
            st.error(f"No intraday data returned for {ticker}. Check the ticker and try again.")
            st.stop()

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.dropna(inplace=True)
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        X = df[['Open', 'High', 'Low', 'Volume']]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        with st.spinner("Training Random Forest model…"):
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        r2  = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Next bar prediction
        last_features = X.iloc[-1:].values
        next_price = model.predict(last_features)[0]
        cur_price  = float(df['Close'].iloc[-1])
        direction  = "BUY 🟢" if next_price > cur_price else "SELL 🔴"
        dir_color  = GREEN if next_price > cur_price else RED

        # Currency for intraday ticker
        _id_sym, _id_cur = get_currency(ticker)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_card("R² Score",      f"{r2:.4f}",  color=GREEN if r2 > 0.7 else GOLD, icon="🎯")
        with m2: metric_card("RMSE",          f"{fmt_price(ticker, rmse)}", color=BLUE, icon="📐")
        with m3: metric_card("MAE",           f"{fmt_price(ticker, mae)}",  color=BLUE, icon="📏")
        with m4: metric_card("Next Bar Signal", direction,
                              delta=f"Predicted: {fmt_price(ticker, next_price)}  |  Current: {fmt_price(ticker, cur_price)}",
                              color=dir_color, icon="⚡")

        # Chart — Actual vs Predicted
        chart_container_open("📈 Actual vs Predicted Price (Test Set)")
        fig, ax = plt.subplots(figsize=(14, 4.5))
        x_axis = range(len(y_test))
        ax.plot(x_axis, y_test.values, color=BLUE,  linewidth=1.8, label='Actual Price',    zorder=3)
        ax.plot(x_axis, y_pred,        color=RED,   linewidth=1.8, label='Predicted Price', zorder=3, linestyle='--')
        ax.fill_between(x_axis, y_test.values, y_pred,
                        where=(np.array(y_pred) >= np.array(y_test.values)),
                        alpha=0.15, color=RED, label='Overestimate')
        ax.fill_between(x_axis, y_test.values, y_pred,
                        where=(np.array(y_pred) <  np.array(y_test.values)),
                        alpha=0.15, color=GREEN, label='Underestimate')
        ax.set_title(f"{ticker} — Intraday Price Prediction (30-min intervals)",
                     fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Time Steps (30-min bars)")
        ax.set_ylabel("Price")
        ax.legend(facecolor='#1c2540', edgecolor='#1e2d4a', labelcolor='#e8eaf6', fontsize=9)
        apply_dark_style(fig)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        chart_container_close()

        # Feature importance
        st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
        section_header("🔬 Feature Importance", "Contribution of each input feature to the model")

        feat_names  = ['Open', 'High', 'Low', 'Volume']
        importances = model.feature_importances_
        sorted_idx  = np.argsort(importances)[::-1]

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        bar_colors = [GREEN, BLUE, GOLD, PURPLE]
        ax2.bar([feat_names[i] for i in sorted_idx],
                [importances[i] for i in sorted_idx],
                color=[bar_colors[j] for j in range(len(feat_names))],
                edgecolor=CHART_BG, linewidth=0.8)
        ax2.set_title("Feature Importance (Random Forest)", fontweight='bold')
        ax2.set_ylabel("Importance Score")
        for j, idx in enumerate(sorted_idx):
            ax2.text(j, importances[idx] + 0.005, f"{importances[idx]:.3f}",
                     ha='center', color='#e8eaf6', fontsize=9)
        apply_dark_style(fig2)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # Recent data preview
        section_header("📋 Recent Price Data")
        st.dataframe(
            df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20)
            .style.format("{:.2f}", subset=['Open', 'High', 'Low', 'Close'])
                  .format("{:,.0f}", subset=['Volume']),
            use_container_width=True,
        )

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
_tech = [("XGBoost","#3d8eff"),("Random Forest","#00e676"),("PyPortfolioOpt","#a855f7"),("yFinance","#ffd700"),("Streamlit","#ff4757")]
_tech_pills = " ".join([f'<span style="background:{c}22;color:{c};border:1px solid {c}44;padding:3px 12px;border-radius:20px;font-size:0.7rem;font-weight:600;">{t}</span>' for t, c in _tech])
st.markdown(f"""
<div style="
    margin-top:3rem;
    border-top:1px solid #1e2d4a;
    padding:1.5rem 0 1rem 0;
">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
        <div>
            <div style="font-size:0.9rem;font-weight:700;background:linear-gradient(135deg,#3d8eff,#00e676);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">⚡ StockEdge Pro</div>
            <div style="font-size:0.65rem;color:#4a5568;margin-top:2px;">For educational purposes only. Not financial advice.</div>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
            {_tech_pills}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
