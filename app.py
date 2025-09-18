# Altara — AI-Powered Trading Dashboard (MVP)
# ------------------------------------------------
# Premium, clean Streamlit app for U.S. stocks & ETFs
# Features: Market Overview, Sector Heatmap, Top Movers, News & AI Recap,
# Earnings Calendar, Altara Opportunities (AI-like shortlist), Ticker Quick Search
#
# Notes
# - Real data via yfinance; optional Finnhub & OpenAI. Graceful fallbacks included.
# - Keep code clear, modular, and production-ready with caching and error handling.
#
# Environment / Secrets:
#   st.secrets["FINNHUB_API_KEY"] or env FINNHUB_API_KEY
#   st.secrets["OPENAI_API_KEY"] or env OPENAI_API_KEY
#
# Disclaimer: Educational purposes only, not financial advice.

import os
import time
import math
import json
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

import streamlit as st
import yfinance as yf

# Optional: OpenAI (only if key provided)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Optional: Plotly for charts/heatmap
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# ---------- THEME ------------
# -----------------------------

ALTARA_BLUE = "#0B3C78"
ACCENT_GOLD = "#D4AF37"

st.set_page_config(
    page_title="Altara — AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Global CSS for premium look
st.markdown(f"""
<style>
/* Global font + card polish */
html, body, [class*="css"]  {{
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
}}

section[data-testid="stSidebar"] > div {{
  background: #0a0f1a;
  border-right: 1px solid rgba(255,255,255,0.08);
}}

div.block-container {{
  padding-top: 1.2rem;
}}

h1, h2, h3 {{
  color: white;
}}

.small-muted {{
  color: rgba(255,255,255,0.6);
  font-size: 0.85rem;
}}

.card {{
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.8rem;
}}

.card-header {{
  font-weight: 600;
  color: #ffffff;
  margin-bottom: 0.2rem;
}}

.badge {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 0.75rem;
  border: 1px solid rgba(255,255,255,0.12);
  color: rgba(255,255,255,0.85);
  margin-left: 6px;
}}

.kpi {{
  display:flex;
  align-items: baseline;
  gap: 10px;
}}
.kpi .val {{
  font-weight: 700;
  font-size: 1.25rem;
  color: white;
}}
.kpi .chg.up {{ color: #16c784; }}
.kpi .chg.down {{ color: #ea3943; }}

a, a:visited {{
  color: {ACCENT_GOLD};
  text-decoration: none;
}}
a:hover {{ text-decoration: underline; }}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# --------- HELPERS -----------
# -----------------------------

def get_secret(name: str) -> Optional[str]:
    """Read from Streamlit secrets OR environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)

def human_pct(x):
    try:
        return f"{x:.2f}%"
    except Exception:
        return "—"

def safe_pct_change(curr, prev):
    try:
        return (curr/prev - 1.0) * 100.0
    except Exception:
        return np.nan

def positive_badge(text: str):
    return f'<span class="badge" style="border-color:#1b6; color:#1b6;">{text}</span>'

def negative_badge(text: str):
    return f'<span class="badge" style="border-color:#e66; color:#e66;">{text}</span>'

def neutral_badge(text: str):
    return f'<span class="badge">{text}</span>'

@st.cache_data(ttl=300, show_spinner=False)
def yf_prices(tickers: List[str], period: str="5d", interval: str="1d") -> pd.DataFrame:
    """Batch download prices, return multi-index DF: columns like ('Close', 'AAPL')."""
    try:
        df = yf.download(tickers, period=period, interval=interval, auto_adjust=False, threads=True, progress=False, group_by='column')
        # yfinance sometimes returns single-index columns when only 1 ticker; normalize:
        if isinstance(df.columns, pd.MultiIndex):
            return df
        else:
            # Create MultiIndex for single ticker
            ticker = tickers[0] if isinstance(tickers, list) and len(tickers) else None
            new_cols = pd.MultiIndex.from_product([df.columns, [ticker]])
            df.columns = new_cols
            return df
    except Exception:
        return pd.DataFrame()

def compute_intraday_change(symbol: str) -> Dict[str, float]:
    """Prev close vs last close % change for symbol. Returns last price and change%."""
    df = yf_prices([symbol], period="5d", interval="1d")
    try:
        close = df[("Close", symbol)].dropna()
        if len(close) >= 2:
            last = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            chg = safe_pct_change(last, prev)
            return {"last": last, "change_pct": chg}
    except Exception:
        pass
    return {"last": np.nan, "change_pct": np.nan}

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def moving_avg(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def volatility(series: pd.Series, window: int = 20) -> float:
    returns = series.pct_change().dropna()
    return float(returns.rolling(window).std().iloc[-1]) if len(returns) >= window else float(returns.std())

# -----------------------------
# ------- STATIC UNIVERSE -----
# -----------------------------

SPY_UNIVERSE = [
    # Large-cap demo universe (subset of S&P 100 + liquid names)
    "AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","TSLA","AVGO","LLY","JPM","V","WMT","XOM","UNH","MA",
    "PG","JNJ","HD","MRK","COST","ORCL","BAC","ABBV","CVX","KO","PEP","ADBE","WFC","CSCO","ACN","MCD","CRM",
    "NFLX","INTC","LIN","ABT","AMD","TXN","TMO","HON","PM","IBM","AMAT","BA","GE","CAT","AMGN","SPY","QQQ","DIA",
    "LOW","BKNG","DE","MS","BLK","GS","NOW","LMT","MDT","ISRG","QCOM","RTX","SYK","ADI","NEE","SCHW","INTU","SBUX",
    "PLD","PFE","BK","MU","ELV","USB","GILD","CB","VRTX","CVS","SO","MMC","REGN","T","PNC","C","ADI","SHOP",
    "UBER","PANW","ARM","SNOW","TSM"
]

SECTOR_ETFS = {
    "Tech": "XLK",
    "Comm": "XLC",
    "Cons Disc": "XLY",
    "Cons Staples": "XLP",
    "Health": "XLV",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE"
}

INDEX_PROXIES = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Dow Jones": "DIA",
    "Russell 2000": "IWM",
    "VIX": "^VIX"
}

# -----------------------------
# ------- NEWS & AI -----------
# -----------------------------

def fetch_company_news(symbol: str, max_items: int = 8) -> List[Dict]:
    """Finnhub if available; else fallback to yfinance Ticker.news (best-effort)."""
    items = []
    token = get_secret("FINNHUB_API_KEY")
    today = date.today()
    start = today - relativedelta(days=7)
    if token:
        try:
            url = "https://finnhub.io/api/v1/company-news"
            params = {"symbol": symbol, "from": start.isoformat(), "to": today.isoformat(), "token": token}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()[:max_items]
            for n in data:
                items.append({
                    "source": n.get("source") or "News",
                    "headline": n.get("headline") or "",
                    "summary": n.get("summary") or "",
                    "url": n.get("url"),
                    "datetime": datetime.fromtimestamp(n.get("datetime", time.time()))
                })
            return items
        except Exception:
            pass
    # yfinance fallback (single symbol)
    try:
        news = yf.Ticker(symbol).news or []
        for n in news[:max_items]:
            items.append({
                "source": n.get("provider") or "News",
                "headline": n.get("title") or "",
                "summary": n.get("summary") or "",
                "url": n.get("link") or n.get("url"),
                "datetime": datetime.fromtimestamp(n.get("providerPublishTime", time.time()))
            })
    except Exception:
        pass
    return items

def ai_market_recap(context: Dict) -> str:
    """Generate 80–120 word recap; uses OpenAI if available, else heuristic."""
    # Heuristic summary
    def heuristic(ctx):
        parts = []
        for name, d in ctx.get("indices", {}).items():
            chg = d.get("change_pct")
            if math.isnan(chg):
                continue
            direction = "up" if chg >= 0 else "down"
            parts.append(f"{name} {direction} {abs(chg):.2f}%")
        sector_winners = ctx.get("sectors_top", [])[:2]
        sector_losers = ctx.get("sectors_bottom", [])[:2]
        line = " • ".join(parts)
        sec_line = ""
        if sector_winners or sector_losers:
            win = ", ".join([s for s,_ in sector_winners]) or "—"
            lose = ", ".join([s for s,_ in sector_losers]) or "—"
            sec_line = f" Sectors: leaders {win}; laggards {lose}."
        return f"Market recap: {line}.{sec_line} Volume and breadth reflect the day’s tone. Keep risk management in mind; this is informational, not financial advice."
    # OpenAI path
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return heuristic(context)
    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""
        You are Altara, an AI market analyst. Write an 80–120 word U.S. market recap in plain English.
        Use this JSON context:\n{json.dumps(context, default=str)}
        Requirements:
        - Mention overall move for S&P 500, Nasdaq 100, Dow, Russell 2000, and VIX if available.
        - Call out 1–2 leading and lagging sectors.
        - Keep neutral tone; no hype. Add one sentence of risk/caution.
        - End with: "Educational use only — not financial advice."
        """
        msg = [
            {"role": "system", "content": "You summarize markets clearly and concisely for retail investors."},
            {"role": "user", "content": prompt.strip()}
        ]
        # Use a lightweight model if available
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=180,
            messages=msg
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception:
        return heuristic(context)

# -----------------------------
# ----- EARNINGS CALENDAR -----
# -----------------------------

def fetch_earnings(days_ahead: int = 7, limit: int = 50) -> pd.DataFrame:
    token = get_secret("FINNHUB_API_KEY")
    start = date.today()
    end = start + relativedelta(days=days_ahead)
    if token:
        try:
            url = "https://finnhub.io/api/v1/calendar/earnings"
            params = {"from": start.isoformat(), "to": end.isoformat(), "token": token}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json().get("earningsCalendar", [])[:limit]
            if data:
                df = pd.DataFrame(data)
                df.rename(columns={"symbol": "Symbol", "date": "Date", "epsEstimate": "Est EPS", "revenueEstimate": "Est Rev"}, inplace=True)
                return df[["Date","Symbol","Est EPS","Est Rev"]]
        except Exception:
            pass
    # Fallback demo rows
    demo = [
        {"Date": (start + relativedelta(days=1)).isoformat(), "Symbol": "AAPL", "Est EPS": 1.41, "Est Rev": 90900000000},
        {"Date": (start + relativedelta(days=2)).isoformat(), "Symbol": "MSFT", "Est EPS": 2.63, "Est Rev": 60400000000},
        {"Date": (start + relativedelta(days=3)).isoformat(), "Symbol": "NVDA", "Est EPS": 5.20, "Est Rev": 28000000000},
    ]
    return pd.DataFrame(demo)

# -----------------------------
# ----- OPPORTUNITY SCANNER ---
# -----------------------------

def scan_opportunities(tickers: List[str], lookback_days: int = 120, max_results: int = 10) -> List[Dict]:
    """Momentum/health scan with simple signals: Breakout, Oversold bounce, Golden Cross."""
    if not tickers:
        return []

    data = yf.download(tickers, period=f"{lookback_days}d", interval="1d", auto_adjust=False, threads=True, progress=False, group_by='column')
    picks = []

    if data.empty:
        return picks

    # Normalize for single ticker vs multi ticker columns
    if not isinstance(data.columns, pd.MultiIndex):
        # single ticker; wrap
        t = tickers[0]
        data.columns = pd.MultiIndex.from_product([data.columns, [t]])

    for t in tickers:
        try:
            df = data.xs(t, axis=1, level=1).dropna()
            if df.empty or len(df) < 50:
                continue
            close = df["Close"]
            high = df["High"]
            vol = df["Volume"]

            ma20 = moving_avg(close, 20)
            ma50 = moving_avg(close, 50)
            r = rsi(close, 14)
            vol20 = vol.rolling(20).mean()

            last = close.iloc[-1]
            prev = close.iloc[-2] if len(close) > 1 else last
            last_rsi = float(r.iloc[-1]) if not np.isnan(r.iloc[-1]) else np.nan
            last_ma20 = float(ma20.iloc[-1])
            last_ma50 = float(ma50.iloc[-1])
            last_vol = float(vol.iloc[-1]) if len(vol) else np.nan
            last_vol20 = float(vol20.iloc[-1]) if not np.isnan(vol20.iloc[-1]) else np.nan
            vol_rel = (last_vol / last_vol20) if last_vol20 and last_vol20 > 0 else np.nan

            # Signals
            breakout = last >= high.rolling(50).max().iloc[-2] * 0.999 if len(high) > 50 else False
            oversold_bounce = (last_rsi < 30) and (last > prev)
            golden_cross = (last_ma20 > last_ma50) and (ma20.iloc[-2] <= ma50.iloc[-2])

            signal = None
            reasons = []
            risks = ["Macro or sector volatility", "False signal in choppy market"]

            if breakout:
                signal = "Breakout setup"
                reasons += [f"Price near 50D high", f"MA20 ≥ MA50 ({last_ma20:.2f} ≥ {last_ma50:.2f})", f"Volume vs avg ~{vol_rel:.2f}x"]
            elif golden_cross:
                signal = "Golden Cross (MA20>MA50)"
                reasons += [f"MA20 crossed above MA50", f"RSI {last_rsi:.1f}", f"Volume vs avg ~{vol_rel:.2f}x"]
            elif oversold_bounce:
                signal = "Oversold bounce"
                reasons += [f"RSI {last_rsi:.1f} < 30", "Green day after selloff", f"Volume vs avg ~{vol_rel:.2f}x"]
            else:
                # Momentum health filter
                if (last > last_ma50) and (last_ma20 >= last_ma50) and (30 < last_rsi < 70):
                    signal = "Momentum uptrend"
                    reasons += [f"Price above MA50", f"MA20 ≥ MA50 ({last_ma20:.2f} ≥ {last_ma50:.2f})", f"RSI {last_rsi:.1f}"]
                else:
                    continue  # skip

            # Risk label by volatility
            vol_20 = volatility(close, 20)
            if vol_20 <= 0.02:
                risk = "Conservative"
            elif vol_20 <= 0.035:
                risk = "Balanced"
            else:
                risk = "Aggressive"

            chg = (last/prev - 1)*100 if prev else np.nan

            picks.append({
                "ticker": t,
                "signal": signal,
                "risk": risk,
                "last": float(last),
                "change_pct": float(chg) if not np.isnan(chg) else np.nan,
                "rsi": float(last_rsi) if not np.isnan(last_rsi) else np.nan,
                "reasons": reasons[:3],
                "risks": risks[:2]
            })
        except Exception:
            continue

    # Sort: prioritize strongest technical posture
    def score(p):
        s = 0
        if p["signal"].startswith("Breakout"): s += 3
        if "Momentum" in p["signal"]: s += 2
        if p["risk"] == "Conservative": s += 2
        if not math.isnan(p["rsi"]):
            s += (70 - abs(55 - p["rsi"])) / 20  # sweet spot around 55
        return s

    picks.sort(key=score, reverse=True)
    return picks[:max_results]

# -----------------------------
# ------- TOP MOVERS -----------
# -----------------------------

@st.cache_data(ttl=180, show_spinner=False)
def compute_movers(universe: List[str]) -> pd.DataFrame:
    """Compute gainers/losers/most active from a ticker universe."""
    if not universe:
        return pd.DataFrame()
    df = yf.download(universe, period="7d", interval="1d", auto_adjust=False, threads=True, progress=False, group_by='ticker')
    rows = []
    # Normalize for single ticker
    multi = isinstance(df.columns, pd.MultiIndex)
    for t in universe:
        try:
            sub = df[t] if multi else df
            sub = sub.dropna()
            if len(sub) < 2: 
                continue
            last = float(sub["Close"].iloc[-1])
            prev = float(sub["Close"].iloc[-2])
            chg = safe_pct_change(last, prev)
            vol = float(sub["Volume"].iloc[-1]) if "Volume" in sub else np.nan
            vol20 = float(sub["Volume"].rolling(20).mean().iloc[-1]) if "Volume" in sub else np.nan
            vol_rel = (vol/vol20) if vol20 and vol20>0 else np.nan
            spark = sub["Close"].tail(10).tolist()
            rows.append({"Ticker": t, "Price": last, "Change %": chg, "Rel Vol": vol_rel, "Spark": spark})
        except Exception:
            continue
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Change %"] = out["Change %"].round(2)
    out["Rel Vol"] = out["Rel Vol"].round(2)
    return out.sort_values("Change %", ascending=False)

# -----------------------------
# ------- UI COMPONENTS -------
# -----------------------------

def kpi_card(label: str, value: float, change: float, help_text: Optional[str] = None):
    chg_class = "up" if change >= 0 else "down"
    change_fmt = f"+{change:.2f}%" if change >= 0 else f"{change:.2f}%"
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-header">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi"><div class="val">{value:,.2f}</div><div class="chg {chg_class}">{change_fmt}</div></div>', unsafe_allow_html=True)
        if help_text:
            st.markdown(f'<div class="small-muted">{help_text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def section_header(title: str, badge: Optional[str] = None):
    st.markdown(f"### {title} " + (f"{badge}" if badge else ""))

def market_overview():
    section_header("Market Overview", neutral_badge("Today"))
    cols = st.columns(5)
    for i, (name, sym) in enumerate(INDEX_PROXIES.items()):
        data = compute_intraday_change(sym)
        with cols[i]:
            kpi_card(name, data["last"], data["change_pct"], help_text=sym)

def sector_heatmap():
    section_header("Sector Performance", neutral_badge("SPDR"))
    syms = list(SECTOR_ETFS.values())
    dfp = yf_prices(syms, period="7d", interval="1d")
    rows = []
    for name, sym in SECTOR_ETFS.items():
        try:
            close = dfp[("Close", sym)].dropna()
            if len(close) >= 2:
                chg = safe_pct_change(float(close.iloc[-1]), float(close.iloc[-2]))
            else:
                chg = np.nan
        except Exception:
            chg = np.nan
        rows.append({"Sector": name, "Symbol": sym, "Change %": chg})
    sdf = pd.DataFrame(rows).sort_values("Change %", ascending=False)

    fig = px.treemap(
        sdf,
        path=["Sector"],
        values=[abs(x) if not math.isnan(x) else 1 for x in sdf["Change %"].tolist()],
        color="Change %",
        color_continuous_scale=[(0, "#ea3943"), (0.5, "#2b2b2b"), (1, "#16c784")],
        hover_data={"Symbol": True, "Change %": ":.2f"}
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=280)
    st.plotly_chart(fig, use_container_width=True)

def top_movers(universe: List[str]):
    section_header("Top Movers", neutral_badge("Demo Universe"))
    df = compute_movers(universe)
    if df.empty:
        st.info("No data available for movers (demo). Try a smaller universe.")
        return
    # Table with sparkline
    st.dataframe(
        df[["Ticker","Price","Change %","Rel Vol","Spark"]].head(50),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="%.2f"),
            "Change %": st.column_config.NumberColumn(format="%.2f"),
            "Rel Vol": st.column_config.NumberColumn(format="%.2f"),
            "Spark": st.column_config.LineChartColumn(width="medium", y_min=None, y_max=None)
        }
    )

def news_and_ai_recap(primary_symbol: str, sectors_sorted: List[tuple], indices_ctx: Dict):
    left, right = st.columns((1,1))
    with left:
        section_header("AI Market Recap", positive_badge("Daily"))
        ctx = {
            "indices": indices_ctx,
            "sectors_top": sectors_sorted[:3],
            "sectors_bottom": list(reversed(sectors_sorted[-3:]))
        }
        recap = ai_market_recap(ctx)
        st.write(recap)

    with right:
        section_header(f"Latest News — {primary_symbol}")
        items = fetch_company_news(primary_symbol, max_items=6)
        if not items:
            st.info("No recent company news available.")
        else:
            for n in items:
                st.markdown(f"**{n['headline']}**  \n<span class='small-muted'>{n['source']} • {n['datetime'].strftime('%Y-%m-%d %H:%M')}</span>  \n[Read more]({n['url']})", unsafe_allow_html=True)
                st.markdown("---")

def earnings_calendar():
    section_header("Upcoming Earnings (Next 7 Days)")
    df = fetch_earnings(days_ahead=7, limit=50)
    if df.empty:
        st.info("No earnings data available.")
        return
    # Pretty display
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(df, use_container_width=True, hide_index=True)

def opportunities(universe: List[str]):
    section_header("Altara Opportunities", positive_badge("Daily Shortlist"))
    with st.spinner("Scanning for healthy momentum setups…"):
        picks = scan_opportunities(universe, lookback_days=150, max_results=10)

    if not picks:
        st.info("No opportunities identified in the current demo universe. Try expanding your symbols or check back later.")
        return

    for p in picks:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            head = f"**{p['ticker']}** — {p['signal']}  {neutral_badge(p['risk'])}"
            st.markdown(head, unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.metric("Last", f"{p['last']:.2f}", f"{p['change_pct']:.2f}%")
            with col2:
                st.metric("RSI(14)", f"{p['rsi']:.1f}")
            with col3:
                st.caption("Educational use only — not financial advice.")
            st.write("**Why it stands out**")
            st.write("- " + "\n- ".join(p["reasons"]))
            st.write("**Risks**")
            st.write("- " + "\n- ".join(p["risks"]))
            st.markdown('</div>', unsafe_allow_html=True)

def ticker_quick_search():
    section_header("Ticker Quick Search")
    q = st.text_input("Type a ticker (e.g., AAPL)", value="AAPL")
    if not q:
        return
    q = q.strip().upper()
    try:
        hist = yf.download(q, period="1mo", interval="1d", progress=False)
        if hist.empty:
            st.warning("No price data found for this ticker.")
            return
        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last
        chg = safe_pct_change(last, prev)
        c1, c2 = st.columns([1,2])
        with c1:
            st.metric(f"{q} Price", f"{last:.2f}", f"{chg:.2f}%")
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name=q))
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=220)
            st.plotly_chart(fig, use_container_width=True)
        # Fast take (heuristic)
        st.caption("Fast take: Simple daily change view. For deeper analysis, open the Ticker Workspace (coming soon).")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# -----------------------------
# ------------- APP -----------
# -----------------------------

def main():
    st.title("Altara — Premium AI Trading Dashboard")
    st.caption("Clean. Clear. Contextual.  \nEducational use only — not financial advice.")

    # Sidebar preferences
    with st.sidebar:
        st.markdown(f"## ⚙️ Settings")
        st.text_input("FINNHUB_API_KEY", value=get_secret("FINNHUB_API_KEY") or "", type="password", key="finnhub_key")
        st.text_input("OPENAI_API_KEY", value=get_secret("OPENAI_API_KEY") or "", type="password", key="openai_key")
        st.selectbox("Ticker Universe Preset", options=["S&P Large-Cap (demo)","Custom"], key="universe_preset")
        if st.session_state["universe_preset"] == "Custom":
            custom = st.text_area("Custom tickers (comma-separated)", value="AAPL,MSFT,NVDA,AMZN,TSLA,AMD,GOOGL,META,SPY,QQQ")
            universe = [x.strip().upper() for x in custom.split(",") if x.strip()]
        else:
            universe = SPY_UNIVERSE

        primary_symbol = st.text_input("Primary news symbol", value="AAPL").strip().upper()
        st.markdown("---")
        st.markdown("**Tips**\n- Add your API keys here or via `st.secrets`.\n- Custom universe speeds up scanning.\n- More features coming soon (Screener, Alerts, Ticker Workspace).")
        st.markdown("---")
        st.markdown("**Theme**")
        st.color_picker("Altara Blue", value=ALTARA_BLUE, key="theme_color")

    # Top: Market Overview + Sector Heatmap
    c_top = st.columns((1.3, 1))
    with c_top[0]:
        market_overview()
    with c_top[1]:
        sector_heatmap()

    # Movers
    top_movers(universe)

    # Build sectors_sorted + indices_ctx for AI recap
    sectors_rows = []
    for s_name, s_sym in SECTOR_ETFS.items():
        data = compute_intraday_change(s_sym)
        sectors_rows.append((s_name, data.get("change_pct", np.nan)))
    sectors_sorted = sorted(sectors_rows, key=lambda x: (float('-inf') if math.isnan(x[1]) else x[1]), reverse=True)

    indices_ctx = {name: compute_intraday_change(sym) for name, sym in INDEX_PROXIES.items()}

    # News & AI Recap
    news_and_ai_recap(primary_symbol, sectors_sorted, indices_ctx)

    # Earnings
    earnings_calendar()

    # Opportunities
    opportunities(universe)

    # Quick Search
    ticker_quick_search()

    # Footer disclaimer
    st.markdown("---")
    st.markdown("**Altara** — © {year} • Educational use only — not financial advice.".format(year=datetime.now().year))
    st.markdown("Data sources: Yahoo Finance (yfinance), optional Finnhub. AI summaries optional via OpenAI.")

if __name__ == "__main__":
    main()