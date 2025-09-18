# Altara — AI-Powered Trading Dashboard (MVP)

Premium, clean Streamlit app for U.S. stocks & ETFs with AI explanations.

## ✨ What’s inside
- **Market Overview**: SPY/QQQ/DIA/IWM and VIX KPIs
- **Sector Performance**: SPDR treemap with color-by-change
- **Top Movers**: Gainers/Losers/Most Active from a demo large-cap universe (fast)
- **News & AI Recap**: Real headlines + optional AI daily market recap
- **Earnings Calendar**: Next 7 days (Finnhub if available → graceful fallback)
- **Altara Opportunities**: 5–10 momentum/health setups with 3 reasons + 2 risks
- **Ticker Quick Search**: Price, % change, and mini chart

> **Educational use only — not financial advice.**

## 🔧 Setup

### Local
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

### GitHub Codespaces
1. Create a new Codespace on this folder.
2. In the terminal:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py --server.address 0.0.0.0 --server.port 7860
   ```
3. Open the forwarded port.

### Streamlit Cloud
1. Push this folder to GitHub.
2. In Streamlit Cloud, point to `app.py`.
3. Add secrets under **Settings → Secrets**:
   ```toml
   FINNHUB_API_KEY="..."
   OPENAI_API_KEY="..."
   ```

### Environment variables (optional)
Instead of Streamlit secrets, you can export:
```bash
export FINNHUB_API_KEY=...
export OPENAI_API_KEY=...
```

## 🧠 Notes on Data & AI
- **Prices**: yfinance (Yahoo Finance). We compute day-over-day %.
- **Sectors**: SPDR ETFs (XLK, XLF, etc.).
- **News**: Finnhub company-news if key provided; falls back to Yahoo news via yfinance.
- **AI Recap**: Uses OpenAI (gpt-4o-mini) if key provided; else a clear heuristic summary.

## 🧪 Troubleshooting
- If charts look empty, ensure internet access is available to the server.
- Movers scan uses a demo universe for speed. Switch to **Custom** in the sidebar to scan your tickers.
- OpenAI/Finnhub requests are optional. The app works without them (with limited features).

## 🛡️ Compliance
Altara is an educational tool and does not provide investment advice. Always include the built-in disclaimer in UI.

---

**Brand**: Altara Blue `#0B3C78`, dark backgrounds, accent gold for highlights.
**UI Goal**: Clean, modern, trustworthy.