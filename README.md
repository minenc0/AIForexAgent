# AI FOREX DECISION AGENT

AI Agent profesional untuk mengambil keputusan BUY, SELL, atau NO TRADE pada trading Forex dengan reasoning berbasis LLM.

## Fitur

- **Multi-Agent Architecture** — 8 AI agents yang saling berkoordinasi
- **Correlation Engine** — Konfirmasi sinyal berdasarkan korelasi pair forex
- **Multi Timeframe Analysis** — H4, H1, M15, M5
- **News Filter** — Deteksi High Impact News dari Forex Factory
- **Technical Indicators** — EMA, RSI, MACD, ATR, ADX, Support/Resistance
- **Candlestick Pattern Recognition** — 10 pola candlestick utama
- **AI Reasoning** — GPT-based reasoning untuk keputusan akhir
- **Risk Management** — R:R 1:2 dengan perhitungan otomatis
- **Backtest Engine** — Win Rate, Profit Factor, Drawdown, Sharpe Ratio
- **AI Memory** — Evaluasi otomatis dan penyesuaian bobot indikator
- **SQLite Database** — Riwayat trade tersimpan

## Deploy ke Streamlit Cloud

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/AI_FOREX_AGENT.git
git push -u origin main
```

Kemudian hubungkan repository di [Streamlit Cloud](https://streamlit.cloud).

## Konfigurasi

1. Copy `.env.example` menjadi `.env`
2. Masukkan OpenAI API Key
3. Jalankan `streamlit run app.py`

## Teknologi

- Python 3.12
- Streamlit
- Plotly
- yfinance
- OpenAI GPT
- SQLite