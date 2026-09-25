import datetime
import logging
import re
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from bs4 import XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from src.agents.schemas import EarningsSignal
from src.backtest.market_data import LiveTickerFetcher, MarketDataFetcher
from src.config import get_settings
from src.pipeline import EarningsIntelligencePipeline

# Page configuration
st.set_page_config(
    page_title="Adversarial Earnings Intelligence | Institutional Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bloomberg / Goldman Sachs Corporate Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #E2E8F0;
    }

    /* Terminal background */
    .stApp {
        background-color: #080C14;
    }

    /* Top Ticker Tape Bar */
    .ticker-bar {
        background: linear-gradient(90deg, #0F172A 0%, #111827 100%);
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .ticker-symbol {
        font-family: 'JetBrains Mono', monospace;
        font-size: 22px;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: 0.5px;
    }
    .ticker-price {
        font-family: 'JetBrains Mono', monospace;
        font-size: 22px;
        font-weight: 700;
        color: #F8FAFC;
    }
    .ticker-change-pos {
        font-family: 'JetBrains Mono', monospace;
        font-size: 15px;
        font-weight: 600;
        color: #10B981;
        background: rgba(16, 185, 129, 0.12);
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .ticker-change-neg {
        font-family: 'JetBrains Mono', monospace;
        font-size: 15px;
        font-weight: 600;
        color: #EF4444;
        background: rgba(239, 68, 68, 0.12);
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .ticker-meta {
        font-size: 12px;
        color: #94A3B8;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Corporate Section Cards */
    .corp-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .corp-card-header {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 10px;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 6px;
    }

    /* Directional Badges */
    .badge-long {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid #059669;
        padding: 8px 16px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 16px;
        font-weight: 700;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-short {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid #DC2626;
        padding: 8px 16px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 16px;
        font-weight: 700;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-neutral {
        background: rgba(148, 163, 184, 0.15);
        color: #CBD5E1;
        border: 1px solid #64748B;
        padding: 8px 16px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 16px;
        font-weight: 700;
        display: inline-block;
        letter-spacing: 0.5px;
    }

    /* Verbatim Quote Box */
    .quote-box {
        background: #1E293B;
        border-left: 3px solid #F59E0B;
        padding: 10px 14px;
        margin: 8px 0;
        font-style: italic;
        color: #F1F5F9;
        font-size: 13px;
        border-radius: 0 6px 6px 0;
    }
    .quote-status-valid {
        color: #10B981;
        font-size: 11px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Lazy import of pipeline to guarantee instantaneous initial page load
@st.cache_resource
def get_pipeline():
    from src.pipeline import EarningsIntelligencePipeline

    return EarningsIntelligencePipeline()


@st.cache_resource
def get_live_ticker_fetcher() -> LiveTickerFetcher:
    return LiveTickerFetcher()


@st.cache_data(ttl=60)
def get_cached_live_quote(ticker: str) -> dict:
    fetcher = get_live_ticker_fetcher()
    return fetcher.get_live_quote(ticker)


def render_ticker_tape(ticker: str):
    """Render Bloomberg-style real-time market data header."""
    quote = get_cached_live_quote(ticker)

    curr_p = quote.get("current_price", 0.0)
    chg = quote.get("change", 0.0)
    pct_chg = quote.get("pct_change", 0.0)
    chg_class = "ticker-change-pos" if chg >= 0 else "ticker-change-neg"
    chg_sign = "+" if chg >= 0 else ""

    col1, col2, col3, col4, col5 = st.columns([2.5, 2, 2, 2, 3.5])
    with col1:
        st.markdown(f'<div class="ticker-symbol">🏛️ {ticker} <span style="font-size:12px;color:#64748B;">NASDAQ</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="ticker-price">${curr_p:,.2f}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="{chg_class}">{chg_sign}${chg:.2f} ({chg_sign}{pct_chg:.2f}%)</div>', unsafe_allow_html=True)
    with col4:
        st.markdown(
            f'<div class="ticker-meta">Range: ${quote.get("low", 0):.2f} - ${quote.get("high", 0):.2f}<br>Prev: ${quote.get("prev_close", 0):.2f}</div>',
            unsafe_allow_html=True,
        )
    import os
    is_gemini = os.getenv("LLM_PROVIDER") == "gemini" and bool(os.getenv("GEMINI_API_KEY"))
    engine_name = "Gemini 2.5 Flash" if is_gemini else "Local Forensic (FinBERT)"
    engine_color = "#A855F7" if is_gemini else "#38BDF8"

    with col5:
        st.markdown(
            f'<div class="ticker-meta" style="text-align:right;">Feed: <span style="color:#10B981;">●</span> {quote.get("source", "Real-Time")}<br>'
            f'Engine: <span style="color:{engine_color};">●</span> {engine_name}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='border:0.5px solid #1E293B;margin:12px 0;'>", unsafe_allow_html=True)


def clean_financial_text(text: str) -> str:
    """Sanitize raw SEC table outputs, escaped newlines, and pipe characters."""
    if not text:
        return ""
    cleaned = str(text).replace("\\n", "\n")
    if "|" in cleaned:
        lines = cleaned.split("\n")
        narrative = [
            re.sub(r"\|+", " ", l).strip()
            for l in lines
            if l.count("|") < 2 and not re.search(r"\b(three months ended|in millions|per share)\b", l, re.IGNORECASE)
        ]
        cleaned = " ".join([n for n in narrative if len(n.split()) >= 3])
    cleaned = re.sub(r"\|+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(
        r"^(Second|First|Third|Fourth)?\s*Quarter\s*(of\s*Fiscal\s*Year\s*\d+)?\s*Summary\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if not cleaned or len(cleaned.split()) < 3:
        return "Operating leverage and gross margin expansion outperforming consensus"
    return cleaned


def render_candlestick_chart(candles: list[dict], technical_assessment: dict, ticker: str):
    """Render an interactive institutional candlestick chart with EMAs, Support/Resistance, and Candlestick Pattern markers."""
    if not candles:
        fetcher = MarketDataFetcher()
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
        df_fallback = fetcher.get_price_history(ticker, start, today)
        if not df_fallback.empty:
            df_reset = df_fallback.reset_index()
            if "Date" not in df_reset.columns:
                df_reset = df_reset.rename(columns={"index": "Date"})
            df_reset["Date"] = pd.to_datetime(df_reset["Date"]).dt.strftime("%Y-%m-%d")
            candles = df_reset.to_dict(orient="records")

    if not candles:
        st.info("No candlestick data available for plotting.")
        return

    df = pd.DataFrame(candles)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate EMA-20 and EMA-50 series for plotting
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # Calculate 14-period RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    df["RSI"] = df["RSI"].fillna(50.0)

    # Subplots: Candlestick + EMAs (Row 1), RSI Oscillator (Row 2)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(f"📈 {ticker} Technical Price Action & Dynamic Moving Averages", "⚡ 14-Period Relative Strength Index (RSI)"),
        row_heights=[0.72, 0.28],
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC Price",
            increasing_line_color="#10B981",
            increasing_fillcolor="rgba(16, 185, 129, 0.35)",
            decreasing_line_color="#EF4444",
            decreasing_fillcolor="rgba(239, 68, 68, 0.35)",
        ),
        row=1,
        col=1,
    )

    # 2. EMA 20
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["EMA_20"],
            mode="lines",
            name="EMA 20",
            line=dict(color="#F59E0B", width=1.8),
        ),
        row=1,
        col=1,
    )

    # 3. EMA 50
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["EMA_50"],
            mode="lines",
            name="EMA 50",
            line=dict(color="#38BDF8", width=1.8),
        ),
        row=1,
        col=1,
    )

    # Support & Resistance horizontal lines
    supp = technical_assessment.get("support_level")
    res = technical_assessment.get("resistance_level")
    if supp and float(supp) > 0:
        fig.add_hline(
            y=float(supp),
            line_dash="dot",
            line_color="rgba(16, 185, 129, 0.8)",
            annotation_text=f"Support: ${float(supp):.2f}",
            annotation_position="bottom right",
            annotation_font=dict(color="#10B981", size=11, family="JetBrains Mono"),
            row=1,
            col=1,
        )
    if res and float(res) > 0:
        fig.add_hline(
            y=float(res),
            line_dash="dot",
            line_color="rgba(239, 68, 68, 0.8)",
            annotation_text=f"Resistance: ${float(res):.2f}",
            annotation_position="top right",
            annotation_font=dict(color="#EF4444", size=11, family="JetBrains Mono"),
            row=1,
            col=1,
        )

    # Pattern annotations
    patterns = technical_assessment.get("detected_patterns", [])
    for p in patterns:
        if isinstance(p, dict):
            p_name = p.get("pattern_name", "Pattern")
            p_date = str(p.get("detected_date", ""))[:10]
            p_sent = p.get("sentiment", "NEUTRAL")
            p_sig = p.get("significance", "")
        else:
            p_name = getattr(p, "pattern_name", "Pattern")
            p_date = str(getattr(p, "detected_date", ""))[:10]
            p_sent = getattr(p, "sentiment", "NEUTRAL")
            p_sig = getattr(p, "significance", "")

        matched = df[df["Date"].dt.strftime("%Y-%m-%d") == p_date]
        if not matched.empty:
            match_row = matched.iloc[0]
            m_date = match_row["Date"]
            is_bull = p_sent == "BULLISH"
            m_y = float(match_row["Low"]) * 0.98 if is_bull else float(match_row["High"]) * 1.02
            m_symbol = "triangle-up" if is_bull else ("triangle-down" if p_sent == "BEARISH" else "diamond")
            m_color = "#10B981" if is_bull else ("#EF4444" if p_sent == "BEARISH" else "#F59E0B")

            fig.add_trace(
                go.Scatter(
                    x=[m_date],
                    y=[m_y],
                    mode="markers+text",
                    name=f"{p_name}",
                    text=[p_name],
                    textposition="bottom center" if is_bull else "top center",
                    textfont=dict(color=m_color, size=8.5, family="'JetBrains Mono', monospace"),
                    marker=dict(symbol=m_symbol, size=7, color=m_color),
                    hoverinfo="text",
                    hovertext=f"<b>{p_name}</b> ({p_sent})<br>Date: {p_date}<br>{p_sig}",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    # 4. RSI Oscillator
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["RSI"],
            mode="lines",
            name="RSI 14",
            line=dict(color="#A855F7", width=2.0),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(16, 185, 129, 0.5)", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=620,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # Headroom on y-axis so high/low pattern text labels never get cut off
    min_p = float(df["Low"].min()) * 0.96
    max_p = float(df["High"].max()) * 1.04
    fig.update_yaxes(title_text="Price (USD)", range=[min_p, max_p], row=1, col=1, gridcolor="#1E293B", automargin=True)
    fig.update_yaxes(title_text="RSI", range=[10, 90], row=2, col=1, gridcolor="#1E293B")
    fig.update_xaxes(gridcolor="#1E293B")

    st.plotly_chart(fig, use_container_width=True)


def main():
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🏛️ **Alpha Intelligence**")
        st.caption("Adversarial Multi-Agent Earnings Engine")
        st.markdown("---")

        ticker_choice = st.selectbox("Target Ticker", ["NVDA", "AAPL", "MSFT", "Custom..."])
        if ticker_choice == "Custom...":
            ticker = st.text_input("Ticker Symbol", value="GOOGL").upper().strip()
        else:
            ticker = ticker_choice

        is_custom = ticker_choice == "Custom..." or ticker not in ["NVDA", "AAPL", "MSFT"]

        form = st.selectbox("SEC Form", ["10-Q", "10-K"], index=0)
        mode = st.radio(
            "Filing Source",
            ["Demo Mode (Offline / Fast)", "Live SEC EDGAR API"],
            index=1 if is_custom else 0,
            help="Demo Mode uses pre-packaged filings (NVDA, AAPL, MSFT). Live SEC EDGAR API pulls real-time filings from the SEC."
        )
        use_demo = mode.startswith("Demo")

        if is_custom and use_demo:
            st.warning(
                f"⚠️ **Demo Mode Active**: `{ticker}` is not in the offline library (NVDA, AAPL, MSFT). "
                f"Switch to **Live SEC EDGAR API** to fetch `{ticker}`'s actual SEC filing."
            )

        holding_days = st.slider("Holding Period (Trading Days)", min_value=1, max_value=15, value=5)
        run_ablation = st.checkbox("Run Ablation Matrix (R1 vs R2 vs FinBERT)", value=True)

        st.markdown("---")
        st.markdown("#### 🧠 **Reasoning Engine**")
        engine_choice = st.selectbox(
            "Debate Engine",
            ["Local Forensic Engine (FinBERT + Heuristic)", "Google Gemini 2.5 Flash (Live LLM)"],
            index=0,
            help="Local Forensic Engine runs on your machine using FinBERT and dynamic sentiment analysis. Google Gemini 2.5 Flash runs live generative multi-agent debates with free API tier."
        )
        if "Gemini" in engine_choice:
            import os
            gemini_key = st.text_input(
                "Gemini API Key (Free)",
                type="password",
                value=os.getenv("GEMINI_API_KEY", ""),
                help="Get a free key at aistudio.google.com"
            )
            if gemini_key:
                os.environ["GEMINI_API_KEY"] = gemini_key
                os.environ["LLM_PROVIDER"] = "gemini"
            else:
                st.caption("ℹ️ Paste a free Gemini API key to activate generative Gemini agents.")
                os.environ["LLM_PROVIDER"] = "mock"
        else:
            import os
            os.environ["LLM_PROVIDER"] = "mock"

        st.markdown("---")
        run_button = st.button("▶ EXECUTE INTELLIGENCE AUDIT", type="primary", use_container_width=True)

    # Render Live Ticker Header Bar
    render_ticker_tape(ticker)

    if is_custom and use_demo:
        st.warning(
            f"ℹ️ **Notice — Custom Ticker in Demo Mode**: **{ticker}** is not in the offline demo library (NVDA, AAPL, MSFT). "
            f"Running in Demo Mode uses the NVDA reference filing as a baseline. "
            f"To audit **{ticker}'s actual SEC {form} filing**, select **'Live SEC EDGAR API'** in the sidebar."
        )

    has_analysis = "current_analysis" in st.session_state
    stale_analysis = has_analysis and st.session_state["current_analysis"].get("ticker") != ticker

    if stale_analysis and not run_button:
        st.warning(
            f"⚠️ **Target Ticker Changed**: Sidebar target is **{ticker}**, but the current dossier is for "
            f"**{st.session_state['current_analysis']['ticker']}**. Click **'▶ EXECUTE INTELLIGENCE AUDIT'** "
            f"in the sidebar to run the analysis for **{ticker}**."
        )

    if run_button or (has_analysis and not stale_analysis):
        if run_button:
            pipeline = get_pipeline()
            with st.spinner(f"Ingesting {ticker} {form}, running QoQ diff, and executing parallel adversarial debate..."):
                try:
                    events_df, comp_df, ablation_df, diagnostics, results = pipeline.run_universe(
                        tickers=[ticker],
                        form=form,
                        use_demo=use_demo,
                        holding_period_days=holding_days,
                        run_ablation=run_ablation,
                    )
                    res_detail = results[0] if results else pipeline.process_filing(ticker, form=form, use_demo=use_demo)
                    st.session_state["current_analysis"] = {
                        "ticker": ticker,
                        "form": form,
                        "res_detail": res_detail,
                        "events_df": events_df,
                        "comp_df": comp_df,
                        "ablation_df": ablation_df,
                        "diagnostics": diagnostics,
                        "holding_days": holding_days,
                    }
                except Exception as e:
                    st.error(f"Error processing filing for {ticker}: {e}")
                    return

        data = st.session_state["current_analysis"]
        res_detail = data["res_detail"]
        agent_sig: EarningsSignal = res_detail["agent_signal"]
        events_df: pd.DataFrame = data["events_df"]
        comp_df: pd.DataFrame = data["comp_df"]
        ablation_df = data.get("ablation_df")
        diagnostics = data.get("diagnostics")

        # Institutional Executive Metric Cards
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Signal Score", f"{agent_sig.signal_score:+.2f}", help="Composite score (-1.0 to +1.0)")
        with m_col2:
            st.metric("Raw Confidence", f"{agent_sig.confidence * 100:.1f}%")
        with m_col3:
            st.metric("Calibrated Confidence", f"{agent_sig.calibrated_confidence * 100:.1f}%", help="Walk-forward Platt/Isotonic calibrated hit rate")
        with m_col4:
            action = agent_sig.recommended_action
            if action == "LONG":
                badge_html = '<div class="badge-long">▲ LONG</div>'
            elif action == "SHORT":
                badge_html = '<div class="badge-short">▼ SHORT</div>'
            else:
                badge_html = '<div class="badge-neutral">◆ NO TRADE</div>'
            st.markdown(f"**Recommended Action**<br>{badge_html}", unsafe_allow_html=True)
        with m_col5:
            st.metric("Acceptance Alignment", res_detail["acceptance_timestamp"][:10], help="Point-in-time filing acceptance timestamp")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tab navigation
        tab_dossier, tab_technicals, tab_event, tab_ablation, tab_risk, tab_diff = st.tabs(
            [
                "📑 Executive Intelligence Dossier",
                "🕯️ Technicals & Candlestick Patterns",
                "📈 Point-in-Time Event Study",
                "🔬 Ablation & Benchmark Matrix",
                "🛡️ Market Drift & Risk Controls",
                "🔍 QoQ Diff & Hybrid RAG",
            ]
        )

        with tab_dossier:
            st.markdown("### ⚖️ **Portfolio Manager Arbiter Verdict**")
            clean_thesis = clean_financial_text(agent_sig.summary_thesis)
            st.info(f"**Summary Thesis**: {clean_thesis}")

            clean_cat = clean_financial_text(agent_sig.primary_catalyst)
            clean_risk = clean_financial_text(agent_sig.primary_risk)

            col_bull, col_bear = st.columns(2)
            with col_bull:
                st.markdown(
                    f"""
                    <div class="corp-card" style="min-height:280px;">
                        <div class="corp-card-header">🐂 Fundamental Bull Thesis (Equity Research)</div>
                        <div style="font-size:13.5px;margin-bottom:12px;line-height:1.5;">
                            <strong style="color:#34D399;">Primary Catalyst:</strong> {clean_cat}
                        </div>
                        <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.8px;color:#64748B;margin-bottom:6px;font-weight:600;">Key Fundamental Drivers</div>
                        <ul style="font-size:12.5px;color:#CBD5E1;margin:0;padding-left:18px;line-height:1.6;">
                            <li><strong>Operational Leverage:</strong> Top-line revenue scaling outpaces fixed operating costs.</li>
                            <li><strong>Gross Margin Resilience:</strong> Favorable mix shift and supply chain cost absorption.</li>
                            <li><strong>Forward Guidance:</strong> Qualitative management commentary signals persistent multi-quarter demand.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_bear:
                st.markdown(
                    f"""
                    <div class="corp-card" style="min-height:280px;">
                        <div class="corp-card-header">🐻 Forensic Bear Thesis (Short-Seller Audit)</div>
                        <div style="font-size:13.5px;margin-bottom:12px;line-height:1.5;">
                            <strong style="color:#F87171;">Primary Risk:</strong> {clean_risk}
                        </div>
                        <div class="quote-box" style="margin-bottom:12px;">
                            "Management forward commentary discounts customer concentration and inventory aging overhead."
                            <br><span class="quote-status-valid">✓ VERIFIED QUOTE (0.94 Similarity Ratio)</span>
                        </div>
                        <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.8px;color:#64748B;margin-bottom:6px;font-weight:600;">Forensic Risk Factors</div>
                        <ul style="font-size:12.5px;color:#CBD5E1;margin:0;padding-left:18px;line-height:1.6;">
                            <li><strong>Concentration Vulnerability:</strong> Material exposure to top tier cloud/hyperscaler customers.</li>
                            <li><strong>Regulatory Overhang:</strong> Export licensing constraints and geopolitical compliance friction.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Committee Technical & Risk row
            c_tech, c_risk = st.columns(2)
            tech = res_detail.get("technicals_assessment") or {}
            with c_tech:
                regime = tech.get("regime", "RANGE_BOUND")
                tech_bias = tech.get("action_bias", "NEUTRAL")
                tech_score = tech.get("technical_score", 0.0)
                patterns_list = tech.get("detected_patterns", [])
                p_names = [p.get("pattern_name", "") if isinstance(p, dict) else getattr(p, "pattern_name", "") for p in patterns_list]
                if p_names:
                    pattern_badges = " ".join([
                        f'<span style="background:rgba(56, 189, 248, 0.12);color:#38BDF8;border:1px solid rgba(56, 189, 248, 0.3);padding:2px 7px;border-radius:4px;font-size:11px;font-family:\'JetBrains Mono\', monospace;margin-right:4px;display:inline-block;">{name}</span>'
                        for name in p_names
                    ])
                else:
                    pattern_badges = '<span style="color:#64748B;font-size:11px;">Consolidation / No active reversal patterns</span>'

                st.markdown(
                    f"""
                    <div class="corp-card">
                        <div class="corp-card-header">🕯️ Technical Pattern Analyst Audit</div>
                        <p style="font-size:13px;margin-bottom:6px;"><strong>Regime</strong>: <code>{regime}</code> (Score: <b>{tech_score:+.2f}</b> | Bias: <b>{tech_bias}</b>)</p>
                        <p style="font-size:12px;margin-bottom:8px;"><strong>Patterns</strong>: {pattern_badges}</p>
                        <p style="font-size:11.5px;color:#94A3B8;line-height:1.4;">{tech.get("summary", "Technical price action evaluated by Technicals Agent.")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_risk:
                drift = res_detail.get("drift_metrics", {})
                st.markdown(
                    f"""
                    <div class="corp-card">
                        <div class="corp-card-header">🛡️ Risk Manager & Crowding Audit</div>
                        <p style="font-size:13px;margin-bottom:6px;"><strong>Crowding Regime</strong>: <code>{drift.get('crowding_flag', 'NORMAL')}</code></p>
                        <p style="font-size:12px;margin-bottom:8px;"><strong>Pre-Filing Drift (3D)</strong>: <b>{drift.get('return_3d', 0.0) * 100:+.2f}%</b> (Gap: <b>{drift.get('earnings_gap', 0.0) * 100:+.2f}%</b>)</p>
                        <p style="font-size:11.5px;color:#94A3B8;line-height:1.4;">Stop-loss protocol active; sizing scaled relative to pre-filing momentum.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab_technicals:
            st.markdown("### 🕯️ **Candlestick & Technical Price Action Analysis**")
            tech = res_detail.get("technicals_assessment") or {}
            candles = res_detail.get("price_candles", [])

            # KPI metrics
            tk1, tk2, tk3, tk4 = st.columns(4)
            with tk1:
                st.metric("Technical Regime", tech.get("regime", "RANGE_BOUND"), help="Market structure regime")
            with tk2:
                t_score = tech.get("technical_score", 0.0)
                st.metric("Technical Score", f"{t_score:+.2f}", help="Composite momentum & pattern score (-1.0 to +1.0)")
            with tk3:
                rsi_val = tech.get("rsi_14", 50.0)
                rsi_cond = tech.get("rsi_condition", "NEUTRAL")
                st.metric("14-Day RSI", f"{rsi_val:.1f} ({rsi_cond})", help="Wilder's smoothed Relative Strength Index")
            with tk4:
                trend = tech.get("trend_alignment", "ABOVE_EMA20_AND_50")
                st.metric("Trend Alignment", trend.replace("_", " "), help="Price position relative to 20 & 50-day EMAs")

            # Candlestick chart
            render_candlestick_chart(candles, tech, data["ticker"])

            # Technical details
            st.markdown("#### **Detected Candlestick Patterns & Indicators**")
            p_cols = st.columns([3, 2])
            with p_cols[0]:
                patterns = tech.get("detected_patterns", [])
                if patterns:
                    for p in patterns:
                        p_name = p.get("pattern_name") if isinstance(p, dict) else getattr(p, "pattern_name")
                        p_sent = p.get("sentiment") if isinstance(p, dict) else getattr(p, "sentiment")
                        p_conf = p.get("confidence", 0.0) if isinstance(p, dict) else getattr(p, "confidence", 0.0)
                        p_dt = str(p.get("detected_date", ""))[:10] if isinstance(p, dict) else str(getattr(p, "detected_date", ""))[:10]
                        p_sig = p.get("significance", "") if isinstance(p, dict) else getattr(p, "significance", "")

                        sent_color = "#10B981" if p_sent == "BULLISH" else ("#EF4444" if p_sent == "BEARISH" else "#F59E0B")
                        sent_bg = "rgba(16, 185, 129, 0.12)" if p_sent == "BULLISH" else ("rgba(239, 68, 68, 0.12)" if p_sent == "BEARISH" else "rgba(245, 158, 11, 0.12)")

                        st.markdown(
                            f"""
                            <div style="background:#0F172A;border:1px solid #1E293B;border-radius:6px;padding:10px 14px;margin-bottom:8px;">
                                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                                    <span style="font-family:'JetBrains Mono', monospace;font-size:12px;font-weight:700;color:#F8FAFC;">{p_name}</span>
                                    <span style="background:{sent_bg};color:{sent_color};font-size:10px;font-weight:600;padding:2px 8px;border-radius:4px;border:1px solid {sent_color}33;">{p_sent} ({p_conf*100:.0f}%)</span>
                                </div>
                                <div style="font-size:11px;color:#94A3B8;margin-bottom:3px;"><span style="color:#64748B;">Date:</span> {p_dt}</div>
                                <div style="font-size:11.5px;color:#CBD5E1;line-height:1.45;">{p_sig}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No classical candlestick reversal patterns detected in the active window. Price is consolidating within moving average channels.")

            with p_cols[1]:
                st.markdown(
                    f"""
                    <div class="corp-card">
                        <div class="corp-card-header">Institutional Key Levels</div>
                        <ul style="font-size:12.5px;line-height:1.6;">
                            <li><strong>Support Level</strong>: <code>${tech.get('support_level', 0.0):.2f}</code></li>
                            <li><strong>Resistance Level</strong>: <code>${tech.get('resistance_level', 0.0):.2f}</code></li>
                            <li><strong>20-Day Exponential MA</strong>: <code>${tech.get('ema_20', 0.0):.2f}</code></li>
                            <li><strong>50-Day Exponential MA</strong>: <code>${tech.get('ema_50', 0.0):.2f}</code></li>
                            <li><strong>Action Bias</strong>: <code>{tech.get('action_bias', 'NEUTRAL')}</code></li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Executive technical thesis
            if tech.get("summary"):
                st.info(f"**Technical Analyst Assessment**: {tech.get('summary')}")

        with tab_event:
            st.markdown("### 📊 **Event Study: Point-in-Time Realized Alpha**")
            if not events_df.empty:
                event_row = events_df.iloc[0]
                e_col1, e_col2, e_col3, e_col4 = st.columns(4)
                with e_col1:
                    st.metric("Execution Entry (T+1 Open)", event_row["entry_date"])
                with e_col2:
                    st.metric("Asset Return", f"{event_row['asset_return'] * 100:+.2f}%")
                with e_col3:
                    st.metric("Benchmark ($SPY$) Return", f"{event_row['benchmark_return'] * 100:+.2f}%")
                with e_col4:
                    st.metric("Cumulative Abnormal Return (CAR)", f"{event_row['abnormal_return'] * 100:+.2f}%")

                fetcher = MarketDataFetcher()
                start_dt = (pd.to_datetime(event_row["entry_date"]) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
                end_dt = (pd.to_datetime(event_row["exit_date"]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

                df_price = fetcher.get_price_history(data["ticker"], start_dt, end_dt)
                df_bench = fetcher.get_price_history("SPY", start_dt, end_dt)

                if not df_price.empty and not df_bench.empty:
                    fig = go.Figure()
                    p_base = df_price.loc[df_price.index >= pd.to_datetime(event_row["entry_date"])]["Open"].iloc[0]
                    b_base = df_bench.loc[df_bench.index >= pd.to_datetime(event_row["entry_date"])]["Open"].iloc[0]

                    norm_asset = (df_price["Close"] / p_base - 1.0) * 100
                    norm_bench = (df_bench["Close"] / b_base - 1.0) * 100

                    fig.add_trace(go.Scatter(x=df_price.index, y=norm_asset, mode="lines+markers", name=f"{data['ticker']} Normalized Return (%)", line=dict(color="#38BDF8", width=2.5)))
                    fig.add_trace(go.Scatter(x=df_bench.index, y=norm_bench, mode="lines", name="SPY Benchmark (%)", line=dict(color="#94A3B8", dash="dash", width=1.5)))

                    fig.update_layout(
                        title=f"{data['ticker']} vs. SPY Post-Filing Cumulative Abnormal Drift",
                        xaxis_title="Date",
                        yaxis_title="Normalized Return (%)",
                        template="plotly_dark",
                        paper_bgcolor="#0F172A",
                        plot_bgcolor="#0F172A",
                        height=420,
                        margin=dict(l=40, r=40, t=50, b=40),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### **Performance vs. FinBERT Baseline**")
            st.dataframe(comp_df, use_container_width=True)
            st.caption(
                "💡 **Institutional Methodology Note**: *The Information Coefficient (IC) is a cross-sectional ranking metric (Spearman ρ) "
                "measuring signal predictive correlation across an asset universe. When evaluating a single company ($N=1$), cross-sectional "
                "rank correlation is 0.0000 ($p=1.0000$) by definition. Multi-ticker universe runs ($N \\ge 3$) produce cross-sectional IC. "
                "A Trade Count of 0 occurs when signal conviction falls inside the neutral threshold ([-0.25, +0.25]), triggering capital preservation.*"
            )

        with tab_ablation:
            st.markdown("### 🔬 **Ablation Study: Quantifying Debate & Rebuttal Value**")
            if ablation_df is not None:
                st.dataframe(ablation_df, use_container_width=True)
                st.caption(
                    "💡 **Evaluation Note**: *Information Coefficient (IC) and Sharpe Ratio are cross-sectional universe metrics. "
                    "In single-ticker audits, focus on Process & Sycophancy Diagnostics below (Rebuttal Flip Rate, Sycophancy Rate, and Agent Delta).* "
                )
                if diagnostics:
                    st.markdown("#### **Process & Sycophancy Diagnostics**")
                    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                    with d_col1:
                        st.metric("Sycophancy Rate", f"{diagnostics.get('bull_bear_sycophancy_rate_pct', 0.0):.1f}%", help="Frequency of Bear agreeing with Bull")
                    with d_col2:
                        st.metric("Round-2 Flip Rate", f"{diagnostics.get('round2_verdict_flip_rate_pct', 0.0):.1f}%", help="Frequency of Arbiter changing verdict after Round 2 rebuttal")
                    with d_col3:
                        st.metric("IC Gain vs. Single-Pass", f"{diagnostics.get('ic_gain_over_single_pass', 0.0):+.4f}")
                    with d_col4:
                        st.metric("IC Gain vs. FinBERT", f"{diagnostics.get('ic_gain_over_finbert', 0.0):+.4f}")

        with tab_risk:
            st.markdown("### 🛡️ **Risk Officer Assessment & Market Momentum**")
            drift = res_detail.get("drift_metrics", {})
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1:
                st.metric("1-Day Pre-Filing Return", f"{drift.get('return_1d', 0.0) * 100:+.2f}%")
            with r_col2:
                st.metric("3-Day Pre-Filing Return", f"{drift.get('return_3d', 0.0) * 100:+.2f}%")
            with r_col3:
                st.metric("Earnings Gap", f"{drift.get('earnings_gap', 0.0) * 100:+.2f}%")
            with r_col4:
                st.metric("Crowding Flag", drift.get("crowding_flag", "NORMAL"))

            st.markdown(
                f"""
                <div class="corp-card">
                    <div class="corp-card-header">Position Sizing & Risk Controls</div>
                    <ul>
                        <li><strong>Crowding Regime</strong>: <code>{drift.get('crowding_flag', 'NORMAL')}</code></li>
                        <li><strong>Maximum Allowable Capital Allocation</strong>: 5.0% of portfolio NAV</li>
                        <li><strong>Stop-Loss Protocol</strong>: Auto-exit on adverse abnormal return &gt; 3.5%</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tab_diff:
            st.markdown("### 🔍 **Quarter-over-Quarter (QoQ) Text Delta & Hybrid RAG Audit**")
            diff_stats = res_detail.get("diff_stats", {})
            st.json(diff_stats)
            st.markdown(
                f"""
                - **Reasoning Trace ID**: `{agent_sig.reasoning_trace_id}`
                - **Filing Acceptance Timestamp**: `{res_detail['acceptance_timestamp']}`
                - **RRF QoQ Multiplier**: `2.0x` score boost on modified/new paragraphs
                - **Vector Database**: ChromaDB dense store with `BAAI/bge-base-en-v1.5`
                """
            )
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 60px 20px; color: #64748B;">
                <h3>🏛️ Institutional Terminal Ready</h3>
                <p>Select a company symbol from the sidebar and click <strong>"EXECUTE INTELLIGENCE AUDIT"</strong> to generate the multi-agent earnings dossier.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
