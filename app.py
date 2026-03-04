import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
# 页面基础配置
st.set_page_config(page_title="Cui Qian 投资决策终端", layout="wide")
st.title("🛡️ 全资产监控 & BTC 底部六因子计分")
# --- 1. 侧边栏：核心资产与手动指标 ---
st.sidebar.header("💰 账户余额")
mma_cash = st.sidebar.number_input("MMA 现金 (USD)", value=500000)
st.sidebar.divider()
st.sidebar.header("📊 BTC 底部因子 (手动观察项)")
mvrv_low = st.sidebar.checkbox("1. MVRV 比率 < 1.0", help="代表持有者整体处于亏损状态，历史底部区间")
social_panic = st.sidebar.checkbox("2. 社交媒体恐慌指数 > 75", help="Twitter/Reddit 极度恐慌")
miner_price = st.sidebar.checkbox("3. 现价接近矿机关机价", help="主流矿机开始亏损，通常是绝对底部")
lth_rising = st.sidebar.checkbox("4. LTH (长期持有者) 占比上升", help="筹码从散户向坚定持有者转移")
# --- 2. 数据获取与处理（分标的下载，失败时用默认值避免整页报错）---
PERIOD = "550d"

def _default_series():
    """下载失败时返回单行序列，避免 iloc[-1] 越界，页面照常展示."""
    return pd.Series([0.0], index=[pd.Timestamp.today().normalize()])


@st.cache_data(ttl=3600)
def get_btc_data():
    """单独下载 BTC：返回 (收盘价序列, 成交量序列)。使用 Close/Volume 列名，auto_adjust=False 保证列名一致."""
    try:
        df = yf.download("BTC-USD", period=PERIOD, auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty or "Close" not in df.columns:
            return _default_series(), _default_series()
        close = df["Close"].dropna()
        vol = df["Volume"].dropna()
        if len(close) == 0:
            close = _default_series()
        if len(vol) == 0:
            vol = _default_series()
        return close, vol
    except Exception:
        return _default_series(), _default_series()


def _download_ticker_close(ticker: str) -> pd.Series:
    """下载单只股票的收盘价。使用 Close 列，auto_adjust=False 确保列名不变."""
    try:
        df = yf.download(ticker, period=PERIOD, auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty or "Close" not in df.columns:
            return _default_series()
        close = df["Close"].dropna()
        return close if len(close) > 0 else _default_series()
    except Exception:
        return _default_series()


@st.cache_data(ttl=3600)
def get_tsla():
    return _download_ticker_close("TSLA")


@st.cache_data(ttl=3600)
def get_aapl():
    return _download_ticker_close("AAPL")


@st.cache_data(ttl=3600)
def get_amzn():
    return _download_ticker_close("AMZN")
def pct_change(series: pd.Series) -> float | None:
    if series is None or len(series) < 2:
        return None
    prev = series.iloc[-2]
    cur = series.iloc[-1]
    if pd.isna(prev) or prev == 0:
        return None
    return (cur / prev - 1) * 100.0
def format_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:+.2f}%"
try:
    # 分别下载：BTC 单独拿收盘价+成交量，TSLA/AAPL/AMZN 各下一份
    btc_p, btc_volume = get_btc_data()
    tsla_p = get_tsla()
    aapl_p = get_aapl()
    amzn_p = get_amzn()

    # 任一标的失败时已用默认单行序列，不再整页 st.stop()；仅保证有数据可做 iloc[-1]
    if len(btc_p) == 0:
        btc_p = _default_series()
    if len(tsla_p) == 0:
        tsla_p = _default_series()
    if len(aapl_p) == 0:
        aapl_p = _default_series()
    if len(amzn_p) == 0:
        amzn_p = _default_series()

    # 对齐成交量到 BTC 收盘价索引（同一次 BTC 下载，索引一致）
    btc_volume = btc_volume.reindex(btc_p.index).dropna()
    if len(btc_volume) == 0:
        btc_volume = pd.Series(dtype=float)
    # --- 3. 自动指标计算 (BTC) ---
    # 因子 5: RSI (14天)
    delta = btc_p.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    rsi_valid = rsi_series.dropna()
    rsi = float(rsi_valid.iloc[-1]) if not rsi_valid.empty else None
    # 因子 6: 缩量判断 (当日成交量 < 30日均量)
    vol_signal = False
    current_vol = None
    avg_vol_30 = None
    if len(btc_volume) > 0 and len(btc_volume) >= 30:
        current_vol = float(btc_volume.iloc[-1])
        avg_vol_30 = float(btc_volume.rolling(window=30).mean().iloc[-1])
        vol_signal = current_vol < avg_vol_30
    # --- 4. 计分系统 ---
    score = 0
    active_factors: list[str] = []
    if rsi is not None and rsi < 30:
        score += 1
        active_factors.append("K线 RSI < 30 (日线超跌)")
    if vol_signal:
        score += 1
        active_factors.append("成交量萎缩 (低于30日均量)")
    if mvrv_low:
        score += 1
        active_factors.append("MVRV < 1.0 (市值低于实现价值)")
    if social_panic:
        score += 1
        active_factors.append("社交媒体极度恐慌 (逆向指标)")
    if miner_price:
        score += 1
        active_factors.append("接近矿机成本价 (支撑线)")
    if lth_rising:
        score += 1
        active_factors.append("LTH 持有占比上升 (筹码整合)")
    # --- 5. 第一行：多资产实时价格看板 ---
    btc_last = float(btc_p.iloc[-1])
    tsla_last = float(tsla_p.iloc[-1])
    aapl_last = float(aapl_p.iloc[-1])
    amzn_last = float(amzn_p.iloc[-1])
    aapl_change = pct_change(aapl_p)
    amzn_change = pct_change(amzn_p)
    tsla_change = pct_change(tsla_p)
    btc_change = pct_change(btc_p)

    def _delta_label(series: pd.Series, pct: float | None, fmt: str) -> str:
        """下载失败时（默认单行 0）显示「数据暂缺」."""
        if len(series) == 1 and (series.iloc[-1] == 0 or pd.isna(series.iloc[-1])):
            return "数据暂缺"
        return fmt(pct) if pct is not None else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AAPL 实时价格", f"${aapl_last:.2f}", _delta_label(aapl_p, aapl_change, format_pct))
    c2.metric("AMZN 实时价格", f"${amzn_last:.2f}", _delta_label(amzn_p, amzn_change, format_pct))
    c3.metric("TSLA 实时价格", f"${tsla_last:.2f}", _delta_label(tsla_p, tsla_change, format_pct))
    c4.metric("BTC 实时价格", f"${btc_last:,.0f}", _delta_label(btc_p, btc_change, format_pct))
    st.divider()
    # --- 6. 第二行：BTC 计分与策略 + TSLA 图表 ---
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("🎯 BTC 底部信号计分")
        st.header(f"{score} / 6")
        st.progress(score / 6)
        # 2% MMA 自动加仓建议逻辑
        if score >= 5:
            budget = mma_cash * 0.02
            st.error(
                f"🚀 信号共振：触发 2% MMA 资金补仓！\n\n"
                f"- 建议动用约 **${budget:,.0f}** 等值 USD 加仓 BTC。"
            )
        elif score >= 3:
            st.warning("⚠️ 筑底迹象：建议开始轻仓定投。")
        else:
            st.success("💤 趋势未现：持有 MMA 现金获取利息。")
        st.markdown("**已触发因子：**")
        if active_factors:
            for f in active_factors:
                st.write(f"- {f}")
        else:
            st.write("暂无触发的底部因子。")
        if rsi is not None:
            st.caption(f"当前 BTC RSI(14): {rsi:.1f}")
        if current_vol is not None and avg_vol_30 is not None:
            st.caption(
                f"当前成交量: {current_vol:,.0f} | 30日均量: {avg_vol_30:,.0f}"
            )
    with col_right:
        st.subheader("📊 TSLA 价格 / MA200 / 持仓成本线")
        if len(tsla_p) == 1 and (tsla_p.iloc[-1] == 0 or pd.isna(tsla_p.iloc[-1])):
            st.caption("TSLA 数据暂缺，请检查网络后刷新。")
        # 计算 TSLA MA200（550 天数据足够，保证不为 NaN）
        tsla_ma200 = tsla_p.rolling(window=200).mean()
        fig = go.Figure()
        # TSLA 价格
        fig.add_trace(
            go.Scatter(
                x=tsla_p.index,
                y=tsla_p,
                name="TSLA 价格",
                line=dict(color="steelblue"),
            )
        )
        # 200 日均线
        fig.add_trace(
            go.Scatter(
                x=tsla_p.index,
                y=tsla_ma200,
                name="200日均线 (MA200)",
                line=dict(color="orange", dash="dash"),
            )
        )
        # 成本线 91.20（红色虚线）
        fig.add_trace(
            go.Scatter(
                x=tsla_p.index,
                y=[91.20] * len(tsla_p),
                name="我的持仓成本 ($91.20)",
                line=dict(color="red", dash="dot"),
            )
        )
        fig.update_layout(
            legend=dict(
                title="图例",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis_title="价格 (USD)",
        )
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"运行出错: {e}")
