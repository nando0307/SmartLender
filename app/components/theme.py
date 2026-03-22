"""SmartLend Design System — Shared CSS and theme utilities."""

# Fintech-inspired design system
# Colors: Deep navy + Emerald accents + Warm gold highlights
# Typography: Inter (headings) + JetBrains Mono (data)
# Style: Dark glassmorphism with subtle gradients

CUSTOM_CSS = """
<style>
/* ===== Google Fonts ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ===== Root Variables ===== */
:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.8);
    --bg-card-hover: rgba(30, 41, 59, 0.9);
    --border-subtle: rgba(99, 102, 241, 0.15);
    --border-glow: rgba(99, 102, 241, 0.4);

    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;

    --accent-emerald: #10b981;
    --accent-emerald-dim: rgba(16, 185, 129, 0.15);
    --accent-blue: #6366f1;
    --accent-blue-dim: rgba(99, 102, 241, 0.15);
    --accent-red: #ef4444;
    --accent-red-dim: rgba(239, 68, 68, 0.15);
    --accent-gold: #f59e0b;
    --accent-gold-dim: rgba(245, 158, 11, 0.15);
    --accent-cyan: #06b6d4;

    --radius: 12px;
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    --transition: all 0.2s ease;
}

/* ===== Global Overrides ===== */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ===== Sidebar Styling ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ===== Custom Card Component ===== */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 16px;
    transition: var(--transition);
}

.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 0 30px rgba(99, 102, 241, 0.08);
}

/* ===== Hero Section ===== */
.hero-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border-radius: 20px;
    padding: 48px 40px;
    margin-bottom: 32px;
    border: 1px solid var(--border-subtle);
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f1f5f9 0%, #6366f1 50%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 12px;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    color: var(--text-secondary);
    font-weight: 400;
    line-height: 1.6;
    max-width: 600px;
}

/* ===== Metric Cards ===== */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 20px 24px;
    text-align: center;
    transition: var(--transition);
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-glow);
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
}

.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

.metric-emerald .metric-value { color: var(--accent-emerald); }
.metric-red .metric-value { color: var(--accent-red); }
.metric-gold .metric-value { color: var(--accent-gold); }
.metric-blue .metric-value { color: var(--accent-blue); }

/* ===== Status Badges ===== */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.03em;
}

.badge-approve {
    background: var(--accent-emerald-dim);
    color: var(--accent-emerald);
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-deny {
    background: var(--accent-red-dim);
    color: var(--accent-red);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.badge-winner {
    background: var(--accent-gold-dim);
    color: var(--accent-gold);
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* ===== Navigation Cards ===== */
.nav-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 28px;
    transition: var(--transition);
    cursor: pointer;
    height: 100%;
}

.nav-card:hover {
    border-color: var(--accent-blue);
    box-shadow: 0 0 30px rgba(99, 102, 241, 0.1);
    transform: translateY(-3px);
}

.nav-icon {
    font-size: 2rem;
    margin-bottom: 12px;
}

.nav-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.nav-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.5;
}

/* ===== Section Headers ===== */
.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 32px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border-subtle);
}

/* ===== Winner Banner ===== */
.winner-banner {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: var(--radius);
    padding: 20px 28px;
    margin-bottom: 24px;
}

.winner-text {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
    font-size: 1.05rem;
}

.winner-name {
    color: var(--accent-gold);
    font-weight: 700;
}

/* ===== Dataframe Styling ===== */
.stDataFrame {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ===== Plotly Chart Container ===== */
.stPlotlyChart {
    border-radius: var(--radius);
    overflow: hidden;
}

/* ===== Streamlit Metric Override ===== */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    padding: 16px 20px !important;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ===== Divider ===== */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
    margin: 32px 0;
}

/* ===== Tag Row ===== */
.tech-tag {
    display: inline-block;
    background: var(--accent-blue-dim);
    color: var(--accent-blue);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 6px;
    border: 1px solid rgba(99, 102, 241, 0.2);
}
</style>
"""

# Plotly dark theme template
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,24,39,0.6)',
        font=dict(family='Inter, sans-serif', color='#94a3b8', size=12),
        title=dict(font=dict(size=16, color='#f1f5f9', family='Inter, sans-serif')),
        xaxis=dict(
            gridcolor='rgba(99,102,241,0.08)',
            zerolinecolor='rgba(99,102,241,0.15)',
        ),
        yaxis=dict(
            gridcolor='rgba(99,102,241,0.08)',
            zerolinecolor='rgba(99,102,241,0.15)',
        ),
        colorway=[
            '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#06b6d4',
            '#8b5cf6', '#ec4899', '#14b8a6',
        ],
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
        ),
        margin=dict(l=40, r=20, t=50, b=40),
    )
)


def inject_css():
    """Inject custom CSS into Streamlit page."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str):
    """Render a gradient hero section."""
    import streamlit as st
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, variant: str = ""):
    """Render a styled metric card. variant: emerald, red, gold, blue"""
    cls = f"metric-{variant}" if variant else ""
    return f"""
    <div class="metric-card {cls}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def section_header(text: str):
    """Render a styled section header."""
    import streamlit as st
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def winner_banner(name: str, metric: str, value: str):
    """Render a winner announcement banner."""
    import streamlit as st
    st.markdown(f"""
    <div class="winner-banner">
        <div class="winner-text">
            🏆 <span class="winner-name">{name}</span> — {metric}: <span style="font-family: 'JetBrains Mono', monospace; color: #f59e0b;">{value}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def badge(text: str, variant: str = "approve"):
    """Return HTML for a status badge."""
    return f'<span class="badge badge-{variant}">{text}</span>'


def divider():
    """Render a subtle gradient divider."""
    import streamlit as st
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
