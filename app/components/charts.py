"""Plotly chart helper functions with fintech dark theme."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from app.components.theme import PLOTLY_TEMPLATE

# Color palette
COLORS = {
    'emerald': '#10b981',
    'red': '#ef4444',
    'blue': '#6366f1',
    'gold': '#f59e0b',
    'cyan': '#06b6d4',
    'purple': '#8b5cf6',
    'pink': '#ec4899',
    'teal': '#14b8a6',
}

ALGO_COLORS = {
    'Logistic Regression': '#6366f1',
    'Decision Tree': '#ef4444',
    'Random Forest': '#10b981',
    'Gradient Boosting': '#f59e0b',
    'AdaBoost': '#06b6d4',
    'XGBoost': '#8b5cf6',
    'LightGBM': '#ec4899',
    'CatBoost': '#14b8a6',
    'Linear Regression': '#6366f1',
}


def _apply_theme(fig):
    """Apply dark fintech theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_TEMPLATE['layout'])
    return fig


def risk_bar_chart(predictions: pd.DataFrame) -> go.Figure:
    colors = [COLORS['red'] if d == 'Deny' else COLORS['emerald'] for d in predictions['decision']]

    fig = go.Figure(go.Bar(
        x=predictions['risk_pct'],
        y=predictions['algorithm'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0),
            opacity=0.85,
        ),
        text=[f"{v:.1f}%" for v in predictions['risk_pct']],
        textposition='auto',
        textfont=dict(family='JetBrains Mono, monospace', size=12, color='#f1f5f9'),
    ))

    fig.add_vline(x=50, line_dash="dot", line_color="rgba(148,163,184,0.4)",
                  annotation_text="50% threshold",
                  annotation_font=dict(color='#64748b', size=11))
    fig = _apply_theme(fig)
    fig.update_layout(
        title='Default Risk by Algorithm',
        xaxis_title='Risk Probability (%)',
        yaxis_title='',
        height=400,
        xaxis=dict(range=[0, 100]),
    )
    return fig


def comparison_bar_chart(df: pd.DataFrame, metric: str, title: str = None) -> go.Figure:
    if title is None:
        title = f'{metric.upper().replace("_", " ")} by Algorithm'

    sorted_df = df.sort_values(metric, ascending=False)
    colors = [ALGO_COLORS.get(a, '#6366f1') for a in sorted_df['algorithm']]

    fig = go.Figure(go.Bar(
        x=sorted_df['algorithm'],
        y=sorted_df[metric],
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        text=[f"{v:.4f}" for v in sorted_df[metric]],
        textposition='outside',
        textfont=dict(family='JetBrains Mono, monospace', size=10, color='#94a3b8'),
    ))

    fig = _apply_theme(fig)
    fig.update_layout(title=title, height=420, xaxis_tickangle=-30)
    return fig


def timing_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if 'train_time_s' in df.columns:
        fig.add_trace(go.Bar(
            name='Training',
            x=df['algorithm'],
            y=df['train_time_s'],
            marker=dict(color=COLORS['blue'], opacity=0.8),
        ))

    if 'infer_time_s' in df.columns:
        fig.add_trace(go.Bar(
            name='Inference',
            x=df['algorithm'],
            y=df['infer_time_s'],
            marker=dict(color=COLORS['cyan'], opacity=0.8),
        ))

    fig = _apply_theme(fig)
    fig.update_layout(
        title='Training vs Inference Time',
        yaxis_title='Time (seconds)',
        barmode='group',
        height=420,
        xaxis_tickangle=-30,
    )
    return fig


def cluster_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str) -> go.Figure:
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col,
        title='Customer Segments — PCA Projection',
        opacity=0.5,
        color_discrete_sequence=[
            COLORS['blue'], COLORS['emerald'], COLORS['gold'],
            COLORS['red'], COLORS['cyan'], COLORS['purple'],
            COLORS['pink'], COLORS['teal'],
        ],
    )
    fig = _apply_theme(fig)
    fig.update_layout(height=600)
    fig.update_traces(marker=dict(size=4))
    return fig
