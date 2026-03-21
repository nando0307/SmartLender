"""Plotly chart helper functions for the Streamlit dashboard."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def risk_bar_chart(predictions: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart of risk probabilities across algorithms.

    Args:
        predictions: DataFrame with columns [algorithm, risk_pct, decision]
    """
    colors = ['#e74c3c' if d == 'Deny' else '#2ecc71' for d in predictions['decision']]

    fig = go.Figure(go.Bar(
        x=predictions['risk_pct'],
        y=predictions['algorithm'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1f}%" for v in predictions['risk_pct']],
        textposition='auto',
    ))

    fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="50% threshold")
    fig.update_layout(
        title='Default Risk by Algorithm',
        xaxis_title='Risk Probability (%)',
        yaxis_title='',
        height=400,
        xaxis=dict(range=[0, 100]),
    )
    return fig


def comparison_bar_chart(df: pd.DataFrame, metric: str, title: str = None) -> go.Figure:
    """
    Create a bar chart comparing a metric across algorithms.
    """
    if title is None:
        title = f'{metric} by Algorithm'

    fig = px.bar(
        df.sort_values(metric, ascending=False),
        x='algorithm', y=metric,
        title=title,
        color=metric,
        color_continuous_scale='Viridis',
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    return fig


def timing_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a grouped bar chart showing training and inference times.
    """
    fig = go.Figure()

    if 'train_time_s' in df.columns:
        fig.add_trace(go.Bar(
            name='Training Time',
            x=df['algorithm'],
            y=df['train_time_s'],
            marker_color='#3498db',
        ))

    if 'infer_time_s' in df.columns:
        fig.add_trace(go.Bar(
            name='Inference Time',
            x=df['algorithm'],
            y=df['infer_time_s'],
            marker_color='#e67e22',
        ))

    fig.update_layout(
        title='Training vs Inference Time',
        xaxis_title='',
        yaxis_title='Time (seconds)',
        barmode='group',
        height=400,
        xaxis_tickangle=-45,
    )
    return fig


def cluster_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str) -> go.Figure:
    """
    Create an interactive scatter plot for cluster visualization.
    """
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col,
        title='Customer Segments — PCA Projection',
        opacity=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=600)
    return fig
