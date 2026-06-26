"""
Visualization module for Solar Production Prediction.
Provides reusable plotting functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import PLOT_CONFIG, PLANT_CONFIG


def setup_plot_style():
    """Setup matplotlib/seaborn style."""
    sns.set_style(PLOT_CONFIG['style'])
    plt.rcParams['figure.dpi'] = PLOT_CONFIG['figure_dpi']


def plot_model_comparison(comparison_df):
    """
    Plot model comparison charts.

    Args:
        comparison_df: DataFrame from create_comparison_dataframe()
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    colors = [PLOT_CONFIG['color_palette']['best_model'] if i == 0
              else PLOT_CONFIG['color_palette']['other_models']
              for i in range(len(comparison_df))]

    # R² Score
    ax1 = axes[0, 0]
    ax1.barh(comparison_df['Model'], comparison_df['Test R²'], color=colors)
    ax1.set_xlabel('R² Score', fontsize=PLOT_CONFIG['fonts']['label'], fontweight='bold')
    ax1.set_title('Model Accuracy (Higher is Better)',
                  fontsize=PLOT_CONFIG['fonts']['subtitle'], fontweight='bold')
    ax1.axvline(x=0.9, color='orange', linestyle='--', alpha=0.7, label='Excellent (>0.9)')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # MAE
    ax2 = axes[0, 1]
    ax2.barh(comparison_df['Model'], comparison_df['Test MAE (kWh)'], color=colors)
    ax2.set_xlabel('Mean Absolute Error (kWh)', fontsize=PLOT_CONFIG['fonts']['label'], fontweight='bold')
    ax2.set_title('Prediction Error (Lower is Better)',
                  fontsize=PLOT_CONFIG['fonts']['subtitle'], fontweight='bold')
    ax2.invert_xaxis()
    ax2.grid(axis='x', alpha=0.3)

    # WAPE (fallback to old MAPE column for older comparison files)
    error_pct_col = 'Test WAPE (%)' if 'Test WAPE (%)' in comparison_df.columns else 'Test MAPE (%)'
    error_pct_label = 'Weighted Absolute Percentage Error' if error_pct_col == 'Test WAPE (%)' else 'Mean Absolute Percentage Error'
    ax3 = axes[1, 0]
    ax3.barh(comparison_df['Model'], comparison_df[error_pct_col], color=colors)
    ax3.set_xlabel(f'{error_pct_label} (%)',
                   fontsize=PLOT_CONFIG['fonts']['label'], fontweight='bold')
    ax3.set_title('Relative Error (Lower is Better)',
                  fontsize=PLOT_CONFIG['fonts']['subtitle'], fontweight='bold')
    ax3.invert_xaxis()
    ax3.grid(axis='x', alpha=0.3)

    # Training Time
    ax4 = axes[1, 1]
    ax4.barh(comparison_df['Model'], comparison_df['Training Time (s)'], color=colors)
    ax4.set_xlabel('Training Time (seconds)', fontsize=PLOT_CONFIG['fonts']['label'], fontweight='bold')
    ax4.set_title('Computational Cost', fontsize=PLOT_CONFIG['fonts']['subtitle'], fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    plt.suptitle('ML Model Comparison with Hyperparameter Tuning',
                 fontsize=PLOT_CONFIG['fonts']['title'], fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


def plot_three_line_comparison(test_df, predictions, best_model_key,
                               start_date=None, end_date=None):
    """
    Plot Actual vs ML Prediction vs Clear-Sky.
    The key visualization showing why ML is better than clear-sky.

    Args:
        test_df: Test dataframe with actual and clearsky values
        predictions: dict with predictions for each model
        best_model_key: Name of best model to plot
        start_date, end_date: Optional date range to plot
    """
    setup_plot_style()

    plot_df = test_df.copy()
    plot_df['ml_predicted_kwh'] = predictions[best_model_key]

    # Filter date range
    if start_date:
        plot_df = plot_df[plot_df['generation_date'] >= start_date]
    if end_date:
        plot_df = plot_df[plot_df['generation_date'] <= end_date]

    # Remove NaN
    plot_df = plot_df.dropna(subset=['ml_predicted_kwh'])

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(22, 14), height_ratios=[3, 1])

    # ===== MAIN PLOT =====
    ax1 = axes[0]

    ax1.plot(plot_df['generation_date'], plot_df['clearsky_expected_kwh'],
            color=PLOT_CONFIG['color_palette']['clearsky'], linewidth=2.5, alpha=0.7,
            label='🟠 Clear-Sky Maximum (Theoretical - Too Optimistic)', linestyle='--')

    ax1.plot(plot_df['generation_date'], plot_df['ml_predicted_kwh'],
            color=PLOT_CONFIG['color_palette']['ml_predicted'], linewidth=2.5, alpha=0.85,
            label=f'🟢 ML Predicted ({best_model_key}) - Weather-Adjusted')

    ax1.plot(plot_df['generation_date'], plot_df['generation_kwh'],
            color=PLOT_CONFIG['color_palette']['actual'], linewidth=2, alpha=0.9,
            label='🔵 Actual Production')

    ax1.set_title(
        f'Solar Production: ML vs Clear-Sky Comparison - {PLANT_CONFIG["name"]}\n'
        f'Green line (ML) follows actual production closely - accounts for clouds & weather\n'
        f'Red line (Clear-Sky) is unrealistically high - assumes perfect conditions',
        fontsize=PLOT_CONFIG['fonts']['title'], fontweight='bold', pad=20
    )
    ax1.set_ylabel('Hourly Energy (kWh)', fontsize=PLOT_CONFIG['fonts']['subtitle'], fontweight='bold')
    ax1.legend(fontsize=PLOT_CONFIG['fonts']['legend'], loc='upper left', framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # ===== RESIDUAL PLOT =====
    ax2 = axes[1]

    residuals = plot_df['generation_kwh'] - plot_df['ml_predicted_kwh']
    threshold = residuals.std() * 2
    colors_res = [PLOT_CONFIG['color_palette']['residual_alert'] if r < -threshold
                  else PLOT_CONFIG['color_palette']['residual_normal']
                  for r in residuals]

    ax2.bar(plot_df['generation_date'], residuals, color=colors_res, alpha=0.6, width=0.04)
    ax2.axhline(y=0, color='black', linewidth=2, linestyle='-')
    ax2.axhline(y=-threshold, color='red', linewidth=2,
               linestyle='--', alpha=0.7, label=f'Alert Threshold (-{threshold:.1f} kWh)')

    n_alerts = sum(r < -threshold for r in residuals)
    ax2.set_title(f'Residuals (Actual - ML) → {n_alerts} potential problems detected (red bars)',
                 fontsize=PLOT_CONFIG['fonts']['subtitle'], fontweight='bold', pad=10)
    ax2.set_xlabel('Date', fontsize=PLOT_CONFIG['fonts']['subtitle'], fontweight='bold')
    ax2.set_ylabel('Residual (kWh)', fontsize=PLOT_CONFIG['fonts']['label'])
    ax2.legend(fontsize=PLOT_CONFIG['fonts']['legend'])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\n" + "="*80)
    print("  COMPARISON STATISTICS")
    print("="*80)

    actual_mean = plot_df['generation_kwh'].mean()
    ml_mean = plot_df['ml_predicted_kwh'].mean()
    clearsky_mean = plot_df['clearsky_expected_kwh'].mean()

    print(f"\nAverage Hourly Production:")
    print(f"  Actual:       {actual_mean:.2f} kWh")
    print(f"  ML Predicted: {ml_mean:.2f} kWh (error: {abs(ml_mean-actual_mean)/actual_mean*100:.1f}%)")
    print(f"  Clear-Sky:    {clearsky_mean:.2f} kWh (error: {abs(clearsky_mean-actual_mean)/actual_mean*100:.1f}%)")

    ml_error = abs(ml_mean - actual_mean) / actual_mean * 100
    clearsky_error = abs(clearsky_mean - actual_mean) / actual_mean * 100

    print(f"\n📊 Accuracy Comparison:")
    print(f"  ML Model:    {ml_error:.1f}% average error")
    print(f"  Clear-Sky:   {clearsky_error:.1f}% average error")
    print(f"\n  ⭐ ML is {clearsky_error/ml_error:.1f}x more accurate than clear-sky!")
    print("="*80)


def plot_feature_importance(results, feature_cols, top_n=15):
    """
    Plot feature importance for tree-based models.

    Args:
        results: dict of model results
        feature_cols: List of feature names
        top_n: Number of top features to show
    """
    setup_plot_style()

    tree_models = ['RandomForest', 'GradientBoosting', 'XGBoost']
    available_tree_models = [m for m in tree_models if m in results]

    if not available_tree_models:
        print("No tree-based models available for feature importance plot")
        return

    fig, axes = plt.subplots(1, len(available_tree_models),
                            figsize=(8*len(available_tree_models), 8))

    if len(available_tree_models) == 1:
        axes = [axes]

    for idx, model_name in enumerate(available_tree_models):
        model = results[model_name]['model']

        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        axes[idx].barh(range(len(importance_df)), importance_df['importance'],
                      color='steelblue')
        axes[idx].set_yticks(range(len(importance_df)))
        axes[idx].set_yticklabels(importance_df['feature'])
        axes[idx].invert_yaxis()
        axes[idx].set_xlabel('Importance', fontsize=PLOT_CONFIG['fonts']['label'], fontweight='bold')
        axes[idx].set_title(f'{model_name}\nTop {top_n} Features',
                           fontsize=PLOT_CONFIG['fonts']['subtitle'], fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_ridge_coefficients(ridge_model, feature_cols, top_n=15):
    """
    Plot Ridge regression coefficients.

    Args:
        ridge_model: Trained Ridge model
        feature_cols: List of feature names
        top_n: Number of top features to show
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Create coefficient dataframe
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': ridge_model.coef_,
        'abs_coefficient': np.abs(ridge_model.coef_)
    }).sort_values('abs_coefficient', ascending=False)

    # Plot 1: Top features by absolute coefficient
    ax1 = axes[0]
    top_features = coef_df.head(top_n)
    colors = ['green' if c > 0 else 'red' for c in top_features['coefficient']]

    ax1.barh(range(len(top_features)), top_features['abs_coefficient'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'])
    ax1.invert_yaxis()
    ax1.set_xlabel('Absolute Coefficient Value', fontsize=PLOT_CONFIG['fonts']['label'], fontweight='bold')
    ax1.set_title(f'Ridge Regression\nTop {top_n} Features by Absolute Coefficient\n(Green=Positive, Red=Negative)',
                 fontsize=PLOT_CONFIG['fonts']['subtitle'], fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Plot 2: Actual coefficient values (with sign)
    ax2 = axes[1]
    colors2 = ['green' if c > 0 else 'red' for c in top_features['coefficient']]

    ax2.barh(range(len(top_features)), top_features['coefficient'], color=colors2, alpha=0.7)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['feature'])
    ax2.invert_yaxis()
    ax2.axvline(x=0, color='black', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Coefficient Value', fontsize=PLOT_CONFIG['fonts']['label'], fontweight='bold')
    ax2.set_title('Ridge Regression\nCoefficient Values with Direction\n(Positive → Increases production, Negative → Decreases production)',
                 fontsize=PLOT_CONFIG['fonts']['subtitle'], fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print top features
    print(f"\n📊 Top 10 Most Important Features (Ridge Regression):")
    print("="*70)
    print(coef_df[['feature', 'coefficient', 'abs_coefficient']].head(10).to_string(index=False))
    print("="*70)


# ============================================================================
# PLOTLY FUNCTIONS FOR STREAMLIT DASHBOARD
# ============================================================================

import plotly.graph_objects as go
from typing import Optional


def plot_forecast_daily_plotly(
    forecast_df: pd.DataFrame,
    historical_df: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Plot daily aggregated forecast with historical data (Plotly version for Streamlit).

    Args:
        forecast_df: DataFrame with hourly predictions
        historical_df: Optional DataFrame with historical production data

    Returns:
        Plotly Figure object
    """
    # Aggregate forecast data
    daily_forecast = forecast_df.groupby(forecast_df.index.date).agg({
        'predicted_kwh': 'sum'
    }).reset_index()
    daily_forecast.columns = ['date', 'forecast_kwh']
    daily_forecast['date'] = pd.to_datetime(daily_forecast['date'])

    fig = go.Figure()

    replay_mode = False

    # Add historical data if provided
    if historical_df is not None:
        agg_dict = {'generation_kwh': 'sum'}
        if 'ml_predicted_kwh' in historical_df.columns:
            agg_dict['ml_predicted_kwh'] = 'sum'

        historical_daily = historical_df.groupby(historical_df.index.date).agg(agg_dict).reset_index()
        historical_daily.columns = ['date', 'actual_kwh', 'predicted_kwh'] if 'ml_predicted_kwh' in historical_df.columns else ['date', 'actual_kwh']
        historical_daily['date'] = pd.to_datetime(historical_daily['date'])

        replay_mode = daily_forecast['date'].max() <= historical_daily['date'].max()
        if replay_mode:
            # No live weather forecast is configured: compare model predictions against
            # the actual latest historical days instead of drawing fake future dates.
            plot_dates = set(daily_forecast['date'])
            last_days = historical_daily[historical_daily['date'].isin(plot_dates)].copy()
        else:
            last_days = historical_daily.tail(5).copy()

        # Add actual production (grey bars)
        fig.add_trace(go.Bar(
            x=last_days['date'],
            y=last_days['actual_kwh'],
            name='Actual Production',
            marker_color='#B0B0B0',
            text=[f"{val:.0f} kWh" for val in last_days['actual_kwh']],
            textposition='outside',
            hovertemplate='<b>Date:</b> %{x|%b %d}<br><b>Actual:</b> %{y:.1f} kWh<extra></extra>'
        ))

        if replay_mode:
            combined_predictions = daily_forecast[['date', 'forecast_kwh']].rename(columns={'forecast_kwh': 'prediction'})
            trace_name = 'Model Replay'
        elif 'ml_predicted_kwh' in historical_df.columns:
            combined_predictions = pd.concat([
                last_days[['date', 'predicted_kwh']].rename(columns={'predicted_kwh': 'prediction'}),
                daily_forecast[['date', 'forecast_kwh']].rename(columns={'forecast_kwh': 'prediction'})
            ]).sort_values('date')
            trace_name = 'Predicted Production'
        else:
            combined_predictions = daily_forecast[['date', 'forecast_kwh']].rename(columns={'forecast_kwh': 'prediction'})
            trace_name = 'Predicted Production'

        fig.add_trace(go.Scatter(
            x=combined_predictions['date'],
            y=combined_predictions['prediction'],
            name=trace_name,
            line=dict(color='#2ca02c', width=3),
            mode='lines+markers',
            marker=dict(size=8, color='#2ca02c'),
            hovertemplate='<b>Date:</b> %{x|%b %d}<br><b>Predicted:</b> %{y:.1f} kWh<extra></extra>'
        ))
    else:
        # No historical data, just show forecast
        fig.add_trace(go.Scatter(
            x=daily_forecast['date'],
            y=daily_forecast['forecast_kwh'],
            name='Predicted Production',
            line=dict(color='#2ca02c', width=3),
            mode='lines+markers',
            marker=dict(size=8, color='#2ca02c'),
            hovertemplate='<b>Date:</b> %{x|%b %d}<br><b>Predicted:</b> %{y:.1f} kWh<extra></extra>'
        ))

    fig.update_layout(
        title='Latest Historical Replay: Actual vs Model Prediction' if replay_mode else '10-Day View: Historical (5 days) + Forecast (5 days)',
        xaxis_title='Date',
        yaxis_title='Energy Production (kWh)',
        height=500,
        hovermode='x unified'
    )

    return fig


def plot_forecast_hourly_plotly(forecast_df: pd.DataFrame, selected_date) -> go.Figure:
    """
    Plot hourly forecast for a specific day (Plotly version for Streamlit).

    Args:
        forecast_df: DataFrame with hourly predictions
        selected_date: Date to plot (datetime.date object)

    Returns:
        Plotly Figure object
    """
    day_data = forecast_df[forecast_df.index.date == selected_date]

    # Filter to daytime hours only using solar elevation
    if 'elevation' in day_data.columns:
        from .config import ML_CONFIG
        day_data = day_data[day_data['elevation'] > ML_CONFIG['min_sun_elevation']]
    else:
        # Fallback to approximate hours if elevation not available
        day_data = day_data[(day_data.index.hour >= 6) & (day_data.index.hour <= 18)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=day_data.index,
        y=day_data['predicted_kwh'],
        name='Predicted Production',
        line=dict(color='#2ca02c', width=3),
        mode='lines+markers'
    ))

    fig.update_layout(
        title=f'Hourly Production Forecast - {selected_date} (Daytime Only)',
        xaxis_title='Time',
        yaxis_title='Energy Production (kWh / hour)',
        height=400,
        hovermode='x unified'
    )

    return fig


def create_feature_diagnostics_table(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create diagnostic summary of key features for debugging.

    Args:
        forecast_df: DataFrame with forecast features

    Returns:
        DataFrame with feature statistics
    """
    important_features = [
        'clouds_all', 'cloud_impact', 'production_lag_24h',
        'production_lag_168h', 'clearsky_expected_kwh', 'solar_elevation'
    ]

    diag_data = []
    for feat in important_features:
        if feat in forecast_df.columns:
            diag_data.append({
                'Feature': feat,
                'Min': f"{forecast_df[feat].min():.2f}",
                'Max': f"{forecast_df[feat].max():.2f}",
                'Mean': f"{forecast_df[feat].mean():.2f}"
            })

    return pd.DataFrame(diag_data)


# ============================================================================
# ADDITIONAL PLOTLY VISUALIZATION FUNCTIONS
# ============================================================================


import calendar

def plot_calendar_heatmap(df, year=None):
    """Create a calendar heatmap showing daily production"""
    # Use latest year from data if not specified
    if year is None:
        year = df.index.max().year

    # Aggregate daily production
    daily_prod = df.groupby(df.index.date)['generation_kwh'].sum().reset_index()
    daily_prod.columns = ['date', 'production']
    daily_prod['date'] = pd.to_datetime(daily_prod['date'])

    # Create complete date range for the year (up to today, not future)
    start_date = pd.Timestamp(f'{year}-01-01')
    end_date = min(pd.Timestamp(f'{year}-12-31'), pd.Timestamp.now().floor('D'))
    all_dates = pd.DataFrame({'date': pd.date_range(start_date, end_date, freq='D')})

    # Merge with actual production data (NaN for missing days)
    daily_prod = all_dates.merge(daily_prod, on='date', how='left')
    daily_prod['production'] = daily_prod['production'].fillna(0)

    daily_prod['year'] = daily_prod['date'].dt.year
    daily_prod['month'] = daily_prod['date'].dt.month
    daily_prod['day'] = daily_prod['date'].dt.day

    # Filter to selected year
    daily_prod = daily_prod[daily_prod['year'] == year]

    # Custom colorscale: grey for 0, then YlOrRd for actual production
    custom_colorscale = [
        [0, '#D3D3D3'],      # Grey for no data (0)
        [0.001, '#FFFFCC'],  # Light yellow for very low production
        [0.25, '#FFEDA0'],
        [0.5, '#FEB24C'],
        [0.75, '#FC4E2A'],
        [1, '#BD0026']       # Dark red for high production
    ]

    # Calculate max production for colorbar range
    max_prod = daily_prod['production'].max()

    # Create calendar matrix
    fig = go.Figure()

    for month in range(1, 13):
        month_data = daily_prod[daily_prod['month'] == month]

        if len(month_data) > 0:
            fig.add_trace(go.Scatter(
                x=month_data['day'],
                y=[calendar.month_name[month]] * len(month_data),
                mode='markers',
                marker=dict(
                    size=15,
                    color=month_data['production'],
                    colorscale=custom_colorscale,
                    showscale=True,
                    colorbar=dict(
                        title="Production<br>(kWh)",
                        tickmode='linear',
                        tick0=0,
                        dtick=150,
                        thickness=12,
                        len=1.0,
                        tickfont=dict(size=10)
                    ),
                    cmin=0,
                    cmax=max_prod
                ),
                text=[f"{p:.0f} kWh" if p > 0 else "No data" for p in month_data['production']],
                hovertemplate='<b>%{y}</b><br>Day: %{x}<br>%{text}<extra></extra>',
                showlegend=False
            ))

    fig.update_layout(
        title=f'Daily Production Calendar - {year}',
        xaxis_title='Day of Month',
        yaxis_title='Month',
        height=600,
        hovermode='closest',
        annotations=[
            dict(
                text='<b>Legend:</b> <span style="color:#D3D3D3">⬤</span> Grey = No data',
                xref="paper", yref="paper",
                x=0.5, y=-0.05,
                xanchor='center', yanchor='top',
                showarrow=False,
                font=dict(size=11, color='#555')
            )
        ]
    )

    return fig



def plot_time_of_day_analysis(df):
    """Show hourly production and prediction for the latest day with usable production data."""
    # Use the latest non-zero production day so missing final records do not dominate the chart.
    daily_totals = df.groupby(df.index.date)['generation_kwh'].sum(min_count=1)
    usable_dates = daily_totals[daily_totals > 0]
    latest_date = usable_dates.index.max() if not usable_dates.empty else df.index.max().date()

    # Filter data for the selected day
    daily_data = df[df.index.date == latest_date].copy()
    daily_data = daily_data.sort_index()

    # Extract hour from index
    hours = daily_data.index.hour
    actual = daily_data['generation_kwh'].values

    fig = go.Figure()

    # Actual production
    fig.add_trace(go.Scatter(
        x=hours,
        y=actual,
        mode='lines+markers',
        name='Actual Production',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate='Hour: %{x}:00<br>Actual: %{y:.2f} kWh<extra></extra>'
    ))

    # Predicted production (if available)
    if 'ml_predicted_kwh' in daily_data.columns:
        predicted = daily_data['ml_predicted_kwh'].values
        fig.add_trace(go.Scatter(
            x=hours,
            y=predicted,
            mode='lines+markers',
            name='ML Prediction',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate='Hour: %{x}:00<br>Predicted: %{y:.2f} kWh<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text=f'Hourly Production - {latest_date.strftime("%Y-%m-%d")}',
            font=dict(size=16)
        ),
        xaxis_title='Hour of Day',
        yaxis_title='Production (kWh)',
        height=500,
        hovermode='x unified',
        annotations=[
            dict(
                text=f"Data from: {latest_date.strftime('%A, %B %d, %Y')}",
                xref="paper", yref="paper",
                x=0.5, y=1.08,
                xanchor='center', yanchor='bottom',
                showarrow=False,
                font=dict(size=12, color='#666')
            )
        ]
    )

    return fig



def plot_weather_correlation(df):
    """Plot correlations between weather and production"""
    # Select relevant columns
    weather_cols = ['temp', 'humidity', 'clouds_all', 'wind_speed', 'pressure']
    available_cols = [col for col in weather_cols if col in df.columns]

    if len(available_cols) == 0:
        st.warning("Weather data not available in predictions")
        return None

    # Calculate correlations
    corr_data = []
    for col in available_cols:
        corr = df[['generation_kwh', col]].corr().iloc[0, 1]
        corr_data.append({'Feature': col, 'Correlation': corr})

    corr_df = pd.DataFrame(corr_data)

    fig = go.Figure()

    colors = ['green' if x > 0 else 'red' for x in corr_df['Correlation']]

    fig.add_trace(go.Bar(
        x=corr_df['Feature'],
        y=corr_df['Correlation'],
        marker_color=colors,
        text=[f"{x:.3f}" for x in corr_df['Correlation']],
        textposition='outside'
    ))

    fig.update_layout(
        title='Weather Variables Correlation with Production',
        xaxis_title='Weather Feature',
        yaxis_title='Correlation Coefficient',
        height=400,
        yaxis_range=[-1, 1]
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    return fig



def plot_monthly_comparison(df):
    """Compare production across months"""
    df_copy = df.copy()
    # Remove timezone before converting to period to avoid warning
    index_no_tz = df_copy.index.tz_localize(None) if df_copy.index.tz else df_copy.index
    df_copy['year_month'] = index_no_tz.to_period('M')

    # Aggregate actual production
    agg_dict = {'generation_kwh': 'sum'}

    # Add predicted if available
    if 'ml_predicted_kwh' in df_copy.columns:
        agg_dict['ml_predicted_kwh'] = 'sum'

    monthly_prod = df_copy.groupby('year_month').agg(agg_dict).reset_index()
    monthly_prod['year_month'] = monthly_prod['year_month'].astype(str)

    fig = go.Figure()

    # Actual production (bars)
    fig.add_trace(go.Bar(
        x=monthly_prod['year_month'],
        y=monthly_prod['generation_kwh'],
        name='Actual Production',
        marker_color='#1f77b4',
        text=[f"{x/1000:.1f}k" for x in monthly_prod['generation_kwh']],
        textposition='outside'
    ))

    # Predicted production (line)
    if 'ml_predicted_kwh' in monthly_prod.columns:
        fig.add_trace(go.Scatter(
            x=monthly_prod['year_month'],
            y=monthly_prod['ml_predicted_kwh'],
            name='Predicted Production',
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=8, symbol='circle')
        ))

    fig.update_layout(
        title='Monthly Production Comparison (Actual vs Predicted)',
        xaxis_title='Month',
        yaxis_title='Total Production (kWh)',
        height=450,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig




def plot_weekly_production_pattern(df):
    """Show hourly production for the last 7 days with actual and predicted values"""
    df_copy = df.copy()

    # Get last 7 days
    latest_date = df_copy.index.max()
    start_date = latest_date - pd.Timedelta(days=6)
    df_last_7 = df_copy[df_copy.index >= start_date].copy()

    # Sort by index to ensure chronological order
    df_last_7 = df_last_7.sort_index()

    # Create datetime strings for display
    df_last_7['datetime_str'] = df_last_7.index.strftime('%m-%d %H:00')

    fig = go.Figure()

    # Add filled area if prediction is available
    if 'ml_predicted_kwh' in df_last_7.columns:
        # Simple approach: fill between the two curves
        # This creates a single continuous fill area
        fig.add_trace(go.Scatter(
            x=df_last_7['datetime_str'],
            y=df_last_7['generation_kwh'],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='baseline'
        ))

        fig.add_trace(go.Scatter(
            x=df_last_7['datetime_str'],
            y=df_last_7['ml_predicted_kwh'],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(150, 150, 150, 0.2)',
            name='Prediction Gap',
            showlegend=True,
            hoverinfo='skip'
        ))

    # Actual production (line)
    fig.add_trace(go.Scatter(
        x=df_last_7['datetime_str'],
        y=df_last_7['generation_kwh'],
        name='Actual Production',
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{x}<br>Actual: %{y:.2f} kWh<extra></extra>'
    ))

    # Predicted production (line)
    if 'ml_predicted_kwh' in df_last_7.columns:
        fig.add_trace(go.Scatter(
            x=df_last_7['datetime_str'],
            y=df_last_7['ml_predicted_kwh'],
            name='ML Prediction',
            mode='lines',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='%{x}<br>Predicted: %{y:.2f} kWh<extra></extra>'
        ))

    fig.update_layout(
        title='Last 7 Days Production (Hourly)',
        xaxis_title='Date & Hour',
        yaxis_title='Hourly Production (kWh)',
        yaxis=dict(
            tickformat='.1f'
        ),
        xaxis=dict(
            tickangle=-45,
            nticks=20
        ),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig




def plot_monthly_production_pattern(df):
    """Show hourly production for the last 30 days with actual and predicted values"""
    df_copy = df.copy()

    # Get last 30 days
    latest_date = df_copy.index.max()
    start_date = latest_date - pd.Timedelta(days=29)
    df_last_30 = df_copy[df_copy.index >= start_date].copy()

    # Sort by index to ensure chronological order
    df_last_30 = df_last_30.sort_index()

    # Create datetime strings for display
    df_last_30['datetime_str'] = df_last_30.index.strftime('%m-%d %H:00')

    fig = go.Figure()

    # Add filled area if prediction is available
    if 'ml_predicted_kwh' in df_last_30.columns:
        # Simple approach: fill between the two curves
        # This creates a single continuous fill area
        fig.add_trace(go.Scatter(
            x=df_last_30['datetime_str'],
            y=df_last_30['generation_kwh'],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='baseline'
        ))

        fig.add_trace(go.Scatter(
            x=df_last_30['datetime_str'],
            y=df_last_30['ml_predicted_kwh'],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(150, 150, 150, 0.2)',
            name='Prediction Gap',
            showlegend=True,
            hoverinfo='skip'
        ))

    # Actual production (line)
    fig.add_trace(go.Scatter(
        x=df_last_30['datetime_str'],
        y=df_last_30['generation_kwh'],
        name='Actual Production',
        mode='lines',
        line=dict(color='#1f77b4', width=1.5),
        hovertemplate='%{x}<br>Actual: %{y:.2f} kWh<extra></extra>'
    ))

    # Predicted production (line)
    if 'ml_predicted_kwh' in df_last_30.columns:
        fig.add_trace(go.Scatter(
            x=df_last_30['datetime_str'],
            y=df_last_30['ml_predicted_kwh'],
            name='ML Prediction',
            mode='lines',
            line=dict(color='#ff7f0e', width=1.5, dash='dash'),
            hovertemplate='%{x}<br>Predicted: %{y:.2f} kWh<extra></extra>'
        ))

    fig.update_layout(
        title='Last 30 Days Production (Hourly)',
        xaxis_title='Date & Hour',
        yaxis_title='Hourly Production (kWh)',
        yaxis=dict(
            tickformat='.1f'
        ),
        xaxis=dict(
            tickangle=-45,
            nticks=25
        ),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig




def plot_feature_importance_proxy(df, feature_columns):
    """Visualize feature distributions and statistics"""
    # Select numeric features only
    numeric_features = [col for col in feature_columns if col in df.columns and df[col].dtype in ['float64', 'int64']]

    if len(numeric_features) == 0:
        return None

    # Calculate basic statistics
    feature_stats = []
    for col in numeric_features[:15]:  # Limit to top 15
        feature_stats.append({
            'Feature': col,
            'Mean': df[col].mean(),
            'Std': df[col].std(),
            'Min': df[col].min(),
            'Max': df[col].max()
        })

    stats_df = pd.DataFrame(feature_stats)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Feature Mean Values', 'Feature Variability (Std Dev)')
    )

    fig.add_trace(
        go.Bar(x=stats_df['Feature'], y=stats_df['Mean'], name='Mean', marker_color='#1f77b4'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=stats_df['Feature'], y=stats_df['Std'], name='Std Dev', marker_color='#ff7f0e'),
        row=1, col=2
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=500, showlegend=False)

    return fig




def plot_top_features_importance(df, feature_columns, top_n=10):
    """Plot top N features by absolute correlation with production."""
    if feature_columns is None or len(feature_columns) == 0:
        return None

    # Select numeric features that exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns and df[col].dtype in ['float64', 'int64']]

    if len(available_features) == 0:
        return None

    # Calculate correlation with production
    correlations = []
    for feature in available_features:
        try:
            corr = df[['generation_kwh', feature]].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations.append({
                    'Feature': feature,
                    'Importance': abs(corr),  # Absolute correlation as importance
                    'Correlation': corr
                })
        except:
            continue

    if len(correlations) == 0:
        return None

    # Sort by absolute correlation strength and take top N
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Importance', ascending=False).head(top_n)

    # Create bar chart
    fig = go.Figure()

    colors = ['#1f77b4' if x > 0 else '#d62728' for x in corr_df['Correlation']]

    fig.add_trace(go.Bar(
        x=corr_df['Feature'],
        y=corr_df['Importance'],
        marker_color=colors,
        text=[f"{x:.3f}" for x in corr_df['Correlation']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Correlation strength: %{y:.3f}<br>Signed correlation: %{text}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Strongest {top_n} Feature Relationships with Production',
        xaxis_title='Engineered feature',
        yaxis_title='Absolute correlation strength',
        height=500,
        xaxis_tickangle=-45,
        showlegend=False
    )

    return fig




def plot_model_comparison_metrics(comparison_df):
    """Create compact model comparison visualizations for core quality metrics."""
    if comparison_df is None or len(comparison_df) == 0:
        return None

    error_col = 'Test WAPE (%)' if 'Test WAPE (%)' in comparison_df.columns else 'Test MAPE (%)'
    error_label = 'WAPE (%)' if error_col == 'Test WAPE (%)' else 'MAPE (%)'

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=(
            'R² (Higher Better)',
            'MAE (Lower Better)',
            'RMSE (Lower Better)',
            f'{error_label} (Lower Better)'
        ),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]],
        horizontal_spacing=0.06
    )

    models = comparison_df['Model'].tolist()

    fig.add_trace(
        go.Bar(x=models, y=comparison_df['Test R²'], marker_color='#2ca02c', name='R²'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=models, y=comparison_df['Test MAE (kWh)'], marker_color='#d62728', name='MAE'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=models, y=comparison_df['Test RMSE (kWh)'], marker_color='#ff7f0e', name='RMSE'),
        row=1, col=3
    )
    fig.add_trace(
        go.Bar(x=models, y=comparison_df[error_col], marker_color='#9467bd', name=error_label),
        row=1, col=4
    )

    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=380, showlegend=False, title_text="Model Quality Comparison")

    return fig
