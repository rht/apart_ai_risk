"""
Visualization module for AI Race Simulations
Provides plotting functions using Plotly for interactive visualizations.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Bloc/player names
BLOCS = ["US", "CN", "EU"]
N_PLAYERS = 3
US, CN, EU = 0, 1, 2


# ---------------------------
# CST.py visualization functions
# ---------------------------
def plot_results(t_eval, K_path, output_dir="plots", show=False, save_png=True):
    """
    Create interactive Plotly visualization for capability evolution (CST model).
    
    Analogous to plot_results() in CST.py but using Plotly.
    
    Args:
        t_eval: Array of time points
        K_path: Capability paths, shape (3, len(t_eval))
        output_dir: Directory to save the plots (default: "plots")
        show: If True, display plots interactively (default: False, returns figure object)
        save_png: If True, save plots as PNG files in addition to HTML (default: True)
    
    Returns:
        fig: Plotly figure object that can be displayed with IPython.display or fig.show()
    """
    # Define color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for US, CN, EU
    
    fig = go.Figure()
    
    for i, bloc in enumerate(BLOCS):
        fig.add_trace(go.Scatter(
            x=t_eval,
            y=K_path[i],
            mode='lines',
            name=f"{bloc} K",
            line=dict(color=colors[i], width=2),
            hovertemplate=f'<b>{bloc}</b><br>Time: %{{x:.2f}}<br>Capability: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Capability Evolution',
        xaxis_title='Time',
        yaxis_title='Capability (K)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        width=800,
        height=400
    )
    
    # Save HTML
    fig.write_html(f"{output_dir}/CST_plotly.html")
    
    # Save PNG if requested
    if save_png:
        fig.write_image(f"{output_dir}/CST_plotly.png", width=800, height=400)
    
    print(f"Saved capability plot to {output_dir}/CST_plotly.html" + 
          (f" and {output_dir}/CST_plotly.png" if save_png else ""))
    
    # Show if requested, otherwise return figure for notebook display
    if show:
        fig.show()
    
    return fig


def plot_actions(t_eval, aX_path, aS_path, aV_path, output_dir="plots", show=False, save_png=True):
    """
    Create interactive Plotly visualization for action evolution over time (CST model).
    
    Analogous to plot_actions() in CST.py but using Plotly.
    Creates a 3-panel subplot showing acceleration, safety, and verification efforts.
    
    Args:
        t_eval: Array of time points
        aX_path: Acceleration effort paths, shape (3, len(t_eval))
        aS_path: Safety effort paths, shape (3, len(t_eval))
        aV_path: Verification effort paths, shape (3, len(t_eval))
        output_dir: Directory to save the plots (default: "plots")
        show: If True, display plots interactively (default: False, returns figure object)
        save_png: If True, save plots as PNG files in addition to HTML (default: True)
    
    Returns:
        fig: Plotly figure object that can be displayed with IPython.display or fig.show()
    """
    # Define color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for US, CN, EU
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Acceleration Effort (aX)', 'Safety Effort (aS)', 'Verification Effort (aV)'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Helper function to create step plot data
    def create_step_data(times, values):
        """Convert data to step plot format."""
        x_step = []
        y_step = []
        for j in range(len(times)):
            if j > 0:
                x_step.append(times[j])
                y_step.append(values[j-1])
            x_step.append(times[j])
            y_step.append(values[j])
        return x_step, y_step
    
    # Plot aX (acceleration) - Row 1
    for i, bloc in enumerate(BLOCS):
        x_step, y_step = create_step_data(t_eval, aX_path[i])
        fig.add_trace(
            go.Scatter(
                x=x_step,
                y=y_step,
                mode='lines',
                name=bloc,
                line=dict(color=colors[i], width=2, shape='hv'),
                legendgroup=bloc,
                hovertemplate=f'<b>{bloc}</b><br>Time: %{{x:.2f}}<br>aX: %{{y:.3f}}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Plot aS (safety) - Row 2
    for i, bloc in enumerate(BLOCS):
        x_step, y_step = create_step_data(t_eval, aS_path[i])
        fig.add_trace(
            go.Scatter(
                x=x_step,
                y=y_step,
                mode='lines',
                name=bloc,
                line=dict(color=colors[i], width=2, shape='hv'),
                legendgroup=bloc,
                showlegend=False,
                hovertemplate=f'<b>{bloc}</b><br>Time: %{{x:.2f}}<br>aS: %{{y:.3f}}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Plot aV (verification) - Row 3
    for i, bloc in enumerate(BLOCS):
        x_step, y_step = create_step_data(t_eval, aV_path[i])
        fig.add_trace(
            go.Scatter(
                x=x_step,
                y=y_step,
                mode='lines',
                name=bloc,
                line=dict(color=colors[i], width=2, shape='hv'),
                legendgroup=bloc,
                showlegend=False,
                hovertemplate=f'<b>{bloc}</b><br>Time: %{{x:.2f}}<br>aV: %{{y:.3f}}<extra></extra>'
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title_text='Action Evolution Over Time',
        hovermode='x unified',
        template='plotly_white',
        height=750,
        width=800,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    # Update y-axes ranges
    fig.update_yaxes(range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], row=3, col=1)
    
    # Update x-axis label (only on bottom subplot)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    
    # Save HTML
    fig.write_html(f"{output_dir}/CST_actions_plotly.html")
    
    # Save PNG if requested
    if save_png:
        fig.write_image(f"{output_dir}/CST_actions_plotly.png", width=800, height=750)
    
    print(f"Saved action plot to {output_dir}/CST_actions_plotly.html" + 
          (f" and {output_dir}/CST_actions_plotly.png" if save_png else ""))
    
    # Show if requested, otherwise return figure for notebook display
    if show:
        fig.show()
    
    return fig


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    print("This module provides visualization functions.")
    print("\nAvailable functions for CST model:")
    print("  - plot_results_cst(t_eval, K_path, ...)")
    print("  - plot_actions_cst(t_eval, aX_path, aS_path, aV_path, ...)")
    print("\nExample usage:")
    print("  from visualizations import plot_results_cst, plot_actions_cst")
    print("  # Show plots and save as both HTML and PNG")
    print("  plot_results_cst(t_eval, K_path)")
    print("  plot_actions_cst(t_eval, aX_path, aS_path, aV_path)")
    print("  # Only save files, don't show")
    print("  plot_results_cst(t_eval, K_path, show=False)")
    print("  # Show only, don't save PNG")
    print("  plot_actions_cst(t_eval, aX_path, aS_path, aV_path, save_png=False)")
