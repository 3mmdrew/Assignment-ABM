# app.py - SolaraViz Frontend for WaterToC Mesa Model
import solara
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional

from src.model import WaterToC

# Reactive variables for model parameters
height = solara.reactive(20)
width = solara.reactive(20)
initial_humans = solara.reactive(10)
initial_ai = solara.reactive(10)
c_payoff = solara.reactive(0.5)
d_payoff = solara.reactive(1.0)
max_water_capacity = solara.reactive(10)
water_cell_density = solara.reactive(0.3)
theta = solara.reactive(3.0)  # Added theta parameter
max_steps = solara.reactive(100)
seed = solara.reactive(42)

# Reactive variables for simulation state
model_data = solara.reactive(None)
is_running = solara.reactive(False)
current_step = solara.reactive(0)

def run_simulation():
    """Run the Mesa model simulation"""
    if WaterToC is None:
        return pd.DataFrame()
    
    try:
        # Create model instance with theta parameter
        model = WaterToC(
            height=height.value,
            width=width.value,
            initial_humans=initial_humans.value,
            initial_ai=initial_ai.value,
            C_Payoff=c_payoff.value,
            D_Payoff=d_payoff.value,
            max_water_capacity=max_water_capacity.value,
            water_cell_density=water_cell_density.value,
            theta=theta.value,  # Added theta parameter
            seed=seed.value
        )
        
        # Run simulation
        for i in range(max_steps.value):
            model.step()
            current_step.value = i + 1
        
        # Get collected data
        data = model.datacollector.get_model_vars_dataframe()
        return data
    
    except Exception as e:
        print(f"Error running simulation: {e}")
        return pd.DataFrame()

@solara.component
def ParameterControls():
    """Component for model parameter controls"""
    with solara.Card("Model Parameters"):
        with solara.Column():
            # Grid parameters
            solara.SliderInt("Grid Height", value=height, min=10, max=30)
            solara.SliderInt("Grid Width", value=width, min=10, max=30)
            
            # Agent parameters
            solara.SliderInt("Initial Humans", value=initial_humans, min=1, max=50)
            solara.SliderInt("Initial AI", value=initial_ai, min=1, max=50)
            
            # Payoff parameters
            solara.SliderFloat("Cooperation Payoff", value=c_payoff, min=0.0, max=2.0, step=0.05)
            solara.SliderFloat("Defection Payoff", value=d_payoff, min=0.0, max=2.0, step=0.05)
            
            # Water parameters
            solara.SliderInt("Max Water Capacity", value=max_water_capacity, min=1, max=20)
            solara.SliderFloat("Water Cell Density", value=water_cell_density, min=0.1, max=1.0, step=0.1)
            
            # Environmental feedback parameter
            solara.SliderFloat("Theta (Environmental Feedback)", value=theta, min=1.0, max=20.0, step=0.1)
            
            # Simulation parameters
            solara.SliderInt("Max Steps", value=max_steps, min=10, max=1000)
            solara.SliderInt("Random Seed", value=seed, min=1, max=1000)

@solara.component
def SimulationControls():
    """Component for simulation controls"""
    def on_run_click():
        is_running.value = True
        current_step.value = 0
        data = run_simulation()
        model_data.value = data
        is_running.value = False
    
    with solara.Card("Simulation Controls"):
        solara.Button(
            "Run Simulation", 
            on_click=on_run_click, 
            disabled=is_running.value,
            color="primary"
        )
        
        if is_running.value:
            solara.Text(f"Running... Step {current_step.value}/{max_steps.value}")
        elif model_data.value is not None:
            solara.Text(f"Simulation completed: {len(model_data.value)} steps")

@solara.component
def TimeSeriesPlots():
    """Enhanced time series visualizations with 4-row layout"""
    if model_data.value is None or model_data.value.empty:
        solara.Markdown("No data available. Run a simulation first.")
        return
    
    data = model_data.value.reset_index()
    
    # Create enhanced plots with 4 rows
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Cooperation Fraction Over Time',
            'Environment State (n) Over Time',
            'Total Water vs Capacity',
            'Strategy Counts Over Time'
        ),
        vertical_spacing=0.08
    )
    
    # Plot 1: Cooperation Fraction
    if 'Coop_Fraction' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Coop_Fraction'],
                name='Cooperation Fraction',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        # Add horizontal line at 1/theta for reference
        theta_line = 1.0 / theta.value
        fig.add_hline(
            y=theta_line,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"1/θ = {theta_line:.3f}",
            row=1, col=1
        )
    
    # Plot 2: Environment State
    if 'Environment_State' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Environment_State'],
                name='Environment State (n)',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
    
    # Plot 3: Water levels
    if 'Total_Water' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Total_Water'],
                name='Current Water',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
    if 'Total_Water_Capacity' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Total_Water_Capacity'],
                name='Max Capacity',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=3, col=1
        )
    
    # Plot 4: Strategy counts
    if 'Cooperators' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Cooperators'],
                name='Cooperators',
                line=dict(color='green', width=2)
            ),
            row=4, col=1
        )
    if 'Defectors' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Defectors'],
                name='Defectors',
                line=dict(color='red', width=2)
            ),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=900,
        title_text="WaterToC Model Dynamics",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time Step", row=4, col=1)
    fig.update_yaxes(title_text="Cooperation Fraction", row=1, col=1)
    fig.update_yaxes(title_text="Environment State", row=2, col=1)
    fig.update_yaxes(title_text="Water Level", row=3, col=1)
    fig.update_yaxes(title_text="Agent Count", row=4, col=1)
    
    solara.FigurePlotly(fig)

@solara.component
def PhaseSpacePlot():
    """Phase space plot showing relationship between cooperation and environment"""
    if model_data.value is None or model_data.value.empty:
        return
    
    data = model_data.value.reset_index()
    
    # Check if required columns exist
    if 'Coop_Fraction' not in data.columns or 'Environment_State' not in data.columns:
        return
    
    fig = go.Figure()
    
    # Create phase space plot
    fig.add_trace(
        go.Scatter(
            x=data['Coop_Fraction'],
            y=data['Environment_State'],
            mode='markers+lines',
            name='Phase Trajectory',
            line=dict(color='purple', width=1),
            marker=dict(
                color=data.index,
                colorscale='Viridis',
                size=4,
                colorbar=dict(title="Time Step")
            )
        )
    )
    
    # Add start and end points
    fig.add_trace(
        go.Scatter(
            x=[data['Coop_Fraction'].iloc[0]],
            y=[data['Environment_State'].iloc[0]],
            mode='markers',
            name='Start',
            marker=dict(color='green', size=10, symbol='star')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[data['Coop_Fraction'].iloc[-1]],
            y=[data['Environment_State'].iloc[-1]],
            mode='markers',
            name='End',
            marker=dict(color='red', size=10, symbol='x')
        )
    )
    
    fig.update_layout(
        title="Phase Space: Cooperation vs Environment State",
        xaxis_title="Cooperation Fraction",
        yaxis_title="Environment State (n)",
        height=500
    )
    
    solara.FigurePlotly(fig)

@solara.component
def SpatialSummary():
    """Summary of spatial patterns"""
    if model_data.value is None or model_data.value.empty:
        return
    
    data = model_data.value
    
    with solara.Card("Spatial Pattern Indicators"):
        if 'Local_Coop_Variance' in data.columns:
            final_variance = data['Local_Coop_Variance'].iloc[-1]
            avg_variance = data['Local_Coop_Variance'].mean()
            
            solara.Text(f"**Final Spatial Cooperation Variance:** {final_variance:.4f}")
            solara.Text(f"**Average Spatial Variance:** {avg_variance:.4f}")
            solara.Text("Higher values indicate stronger spatial clustering/patterns")
        else:
            solara.Text("Add 'Local_Coop_Variance' to model reporters to see spatial patterns")

@solara.component
def DataTable():
    """Component to display raw data table"""
    if model_data.value is None or model_data.value.empty:
        return
    
    with solara.Card("Recent Simulation Data"):
        # Show last 10 rows of data
        display_data = model_data.value.tail(10).round(3)
        solara.DataFrame(display_data)

@solara.component
def SummaryStats():
    """Enhanced summary statistics with environmental feedback analysis"""
    if model_data.value is None or model_data.value.empty:
        return
    
    data = model_data.value
    
    with solara.Card("Summary Statistics"):
        with solara.Columns([1, 1]):
            with solara.Column():
                if 'Total_Water' in data.columns:
                    solara.Text(f"**Final Water Level:** {data['Total_Water'].iloc[-1]:.2f}")
                if 'Total_Water_Capacity' in data.columns:
                    solara.Text(f"**Water Capacity:** {data['Total_Water_Capacity'].iloc[-1]:.2f}")
                if 'Environment_State' in data.columns:
                    solara.Text(f"**Final Environment State:** {data['Environment_State'].iloc[-1]:.3f}")
            
            with solara.Column():
                if 'Coop_Fraction' in data.columns:
                    final_coop_rate = data['Coop_Fraction'].iloc[-1] * 100
                    avg_coop_rate = data['Coop_Fraction'].mean() * 100
                    solara.Text(f"**Final Cooperation Rate:** {final_coop_rate:.1f}%")
                    solara.Text(f"**Average Cooperation Rate:** {avg_coop_rate:.1f}%")
                    solara.Text(f"**Cooperation Threshold (1/θ):** {(1/theta.value):.3f}")

@solara.component
def Page():
    """Main page component with enhanced layout"""
    solara.Title("WaterToC Mesa Model - Weitz et al. Implementation")
    
    with solara.Sidebar():
        ParameterControls()
        SimulationControls()
        SummaryStats()
        SpatialSummary()
    
    # Main content area
    with solara.Column():
        TimeSeriesPlots()
        PhaseSpacePlot()
        DataTable()

# Run the app
if __name__ == "__main__":
    # You can run this with: solara run server.py
    Page()