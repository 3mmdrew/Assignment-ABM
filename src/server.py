"""
TO BE WRITTEN
"""
import mesa
from mesa.visualization import SolaraViz
import solara
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import math

# Import the model and agents
from models.water_competition_model import WaterCompetitionModel
from agents.human_agent import HumanAgent
from agents.ai_agent import AIAgent

# SOLARA COMPONENTS can be added like the following so as to show headers/params/graphs etc...
@solara.component
def PlotlySpaceComponent(model):
    """Plotly-based space visualization replacing Mesa's make_space_component"""
    from mesa.visualization.utils import update_counter
    
    # Register for model updates
    update_counter.get()
    
    # Calculate responsive height based on grid aspect ratio
    grid_aspect_ratio = model.height / model.width
    base_width = 800  # Base width for calculations
    calculated_height = base_width * grid_aspect_ratio
    
    # Ensure minimum height for visibility
    responsive_height = max(calculated_height, 400)
    
    # Calculate card dimensions (1.5x the plotly dimensions)
    card_width = base_width 
    card_height = responsive_height 
    
    # Create the main scatter plot
    fig = go.Figure()
    
    # Add water grid visualization as a heatmap
    water_levels = np.array([[cell.current_water for cell in row] for row in model.water_grid])
    max_water = np.max(water_levels)
    
    # Create a heatmap for water levels with blue color and transparency
    fig.add_trace(go.Heatmap(
        z=water_levels,
        colorscale=[[0, 'rgba(0,0,255,0)'], [0.2, 'rgba(0,0,255,0.2)'], 
                   [0.4, 'rgba(0,0,255,0.4)'], [0.6, 'rgba(0,0,255,0.6)'],
                   [0.8, 'rgba(0,0,255,0.8)'], [1, 'rgba(0,0,255,1)']],
        showscale=False,
        hoverinfo='z',
        hovertemplate='Water Level: %{z:.2f}<extra></extra>',
        zmin=0,
        zmax=max_water
    ))
    
    # Collect and categorize agents based on original agent_portrayal logic
    human_agents = [agent for agent in model.agents if isinstance(agent, HumanAgent)]
    ai_agents = [agent for agent in model.agents if isinstance(agent, AIAgent)]
    
    # Add human agents with styling matching original agent_portrayal
    if human_agents:
        # Categorize humans based on original logic
        unsatisfied = [a for a in human_agents if not a.water_satisfied]
        high_coop = [a for a in human_agents if a.water_satisfied and a.cooperation_level > 0.7]
        med_coop = [a for a in human_agents if a.water_satisfied and 0.4 < a.cooperation_level <= 0.7]
        low_coop = [a for a in human_agents if a.water_satisfied and a.cooperation_level <= 0.4]
        
        # Unsatisfied humans (red, larger)
        if unsatisfied:
            fig.add_trace(go.Scatter(
                x=[a.pos[0] for a in unsatisfied],
                y=[a.pos[1] for a in unsatisfied],
                mode='markers',
                marker=dict(size=12, color='red', opacity=0.8, 
                           line=dict(width=2, color='black')),
                name='Unsatisfied',
                hovertemplate='<b>Unsatisfied Human</b><br>ID: %{text}<br>Position: (%{x}, %{y})<extra></extra>',
                text=[str(a.unique_id) for a in unsatisfied]
            ))
        
        # High cooperation humans (green)
        if high_coop:
            fig.add_trace(go.Scatter(
                x=[a.pos[0] for a in high_coop],
                y=[a.pos[1] for a in high_coop],
                mode='markers',
                marker=dict(size=10, color='green', opacity=0.8),
                name='High Cooperation',
                hovertemplate='<b>High Cooperation Human</b><br>ID: %{text}<br>Position: (%{x}, %{y})<extra></extra>',
                text=[str(a.unique_id) for a in high_coop]
            ))
        
        # Medium cooperation humans (orange)
        if med_coop:
            fig.add_trace(go.Scatter(
                x=[a.pos[0] for a in med_coop],
                y=[a.pos[1] for a in med_coop],
                mode='markers',
                marker=dict(size=10, color='orange', opacity=0.8),
                name='Med Cooperation',
                hovertemplate='<b>Medium Cooperation Human</b><br>ID: %{text}<br>Position: (%{x}, %{y})<extra></extra>',
                text=[str(a.unique_id) for a in med_coop]
            ))
        
        # Low cooperation humans (dark red)
        if low_coop:
            fig.add_trace(go.Scatter(
                x=[a.pos[0] for a in low_coop],
                y=[a.pos[1] for a in low_coop],
                mode='markers',
                marker=dict(size=10, color='darkred', opacity=0.8),
                name='Low Cooperation',
                hovertemplate='<b>Low Cooperation Human</b><br>ID: %{text}<br>Position: (%{x}, %{y})<extra></extra>',
                text=[str(a.unique_id) for a in low_coop]
            ))
    
    # Add AI agents (blue squares)
    if ai_agents:
        fig.add_trace(go.Scatter(
            x=[a.pos[0] for a in ai_agents],
            y=[a.pos[1] for a in ai_agents],
            mode='markers',
            marker=dict(size=8, color='purple', symbol='square', opacity=0.9,
                       line=dict(width=1, color='navy')),
            name='AI Agent',
            hovertemplate='<b>AI Agent</b><br>ID: %{text}<br>Position: (%{x}, %{y})<extra></extra>',
            text=[str(a.unique_id) for a in ai_agents]
        ))
    
    # Add data center (large purple square)
    fig.add_trace(go.Scatter(
        x=[model.dc_pos[0]],
        y=[model.dc_pos[1]],
        mode='markers',
        marker=dict(size=20, color='purple', symbol='square', opacity=0.8,
                   line=dict(width=2, color='black')),
        name='Data Center',
        hovertemplate='<b>Data Center</b><br>Position: (%{x}, %{y})<br>Water Reserve: %{text}<extra></extra>',
        text=[f"{model.data_center.water_reserve:.1f}"]
    ))
    
    # Add influence radius circle (dashed purple circle)
    theta = np.linspace(0, 2*np.pi, 100)
    radius_x = model.dc_pos[0] + model.dc_influence_radius * np.cos(theta)
    radius_y = model.dc_pos[1] + model.dc_influence_radius * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=radius_x,
        y=radius_y,
        mode='lines',
        line=dict(color='purple', width=2, dash='dash'),
        name='DC Influence',
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Customize layout with proper margins and maintain aspect ratio
    fig.update_layout(
        title=dict(
            text=f"Water Competition Grid",
            font=dict(size=16, weight='bold')
        ),
        xaxis=dict(
            title='X Coordinate',
            range=[-2, model.width + 2],
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            title='Y Coordinate', 
            range=[-2, model.height + 2],
            gridcolor='lightgray',
            gridwidth=1,
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='#f8f8f8',
        paper_bgcolor='white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        ),
        width=card_width,  # Use 4x card width
        height=card_height,  # Use 4x card height  
        margin=dict(l=60, r=40, t=60, b=60),  # Restored proper margins for readability
        autosize=False  # Disable autosize to use explicit dimensions
    )
    
    # Use WebGL for better performance with many agents
    if len(model.agents) > 100:
        for trace in fig.data:
            if hasattr(trace, 'mode') and trace.mode == 'markers':
                trace.update()
    
    solara.FigurePlotly(fig)

@solara.component
def MainVisualizationLayout(model):
    """Main layout component with maximum grid space and no header"""
    
    # Calculate card dimensions based on model grid
    grid_aspect_ratio = model.height / model.width
    base_width = 800
    calculated_height = base_width * grid_aspect_ratio
    responsive_height = max(calculated_height, 400)
    card_width = base_width * 4
    card_height = responsive_height * 4
    
    # Main container with full viewport and no header
    with solara.Column(style="width: 100vw; height: 100vh; margin: 0; padding: 0px; display: flex; flex-direction: column; overflow: auto;"):
        # Grid takes all available space with explicit 4x dimensions
        with solara.Column(style=f"width: {card_width}px; height: {card_height}px; margin: 0 auto; padding: 0; overflow: auto;"):
            # Direct plot component with maximum space
            PlotlySpaceComponent(model)


def create_water_competition_visualization():
    """Create the Mesa 3.x SolaraViz visualization with Plotly components"""
    
    # Model parameters that can be adjusted via controls (toolbar will be minimal and on left by default)
    model_params = {
        "num_humans": {
            "type": "SliderInt",
            "value": 200,
            "label": "Humans",
            "min": 50,
            "max": 500,
            "step": 25
        },
        "num_ai_agents": {
            "type": "SliderInt",
            "value": 20,
            "label": "AI Agents",
            "min": 1,
            "max": 100,
            "step": 1
        },
        "dc_influence_radius": {
            "type": "SliderInt",
            "value": 25,
            "label": "DC Radius",
            "min": 10,
            "max": 50,
            "step": 5
        },
        "ai_impact_coeff": {
            "type": "SliderFloat",
            "value": 0.1,
            "label": "AI Impact",
            "min": 0.01,
            "max": 0.5,
            "step": 0.01
        },
        # Fixed parameters (not adjustable in UI)
        "dc_pos": (50, 50),  # Fixed to center position
        "width": 100,
        "height": 100
    }
    
    # Create initial model instance
    model = WaterCompetitionModel(
        width=100,
        height=100,
        num_humans=200,
        num_ai_agents=10,
        dc_influence_radius=25,
        ai_impact_coeff=0.1,
        dc_pos=(50, 50)
    )
    
    # Use the main layout component
    components = [MainVisualizationLayout]
    
    # Create the SolaraViz page
    page = SolaraViz(
        model,
        components=components,
        model_params=model_params,
        name="Water Competition Model",
        play_interval=500,  # Milliseconds between steps when playing
    )
    
    return page


def print_visualization_guide():
    """Print usage guide for the visualization"""
    print("Human-AI Water Competition Model Visualization")
    print("=" * 60)
    print("\nVisualization Guide:")
    print("- Green circles: Highly cooperative humans")
    print("- Orange circles: Moderately cooperative humans") 
    print("- Red circles: Unsatisfied/uncooperative humans")
    print("- Dark red circles: Low cooperation humans")
    print("- Blue squares: AI agents")
    print("- Purple square: Data center")
    print("- Dashed purple circle: Data center influence radius")
    print("\nControls:")
    print("- Use sliders to adjust model parameters (toolbar on left)")
    print("- Click 'Reset' to restart with new parameters")
    print("- Click 'Step' for single steps or 'Play' for continuous run")
    print("- Charts temporarily hidden - grid is now central focus")
    print("- Hover over agents for details")
    print("- Interactive zoom and pan on grid")
    print("\nFor Jupyter Notebook:")
    print("- Simply call create_water_competition_visualization()")
    print("- The visualization will display inline")
    print("\nFor Standalone Web App:")
    print("- Save this file and run: solara run server.py")
    print("- Navigate to http://localhost:8765/")


# Create the visualization page
page = create_water_competition_visualization()

# For Jupyter notebook usage
def launch_in_notebook():
    """Launch visualization in Jupyter notebook"""
    print_visualization_guide()
    return page

# For standalone web application
if __name__ == "__main__":
    print_visualization_guide()
    print("\nStarting Solara server...")
    print("Note: Run with 'solara run server.py' for full web interface")
    
    # Display the page
    page