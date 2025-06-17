"""
Simplified Water Competition Model Visualization Server
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

@solara.component
def PlotlySpaceComponent(model):
    from mesa.visualization.utils import update_counter
    
    update_counter.get()
    
    fig = go.Figure()
    
    # TRACE 0: Water heatmap (always present)
    water_levels = np.array([[cell.current_water for cell in row] for row in model.water_grid])
    max_water = np.max(water_levels) if np.max(water_levels) > 0 else 1
    
    fig.add_trace(go.Heatmap(
        z=water_levels,
        colorscale='Teal',
        showscale=True,
        colorbar=dict(title="Water"),
        hovertemplate='Water: %{z:.1f}<extra></extra>',
        zmin=0,
        zmax=model.max_water_capacity,
        name='Water Levels',
        showlegend=False,
        hoverinfo='text',
        hoverlabel=dict(bgcolor="white", font_size=12)
    ))
    
    # TRACE 1: ALL humans (always present, even if empty)
    human_agents = [agent for agent in model.agents if isinstance(agent, HumanAgent)]
    
    # Create arrays for humans
    human_x = [a.pos[0] for a in human_agents] if human_agents else []
    human_y = [a.pos[1] for a in human_agents] if human_agents else []
    human_colors = []
    human_texts = []
    human_cooperation = []
    
    for agent in human_agents:
        if not agent.water_satisfied:
            human_colors.append('red')
        elif agent.cooperation_level > 0.7:
            human_colors.append('green')
        elif agent.cooperation_level > 0.4:
            human_colors.append('orange')
        else:
            human_colors.append('darkred')
        human_texts.append(str(agent.unique_id))
        human_cooperation.append(agent.cooperation_level)
    
    fig.add_trace(go.Scatter(
        x=human_x,
        y=human_y,
        mode='markers',
        marker=dict(size=10, color=human_colors, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
        name='Human Agents',
        hovertemplate='<b>Human</b><br>ID: %{text}<br>Cooperation: %{customdata:.2f}<extra></extra>',
        text=human_texts,
        customdata=human_cooperation,
        visible=True if human_agents else 'legendonly',
        showlegend=False,
        hoverinfo='text',
        hoverlabel=dict(bgcolor="white", font_size=12)
    ))
    
    # TRACE 2: ALL AI agents (always present, even if empty)
    ai_agents = [agent for agent in model.agents if isinstance(agent, AIAgent)]
    
    fig.add_trace(go.Scatter(
        x=[a.pos[0] for a in ai_agents] if ai_agents else [],
        y=[a.pos[1] for a in ai_agents] if ai_agents else [],
        mode='markers',
        marker=dict(size=12, color='purple', symbol='square', opacity=0.9, line=dict(width=1, color='DarkSlateGrey')),
        name='AI Agents',
        hovertemplate='<b>AI Agent</b><br>ID: %{text}<extra></extra>',
        text=[str(a.unique_id) for a in ai_agents] if ai_agents else [],
        visible=True if ai_agents else 'legendonly',
        showlegend=False,
        hoverinfo='text',
        hoverlabel=dict(bgcolor="white", font_size=12)
    ))
    
    # Update layout with proper configuration for event handling
    fig.update_layout(
        title=f"Water Competition - Step {model.steps}",
        xaxis=dict(range=[0, model.width], title='X'),
        yaxis=dict(range=[0, model.height], title='Y', scaleanchor="x"),
        width=max(800, model.width * 4),
        height=max(650, model.height * 3),
        showlegend=False,
        # Add configuration to prevent event handling issues
        hovermode='closest',
        clickmode='event',
        dragmode='pan'
    )
    
    solara.FigurePlotly(fig)

@solara.component
def SimpleMetricsComponent(model):
    """Simple metrics display for the model with integrated legend"""
    from mesa.visualization.utils import update_counter
    
    # Register for model updates
    update_counter.get()
    
    # Calculate basic metrics
    human_agents = [agent for agent in model.agents if isinstance(agent, HumanAgent)]
    ai_agents = [agent for agent in model.agents if isinstance(agent, AIAgent)]
    
    unsatisfied_humans = sum(1 for a in human_agents if not a.water_satisfied)
    avg_cooperation = np.mean([a.cooperation_level for a in human_agents]) if human_agents else 0
    total_water = sum(sum(cell.current_water for cell in row) for row in model.water_grid)
    avg_ai_activity = np.mean([a.activity_level for a in ai_agents]) if ai_agents else 0
    
    # Display metrics and legend in a layout
    with solara.Column():
        # Legend section
        with solara.Row():
            solara.Text("Legend:", style="font-weight: bold; font-size: 16px;")
            solara.Text("🟢 Human Agent (High Satisfaction); 🟠 Human Agent (Medium Satisfaction); 🔴 Human Agent (Low Satisfaction); 🟣 AI Agent; 💧 Water Levels")

        # Metrics section
        with solara.Row():
            solara.Text(f"Step: {model.steps}", style="font-weight: bold; font-size: 18px;")
            solara.Text(f"Total Water: {total_water:.1f}; Humans: {len(human_agents)}; Unsatisfied: {unsatisfied_humans}; AI Agents: {len(ai_agents)}; Avg Cooperation: {avg_cooperation:.2f}")
           
        

@solara.component
def MainVisualizationLayout(model):
    """Main layout component with metrics and grid"""
    
    # Main container
    with solara.Column(style="width: 100%; height: 100vh; margin: 0;"):

        with solara.Row():
            SimpleMetricsComponent(model)
        with solara.Row(style="flex: 1;"):
            PlotlySpaceComponent(model)
        


def create_water_competition_visualization(width,height):
    """Create the Mesa visualization with simplified parameters"""
    
    # Simplified model parameters
    model_params = {
        "num_humans": {
            "type": "SliderInt",
            "value": 200,
            "label": "Number of Humans",
            "min": 50,
            "max": 500,
            "step": 25
        },
        "num_ai_agents": {
            "type": "SliderInt",
            "value": 10,
            "label": "Number of AI Agents",
            "min": 1,
            "max": 50,
            "step": 1
        },
        # Fixed parameters (not adjustable in UI)
        "width": width,
        "height": height
    }
    
    # Create initial model instance with simplified parameters
    model = WaterCompetitionModel(
        width=width,
        height=height,
        num_humans=200,
        num_ai_agents=10,
        max_water_capacity= 25,
        base_water_replenishment= 0.01,
    )
    
    # Use the main layout component
    components = [MainVisualizationLayout]
    
    # Create the SolaraViz page
    page = SolaraViz(
        model,
        components=components,
        model_params=model_params,
        name="Simplified Water Competition Model",
        play_interval=500,  # Milliseconds between steps when playing
    )
    
    return page


def print_visualization_guide():
    """Print usage guide for the simplified visualization"""
    print("Human-AI Water Competition Model Visualization")
    print("=" * 60)
    print("\nSimplified Model Features:")
    print("- Direct competition between Human and AI agents for water")
    print("- Agents placed randomly across grid")
    print("- Natural water replenishment only")
    print("\nVisualization Guide:")
    print("- Green circles: Highly cooperative humans (satisfied)")
    print("- Orange circles: Moderately cooperative humans (satisfied)") 
    print("- Dark red circles: Low cooperation humans (satisfied)")
    print("- Red circles (larger): Unsatisfied humans")
    print("- Purple squares: AI agents")
    print("- Blue heatmap: Water levels across the grid")
    print("\nControls:")
    print("- Use sliders to adjust number of humans and AI agents")
    print("- Click 'Reset' to restart with new parameters")
    print("- Click 'Step' for single steps or 'Play' for continuous run")
    print("- Hover over agents for details")
    print("- Interactive zoom and pan on grid")
    print("\nMetrics Displayed:")
    print("- Current step number")
    print("- Total water available")
    print("- Number of human and AI agents")
    print("- Number of unsatisfied humans")
    print("- Average cooperation level")
    print("\nFor Jupyter Notebook:")
    print("- Simply call create_water_competition_visualization()")
    print("- The visualization will display inline")
    print("\nFor Standalone Web App:")
    print("- Save this file and run: solara run server.py")
    print("- Navigate to http://localhost:8765/")


# Create the visualization page
page = create_water_competition_visualization(width=100,height=100)

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