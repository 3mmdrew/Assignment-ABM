"""
Solara-based visualization for the Weitz-style Environmental Feedback PD Model.
"""
from model import PdGridEnvModel  # your own model
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

from mesa.visualization.solara import (
    Slider,
    SolaraViz,
    make_plot_component,
    make_space_component,
)



def pd_agent_portrayal(agent):
    """
    Portrayal function for rendering PD agents in the visualization.
    Blue = Cooperate, Red = Defect. Environment sets alpha/opacity.
    """
    color = "blue" if agent.move == "C" else "red"
    # Optional: make environment visible by changing transparency
    alpha = agent.cell.env if hasattr(agent.cell, "env") else 1.0

    return {
        "color": color,
        "marker": "s",  # square marker
        "size": 25,
        "alpha": alpha
    }


# Model parameters
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "width": Slider("Grid Width", value=50, min=10, max=100, step=1),
    "height": Slider("Grid Height", value=50, min=10, max=100, step=1),
    "theta": Slider("Environmental Feedback θ", value=1.0, min=0.1, max=5.0, step=0.1),
    "activation_order": {
        "type": "Select",
        "value": "Random",
        "values": PdGridEnvModel.activation_regimes,
        "label": "Activation Regime",
    },
}


# Visualize agents on a grid
grid_viz = make_space_component(agent_portrayal=pd_agent_portrayal)

# Time series plot of number of cooperators
plot_component = make_plot_component("Cooperators")

# Initialize model
initial_model = PdGridEnvModel()

# Create Solara app
page = SolaraViz(
    model=initial_model,
    components=[grid_viz, plot_component],
    model_params=model_params,
    name="Environmental Feedback PD (Weitz-style)",
)
page
