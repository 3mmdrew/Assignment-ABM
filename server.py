# server.py
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Slider

from model import WaterConflictModel
from agents import WaterConflictAgent

def composite_portrayal(agent):
    portrayals = []

    # Background cell (water)
    if agent is not None:
        x, y = agent.pos
        water = agent.model.water_grid[x][y]
        blue = int(255 * water)
        portrayals.append({
            "Shape": "rect",
            "Color": f"rgb(0,0,{blue})",
            "Filled": "true",
            "Layer": 0,
            "w": 1, "h": 1
        })

        # Agent on top
        color = "blue" if agent.agent_type == "Human" else "red"
        if agent.strategy == "D":
            color = "black"

        portrayals.append({
            "Shape": "circle",
            "Color": color,
            "Filled": True,
            "Layer": 1,
            "r": 0.5,
        })

    return portrayals

canvas = CanvasGrid(composite_portrayal, 50, 50, 500, 500)

chart = ChartModule([
    {"Label": "GlobalEnv", "Color": "green"},
    {"Label": "CooperationRate", "Color": "blue"},
    {"Label": "DefectorCount", "Color": "red"},
    {"Label": "TotalWaterExtracted", "Color": "gray"},
])

model_params = {
    "width": 50,
    "height": 50,
    "num_humans": Slider("# Humans", 100, 10, 300),
    "num_ais": Slider("# AI agents", 50, 10, 200),
    "water_patch_ratio": Slider("% Water Cells", 0.3, 0.0, 1.0, 0.05),
    "initial_coop_rate": Slider("Initial Coop Rate", 0.5, 0.0, 1.0, 0.05),
    "theta": Slider("Environment Feedback Strength", 1.0, 0.0, 2.0, 0.1),
}

server = ModularServer(
    WaterConflictModel,
    [canvas, chart],
    "AI-Human Water Conflict Model",
    model_params
)

server.port = 8524
server.launch()
