# model.py
import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from agents import WaterConflictAgent

# Environment-dependent payoff matrix from Weitz et al.
def weitz_matrix_env(n, T=2.0, R=1.5, S=0.5, P=0.1):
    a11 = T - (T - R) * n
    a12 = P - (P - S) * n
    a21 = R + (T - R) * n
    a22 = S + (P - S) * n
    return np.array([[a11, a12], [a21, a22]])

class WaterConflictModel(Model):
    def __init__(self, width=50, height=50, num_humans=100, num_ais=50,
                 water_patch_ratio=0.3, initial_coop_rate=0.5, theta=1.0):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = SimultaneousActivation(self)
        self.global_env = 1.0  # starts pristine
        self.theta = theta

        # Initialize water endowment grid
        self.water_grid = np.zeros((width, height))
        num_water_cells = int(water_patch_ratio * width * height)
        water_indices = self.random.sample(range(width * height), num_water_cells)
        for idx in water_indices:
            x = idx % width
            y = idx // width
            self.water_grid[x][y] = self.random.uniform(0.5, 1.0)

        # Create agents
        agent_id = 0
        for _ in range(num_humans):
            x, y = self.random.randrange(width), self.random.randrange(height)
            strategy = "C" if self.random.random() < initial_coop_rate else "D"
            agent = WaterConflictAgent(agent_id, self, agent_type="Human", strategy=strategy)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)
            agent_id += 1

        for _ in range(num_ais):
            x, y = self.random.randrange(width), self.random.randrange(height)
            strategy = "C" if self.random.random() < initial_coop_rate else "D"
            agent = WaterConflictAgent(agent_id, self, agent_type="AI", strategy=strategy)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)
            agent_id += 1

        self.total_extracted = 0

        self.datacollector = DataCollector({
            "GlobalEnv": lambda m: m.global_env,
            "CooperationRate": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "C") / len(m.schedule.agents),
            "DefectorCount": lambda m: sum(1 for a in m.schedule.agents if a.strategy == "D"),
            "TotalWaterExtracted": lambda m: m.total_extracted,
            "AvgWater": lambda m: np.mean(m.water_grid),
        })

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.total_extracted = 0
        self.schedule.step()
        self.update_environment_feedback()
        self.datacollector.collect(self)

    def update_environment_feedback(self):
        # Update environment state based on cooperation rate
        x = sum(1 for a in self.schedule.agents if a.strategy == "C") / len(self.schedule.agents)
        delta_n = self.theta * (x - 0.5)
        self.global_env += delta_n
        self.global_env = np.clip(self.global_env, 0, 1)

        # Compute environment-dependent payoff matrix
        payoff_matrix = weitz_matrix_env(self.global_env)

        # Assign fitness (expected payoff) for replicator dynamics
        for agent in self.schedule.agents:
            neighbors = self.grid.get_neighbors(agent.pos, moore=True, include_center=False)
            if not neighbors:
                agent.payoff = 0
                continue
            coop_neighbors = sum(1 for n in neighbors if n.strategy == "C") / len(neighbors)
            defect_neighbors = 1 - coop_neighbors
            if agent.strategy == "C":
                agent.payoff = payoff_matrix[0][0] * coop_neighbors + payoff_matrix[0][1] * defect_neighbors
            else:
                agent.payoff = payoff_matrix[1][0] * coop_neighbors + payoff_matrix[1][1] * defect_neighbors

        # Replicator dynamics update
        avg_payoff = sum(a.payoff for a in self.schedule.agents) / len(self.schedule.agents)
        for agent in self.schedule.agents:
            p_C = agent.payoff if agent.strategy == "C" else 0
            p_D = agent.payoff if agent.strategy == "D" else 0
            if agent.strategy == "C":
                change = agent.strategy == "C" and p_C * (1 - p_C / avg_payoff) > 0.01
                if self.random.random() < p_C * (1 - p_C / avg_payoff):
                    agent.strategy = "D"
            else:
                if self.random.random() < p_D * (1 - p_D / avg_payoff):
                    agent.strategy = "C"

        # Water extraction phase
        for agent in self.schedule.agents:
            x, y = agent.pos
            closest = None
            min_dist = float("inf")

            for i in range(self.width):
                for j in range(self.height):
                    if self.water_grid[i][j] > 0:
                        dist = abs(x - i) + abs(y - j)
                        if dist < min_dist:
                            min_dist = dist
                            closest = (i, j)

            if closest:
                cx, cy = closest
                amount = 0.02 if agent.strategy == "C" else 0.1
                extracted = min(amount, self.water_grid[cx][cy])
                self.water_grid[cx][cy] -= extracted
                self.total_extracted += extracted
                self.global_env = max(0, self.global_env - extracted * 0.005)
