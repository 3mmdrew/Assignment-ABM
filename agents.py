# agents.py
from mesa import Agent
import numpy as np

class WaterConflictAgent(Agent):
    def __init__(self, unique_id, model, agent_type="Human", strategy=None):
        super().__init__(unique_id, model)
        self.agent_type = agent_type  # "Human" or "AI"
        self.strategy = strategy if strategy else self.random.choice(["C", "D"])
        self.payoff = 0

    def step(self):
        x, y = self.pos
        current_water = self.model.water_grid[x][y]

    # If the current cell has enough water, stay and extract
        if current_water > 0.01:
            amount = 0.02 if self.strategy == "C" else 0.1
            extracted = min(amount, self.model.water_grid[x][y])
            self.model.water_grid[x][y] -= extracted
            self.model.total_extracted += extracted
            self.model.global_env = max(0, self.model.global_env - extracted * 0.005)
            self.payoff = extracted
        else:
        # Move toward the nearest cell with water
            closest = None
            min_dist = float("inf")

        for i in range(self.model.grid.width):
            for j in range(self.model.grid.height):
                if self.model.water_grid[i][j] > 0.01:
                    dist = abs(x - i) + abs(y - j)
                    if dist < min_dist:
                        min_dist = dist
                        closest = (i, j)

        if closest:
            dx = np.sign(closest[0] - x)
            dy = np.sign(closest[1] - y)
            new_pos = (x + dx, y + dy)

            if self.model.grid.is_cell_empty(new_pos):
                self.model.grid.move_agent(self, new_pos)


        for i in range(self.model.grid.width):
            for j in range(self.model.grid.height):
                if self.model.water_grid[i][j] > 0:
                    dist = abs(x - i) + abs(y - j)
                    if dist < min_dist:
                        min_dist = dist
                        closest = (i, j)

        if closest:
            cx, cy = closest
            amount = 0.02 if self.strategy == "C" else 0.1
            self.model.water_grid[cx][cy] = max(0, self.model.water_grid[cx][cy] - amount)
            self.model.global_env = max(0, self.model.global_env - amount * 0.001)
            self.payoff = amount  # placeholder: could be based on payoff matrix

    def advance(self):
        # Basic imitation of best neighbor strategy (can be replaced with replicator dynamics)
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        best = max([self] + neighbors, key=lambda a: a.payoff)
        self.strategy = best.strategy
