# Enhanced model.py with deviation parameter
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler
import numpy as np
import random
import math
from src.agents import AI, Human, Strategy

"""
Defines the main `WaterToC` model class for the agent-based simulation.

This module enables the simnulation of cooperation dynamics between Human and AI
agents who share a spatially-explicit, replenishable (water) resource. The model manages
the environment, agent scheduling, and the core simulation loop.

A key feature is the environmental feedback mechanism, where the collective behavior
of agents (cooperation vs. defection) directly impacts the replenishment rate of the
water resource in their local area. The model uses Mesa's `DataCollector` to track
a wide range of metrics for analysis.
"""

class WaterToC(Model):
    """
    An agent-based model exploring cooperation dynamics in a shared resource environment.
    Agents (humans and AI) decide to cooperate or defect in harvesting a replenishable water resource.
    The model incorporates environmental feedback, where cooperation levels influence the resource's replenishment rate.
    """
    def __init__(self,
                 height=20,
                 width=20,
                 initial_humans=50,
                 initial_ai=50,
                 human_C_allocation=0.1,
                 human_D_allocation=0.15,
                 ai_C_allocation=2,
                 ai_D_allocation=3,
                 max_water_capacity=20,
                 water_cell_density=0.3,
                 theta=3,
                 deviation_rate=0.1,
                 seed=None):
        super().__init__(seed=seed)
        self.height = height
        self.width = width
        self.initial_humans = initial_humans
        self.initial_ai = initial_ai
        self.max_water_capacity = max_water_capacity
        self.water_cell_density = water_cell_density
        self.theta = theta  
        self.deviation_rate = deviation_rate  

        self.human_C_allocation = human_C_allocation
        self.human_D_allocation = human_D_allocation
        self.ai_C_allocation = ai_C_allocation
        self.ai_D_allocation = ai_D_allocation

        self.grid = MultiGrid(self.width, self.height, True)
        self.schedule = BaseScheduler(self)
        self.agents = []
        self.running = True

        self._create_water_environment()
        self._create_agents()

        # setup data collection
        reporters = {
            "Total_Water": self._get_total_water,
            "Total_Water_Capacity": self._get_total_water_capacity,
            "Environment_State": self._get_environment_state,
            "Human_Count": lambda m: len([a for a in m.agents if isinstance(a, Human)]),
            "AI_Count": lambda m: len([a for a in m.agents if isinstance(a, AI)]),
            "Cooperators": self._count_cooperators,
            "Defectors": self._count_defectors,
            "Coop_Fraction": self._get_coop_fraction,
            "Human_Cooperators": self._count_human_cooperators,
            "Human_Coop_Fraction": self._get_human_coop_fraction,
            "AI_Cooperators": self._count_ai_cooperators,
            "AI_Coop_Fraction": self._get_ai_coop_fraction,
            "theta": lambda m: m.theta,
            "deviation_rate": lambda m: m.deviation_rate,  # track deviation rate
            "Avg_Water_Per_Cell": self._get_avg_water_per_cell,
            "Local_Coop_Variance": lambda m: np.var(m._get_local_cooperation_map()),
            "Optimal_Cooperators": self._count_optimal_cooperators,  # count agents that should cooperate optimally
            "Deviation_Rate_Actual": self._get_actual_deviation_rate,  # actual observed deviation rate
        }

        # sample k water cells for local water level reporting
        water_positions = list(zip(*np.where(self.has_water)))
        k = 3
        self.sample_cells = self.random.sample(water_positions, k=min(k, len(water_positions)))
        for i, (x, y) in enumerate(self.sample_cells):
            reporters[f"n_cell_{i}"] = (lambda m, x=x, y=y:
                                          m.water_levels[x, y] / m.water_capacity[x, y])

        self.datacollector = DataCollector(model_reporters=reporters)
        self.datacollector.collect(self)

    def _create_water_environment(self):
        """create the water resource grid."""
        # initialize grid arrays for water levels, capacity, etc.
        self.water_levels = np.zeros((self.width, self.height))
        self.water_capacity = np.zeros((self.width, self.height))
        self.replenishment_rates = np.zeros((self.width, self.height))
        self.has_water = np.zeros((self.width, self.height), dtype=bool)

        # randomly assign water to cells based on density
        for x in range(self.width):
            for y in range(self.height):
                if self.random.random() < self.water_cell_density:
                    self.has_water[x, y] = True
                    self.water_capacity[x, y] = self.max_water_capacity
                    self.water_levels[x, y] = self.max_water_capacity  # water cells start at full capacity
                    self.replenishment_rates[x, y] = self.random.uniform(0.1, 0.5)

    def _create_agents(self):
        """create and place human and AI agents."""
        for _ in range(self.initial_humans):
            strategy = self.random.choice([Strategy.COOPERATE, Strategy.DEFECT])
            agent = Human(self.next_id(), self, strategy)
            agent.set_game(R=3, S=0, T=5, P=1)
            agent.C_allocation = self.human_C_allocation
            agent.D_allocation = self.human_D_allocation
            pos = self._get_random_empty_cell()
            if pos:
                self.grid.place_agent(agent, pos)
                agent.pos = pos
                self.agents.append(agent)

        for _ in range(self.initial_ai):
            strategy = self.random.choice([Strategy.COOPERATE, Strategy.DEFECT])
            agent = AI(self.next_id(), self, strategy)
            agent.set_game(R=3, S=0, T=5, P=1)
            agent.C_allocation = self.ai_C_allocation
            agent.D_allocation = self.ai_D_allocation
            pos = self._get_random_empty_cell()
            if pos:
                self.grid.place_agent(agent, pos)
                agent.pos = pos
                self.agents.append(agent)

    def _get_random_empty_cell(self):
        """find a random cell for agent placement."""
        attempts = 0
        while attempts < 100:
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            pos = (x, y)
            cell_contents = self.grid.get_cell_list_contents([pos])
            if len(cell_contents) < 3:  # allow up to 3 agents per grid cell
                return pos
            attempts += 1
        return (0, 0)  # default if no empty cell is found

    def replenish_water(self):
        """replenish water based on local cooperation."""
        # calculate local cooperation around each water cell
        for x in range(self.width):
            for y in range(self.height):
                if not self.has_water[x, y]:
                    continue

                # count cooperators and defectors in the cell's neighborhood
                local_cooperators = 0
                local_defectors = 0
                neighbors = self.grid.get_neighbors(
                    (x, y), moore=True, radius=3, include_center=True
                )
                for agent in neighbors:
                    if hasattr(agent, 'strategy'):
                        if agent.strategy == Strategy.COOPERATE:
                            local_cooperators += 1
                        else:
                            local_defectors += 1

                # calculate the fraction of local cooperators
                local_total = local_cooperators + local_defectors
                if local_total == 0:
                    # if no agents, use the base replenishment rate
                    effective_replenishment = self.replenishment_rates[x, y]
                else:
                    local_coop_fraction = local_cooperators / local_total

                    # apply feedback based on local cooperation
                    cooperation_threshold = 1.0 / self.theta if self.theta > 0 else 0.5
                    local_feedback_strength = self.theta * local_coop_fraction - 1.0

                    # calculate the environmental capacity factor for the cell
                    current_n = (self.water_levels[x, y] / self.water_capacity[x, y]
                                 if self.water_capacity[x, y] > 0 else 0)
                    env_capacity_factor = current_n * (1 - current_n)

                    # calculate the local feedback term
                    local_feedback_term = 1.0 + 0.5 * env_capacity_factor * local_feedback_strength
                    local_feedback_term = np.clip(local_feedback_term, 0.1, 3.0)

                    # determine the effective replenishment rate
                    effective_replenishment = self.replenishment_rates[x, y] * local_feedback_term

                # update the cell's water level
                self.water_levels[x, y] = min(
                    self.water_capacity[x, y],
                    self.water_levels[x, y] + effective_replenishment
                )

    def _get_local_cooperation_map(self):
        """Create a map of local cooperation rates."""
        coop_map = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                neighbors = self.grid.get_neighbors(
                    (x, y), moore=True, radius=2, include_center=True
                )
                local_cooperators = 0
                local_total = 0
                for agent in neighbors:
                    if hasattr(agent, 'strategy'):
                        local_total += 1
                        if agent.strategy == Strategy.COOPERATE:
                            local_cooperators += 1

                if local_total > 0:
                    coop_map[x, y] = local_cooperators / local_total
                else:
                    coop_map[x, y] = 0.5  # neutral (0.5) if no agents are present
        return coop_map

    def get_water_at(self, pos):
        """Get water level at a specific position."""
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height and self.has_water[x, y]:
            return self.water_levels[x, y]
        return 0

    def consume_water_at(self, pos, amount):
        """Consume water at a position and return the amount consumed."""
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height and self.has_water[x, y]:
            actual = min(amount, self.water_levels[x, y])
            self.water_levels[x, y] -= actual
            return actual
        return 0

    def add_water_at(self, pos, amount):
        """Add water at a position, up to its capacity."""
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height and self.has_water[x, y]:
            self.water_levels[x, y] = min(
                self.water_capacity[x, y],
                self.water_levels[x, y] + amount
            )

    def get_water_positions_near(self, pos, radius=1):
        """Get positions with water near a given location."""
        neighbors = self.grid.get_neighborhood(
            pos, moore=True, include_center=False, radius=radius
        )
        return [neighbor for neighbor in neighbors if self.get_water_at(neighbor) > 0]

    def step(self):
        """Advance the model by one step."""
        # first, replenish the water
        self.replenish_water()

        # activate all agents to determine their next action
        for agent in self.agents:
            agent.step()
        # advance all agents to their new state
        for agent in self.agents:
            agent.advance()

        # collect data for this step
        self.datacollector.collect(self)
        # advance the step counter
        self.schedule.steps += 1

    # --- Data Collector Reporter Methods ---
    def _get_total_water(self):
        """Get total water across all water cells."""
        return np.sum(self.water_levels[self.has_water])

    def _get_total_water_capacity(self):
        """Get total water capacity across all water cells."""
        return np.sum(self.water_capacity[self.has_water])

    def _get_environment_state(self):
        """Get normalized environment state (0-1)."""
        total_capacity = self._get_total_water_capacity()
        total_water = self._get_total_water()
        return total_water / total_capacity if total_capacity > 0 else 0

    def _get_avg_water_per_cell(self):
        """Get average water per water-containing cell."""
        water_cells_count = np.sum(self.has_water)
        if water_cells_count > 0:
            return np.sum(self.water_levels[self.has_water]) / water_cells_count
        return 0

    def _count_cooperators(self):
        """Count all agents currently cooperating."""
        return len([a for a in self.agents
                    if hasattr(a, 'strategy') and a.strategy == Strategy.COOPERATE])

    def _count_defectors(self):
        """Count all agents currently defecting."""
        return len([a for a in self.agents
                    if hasattr(a, 'strategy') and a.strategy == Strategy.DEFECT])

    def _get_coop_fraction(self):
        """Get the overall fraction of cooperators."""
        coop = self._count_cooperators()
        total = len(self.agents)
        return coop / total if total > 0 else 0

    def _count_human_cooperators(self):
        """Count human agents that are cooperating."""
        return len([a for a in self.agents if isinstance(a, Human) and a.strategy == Strategy.COOPERATE])

    def _get_human_coop_fraction(self):
        """Get the fraction of humans who are cooperating."""
        cooperators = self._count_human_cooperators()
        total = len([a for a in self.agents if isinstance(a, Human)])
        return cooperators / total if total > 0 else 0

    def _count_ai_cooperators(self):
        """Count AI agents that are cooperating."""
        return len([a for a in self.agents if isinstance(a, AI) and a.strategy == Strategy.COOPERATE])

    def _get_ai_coop_fraction(self):
        """Get the fraction of AIs who are cooperating."""
        cooperators = self._count_ai_cooperators()
        total = len([a for a in self.agents if isinstance(a, AI)])
        return cooperators / total if total > 0 else 0

    def _count_optimal_cooperators(self):
        """Count agents whose optimal strategy, given their context, is to cooperate."""
        optimal_cooperators = 0
        for agent in self.agents:
            if not hasattr(agent, 'target_pos') or agent.target_pos is None:
                continue

            # get water level at the agent's target
            water_level = self.get_water_at(agent.target_pos)
            max_capacity = self.water_capacity[agent.target_pos[0], agent.target_pos[1]]
            n = water_level / max_capacity if max_capacity > 0 else 0

            # determine the optimal strategy based on the environment
            env_game = agent.weitz_matrix_env(n)
            optimal_strategy = agent.choose_best_action(env_game)

            if optimal_strategy == Strategy.COOPERATE:
                optimal_cooperators += 1
        return optimal_cooperators

    def _get_actual_deviation_rate(self):
        """Calculate the observed rate of agents deviating from their optimal strategy."""
        total_agents_with_target = 0
        deviating_agents = 0
        for agent in self.agents:
            if not hasattr(agent, 'target_pos') or agent.target_pos is None:
                continue

            total_agents_with_target += 1

            # get water level at the agent's target
            water_level = self.get_water_at(agent.target_pos)
            max_capacity = self.water_capacity[agent.target_pos[0], agent.target_pos[1]]
            n = water_level / max_capacity if max_capacity > 0 else 0

            # determine the optimal strategy
            env_game = agent.weitz_matrix_env(n)
            optimal_strategy = agent.choose_best_action(env_game)

            # check if the agent's current strategy is sub-optimal
            if agent.strategy != optimal_strategy:
                deviating_agents += 1

        return deviating_agents / total_agents_with_target if total_agents_with_target > 0 else 0