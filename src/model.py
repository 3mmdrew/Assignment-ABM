from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import random

from src.agents import AI, Human, Strategy

import math

class WaterCell:
    """Store water data directly in grid cell properties"""
    
    def __init__(self, width=20, height=20, **kwargs):
        super().__init__(**kwargs)
        self.grid = MultiGrid(width, height, True)
        
        # Initialize water data as grid properties
        self.water_levels = np.full((width, height), 10.0)  # Initial water
        self.water_capacity = np.full((width, height), 10.0)  # Max capacity
        self.replenishment_rates = np.random.uniform(0.1, 0.5, (width, height))
        
        # Track which cells have water
        self.water_cells = np.random.random((width, height)) < 0.3  # 30% have water
        
    def replenish_water(self):
        """Replenish water in all cells"""
        # Only replenish where there are water sources
        self.water_levels = np.where(
            self.water_cells,
            np.minimum(
                self.water_capacity,
                self.water_levels + self.replenishment_rates
            ),
            0  # No water in non-water cells
        )
    
    def get_water_at(self, pos):
        """Get water level at position"""
        x, y = pos
        return self.water_levels[x, y] if self.water_cells[x, y] else 0
    
    def consume_water_at(self, pos, amount):
        """Consume water at position"""
        x, y = pos
        if self.water_cells[x, y]:
            actual = min(amount, self.water_levels[x, y])
            self.water_levels[x, y] -= actual
            return actual
        return 0
    
    def step(self):
        self.replenish_water()
        self.agents.do("step")
        self.agents.do("advance")

class WaterToC(Model):
    """
    Water Tragedy of Commons Model
    """
    def __init__(self, 
                 height=20, 
                 width=20,
                 initial_humans=10, 
                 initial_ai=10,
                 C_Payoff=0.1,
                 D_Payoff=1,
                 initial_payoff_matrix=0.04, 
                 max_water_capacity=10,
                 water_cell_density=0.3,
                 theta=3,  # Add theta parameter
                 seed=None):
        '''
        Initialize the Water Tragedy of Commons model
        '''
        super().__init__(seed=seed)  # Required in Mesa 3.0

        self.height = height
        self.width = width
        self.initial_humans = initial_humans
        self.initial_ai = initial_ai
        self.initial_payoff_matrix = initial_payoff_matrix
        self.max_water_capacity = max_water_capacity
        self.water_cell_density = water_cell_density
        self.theta = theta  # Store theta
        self.C_Payoff = C_Payoff
        self.D_Payoff = D_Payoff

        # Initialize grid (no scheduler needed in Mesa 3.0)
        self.grid = MultiGrid(self.width, self.height, True)

        # Track running state
        self.running = True

        # Create water cells
        self._create_water_environment()
        
        # Create agents
        self._create_agents()

        # Data collection (add enhanced reporters)
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Water": self._get_total_water,
                "Human_Count": lambda m: len([a for a in m.agents if isinstance(a, Human)]),
                "AI_Count": lambda m: len([a for a in m.agents if isinstance(a, AI)]),
                "Cooperators": self._count_cooperators,
                "Defectors": self._count_defectors,
                "Total_Water_Capacity": self._get_total_water_capacity,
                "Environment_State": self._get_environment_state,
                "Coop_Fraction": self._get_coop_fraction,
                "theta": lambda m: m.theta,
                "Avg_Water_Per_Cell": self._get_avg_water_per_cell,
                "Local_Coop_Variance": lambda m: np.var(m._get_local_cooperation_map()),
            }
        )

    def _create_water_environment(self):
            """Create water environment using grid properties"""
            # Initialize arrays for water data
            self.water_levels = np.zeros((self.width, self.height))
            self.water_capacity = np.zeros((self.width, self.height))
            self.replenishment_rates = np.zeros((self.width, self.height))
            self.has_water = np.zeros((self.width, self.height), dtype=bool)
            
            # Randomly place water sources
            for x in range(self.width):
                for y in range(self.height):
                    if self.random.random() < self.water_cell_density:
                        self.has_water[x, y] = True
                        self.water_capacity[x, y] = self.max_water_capacity
                        self.water_levels[x, y] = self.max_water_capacity
                        self.replenishment_rates[x, y] = self.random.uniform(0.1, 0.5)

    def _create_agents(self):
        """Create human and AI agents"""
        
        # Create humans
        for _ in range(self.initial_humans):
            strategy = self.random.choice([Strategy.COOPERATE, Strategy.DEFECT])
            agent = Human(self, strategy)
            
            # Set game matrix for agent
            agent.set_game(R=3, S=0, T=5, P=1)  # Standard prisoner's dilemma
            
            # Place agent randomly
            pos = self._get_random_empty_cell()
            if pos:
                self.grid.place_agent(agent, pos)
                agent.pos = pos
                # No need for schedule.add() in Mesa 3.0 - automatic

        # Create AI agents
        for _ in range(self.initial_ai):
            strategy = self.random.choice([Strategy.COOPERATE, Strategy.DEFECT])
            agent = AI(self, strategy)
            
            # Set game matrix for agent
            agent.set_game(R=3, S=0, T=5, P=1)  # Standard prisoner's dilemma
            
            # Place agent randomly
            pos = self._get_random_empty_cell()
            if pos:
                self.grid.place_agent(agent, pos)
                agent.pos = pos
                # No need for schedule.add() in Mesa 3.0 - automatic

    def _get_random_empty_cell(self):
        """Find a random cell for agent placement"""
        attempts = 0
        while attempts < 100:
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            pos = (x, y)
            
            cell_contents = self.grid.get_cell_list_contents([pos])
            if len(cell_contents) < 3:  # Allow up to 3 agents per cell
                return pos
            attempts += 1
        return (0, 0)

    def replenish_water(self):
        """Replenish water with LOCAL environmental feedback"""
        for x in range(self.width):
            for y in range(self.height):
                if not self.has_water[x, y]:
                    continue
                # Count local cooperators and defectors around this water cell
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
                # Calculate local cooperation fraction
                local_total = local_cooperators + local_defectors
                if local_total == 0:
                    # No agents nearby - use base replenishment
                    effective_replenishment = self.replenishment_rates[x, y]
                else:
                    local_coop_fraction = local_cooperators / local_total
                    # Weitz-style feedback
                    cooperation_threshold = 1.0 / self.theta if self.theta > 0 else 0.5
                    local_feedback_strength = self.theta * local_coop_fraction - 1.0
                    current_n = (self.water_levels[x, y] / self.water_capacity[x, y] 
                                 if self.water_capacity[x, y] > 0 else 0)
                    env_capacity_factor = current_n * (1 - current_n)
                    local_feedback_term = 1.0 + 0.5 * env_capacity_factor * local_feedback_strength
                    local_feedback_term = np.clip(local_feedback_term, 0.1, 3.0)
                    effective_replenishment = self.replenishment_rates[x, y] * local_feedback_term
                # Update this specific water cell
                self.water_levels[x, y] = min(
                    self.water_capacity[x, y],
                    self.water_levels[x, y] + effective_replenishment
                )

    def get_water_at(self, pos):
        """Get water level at position"""
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height and self.has_water[x, y]:
            return self.water_levels[x, y]
        return 0

    def consume_water_at(self, pos, amount):
        """Consume water at position and return actual amount consumed"""
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height and self.has_water[x, y]:
            actual = min(amount, self.water_levels[x, y])
            self.water_levels[x, y] -= actual
            return actual
        return 0

    def add_water_at(self, pos, amount):
        """Add water at position (up to capacity)"""
        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height and self.has_water[x, y]:
            self.water_levels[x, y] = min(
                self.water_capacity[x, y],
                self.water_levels[x, y] + amount
            )

    def get_water_positions_near(self, pos, radius=1):
        """Get positions with water near a given position"""
        neighbors = self.grid.get_neighborhood(
            pos, moore=True, include_center=False, radius=radius
        )
        return [neighbor for neighbor in neighbors if self.get_water_at(neighbor) > 0]

    def step(self):
        """Advance the model by one step"""
        # Replenish water first
        self.replenish_water()
        
        # Activate agents (simultaneous activation)
        self.agents.do("step")
        self.agents.do("advance")
        
        # Collect data
        self.datacollector.collect(self)

    def _get_total_water(self):
        """Get total water across all water cells"""
        return np.sum(self.water_levels[self.has_water])

    def _count_cooperators(self):
        """Count agents currently using cooperate strategy"""
        return len([a for a in self.agents 
                   if hasattr(a, 'strategy') and a.strategy == Strategy.COOPERATE])

    def _count_defectors(self):
        """Count agents currently using defect strategy"""
        return len([a for a in self.agents 
                   if hasattr(a, 'strategy') and a.strategy == Strategy.DEFECT])

    def _get_total_water_capacity(self):
        """Get total water capacity across all water cells"""
        return np.sum(self.water_capacity[self.has_water])

    def _get_environment_state(self):
        """Get normalized environment state (0-1)"""
        total_capacity = self._get_total_water_capacity()
        total_water = self._get_total_water()
        return total_water / total_capacity if total_capacity > 0 else 0

    def _get_coop_fraction(self):
        """Get cooperation fraction across all water cells"""
        total_cooperators = self._count_cooperators()
        total_agents = self._count_cooperators() + self._count_defectors()
        return total_cooperators / total_agents if total_agents > 0 else 0

    def _get_avg_water_per_cell(self):
        """Get average water per water cell"""
        water_cells_count = np.sum(self.has_water)
        if water_cells_count > 0:
            return np.sum(self.water_levels[self.has_water]) / water_cells_count
        return 0

    def _get_local_cooperation_map(self):
        """Get a map of local cooperation rates for visualization"""
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
                    coop_map[x, y] = 0.5  # Neutral when no agents
        return coop_map