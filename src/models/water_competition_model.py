from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import math

# Import the agent classes
from agents.human_agent import HumanAgent
from agents.ai_agent import AIAgent
from environment.water_cell import WaterCell

class WaterCompetitionModel(Model):
    """Simplified model class for human-AI water competition - Mesa 3.x compatible"""
    
    def __init__(self, 
                 width=100, 
                 height=100, 
                 num_humans=200, 
                 num_ai_agents=10, 
                 max_water_capacity=1, 
                 base_water_replenishment=0.1,
                 ai_consumption_rate=0.5,
                 human_water_needs=5, 
                 seed=None, 
                 **kwargs):
        # CRITICAL: Must call super().__init__(seed=seed) for Mesa 3.x
        super().__init__(seed=seed)
        
        # Model parameters
        self.width = width
        self.height = height
        self.num_humans = num_humans
        self.num_ai_agents = num_ai_agents
        self.max_water_capacity = max_water_capacity
        self.base_water_replenishment = base_water_replenishment

        self.ai_consumption_rate = ai_consumption_rate
        self.human_water_needs = human_water_needs
        
        # Initialize grid (no scheduler needed in Mesa 3.x)
        self.grid = MultiGrid(width, height, True)
        
        # Initialize water grid
        self.initialize_water_grid()
        
        # Create agents (they automatically get added to self.agents)
        self.create_human_agents()
        self.create_ai_agents()
        
        # Data collection - simplified for Mesa 3.x
        self.datacollector = DataCollector(
            model_reporters={
                "Average_Human_Cooperation": lambda m: np.mean([
                    agent.cooperation_level for agent in m.agents 
                    if isinstance(agent, HumanAgent)
                ]) if any(isinstance(agent, HumanAgent) for agent in m.agents) else 0,
                "Total_Water_Available": lambda m: sum(
                    sum(cell.current_water for cell in row) 
                    for row in m.water_grid
                ),
                "Unsatisfied_Humans": lambda m: sum(
                    1 for agent in m.agents 
                    if isinstance(agent, HumanAgent) and not agent.water_satisfied
                ),
                "Unsatisfied_Humans_Ratio": lambda m: (
                    sum(1 for agent in m.agents 
                        if isinstance(agent, HumanAgent) and not agent.water_satisfied) /
                    max(1, sum(1 for agent in m.agents if isinstance(agent, HumanAgent)))
                ),
                "Total_AI_Agents": lambda m: sum(
                    1 for agent in m.agents if isinstance(agent, AIAgent)
                ),
                "Total_Human_Agents": lambda m: sum(
                    1 for agent in m.agents if isinstance(agent, HumanAgent)
                ),
                "Average_AI_Activity": lambda m: np.mean([
                    agent.activity_level for agent in m.agents 
                    if isinstance(agent, AIAgent)
                ]) if any(isinstance(agent, AIAgent) for agent in m.agents) else 0
            },
            agent_reporters={
                "Cooperation": lambda a: a.cooperation_level if isinstance(a, HumanAgent) else None,
                "Water_Satisfied": lambda a: a.water_satisfied if isinstance(a, HumanAgent) else None,
                "Position": lambda a: a.pos if hasattr(a, 'pos') and a.pos is not None else None,
                "AI_Activity": lambda a: a.activity_level if isinstance(a, AIAgent) else None,
                "Agent_Type": lambda a: "Human" if isinstance(a, HumanAgent) else "AI"
            }
        )
        
        self.running = True
        
        # Initialize step counter (Mesa 3.x doesn't have schedule.steps)
        self.steps = 0
    
    def initialize_water_grid(self, use_clustering=False):
        """Initialize grid with water cells. By default, uses uniform distribution.
        
        Args:
            use_clustering (bool): If True, creates clustered water distribution.
                                 If False (default), creates uniform water distribution.
        """
        if use_clustering:
            # Create initial random water levels
            water_levels = np.zeros((self.width, self.height))
            
            # Create random clusters
            num_clusters = 10  # Number of water-rich clusters
            cluster_radius = 15  # Radius of influence for each cluster
            
            for _ in range(num_clusters):
                # Random cluster center
                center_x = self.random.randrange(self.width)
                center_y = self.random.randrange(self.height)
                
                # Add water to cells within cluster radius
                for x in range(max(0, center_x - cluster_radius), min(self.width, center_x + cluster_radius)):
                    for y in range(max(0, center_y - cluster_radius), min(self.height, center_y + cluster_radius)):
                        # Calculate distance from cluster center
                        distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if distance <= cluster_radius:
                            # Add water with decreasing amount based on distance
                            water_amount = 80 * (1 - distance/cluster_radius)
                            water_levels[x][y] += water_amount
            
            # Add some random noise to create variation
            noise = np.random.normal(0, 10, (self.width, self.height))
            water_levels = np.clip(water_levels + noise, 0, 100)
        else:
            # Create uniform water distribution with small random variations
            base_water = self.max_water_capacity  # Base water level for all cells
            variation = np.random.normal(0, 0, (self.width, self.height))  # Small random variations
            water_levels = np.clip(base_water + variation, 0, self.max_water_capacity+1)
        
        # Initialize water grid with values
        self.water_grid = []
        for x in range(self.width):
            row = []
            for y in range(self.height):
                water_cell = WaterCell((x, y), max_capacity=self.max_water_capacity,base_replenishment=self.base_water_replenishment)
                water_cell.current_water = water_levels[x][y]
                row.append(water_cell)
            self.water_grid.append(row)
    
    def create_human_agents(self):
        """Create and place human agents randomly"""
        for i in range(self.num_humans):
            # Find valid position with nearby water
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)
                
                # Check if position is empty and has water access
                if self.grid.is_cell_empty((x, y)):
                    neighborhood = self.grid.get_neighborhood(
                        (x, y), moore=True, include_center=False, radius=1
                    )
                    has_water_access = any(
                        self.water_grid[pos[0]][pos[1]].current_water > 0 
                        for pos in neighborhood
                    )
                    
                    if has_water_access:
                        # Create agent without position
                        human = HumanAgent(model=self,water_needed=self.human_water_needs)
                        
                        # Place agent on grid (this sets human.pos automatically)
                        self.grid.place_agent(human, (x, y))
                        
                        # Verify position was set correctly
                        assert hasattr(human, 'pos'), f"Human agent {human.unique_id} missing pos attribute"
                        assert human.pos == (x, y), f"Human agent {human.unique_id} pos mismatch"
                        
                        # Set original position after placement
                        human.original_pos = human.pos
                        break
                
                attempts += 1
            
            # If we couldn't find a valid position, place randomly as fallback
            if attempts >= 100:
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)
                human = HumanAgent(model=self)
                self.grid.place_agent(human, (x, y))
                human.original_pos = human.pos
                print(f"Warning: Human agent {human.unique_id} placed without water access verification")
    
    def create_ai_agents(self):
        """Create AI agents and place them randomly on the grid"""
        for i in range(self.num_ai_agents):
            ai_agent = AIAgent(model=self, activity_level=1.0, water_consumption_rate=self.ai_consumption_rate)
            
            # Place AI agents randomly on the grid
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            
            # Find an empty cell
            attempts = 0
            while not self.grid.is_cell_empty((x, y)) and attempts < 100:
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)
                attempts += 1
            
            # Place AI agent on the grid
            self.grid.place_agent(ai_agent, (x, y))
            
            # Verify position was set correctly
            assert hasattr(ai_agent, 'pos'), f"AI agent {ai_agent.unique_id} missing pos attribute"
            assert ai_agent.pos == (x, y), f"AI agent {ai_agent.unique_id} pos mismatch"
    
    def update_water_system(self):
        """Update all water cells - natural replenishment only"""
        # Simple natural replenishment for all cells
        for row in self.water_grid:
            for water_cell in row:
                water_cell.replenish_water()
    
    def step(self):
        """Single step of the model"""
        # Update water system first
        self.update_water_system()
        
        # Use shuffle_do for random activation (equivalent to RandomActivation)
        self.agents.shuffle_do("step")
        
        # Collect data
        self.datacollector.collect(self)
        
        # Increment step counter
        self.steps += 1
    
    def run_model(self, step_count=100):
        """Run the model for specified number of steps"""
        for i in range(step_count):
            self.step()
            
            # Optional: Add stopping conditions
            if self.check_stopping_conditions():
                break
    
    def check_stopping_conditions(self):
        """Check if model should stop (optional)"""
        human_agents = [agent for agent in self.agents 
                       if isinstance(agent, HumanAgent)]
        
        if human_agents:
            avg_cooperation = np.mean([agent.cooperation_level for agent in human_agents])
            if avg_cooperation < 0.1:  # Very low cooperation
                return True
                
        return False
    
    def reset_randomizer(self, seed=None):
        """Reset the model's random number generator"""
        super().reset_randomizer(seed)
        self.steps = 0
    
    def get_agent_positions(self):
        """Helper method to get all agent positions for debugging"""
        positions = {}
        for agent in self.agents:
            if hasattr(agent, 'pos') and agent.pos is not None:
                agent_type = "Human" if isinstance(agent, HumanAgent) else "AI"
                positions[agent.unique_id] = {
                    'type': agent_type,
                    'pos': agent.pos
                }
            else:
                positions[agent.unique_id] = {
                    'type': "Unknown",
                    'pos': None
                }
        return positions