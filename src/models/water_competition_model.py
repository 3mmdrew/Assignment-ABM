from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import math

# Import the agent classes
from agents.human_agent import HumanAgent
from agents.ai_agent import AIAgent
from environment.data_center import DataCenter
from environment.water_cell import WaterCell

class WaterCompetitionModel(Model):
    """Main model class for human-AI water competition - Mesa 3.x compatible"""
    
    def __init__(self, width=100, height=100, num_humans=200, num_ai_agents=10,
                 dc_pos=(50, 50), dc_influence_radius=25, ai_impact_coeff=0.1, 
                 seed=None, **kwargs):
        # CRITICAL: Must call super().__init__(seed=seed) for Mesa 3.x
        super().__init__(seed=seed)
        
        # Model parameters
        self.width = width
        self.height = height
        self.num_humans = num_humans
        self.num_ai_agents = num_ai_agents
        self.dc_pos = dc_pos
        self.dc_influence_radius = dc_influence_radius
        self.ai_impact_coeff = ai_impact_coeff
        
        # Initialize grid (no scheduler needed in Mesa 3.x)
        self.grid = MultiGrid(width, height, True)
        
        # Initialize water grid
        self.initialize_water_grid()
        
        # Initialize data center
        self.data_center = DataCenter(dc_pos)
        self.dc_water_available = True
        
        # Create agents (they automatically get added to self.agents)
        self.create_human_agents()
        self.create_ai_agents()
        
        # Data collection - updated for Mesa 3.x
        self.datacollector = DataCollector(
            model_reporters={
                "Total_AI_Consumption": lambda m: m.data_center.total_consumption,
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
                "Humans_Near_DC": lambda m: sum(
                    1 for agent in m.agents 
                    if isinstance(agent, HumanAgent) and 
                    hasattr(agent, 'pos') and agent.pos is not None and
                    math.sqrt((agent.pos[0] - m.dc_pos[0])**2 + 
                             (agent.pos[1] - m.dc_pos[1])**2) <= m.dc_influence_radius
                ),
                "Humans_Near_DC_Ratio": lambda m: (
                    sum(1 for agent in m.agents 
                        if isinstance(agent, HumanAgent) and 
                        hasattr(agent, 'pos') and agent.pos is not None and
                        math.sqrt((agent.pos[0] - m.dc_pos[0])**2 + 
                                 (agent.pos[1] - m.dc_pos[1])**2) <= m.dc_influence_radius) /
                    max(1, sum(1 for agent in m.agents if isinstance(agent, HumanAgent)))
                ),
                "DC_Water_Reserve": lambda m: m.data_center.water_reserve
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
    
    def initialize_water_grid(self):
        """Initialize grid with water cells using random clustering"""
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
        
        # Initialize water grid with clustered values
        self.water_grid = []
        for x in range(self.width):
            row = []
            for y in range(self.height):
                water_cell = WaterCell((x, y), max_capacity=100, base_replenishment=0.01)
                water_cell.current_water = water_levels[x][y]
                water_cell.calculate_distance_to_dc(self.dc_pos)
                row.append(water_cell)
            self.water_grid.append(row)
    
    def create_human_agents(self):
        """Create and place human agents"""
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
                        human = HumanAgent(model=self)
                        
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
        """Create AI agents and place them near the data center"""
        dc_x, dc_y = self.dc_pos
        
        for i in range(self.num_ai_agents):
            ai_agent = AIAgent(model=self, activity_level=1.0)
            
            # Place AI agents randomly within a small radius of the data center
            angle = self.random.random() * 2 * math.pi
            distance = self.random.random() * (self.dc_influence_radius)  # Within the influence radius
            x = int(dc_x + distance * math.cos(angle))
            y = int(dc_y + distance * math.sin(angle))
            
            # Ensure position is within grid bounds
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            
            # Place AI agent on the grid
            self.grid.place_agent(ai_agent, (x, y))
            
            # Verify position was set correctly
            assert hasattr(ai_agent, 'pos'), f"AI agent {ai_agent.unique_id} missing pos attribute"
            assert ai_agent.pos == (x, y), f"AI agent {ai_agent.unique_id} pos mismatch"
            
            # Also add to data center for water consumption tracking
            self.data_center.add_ai_agent(ai_agent)
    
    def update_water_system(self):
        """Update all water cells based on AI consumption"""
        total_ai_consumption = self.data_center.calculate_total_consumption()
        
        # Update water availability at data center
        self.dc_water_available = self.data_center.consume_water()
        
        # Update replenishment rates for all water cells
        for row in self.water_grid:
            for water_cell in row:
                water_cell.update_replenishment_rate(
                    total_ai_consumption, 
                    self.dc_influence_radius, 
                    self.ai_impact_coeff
                )
                
        # Replenish water in all cells
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
        """Reset the model's random number generator (Mesa 3.x feature)"""
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