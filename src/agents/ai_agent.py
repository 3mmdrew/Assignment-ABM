from mesa import Agent
import random

class AIAgent(Agent):
    """AI agent that consumes water through data center - Mesa 3.x compatible"""
    
    def __init__(self, model, activity_level=1.0, unique_id=None, **kwargs):
        super().__init__(model)
        
        if unique_id is not None:
            self.unique_id = unique_id
            
        self.activity_level = activity_level
        self.water_consumption_rate = 0.5  # Base water consumption per step
        self.growth_rate = 0.02
        self.max_activity = 10.0
        
        # Water consumption tracking
        self.total_water_consumed = 0.0
        self.current_step_consumption = 0.0
        self.efficiency_factor = random.uniform(0.8, 1.2)  # Individual efficiency variation
        
    def get_position(self):
        """Helper method to safely get position"""
        return getattr(self, 'pos', None)
    
    def get_neighbors(self, radius=1):
        """Get neighboring agents using Mesa's space methods"""
        if self.pos is not None:
            neighbors = self.model.grid.get_neighbors(
                self.pos, 
                moore=True,  # Include diagonal neighbors
                include_center=False,
                radius=radius
            )
            return neighbors
        return []
    
    def move_randomly(self):
        """Move to a random adjacent cell"""
        if hasattr(self.model, 'space') and self.pos is not None:
            possible_steps = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=False
            )
            new_position = self.random.choice(possible_steps)
            self.model.space.move_agent(self, new_position)
    
    def calculate_water_consumption(self):
        """Calculate water consumption based on activity level and neighbors"""
        base_consumption = self.water_consumption_rate * self.activity_level
        
        # Factor in efficiency
        consumption = base_consumption * self.efficiency_factor
        
        # Neighbor effect - more neighbors = more data processing = more water
        neighbors = self.get_neighbors()
        neighbor_factor = 1 + (len(neighbors) * 0.1)
        consumption *= neighbor_factor
        
        # Environmental factor (simulating cooling needs)
        if hasattr(self.model, 'environmental_factor'):
            consumption *= self.model.environmental_factor
        
        return consumption
    
    def update_activity(self):
        """Update activity level with some randomness and growth"""
        # Random fluctuation
        fluctuation = random.uniform(-0.1, 0.1)
        
        # Growth component
        growth = self.activity_level * self.growth_rate
        
        # Update with bounds
        self.activity_level = max(0.1, min(self.max_activity, 
                                         self.activity_level + growth + fluctuation))
    
    def consume_water(self):
        """Execute water consumption for this step"""
        self.current_step_consumption = self.calculate_water_consumption()
        self.total_water_consumed += self.current_step_consumption
        
        # Update model's total water consumption
        if hasattr(self.model, 'total_water_consumed'):
            self.model.total_water_consumed += self.current_step_consumption
    
    def step(self):
        """Main step function"""
        self.update_activity()
        self.consume_water()
        
        # Access position information
        current_pos = self.get_position()
        if current_pos:
            x, y = current_pos
            
            # Move if water consumption is high (seeking efficiency)
            if self.current_step_consumption > 2.0:
                self.move_randomly()
