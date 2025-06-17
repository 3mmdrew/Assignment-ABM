from mesa import Agent
import random

class HumanAgent(Agent):
    """Human agent that consumes water and adjusts cooperation level - Mesa 3.x compatible"""
    
    def __init__(self,model,water_needed=5,pos=None,unique_id=None,**kwargs):
        super().__init__(model)

        # self.pos will be set by grid.place_agent()
        self.cooperation_level = 1.0  # Start fully cooperative
        self.water_satisfied = True
        self.memory = []  # Store recent water availability
        self.movement_state = "stationary"  # or "searching"
        self.water_need = water_needed  # units per timestep
        self.original_pos = None  # Will be set when agent is placed
        
        # Decision parameters
        self.cooperation_increase = 0.1
        self.cooperation_decrease = 0.1
        self.memory_length = 5
        self.satisfaction_threshold = 0.75
    
    def assess_local_water(self):
        """Assess water availability in local neighborhood"""
        neighborhood = self.get_moore_neighborhood()
        total_local_water = 0
        available_cells = 0
        
        for cell_pos in neighborhood:
            water_cell = self.model.water_grid[cell_pos[0]][cell_pos[1]]
            if water_cell.current_water > 0:
                total_local_water += water_cell.current_water
                available_cells += 1
                
        return total_local_water, available_cells
    
    def consume_water(self):
        """Search through neighborhood cells in random order for water"""

        neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=1)

        # Shuffle neighborhood to get random order
        shuffled_neighborhood = list(neighborhood).copy()
        self.model.random.shuffle(shuffled_neighborhood)
        
        total_consumed = 0
        
        # Go through each cell in random order until we have enough water
        for cell_pos in shuffled_neighborhood:
            water_cell = self.model.water_grid[cell_pos[0]][cell_pos[1]]
            
            # Calculate how much more water we need
            remaining_need = self.water_need - total_consumed
            
            if remaining_need <= 0:
                break  # We have enough water
            
            # Consume water from this cell
            consumed_from_cell = water_cell.consume_water(remaining_need)
            total_consumed += consumed_from_cell
        
        # Now update satisfaction based on total consumption
        satisfaction_ratio = total_consumed / self.water_need
        self.water_satisfied = satisfaction_ratio >= self.satisfaction_threshold
        
        # Update memory
        self.memory.append(satisfaction_ratio)
        if len(self.memory) > self.memory_length:
            self.memory.pop(0)
            
        return total_consumed
    
    def update_cooperation_level(self):
        """Update cooperation level based on recent water satisfaction"""
        if self.water_satisfied:
            self.cooperation_level = min(1.0, 
                self.cooperation_level + self.cooperation_increase)
            self.movement_state = "stationary"
        else:
            self.cooperation_level = max(0.0, 
                self.cooperation_level - self.cooperation_decrease)
            self.movement_state = "searching"
    
    def decide_movement(self):
        """Decide whether and where to move - only if unsatisfied"""
        if not hasattr(self, 'pos') or self.pos is None:
            return  # Can't move if agent has no position
        
        # Only allow movement if agent is unsatisfied
        if not self.water_satisfied:
            # Look for next move 
            possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            new_pos = random.choice(possible_moves)
            self.model.grid.move_agent(self,new_pos)
    
    def step(self):
        """Main step function called each timestep"""
        # Skip if agent doesn't have a position yet
        if not hasattr(self, 'pos') or self.pos is None:
            return
            
        # Assess and consume water
        consumed = self.consume_water()
        
        # Update cooperation and movement state based on satisfaction
        self.update_cooperation_level()
        
        # Only decide on movement if we're still unsatisfied after all updates
        if not self.water_satisfied:
            self.decide_movement()