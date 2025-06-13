from mesa import Agent

class HumanAgent(Agent):
    """Human agent that consumes water and adjusts cooperation level - Mesa 3.x compatible"""
    
    def __init__(self, model, pos=None, unique_id=None, **kwargs):
        super().__init__(model)

        # self.pos will be set by grid.place_agent()
        self.cooperation_level = 1.0  # Start fully cooperative
        self.water_satisfied = True
        self.memory = []  # Store recent water availability
        self.movement_state = "stationary"  # or "searching"
        self.water_need = 5  # units per timestep
        self.original_pos = None  # Will be set when agent is placed
        
        # Decision parameters
        self.cooperation_increase = 0.1
        self.cooperation_decrease = 0.15
        self.memory_length = 5
        self.satisfaction_threshold = 0.8
        
    def get_moore_neighborhood(self):
        """Get Moore neighborhood cells (8-connected)"""
        if not hasattr(self, 'pos') or self.pos is None:
            return []  # Return empty list if agent has no position for soft failure
        
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False, radius=1
        )
        return neighborhood
    
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
        """Attempt to consume water from local neighborhood"""
        neighborhood = self.get_moore_neighborhood()
        remaining_need = self.water_need
        consumed = 0
        
        # Try to consume from neighborhood cells
        for cell_pos in neighborhood:
            if remaining_need <= 0:
                break
            water_cell = self.model.water_grid[cell_pos[0]][cell_pos[1]]
            consumption = water_cell.consume_water(remaining_need)
            consumed += consumption
            remaining_need -= consumption
        
        # Update satisfaction
        satisfaction_ratio = consumed / self.water_need
        self.water_satisfied = satisfaction_ratio >= self.satisfaction_threshold
        
        # Update memory
        self.memory.append(satisfaction_ratio)
        if len(self.memory) > self.memory_length:
            self.memory.pop(0)
            
        return consumed
    
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
        """Decide whether and where to move"""
        if not hasattr(self, 'pos') or self.pos is None:
            return  # Can't move if agent has no position
            
        if self.movement_state == "searching":
            # Look for better water availability in extended neighborhood
            possible_moves = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False, radius=2
            )
            
            best_pos = None
            best_water = 0
            
            for candidate_pos in possible_moves:
                # Check if position is empty
                if self.model.grid.is_cell_empty(candidate_pos):
                    # Evaluate water availability around candidate position
                    candidate_neighborhood = self.model.grid.get_neighborhood(
                        candidate_pos, moore=True, include_center=False, radius=1
                    )
                    
                    total_water = sum(
                        self.model.water_grid[pos[0]][pos[1]].current_water 
                        for pos in candidate_neighborhood
                    )
                    
                    if total_water > best_water:
                        best_water = total_water
                        best_pos = candidate_pos
            
            # Move to better position if found
            if best_pos and best_water > 0:
                self.model.grid.move_agent(self, best_pos)
                
            # Check if original position is now good
            elif self.original_pos and self.pos != self.original_pos:
                original_neighborhood = self.model.grid.get_neighborhood(
                    self.original_pos, moore=True, include_center=False, radius=1
                )
                original_water = sum(
                    self.model.water_grid[pos[0]][pos[1]].current_water 
                    for pos in original_neighborhood
                )
                
                # Return to original with some probability if it's good again
                if original_water > self.water_need and self.model.random.random() < 0.3:
                    if self.model.grid.is_cell_empty(self.original_pos):
                        self.model.grid.move_agent(self, self.original_pos)
    
    def step(self):
        """Main step function called each timestep"""
        # Skip if agent doesn't have a position yet
        if not hasattr(self, 'pos') or self.pos is None:
            return
            
        # Assess and consume water
        consumed = self.consume_water()
        
        # Update cooperation based on satisfaction
        self.update_cooperation_level()
        
        # Decide on movement
        self.decide_movement()