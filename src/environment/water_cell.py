import math

class WaterCell:
    """Represents a water resource cell in the grid"""
    def __init__(self, pos, max_capacity=100, base_replenishment=1):
        self.pos = pos
        self.current_water = max_capacity
        self.max_capacity = max_capacity
        self.base_replenishment = base_replenishment
        self.replenishment_rate = base_replenishment
        self.distance_to_dc = None
        
    def calculate_distance_to_dc(self, dc_pos):
        """Calculate Euclidean distance to data center"""
        self.distance_to_dc = math.sqrt(
            (self.pos[0] - dc_pos[0])**2 + (self.pos[1] - dc_pos[1])**2
        )
    
    def update_replenishment_rate(self, total_ai_consumption, dc_influence_radius, ai_impact_coeff):
        """Update replenishment rate based on AI activity and distance"""
        if self.distance_to_dc <= dc_influence_radius:
            distance_effect = max(0, 1 - (self.distance_to_dc / dc_influence_radius))
            ai_suppression = ai_impact_coeff * total_ai_consumption * distance_effect
            self.replenishment_rate = self.base_replenishment * (1 - ai_suppression)
        else:
            self.replenishment_rate = self.base_replenishment
    
    def replenish_water(self):
        """Add water based on current replenishment rate"""
        self.current_water = min(
            self.max_capacity, 
            self.current_water + max(0, self.replenishment_rate)
        )
    
    def consume_water(self, amount):
        """Consume water and return actual amount consumed"""
        actual_consumption = min(amount, self.current_water)
        self.current_water -= actual_consumption
        return actual_consumption