import math

class WaterCell:
    """Represents a water resource cell in the grid"""
    def __init__(self, pos, max_capacity=100, base_replenishment=1):
        self.pos = pos
        self.current_water = max_capacity
        self.max_capacity = max_capacity
        self.base_replenishment = base_replenishment
        self.replenishment_rate = base_replenishment
    
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