class DataCenter:
    """Manages AI agents and tracks total water consumption"""
    
    def __init__(self, pos, initial_water_reserve=1000):
        self.pos = pos
        self.water_reserve = initial_water_reserve
        self.total_consumption = 0
        self.ai_agents = []
        
    def add_ai_agent(self, ai_agent):
        """Add AI agent to data center"""
        self.ai_agents.append(ai_agent)
    
    def calculate_total_consumption(self):
        """Calculate total water consumption by all AI agents"""
        self.total_consumption = sum(
            agent.calculate_water_consumption() 
            for agent in self.ai_agents
        )
        return self.total_consumption
    
    def consume_water(self):
        """Consume water from reserve"""
        consumption = self.calculate_total_consumption()
        if self.water_reserve >= consumption:
            self.water_reserve -= consumption
            return True  # Water available
        else:
            self.water_reserve = 0
            return False  # Water constrained
