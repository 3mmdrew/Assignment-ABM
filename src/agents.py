from mesa import Agent
from enum import Enum
from pydantic import BaseModel
import random
import numpy as np


# STRATEGY ENUM
class Strategy(str, Enum):
    COOPERATE = "C"
    DEFECT = "D"

# BaseAgent Class to be inherited by both Humans and AI
class BaseAgent(Agent):
    def __init__(self, unique_id, model, pos=None):
        super().__init__(unique_id, model)
        self.pos = pos

    
    @property
    def COOPERATE(self):
        return Strategy.COOPERATE

    @property
    def DEFECT(self):
        return Strategy.DEFECT
    
    def set_game(self,R,S,T,P):
        self.game = np.array([[R,S],[T,P]])
        return True

    def weitz_matrix_env(self, n):
        """
        Environment-dependent payoff matrix from Weitz et al. 2016
        A(n) = (1-n) * [[T, P], [R, S]] + n * [[R, S], [T, P]]
    
        Args:
            n: environmental state (0 = degraded, 1 = pristine)
        
        Returns:
            env_matrix: 2x2 numpy array representing the environment-dependent payoff matrix
        """
    # Read payoffs correctly from the matrix
        R = self.game[0][0]  # Reward for mutual cooperation
        S = self.game[0][1]  # Sucker's payoff
        T = self.game[1][0]  # Temptation to defect
        P = self.game[1][1]  # Punishment for mutual defection
    
    # Weitz formula: A(n) = (1-n) * degraded_matrix + n * pristine_matrix
        degraded_matrix = np.array([[T, P], [R, S]])
        pristine_matrix = np.array([[R, S], [T, P]])
    
    # Linear interpolation between degraded and pristine environments
        env_matrix = (1 - n) * degraded_matrix + n * pristine_matrix
    
        return env_matrix

    def choose_best_action(self,env_game):
        """
        Chooses the best action (COOPERATE or DEFECT) based on the sum of payoffs in the weitz env game matrix.
        Returns:
            strategy_name (str): 'COOPERATE' or 'DEFECT'
        """
        row_sums = env_game.sum(axis=1)
        best_row = int(np.argmax(row_sums))
        strategy = self.COOPERATE if best_row == 0 else self.DEFECT
        return strategy
    
    def get_neighbors(self, radius=1, include_center=False):
        """Unified neighbor detection"""
        if self.pos is None:
            return []
        return self.model.grid.get_neighbors(self.pos, moore=True, include_center=include_center, radius=radius)

class Human(BaseAgent):
    def __init__(self, unique_id, model, strategy: Strategy):
        super().__init__(unique_id, model, pos=None)
        self.strategy = strategy
        self.C_Payoff = model.C_Payoff
        self.D_Payoff = model.D_Payoff
        
        # Store planned action for simultaneous activation
        self.planned_action = None
        self.target_pos = None

    def step(self):
        """Pick a random nearby water cell and plan action based on it"""
        water_positions = self.model.get_water_positions_near(self.pos)

        if not water_positions:
            return  # No water nearby

    # Pick a random water cell
        self.target_pos = self.model.random.choice(water_positions)

    # Compute n for the selected water cell
        water_level = self.model.get_water_at(self.target_pos)
        max_capacity = self.model.water_capacity[self.target_pos[0], self.target_pos[1]]
        n = water_level / max_capacity if max_capacity > 0 else 0

    # Compute strategy from Weitz matrix
        env_game = self.weitz_matrix_env(n)
        self.strategy = self.choose_best_action(env_game)
        self.planned_action = self.strategy

    def advance(self):
        """Execute the planned action"""
        if self.planned_action and self.target_pos:
            if self.planned_action == self.COOPERATE:
                self.model.consume_water_at(self.target_pos, self.C_Payoff)
            else:
                self.model.consume_water_at(self.target_pos, self.D_Payoff)
        
        # Reset for next step
        self.planned_action = None
        self.target_pos = None


class AI(BaseAgent):
    def __init__(self, unique_id, model, strategy: Strategy):
        super().__init__(unique_id, model, pos=None)
        self.strategy = strategy
        self.C_Payoff = model.C_Payoff
        self.D_Payoff = model.D_Payoff

        self.planned_action = None
        self.target_pos = None

    def step(self):
        """Pick a random nearby water cell and plan action based on it"""
        water_positions = self.model.get_water_positions_near(self.pos)

        if not water_positions:
            return  # No water nearby

    # Pick a random water cell
        self.target_pos = self.model.random.choice(water_positions)

    # Compute n for the selected water cell
        water_level = self.model.get_water_at(self.target_pos)
        max_capacity = self.model.water_capacity[self.target_pos[0], self.target_pos[1]]
        n = water_level / max_capacity if max_capacity > 0 else 0

    # Compute strategy from Weitz matrix
        env_game = self.weitz_matrix_env(n)
        self.strategy = self.choose_best_action(env_game)
        self.planned_action = self.strategy


    def advance(self):
        """Execute the planned action"""
        if self.planned_action and self.target_pos:
            if self.planned_action == self.COOPERATE:
                self.model.consume_water_at(self.target_pos, self.C_Payoff)
            else:
                self.model.consume_water_at(self.target_pos, self.D_Payoff)

        self.planned_action = None
        self.target_pos = None