"""
STRANGER THINGS XODIA - SUBMISSION TEMPLATE
============================================
Team: Code Monk

TASK: Create 3 separate RL agents (one per universe)
- Universe 1: Find Will (6 Demogorgons)
- Universe 2: Escape Room (5 Guards)
- Universe 3: Fight Vecna (6 Demogorgons + Vecna)
"""

import numpy as np
import random


# UNIVERSE 1: FIND WILL

def discretize_universe1(observation):
    """
    Discretize state for Universe 1 using relative distances.
    """
    agent_x, agent_y = observation[0], observation[1]
    will_x, will_y = observation[2], observation[3]
    demo_x, demo_y = observation[4], observation[5]
    
    # Calculate relative distances instead of absolute map coordinates
    rel_will_x = int(will_x - agent_x)
    rel_will_y = int(will_y - agent_y)
    rel_demo_x = int(demo_x - agent_x)
    rel_demo_y = int(demo_y - agent_y)
    
    # State is now where Will and the Demogorgon are relative to the agent
    state = (rel_will_x, rel_will_y, rel_demo_x, rel_demo_y)
    
    return state


def create_universe1_agent():
    """Create Q-learning agent for Universe 1"""
    Q1 = {}
    
    # Hyperparameters
    alpha = 0.1      # Learning rate: How much new info overrides old info
    gamma = 0.95     # Discount factor: Importance of future rewards
    epsilon = 0.1    # Exploration rate: 10% chance to take a random move
    
    # Helper function to safely get or create Q-values for a new state
    def get_q_values(state):
        if state not in Q1:
            Q1[state] = [0.0, 0.0, 0.0, 0.0]  # 4 actions: Right, Up, Left, Down
        return Q1[state]
    
    def select_action(observation):
        # First, convert the raw observation into our simplified state
        state = discretize_universe1(observation)
        
        # Epsilon-greedy: Explore or Exploit?
        if random.random() < epsilon:
            # EXPLORE: Take a random action
            return random.randint(0, 3) 
        else:
            # EXPLOIT: Pick the action with the highest Q-value for this state
            q_values = get_q_values(state)
            return int(np.argmax(q_values))
    
    def update(state, action, reward, next_state):
        # Fetch current and future Q-values
        q_values = get_q_values(state)
        next_q_values = get_q_values(next_state)
        
        # Find the max possible reward for the next state
        best_next_q = max(next_q_values)
        
        # Q-learning Bellman Update
        q_values[action] = q_values[action] + alpha * (reward + gamma * best_next_q - q_values[action])
        
        # Save back to the Q-table (Optional since lists are mutable, but good practice)
        Q1[state] = q_values
        
    return select_action, update, Q1


# UNIVERSE 2: ESCAPE ROOM

# UNIVERSE 2: ESCAPE ROOM

def discretize_universe2(observation):
    """
    Discretize state for Universe 2 using relative distances.
    """
    agent_x, agent_y = observation[0], observation[1]
    exit_x, exit_y = observation[2], observation[3]
    guard_x, guard_y = observation[4], observation[5]
    
    # Calculate relative distances for the Exit and the Guard
    rel_exit_x = int(exit_x - agent_x)
    rel_exit_y = int(exit_y - agent_y)
    
    rel_guard_x = int(guard_x - agent_x)
    rel_guard_y = int(guard_y - agent_y)
    
    # The state tells the agent exactly where the goal and the danger are
    state = (rel_exit_x, rel_exit_y, rel_guard_x, rel_guard_y)
    
    return state


def create_universe2_agent():
    """Create Q-learning agent for Universe 2"""
    Q2 = {}
    
    # Hyperparameters
    alpha = 0.1      # Learning rate
    gamma = 0.95     # Discount factor (cares about long-term goal of escaping)
    epsilon = 0.1    # Exploration rate
    
    # Helper to initialize states
    def get_q_values(state):
        if state not in Q2:
            Q2[state] = [0.0, 0.0, 0.0, 0.0]  # Right, Up, Left, Down
        return Q2[state]
    
    def select_action(observation):
        state = discretize_universe2(observation)
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            return random.randint(0, 3)  # Explore
        else:
            q_values = get_q_values(state)
            return int(np.argmax(q_values))  # Exploit best known action
    
    def update(state, action, reward, next_state):
        q_values = get_q_values(state)
        next_q_values = get_q_values(next_state)
        
        # Q-learning update logic
        best_next_q = max(next_q_values)
        q_values[action] = q_values[action] + alpha * (reward + gamma * best_next_q - q_values[action])
        
        Q2[state] = q_values
        
    return select_action, update, Q2

# UNIVERSE 3: FIGHT VECNA

def discretize_universe3(observation):
    """
    Discretize state for Universe 3 using relative distances and rescue flags.
    """
    agent_x, agent_y = observation[0], observation[1]
    c1_x, c1_y = observation[2], observation[3]
    c2_x, c2_y = observation[4], observation[5]
    vecna_x, vecna_y = observation[6], observation[7]
    child1_rescued = int(observation[8])
    child2_rescued = int(observation[9])
    proj_x, proj_y = observation[10], observation[11]
    
    # Calculate relative distances for Vecna and the Projectile
    rel_vecna_x = int(vecna_x - agent_x)
    rel_vecna_y = int(vecna_y - agent_y)
    rel_proj_x = int(proj_x - agent_x)
    rel_proj_y = int(proj_y - agent_y)
    
    # OPTIMIZATION TRICK: 
    # Only calculate relative distance for a child if they haven't been rescued yet.
    # If they are rescued, lock their distance to (0,0) so the AI stops worrying about them.
    if child1_rescued == 0:
        rel_c1_x = int(c1_x - agent_x)
        rel_c1_y = int(c1_y - agent_y)
    else:
        rel_c1_x, rel_c1_y = 0, 0
        
    if child2_rescued == 0:
        rel_c2_x = int(c2_x - agent_x)
        rel_c2_y = int(c2_y - agent_y)
    else:
        rel_c2_x, rel_c2_y = 0, 0

    # The new state includes all relative distances PLUS the rescue flags
    state = (
        rel_c1_x, rel_c1_y, 
        rel_c2_x, rel_c2_y, 
        rel_vecna_x, rel_vecna_y, 
        rel_proj_x, rel_proj_y, 
        child1_rescued, child2_rescued
    )
    
    return state


def create_universe3_agent():
    """Create Q-learning agent for Universe 3"""
    Q3 = {}
    
    # Hyperparameters
    alpha = 0.1      # Learning rate
    gamma = 0.95     # Discount factor
    epsilon = 0.1    # Exploration rate
    
    def get_q_values(state):
        if state not in Q3:
            Q3[state] = [0.0, 0.0, 0.0, 0.0]  # Right, Up, Left, Down
        return Q3[state]
    
    def select_action(observation):
        state = discretize_universe3(observation)
        
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            q_values = get_q_values(state)
            return int(np.argmax(q_values))
    
    def update(state, action, reward, next_state):
        q_values = get_q_values(state)
        next_q_values = get_q_values(next_state)
        
        best_next_q = max(next_q_values)
        q_values[action] = q_values[action] + alpha * (reward + gamma * best_next_q - q_values[action])
        
        Q3[state] = q_values
        
    return select_action, update, Q3


# REQUIRED FUNCTIONS

def get_agent_for_universe(universe_name):
    """Router function - DO NOT MODIFY"""
    if "FIND WILL" in universe_name:
        return create_universe1_agent(), discretize_universe1
    elif "ESCAPE" in universe_name:
        return create_universe2_agent(), discretize_universe2
    elif "VECNA" in universe_name or "FIGHT" in universe_name:
        return create_universe3_agent(), discretize_universe3
    else:
        raise ValueError(f"Unknown universe: {universe_name}")
