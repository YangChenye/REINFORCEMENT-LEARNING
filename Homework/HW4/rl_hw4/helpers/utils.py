import numpy as np
def action_evaluation(env, gamma, v):
    '''
    Convert V value to Q value with model.
    
    Inputs:
    env: OpenAI Gym environment
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    gamma: float
            Discount value
    v: numpy ndarray
            Values of states
            
    Outputs:
    q: numpy ndarray
            Q values of all state-action pairs
    '''
    
    nS = env.nS
    nA = env.nA
    q = np.zeros((nS, nA))
    P=env.P
    for s in range(nS):
        for a in range(nA):
            ############################
            # YOUR CODE STARTS HERE
            #[probability, nextstate, reward, terminal]=P[s][a]
            q_s_a = 0
            for i in range(len(P[s][a])):
                next_state_tuple = P[s][a][i]
                v_next_state = v[next_state_tuple[1]]
                p_next_state = next_state_tuple[0]
                reward_next_state = next_state_tuple[2]
                q_s_a+=  p_next_state*(reward_next_state + gamma*v_next_state)
            q[s][a] = q_s_a
            # YOUR CODE ENDS HERE
            ############################
    return q

def action_selection(q):
    '''
    Select action from the Q value
    
    Inputs:
    q: numpy ndarray
    
    Outputs:
    actions: int
            The chosen action of each state
    '''

    actions = np.argmax(q, axis = 1)    
    return actions 

def extract_policy(env, v, gamma):
    
    """ Extract the optimal policy given the optimal value-function 
    Parameters:
    ----------
    env: OpenAI env.

    v: np.ndarray
        value function

    gamma: float
        Discount factor. Number in range [0, 1)
    Returns:
    ----------
    policy: np.ndarray
    """
    
    policy = np.zeros(env.nS, dtype=int)

    ############################
    # YOUR CODE STARTS HERE
    q = action_evaluation(env, gamma, v)
    policy = action_selection(q)
    # YOUR CODE ENDS HERE
    ############################
    return policy