import time
def example(env):
    """Run an example of the game
    Parameters
    ----------
    env: OpenAI env.
    """
    env.seed(0)
    ob = env.reset()
    for t in range(100):
        env.render()
        a = env.action_space.sample()
        ob, rew, done, _ = env.step(a)
        if done:
            break
    assert done
    env.render()

def render_episode(env, policy):
    """Run one episode for given policy on environment. 
    Parameters
    ----------
    env: OpenAI env.
        
    Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(100):
        env.render()
        time.sleep(0.5)  
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    assert done
    env.render()
    print("Episode reward: %f" % episode_reward)

def avg_performance(env, policy):
    """ Evaluate the average rewards 
    Parameters
    ----------
    env: OpenAI env.
        
    Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    sum_reward = 0.
    episode = 100
    max_iteration = 6000
    for i in range(episode):
        done = False
        ob = env.reset()
        for j in range(max_iteration):
            a = policy[ob]
            ob, reward, done, _ = env.step(a)
            sum_reward += reward
            if done:
                break

    return sum_reward / i