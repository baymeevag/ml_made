from utils import estimate_return, compute_policy, sample_action, random_move
from collections import defaultdict
import numpy as np

def q_learning(env, 
                crosses=True,
                n_episodes=200000, 
                eps=0.01, 
                gamma=1., 
                alpha=0.05, 
                logging_freq=10000, 
                estimate_episodes=100000):

    Q = defaultdict(lambda: np.zeros(env.n_rows * env.n_cols))
    rewards = []
    
    for i in range(1, n_episodes + 1):
        if not i % logging_freq:
            print(i)
            pi = compute_policy(Q)
            rewards.append(estimate_return(env, pi, n_episodes=estimate_episodes))
            
        env.reset()
        done = False
        s = env.getHash()
        a, a_int = None, None

        while not done:

            if crosses:
                # crosses: our move
                s = env.getHash()
                a = sample_action(env, Q, eps)
                a_int = env.int_from_action(a)
                s_prime, r, done, _ = env.step(a)

                if done:
                    # we won
                    Q[s][a_int] += alpha * (r - Q[s][a_int])
                    break

                # naughts: their move
                s_prime, r, done = random_move(env)
                if done:
                    # we lost
                    Q[s][a_int] += alpha * (-r - Q[s][a_int])
                    break

                Q[s][a_int] += alpha * (r + gamma * max(Q[s_prime[0]]) - Q[s][a_int])

            else:
                # crosses: their move
                s_prime, r, done = random_move(env)

                if done:
                    # we lost
                    Q[s][a_int] += alpha * (-r - Q[s][a_int])
                    break
                
                if a_int is not None:
                    Q[s][a_int] += alpha * (r + gamma * max(Q[s_prime[0]]) - Q[s][a_int])

                # naughts: our move
                #s = env.getHash()
                a = sample_action(env, Q, eps)
                a_int = env.int_from_action(a)
                s_prime, r, done, _ = env.step(a)

                if done:
                    # we won
                    Q[s][a_int] += alpha * (r - Q[s][a_int])
                    break

                s = env.getHash()
            
                

    return Q, rewards
