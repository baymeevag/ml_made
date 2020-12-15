import numpy as np
import random
import torch

def sample_action(env, Q, eps=None):
    s = env.getHash()
    greedy_step = eps and random.random() < eps
    # select action in a greedy manner if eps is not None

    if s in Q and not greedy_step:
        a_int = np.argmax(Q[s])
        a = env.action_from_int(a_int)
        return np.array(a)

    actions = env.getEmptySpaces().tolist()
    return random.choice(actions)

def sample_action_dqn(model, env, eps=0.1):
    greedy_step = eps and random.random() < eps
    
    if greedy_step:
        actions = env.getEmptySpaces().tolist()
        return random.choice(actions)
    else:
        board = env.tensor_board()
        a_int = model(board).data[0].argmax()
        return env.action_from_int(a_int)

def compute_policy(Q):
    return {k: np.argmax(v) for k, v in Q.items()}

def random_move(env):
    actions = env.getEmptySpaces().tolist()
    a = random.choice(actions)
    s_prime, r, done, _ = env.step(a)
    return s_prime[0], r, done

def sample_from_policy(env, pi):
    s = env.getHash()

    if s in pi:
        return env.action_from_int(pi[s])
    else:
        actions = env.getEmptySpaces().tolist()
        return random.choice(actions)

def estimate_return(env, pi, crosses=True, n_episodes=20000):
    reward = 0.
    
    for _ in range(n_episodes):
        env.reset()
        done = False

        while not done:

            if crosses:
                # crosses: our move
                a = sample_from_policy(env, pi)
                _, r, done, _ = env.step(a)

                if done:
                    # we won
                    reward += r
                    break

                # naughts: their move
                _, r, done = random_move(env)

                if done:
                    # we lost
                    reward -= r
                    break
            else:
                # crosses: their move
                _, r, done = random_move(env)

                if done:
                    # we lost
                    reward -= r
                    break

                # naughts: our move
                a = sample_from_policy(env, pi)
                _, r, done, _ = env.step(a)

                if done:
                    # we won
                    reward += r
                    break
    
    return reward / n_episodes

def estimate_return_dqn(env, model, crosses=True, n_episodes=2000):
    
    reward = 0.

    for _ in range(n_episodes):
        env.reset()
        done = False

        while not done:

            if crosses:
                # crosses: our move
                board = env.tensor_board()
                a_int = model(board).data[0].argmax()
                a = env.action_from_int(a_int)
                _, r, done, _ = env.step(a)

                if done:
                    # we won
                    reward += r
                    break

                # naughts: their move
                _, r, done = random_move(env)

                if done:
                    # we lost
                    reward -= r
                    break
            else:
                # crosses: their move
                _, r, done = random_move(env)

                if done:
                    # we lost
                    reward -= r
                    break

                # naughts: our move
                board = env.tensor_board()
                a_int = model(board).data[0].argmax()
                a = env.action_from_int(a_int)
                _, r, done, _ = env.step(a)

                if done:
                    # we won
                    reward += r
                    break
                                
    return reward / n_episodes