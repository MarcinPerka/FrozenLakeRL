import gym
import torch
import random
from gym.envs.registration import register
from frozenlake.utils import generate_plt, modify_reward, decrease_epsilon_with_large_num_of_epochs, TypeOfAlgortihm


# Q(s,a) = Q(s,a) + a * (reward + y*Q(s',a') - Q(s,a))
def sarsa_algorithm(Q, current_state, action, reward, next_state, next_action, done):
    Q[current_state][action] = Q[current_state][action] + ALPHA * (
            reward + GAMMA * (1 - done) * Q[next_state][next_action] - Q[current_state][action])  # SARSA


# Q(s,a) = Q(s,a) + a * (reward + y*max(Q(s',a)) - Q(s,a))
def q_learning_algorithm(Q, action, current_state, next_state, reward, done):
    Q[current_state][action] = Q[current_state][action] + ALPHA * (
            reward + GAMMA * (1 - done) * torch.max(Q[next_state]) - Q[current_state][action])  # Q-Learning


# Funkcja eksploracja lub eksploatacja
def greedy_epsilon_policy(q_values, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # eksploracja
    else:
        return torch.argmax(q_values).item()  # eksploatacja


def test_q_table():
    current_state = env.reset()
    moves_in_episode = 0
    done = False
    while not done:
        q_values = Q[current_state]
        action = torch.argmax(q_values).item()
        next_state, reward, done, info = env.step(action)
        current_state = next_state
        moves_in_episode += 1
        if done and reward > 0:
            print("You won!")
            return True
        elif done:
            print("Game lost;")
            return False
    return False


# Funkcja odpowiadające za przeprowadzenie max_games gier testowych.
def get_test_game_stats(max_games=1000):
    wins = []
    for i in range(max_games):
        wins.append(test_q_table())
    print("Test: Percent of games won: %.2f" % (sum(wins) / max_games))


def train(algorithm=TypeOfAlgortihm.Q_LEARNING):
    global epsilon
    episodes_moves, episodes_results = [], []

    for i_episode in range(TRAIN_EPISODES):

        current_state = env.reset()
        moves_in_episode = 0
        done = False

        while not done:
            moves_in_episode += 1
            if moves_in_episode == 1:
                q_values = Q[current_state]
                action = greedy_epsilon_policy(q_values, epsilon)
            next_state, reward, done, info = env.step(action)

            next_q_values = Q[next_state]
            next_action = greedy_epsilon_policy(next_q_values, epsilon)
            reward = modify_reward(reward, done)

            if algorithm == TypeOfAlgortihm.Q_LEARNING:
                q_learning_algorithm(Q, action, current_state, next_state, reward, done)
            else:
                sarsa_algorithm(Q, current_state, action, reward, next_state, next_action, done)

            current_state = next_state
            action = next_action

            if done:
                episodes_moves.append(moves_in_episode)
                if reward > 0:
                    result = True
                else:
                    result = False
                episodes_results.append(result)
                print('Train: Episode: {} Steps: {} Result: {} Epsilon: {}'.format(i_episode, moves_in_episode, result, epsilon))
                break
        epsilon = decrease_epsilon_with_large_num_of_epochs(epsilon, MIN_EPSILON, DECAY_RATE)
        env.close()
    print("Train: Percent of episodes finished successfully: {0}".format(sum(episodes_results) / TRAIN_EPISODES))
    print("Train: Average number of steps: %.2f" % (sum(episodes_moves) / TRAIN_EPISODES))
    generate_plt(episodes_moves, "Moves taken")


if __name__ == '__main__':
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '8x8', 'is_slippery': False},
        max_episode_steps=50)
    env = gym.make("FrozenLakeNotSlippery-v0")
    ACTION_SPACE = env.action_space.n
    STATE_SPACE = env.observation_space.n
    GAMMA = 0.9
    epsilon = 0.9
    MIN_EPSILON = 0.05
    DECAY_RATE = 0.999
    TRAIN_EPISODES = 5000
    ALPHA = 0.1

    # inicjacja tablicy q-table
    Q = torch.zeros([STATE_SPACE, ACTION_SPACE])

    train(algorithm=TypeOfAlgortihm.Q_LEARNING)
    get_test_game_stats(max_games=100)  # gry testowe
