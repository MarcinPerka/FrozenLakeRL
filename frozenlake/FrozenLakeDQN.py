import random

import gym
import numpy as np
import copy
import torch
from IPython.core.display import clear_output
from gym.envs.registration import register

from frozenlake.NeuralNetworks import LinearNetworkWithTwoHiddenLayers
from frozenlake.utils import modify_reward, decrease_epsilon_with_large_num_of_epochs, generate_plt, TypeOfAlgortihm
from frozenlake.ReplayMemory import ReplayMemory


# Funkcja eksploracja lub eksploatacja
def greedy_epsilon_policy(q_values, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return torch.argmax(q_values).item()  # .item() cuz tensor to num


# Metoda zwracająca vector z jedną 1  oraz wartościami określonymi jako szum(niskim wartościami różnymi od zera
def one_hot(state_space, current_state_position):
    vector = torch.zeros(1, state_space)
    vector[0][current_state_position] = 1
    return vector


def test_model(neural_network):
    current_state = env.reset()
    moves_in_episode = 0
    done = False
    while not done:
        q_values = neural_network.forward(one_hot(STATE_SPACE, current_state))
        action = torch.argmax(q_values).item()
        next_state, reward, done, info = env.step(action)
        current_state = next_state
        moves_in_episode += 1
        if done and reward > 0:
            print("You won!")
            return True
        elif done:
            print("Game lost")
            return False
    return False


def deep_q_learning_algorithm(action, reward, q_values, next_q_values, done):
    max_next_q_value = torch.max(next_q_values)
    update = reward + GAMMA * (1 - done) * max_next_q_value
    propagate_loss(action, q_values, update)


def propagate_loss(action, q_values, update):
    target_q = torch.zeros(1, 4)
    target_q[:] = q_values[:]
    target_q[0][action] = update
    loss = neural_network.loss_fn(q_values, target_q)
    losses.append(loss.data)
    neural_network.optimizer.zero_grad()
    loss.backward(retain_graph=True)
    neural_network.optimizer.step()


def deep_sarsa_algorithm(action, reward, q_values, next_q_values, next_action, done):
    update = reward + GAMMA * (1 - done) * next_q_values[0][next_action]
    propagate_loss(action,q_values, update)


def deep_q_learning_algorithm_with_replay_memory(total_step, current_state, action, next_state, reward, done, next_action):
    memory.push(current_state, action, next_state, reward, done, next_action)  # None empty value

    if memory.__len__() > BATCH_SIZE:
        minibatch = memory.sample()
        action_batch, current_state_batch, done_batch, _, next_state_batch, reward_batch = get_experience_from_batch(
            minibatch)
        Q1 = neural_network.forward(current_state_batch)
        with torch.no_grad():
            Q2 = neural_network_deep_copy.forward(next_state_batch)

        Y = reward_batch + GAMMA * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

        propagate_batch_loss(X, Y)

        sync_target_network(total_step)


def sync_target_network(total_step):
    if total_step % SYNC_FREQ == 0:
        neural_network_deep_copy.load_state_dict(neural_network.model.state_dict())


def deep_sarsa_algorithm_with_replay_memory(total_step, current_state, action, next_state, reward, done, next_action):
    memory.push(current_state, action, next_state, reward, done, next_action)

    if memory.__len__() > BATCH_SIZE:
        minibatch = memory.sample()
        action_batch, current_state_batch, done_batch, next_action_batch, next_state_batch, reward_batch = get_experience_from_batch(
            minibatch)

        Q1 = neural_network.forward(current_state_batch)
        with torch.no_grad():
            Q2 = neural_network_deep_copy.forward(next_state_batch)

        Y = reward_batch + GAMMA * ((1 - done_batch) * Q2.gather(dim=1, index=next_action_batch.long().unsqueeze(dim=1)).squeeze())
        X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

        propagate_batch_loss(X, Y)

        sync_target_network(total_step)


def propagate_batch_loss(X, Y):
    loss = neural_network.loss_fn(X, Y.detach())
    losses.append(loss.item())
    clear_output(wait=True)
    neural_network.optimizer.zero_grad()
    loss.backward()
    neural_network.optimizer.step()


def get_experience_from_batch(minibatch):
    current_state_batch = torch.cat([transition[0] for transition in minibatch])
    action_batch = torch.Tensor([transition[1] for transition in minibatch])
    next_state_batch = torch.cat([transition[2] for transition in minibatch])
    reward_batch = torch.Tensor([transition[3] for transition in minibatch])
    done_batch = torch.Tensor([transition[4] for transition in minibatch])
    next_action_batch = torch.Tensor([transition[5] for transition in minibatch])
    return action_batch, current_state_batch, done_batch, next_action_batch, next_state_batch, reward_batch


def get_test_game_stats(neural_network, max_games=1000):
    wins = []
    for i in range(max_games):
        wins.append(test_model(neural_network))
    print("Test: Percent of games won: %.2f" % (sum(wins) / max_games))


def train(algorithm=TypeOfAlgortihm.Q_LEARNING_WITH_RM):
    global epsilon

    total_step = 0

    for i_episode in range(TRAIN_EPISODES):

        current_state_ = env.reset()
        current_state = one_hot(STATE_SPACE, current_state_)

        moves_in_episode = 0
        done = False
        while not done:
            moves_in_episode += 1
            total_step += 1

            if moves_in_episode == 1:
                q_values = neural_network.forward(current_state)
                action = greedy_epsilon_policy(q_values, epsilon)

            next_state_, reward, done, info = env.step(action)
            reward = modify_reward(reward, done)
            next_state = one_hot(STATE_SPACE, next_state_)
            next_q_values = neural_network.forward(next_state)
            next_action = greedy_epsilon_policy(next_q_values, epsilon)

            if algorithm == TypeOfAlgortihm.Q_LEARNING:
                deep_q_learning_algorithm(action, reward, q_values, next_q_values, done)
            elif algorithm == TypeOfAlgortihm.Q_LEARNING_WITH_RM:
                deep_q_learning_algorithm_with_replay_memory(total_step, current_state, action, next_state, reward,
                                                             done, next_action)
            elif algorithm == TypeOfAlgortihm.SARSA_WITH_RM:
                deep_sarsa_algorithm_with_replay_memory(total_step, current_state, action, next_state, reward,
                                                             done, next_action)
            elif algorithm == TypeOfAlgortihm.SARSA:
                deep_sarsa_algorithm(action, reward, q_values, next_q_values, next_action, done)

            q_values = next_q_values
            current_state = next_state
            action = next_action
            episodes_rewards.append(reward)

            if done:
                episodes_moves.append(moves_in_episode)
                result = check_if_won(reward)
                episodes_results.append(result)
                print('Train: Episode: {} Steps: {} Result: {} Epsilon: {}'.format(i_episode, moves_in_episode, result,
                                                                                   epsilon))
                break
        epsilon = decrease_epsilon_with_large_num_of_epochs(epsilon, MIN_EPSILON, DECAY_RATE)
        env.close()
    print("Train: Percent of episodes finished successfully: {}".format(sum(episodes_results) / TRAIN_EPISODES))
    print("Train: Average number of steps: %.2f" % (sum(episodes_moves) / TRAIN_EPISODES))

    generate_plt(np.array(losses), "Loss")
    generate_plt(episodes_moves, "Moves taken")


def check_if_won(reward):
    if reward > 0:
        result = True
    else:
        result = False
    return result


if __name__ == '__main__':
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '8x8', 'is_slippery': False},
        max_episode_steps=50)
    env = gym.make("FrozenLakeNotSlippery-v0")

    ACTION_SPACE = env.action_space.n
    STATE_SPACE = env.observation_space.n
    GAMMA = 0.9  # discount factor q learning
    epsilon = 0.9
    MIN_EPSILON = 0.05
    DECAY_RATE = 0.999
    TRAIN_EPISODES = 5000
    HIDDEN_NODES_L1 = 150
    HIDDEN_NODES_L2 = 100
    LEARNING_RATE = 0.001
    REPLAY_SIZE = 500  # experience replay buffer size
    BATCH_SIZE = 100  # size of minibatch
    SYNC_FREQ = 20
    memory = ReplayMemory(REPLAY_SIZE, BATCH_SIZE)

    neural_network = LinearNetworkWithTwoHiddenLayers(STATE_SPACE, ACTION_SPACE, HIDDEN_NODES_L1,
                                                      HIDDEN_NODES_L2,
                                                      LEARNING_RATE)

    neural_network_deep_copy = copy.deepcopy(neural_network.model)
    neural_network_deep_copy.load_state_dict(neural_network.model.state_dict())

    losses, episodes_rewards, episodes_moves, episodes_results = [], [], [], []

    train(algorithm=TypeOfAlgortihm.Q_LEARNING_WITH_RM)
    get_test_game_stats(neural_network, 100)
