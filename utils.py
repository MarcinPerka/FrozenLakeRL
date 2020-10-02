from enum import Enum

import matplotlib.pyplot as plt


# Generowanie podstawowego wykresu do wizualizacji wyników.
def generate_plt(data, y_label, x_label="Training Episodes"):
    plt.figure(figsize=(10, 5))
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.plot(data)
    plt.show()

# Modyfikacja nagrody frozen lake powoduje, że agent wydajniej się uczy. Porażka daje mu negatywny sygnał zamiast
# neutralnego - 0.0 nagroda.
def modify_reward(reward, done):
    if done and reward == 1.0:
        return 10.0
    if done and reward == 0.0:
        return -10.0
    else:
        return -1.0


# Funkcja zmniejszająca wartośc epsilonu odpowiadajacego za eksplorację
def decrease_epsilon_with_large_num_of_epochs(epsilon, min_epsilon, decay_rate):
    if epsilon > min_epsilon:
        epsilon *= decay_rate
    return epsilon


# Funkcja zmniejszająca wartośc epsilonu odpowiadajacego za eksplorację, kiedy wystepuje mala liczba epizodow
def decrease_epsilon_with_small_num_of_epochs(epsilon, min_epsilon, epochs):
    if epsilon > min_epsilon:
        epsilon -= (1 / epochs)
    return epsilon


class TypeOfAlgortihm(Enum):
    Q_LEARNING = 0
    SARSA = 1,
    Q_LEARNING_WITH_RM = 2,
    SARSA_WITH_RM = 3