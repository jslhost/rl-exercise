import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# 1. Configuration de l'environnement
# 'is_slippery=False' : l'agent va exactement là où il demande
env = gym.make(
    "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode=None
)

# 2. Initialisation de la Q-Table
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# 3. Hyperparamètres
learning_rate = 0.8  # Alpha
discount_rate = 0.95  # Gamma
epsilon = 1.0  # Taux d'exploration initial
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005  # Décroissance de l'exploration
total_episodes = 10_000

# 4. Boucle d'apprentissage
rewards = []

for episode in range(total_episodes):
    state, info = env.reset()
    done = False
    total_rewards = 0

    for step in range(100):
        # --- STRATÉGIE EPSILON-GREEDY ---
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        # Exécuter l'action
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- SOLUTION : MISE À JOUR DE LA Q-TABLE (Équation de Bellman) ---
        # On calcule la différence entre la nouvelle estimation et l'ancienne (TD Error)
        best_future_q = np.max(q_table[new_state, :])

        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_rate * best_future_q - q_table[state, action]
        )

        total_rewards += reward
        state = new_state

        if done:
            break

    # Réduction de l'exploration (Epsilon Decay)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print("Entraînement terminé !")

# --- VISUALISATION DE LA PROGRESSION ---
plt.figure(figsize=(10, 5))
plt.plot(
    np.convolve(rewards, np.ones(50) / 50, mode="valid")
)  # Moyenne mobile sur 50 épisodes
plt.title("Évolution des récompenses (Moyenne mobile 50 épisodes)")
plt.xlabel("Épisodes")
plt.ylabel("Succès (1.0 = Arrivée au but)")
plt.show()

# --- TEST DE L'AGENT ---
# Note : Nécessite 'uv add pygame' ou 'pip install pygame'
env_visu = gym.make(
    "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="human"
)
state, info = env_visu.reset()
print("\nTest de l'agent entraîné...")

for _ in range(20):
    action = np.argmax(q_table[state, :])  # On ne prend que la meilleure action connue
    state, reward, terminated, truncated, info = env_visu.step(action)
    if terminated or truncated:
        if reward == 1:
            print("Victoire ! L'agent a atteint le cadeau.")
        else:
            print("Défaite... L'agent est tombé dans un trou.")
        break

env_visu.close()


import seaborn as sns


def plot_q_table(q_table):
    plt.figure(figsize=(8, 6))
    # On prend la valeur max de chaque état (la meilleure action possible)
    q_max = np.max(q_table, axis=1).reshape((4, 4))
    sns.heatmap(q_max, annot=True, cmap="YlGnBu")
    plt.title("Valeur maximale des états (Q-Table)")
    plt.show()


plot_q_table(q_table)
