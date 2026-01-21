import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# 1. Configuration de l'environnement
# 'is_slippery=False' rend le jeu déterministe pour débuter
env = gym.make(
    "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode=None
)

# 2. Initialisation de la Q-Table
# Elle doit être de taille (nombre d'états x nombre d'actions)
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
total_episodes = 1000  # Nombre d'essais

# 4. Boucle d'apprentissage
rewards = []

for episode in range(total_episodes):
    state, info = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(100):
        # --- STRATÉGIE EPSILON-GREEDY ---
        # Choisir entre exploration (hasard) et exploitation (Q-table)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        # Exécuter l'action
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- MISE À JOUR DE LA Q-TABLE (Équation de Bellman) ---
        ### À COMPLÉTER PAR LES ÉTUDIANTS ###
        # Formule : Q(s,a) = Q(s,a) + alpha * [Reward + gamma * max(Q(s',a')) - Q(s,a)]
        # q_table[state, action] = ...

        total_rewards += reward
        state = new_state

        if done:
            break

    # Réduction de l'exploration (Epsilon Decay)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print("Entraînement terminé !")

# Test de l'agent
env_visu = gym.make(
    "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="human"
)
state, info = env_visu.reset()
for _ in range(20):
    action = np.argmax(q_table[state, :])
    state, reward, terminated, truncated, info = env_visu.step(action)
    if terminated or truncated:
        break
env_visu.close()
