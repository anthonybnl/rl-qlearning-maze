from os import path
from agent import Agent
from environment import Environment
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def main():

    lab_size = 10

    env = Environment(size=lab_size, seed=None)

    nb_episode = 3*lab_size

    agent = Agent(
        initial_epsilon=1.0,
        epsilon_decay=(
            1.0 / (0.5 * nb_episode)
        ),  # en 50% des episodes, epsilon vaut final_epsilon
        final_epsilon=0.1,
    )

    diff_np_x = np.arange(nb_episode)
    diff_np_y = []

    # train

    for i in range(nb_episode):

        (obs,) = env.reset()

        total_difference = 0

        done = False
        while not done:

            (state, actions_possibles) = obs

            # l'agent choisit une action
            action = agent.choisir_action(state, actions_possibles)

            # avance
            obs, reward, done = env.step(action)

            next_state, _ = obs

            # update
            difference = agent.update(state, action, reward, next_state)
            total_difference += abs(difference)

        agent.decay_epsilon()

        diff_np_y.append(total_difference)

    # stats

    print(f"dernière différence : {total_difference}")

    diff_np_y = np.array(diff_np_y)
    plt.title("Q-Values update error")
    plt.xlabel("épisode")
    plt.ylabel("erreur")
    plt.plot(diff_np_x, diff_np_y)
    plt.show()

    # test

    (obs,) = env.reset()
    env.render()

    agent.epsilon = 0  # pas d'exploration aléatoire

    done = False
    while not done:

        (state, actions_possibles) = obs

        # l'agent choisit une action
        action = agent.choisir_action(state, actions_possibles)

        # avance
        obs, reward, done = env.step(action)
        env.render()

        next_state, _ = obs

        # update
        agent.update(state, action, reward, next_state)

        if env.quit_request or done:
            done = True

    # generate image from the maze

    arr = env.img
    img = Image.fromarray(arr)
    img.save(path.join(".", "maze.png"))


if __name__ == "__main__":
    main()
