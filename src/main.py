from os import path, getcwd
from agent import Agent
from environment import Environment
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def train_q_learning(env, nb_episode, agent, show_stats=True):

    mae_x = np.arange(nb_episode)
    mae_y = []
    episode_error = 0
    n = 0


    for i in range(nb_episode):

        (obs,) = env.reset()

        episode_error = 0
        n = 0

        done = False
        while not done:

            (state, actions_possibles) = obs

            # l'agent choisit une action
            action = agent.choisir_action(state, actions_possibles)

            # avance
            obs, reward, done = env.step(action)

            next_state, _ = obs

            # update
            step_error = agent.update(state, action, reward, next_state)

            # calcul du Mean Absolute Error
            episode_error += abs(step_error)
            n += 1

        agent.decay_epsilon()

        mae_y.append(episode_error / n)

    if show_stats:

        print(f"dernière différence : {episode_error}")

        mae_y = np.array(mae_y)
        plt.title("Mean Absolute Error")
        plt.xlabel("Épisode")
        plt.ylabel("MAE")
        plt.plot(mae_x, mae_y)
        plt.show()


def test(env, agent):

    agent.set_epsilon_to_zero()  # pas d'exploration aléatoire

    (obs,) = env.reset()
    env.render()

    frames = []

    done = False
    while not done:

        (state, actions_possibles) = obs

        # l'agent choisit une action
        action = agent.choisir_action(state, actions_possibles)

        # avance
        obs, reward, done = env.step(action)
        env.render()

        frames.append(Image.fromarray(env.img))

        next_state, _ = obs

        # update
        agent.update(state, action, reward, next_state)

        if env.quit_request or done:
            done = True

    # # generate image from the maze

    # arr = env.img
    # img = Image.fromarray(arr)
    # img.save(path.join(".", "maze.png"))

    # generate gif
    frames[0].save(
        path.join(getcwd(), "maze.gif"),
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=200,  # 200ms each image
        loop=0,
    )


def main():

    lab_size = 10

    env = Environment(size=lab_size, seed=None)

    nb_episode = 60

    agent = Agent(
        initial_epsilon=1.0,
        epsilon_decay=(
            1.0 / (0.5 * nb_episode)
        ),  # en 50% des episodes, epsilon vaut final_epsilon
        final_epsilon=0.1,
    )

    train_q_learning(env, nb_episode, agent)

    test(env, agent)


if __name__ == "__main__":
    main()
