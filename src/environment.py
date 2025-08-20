import random
import pygame
import pygame.locals
from labyrinthe import Labyrinthe
import numpy as np


class Actions:
    HAUT = 0
    DROITE = 1
    BAS = 2
    GAUCHE = 3


class Environment:
    def __init__(self, size, seed=None):

        random.seed(seed)

        self.LAB_SIZE = size
        self.labyrinthe = Labyrinthe(size)
        self.final_state = size * size - 1
        self.quit_request = False

        self.state = 0

        # ui stuff

        self.window = None
        self.clock = None
        self.fps = 4

        # pixels

        self.cell_size_px = 50
        lab_width_px = size * self.cell_size_px

        min_window_size = lab_width_px + 20

        self.window_size = 512 if 512 > min_window_size else min_window_size
        self.margin_left_lab_px = int((self.window_size - lab_width_px) / 2)
        self.margin_top_lab_px = int((self.window_size - lab_width_px) / 2)

        # numpy array of img

        self.img: np.ndarray | None = None

    def actions_possibles_depuis(self, state):
        next_state_possibles = self.labyrinthe.transition[state]
        actions_possibles = []
        for i in next_state_possibles:
            if i == self.state - self.LAB_SIZE:
                actions_possibles.append(Actions.HAUT)
            elif i == self.state + self.LAB_SIZE:
                actions_possibles.append(Actions.BAS)
            elif i == self.state + 1:
                actions_possibles.append(Actions.DROITE)
            elif i == self.state - 1:
                actions_possibles.append(Actions.GAUCHE)
            else:
                raise Exception("transition impossible")

        return actions_possibles

    def get_observation(self):
        actions_possibles = self.actions_possibles_depuis(self.state)
        return (self.state, actions_possibles)

    def reset(self):
        self.state = 0
        self.quit_request = False

        obs = self.get_observation()

        return (obs,)

    def step(self, action):
        actions_possibles = self.actions_possibles_depuis(self.state)

        if not action in actions_possibles:
            raise Exception("action impossible")

        next_state = self.state
        if action == Actions.HAUT:
            next_state -= self.LAB_SIZE
        elif action == Actions.BAS:
            next_state += self.LAB_SIZE
        elif action == Actions.DROITE:
            next_state += 1
        elif action == Actions.GAUCHE:
            next_state -= 1

        self.state = next_state

        reward = 10 if self.state == self.final_state else -1
        obs = self.get_observation()

        done = (self.state == self.final_state) or self.quit_request

        return (obs, reward, done)

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        self.render_labyrinthe(canvas)

        x = (
            (self.state % self.LAB_SIZE) + 0.5
        ) * self.cell_size_px + self.margin_left_lab_px
        y = (
            int(self.state / self.LAB_SIZE) + 0.5
        ) * self.cell_size_px + self.margin_left_lab_px

        # l'agent

        pygame.draw.circle(canvas, (255, 0, 0), (x, y), self.cell_size_px / 4)

        # dépile events

        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                self.quit_request = True
                return

        # make image

        img = pygame.surfarray.array3d(canvas)
        self.img = np.transpose(
            np.array(img), axes=(1, 0, 2)
        )  # inversion des deux premiers axes, car array3d donne un array indexé par x, puis par y

        # canvas -> window

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()

        self.clock.tick(self.fps)

    def render_labyrinthe(self, canvas):
        rect_labyrinthe = [
            self.margin_left_lab_px,
            self.margin_top_lab_px,
            self.LAB_SIZE * self.cell_size_px,
            self.LAB_SIZE * self.cell_size_px,
        ]
        pygame.draw.rect(
            canvas,
            (255, 255, 255),
            rect_labyrinthe,
            0,
        )
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            rect_labyrinthe,
            1,
        )

        for mur in self.labyrinthe.murs:
            u, v = mur
            j = u % self.LAB_SIZE
            i = int((u - j) / self.LAB_SIZE)

            x_start = self.margin_left_lab_px + j * self.cell_size_px
            y_start = self.margin_top_lab_px + i * self.cell_size_px

            if v - u == 1:  # mur vertical
                x_start += self.cell_size_px
                (x_end, y_end) = (x_start, y_start + self.cell_size_px - 1)
            elif v - u == self.LAB_SIZE:
                y_start += self.cell_size_px
                (x_end, y_end) = (x_start + self.cell_size_px - 1, y_start)
            else:
                raise Exception("mauvais labyrinthe")

            pygame.draw.line(canvas, (0, 0, 0), (x_start, y_start), (x_end, y_end))
