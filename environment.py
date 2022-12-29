import numpy as np
import tkinter as tk
import time
from PIL import ImageTk, Image

unit = 50
height = 7
width = 7
PhotoImage = ImageTk.PhotoImage
np.random.seed(1)

class Environment(tk.Tk):
    def __init__(self):
        super(Environment, self).__init__()
        self.title('Mouse-Cat-Home')
        self.geometry('{0}x{1}'.format(height*unit, width*unit))
        self.action_space = ['up', 'down', 'left', 'right']
        self.action_size = len(self.action_space)
        self.shapes = self.load_images()
        self.window = self.build_window()
        self.counter = 0
        self.rewards = []
        self.goal = []
        self.set_reward([0, 1], -1) # cat
        self.set_reward([1, 3], -1) # cat
        self.set_reward([2, 5], -1) # cat
        self.set_reward([6, 6], 1)  # home

    # rectangle boulding
    def build_window(self):
        self.reward = []
        self.goal = []
        window = tk.Canvas(self, bg = 'white', height = height*unit, width = width*unit)

        for i in range(0, width*unit, unit):
            a = i
            b = 0
            c = i
            d = width*unit
            window.create_line(a, b, c ,d)
        for k in range(0, height*unit, unit):
            a = 0
            b = k
            c = height*unit
            d = k
            window.create_line(a, b, c ,d)

        self.rectangle = window.create_image(unit/2, unit/2, image = self.shapes[0])
        window.pack()
        return window

    def load_images(self):
        mouse = PhotoImage(Image.open("./mouse.jpg").resize((30, 30)))
        cat = PhotoImage(Image.open("./cat.jpg").resize((30, 30)))
        home = PhotoImage(Image.open("./home.jpg").resize((30, 30)))
        return mouse, cat, home

    # position and rewards when the mouse reach these cells
    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])
        t = {}

        if reward > 0:
            t['figure'] = self.window.create_image((unit*x) + unit/2, (unit*y) + unit/2, image = self.shapes[2])
            self.goal.append(t['figure'])
        elif reward < 0:
            t['direction'] = -1
            t['figure'] = self.window.create_image((unit*x) + unit/2, (unit*y) + unit/2, image = self.shapes[1])

        t['coordinates'] = self.window.coords(t['figure'])
        t['state'] = state
        t['reward'] = reward
        self.rewards.append(t)

    def check_reward(self, state):
        list = dict()
        list['goal'] = False
        rewards = 0

        for reward in self.rewards:
            if reward['state'] == state:
                rewards = rewards + reward['reward']
                if reward['reward'] == 1:
                    list['goal'] = True # mouse reached home

        list['rewards'] = rewards
        return list

    def coords_to_state(self, coordinates):
        x = int((coordinates[0] - unit/2)/unit)
        y = int((coordinates[1] - unit/2)/unit)
        return [x, y]

    # reset the states of the characteristics at the end of the episode
    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.window.coords(self.rectangle)
        self.window.move(self.rectangle, unit/2 - x, unit/2 - y)
        for reward in self.rewards:
            self.window.delete(reward['figure'])
        self.rewards.clear()
        self.goal.clear()
        self.set_reward([0, 1], -1)
        self.set_reward([1, 3], -1)
        self.set_reward([2, 5], -1)
        self.set_reward([6, 6], 1)

        return self.get_state()

    # check position on the grid
    def step(self, action):
        self.counter = self.counter + 1
        time.sleep(0.05)
        self.update()

        if self.counter % 2 == 1:
            self.rewards = self.move_rewards()

        next = self.move(self.rectangle, action)
        check = self.check_reward(self.coords_to_state(next))
        end = check['goal']
        reward = check['rewards']
        self.window.tag_raise(self.rectangle)
        get = self.get_state()
        return get, reward, end

    def get_state(self):
        location = self.coords_to_state(self.window.coords(self.rectangle))
        agent_x = location[0]
        agent_y = location[1]
        states = list()

        for reward in self.rewards:
            place = reward['state']
            states.append(place[0] - agent_x)
            states.append(place[1] - agent_y)
            if reward['reward'] < 0:
                states.append(-1)
                states.append(reward['direction'])
            else:
                states.append(1)

        return states

    def move_rewards(self):
        new_rewards = []

        for t in self.rewards:
            if t['reward'] == 1:
                new_rewards.append(t)
                continue
            t['coordinates'] = self.move_const(t)
            t['state'] = self.coords_to_state(t['coordinates'])
            new_rewards.append(t)

        return new_rewards

    # cats move
    def move_const(self, target):
        s = self.window.coords(target['figure'])
        base_action = np.array([0, 0])

        if s[0] == (width - 1)*unit + unit/2:
            target['direction'] = 1
        elif s[0] == unit/2:
            target['direction'] = -1

        if target['direction'] == -1:
            base_action[0] = base_action[0] + unit
        elif target['direction'] == 1:
            base_action[0] = base_action[0] - unit

        if (target['figure']is not self.rectangle and s == [(width - 1)*unit, (height -1)*unit]):
            base_action = np.array([0, 0])

        self.window.move(target['figure'], base_action[0], base_action[1])
        get = self.window.coords(target['figure'])
        return get

    #mouse moves
    def move(self, target, action):
        s = self.window.coords(target)
        base_action = np.array([0, 0])

        if action == 0: #up
            if s[1] > unit:
                base_action[1] = base_action[1] - unit
        elif action == 1: #down
            if s[1] < (height - 1)*unit:
                base_action[1] = base_action[1] + unit
        elif action == 2:  # right
            if s[0] < (width - 1)*unit:
                base_action[0] = base_action[0] + unit
        elif action == 3:  # left
            if s[0] > unit:
                base_action[0] = base_action[0] - unit

        self.window.move(target, base_action[0], base_action[1])
        get = self.window.coords(target)
        return get
