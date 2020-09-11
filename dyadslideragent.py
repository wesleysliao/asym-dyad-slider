from collections import deque

import numpy as np
from scipy.special import expit as sigmoid


class DyadSliderAgent(object):

    def __init__(self,
                 force_max = 1.0,
                 force_min = -1.0,
                 c_error = 1.0,
                 c_effort = 1.0,
                 perspective = 0,
                 history_length = 2):

        self.perspective = perspective

        self.force = 0.0
        self.force_max = force_max
        self.force_min = force_min

        self.observation_history = deque(maxlen = history_length)
        self.action_history = deque(maxlen = history_length)
        self.reward_history = deque(maxlen = history_length)

        self.c_error = c_error
        self.c_effort = c_effort


    def get_force(self, environment_state, record_history = True):
        observation = self.observe(environment_state)

        action = self.action_policy(observation)

        if record_history:
            self.observation_history.append(observation)
            self.action_history.append(action)

        self.force = self.action_to_force(action)

        return self.force


    def give_reward(self, reward, is_terminal, next_environment_state = None):
        self.reward_history.append(reward)

        subj_reward = self.subjective_reward(reward)


        if is_terminal:
            pass
        else:
            pass


    def observe(self, environment_state):
        x, x_dot, r, r_dot, \
        agent1_interaction_force, agent1_interaction_force_dot, \
        agent2_interaction_force, agent2_interaction_force_dot = environment_state

        if self.perspective % 2 == 0:
            error = x - r
            error_dot = x_dot - r_dot
            force_interaction = agent1_interaction_force
            force_interaction_dot = agent1_interaction_force_dot

        else:
            error = r - x
            error_dot = r_dot - x_dot
            force_interaction = -agent2_interaction_force
            force_interaction_dot = -agent2_interaction_force_dot

        observation = np.array([error, error_dot,
                                force_interaction, force_interaction_dot])

        return observation


    def action_policy(self, observation):
        return 0


    def action_to_force(self, action):
        force = action
        return force


    def effort(self, action):
        return np.sum(action)


    def subjective_reward(self, environment_reward):
        last_action = self.action_history[-1]

        reward = ((self.c_error * environment_reward)
                  + (self.c_effort * self.effort(last_action)))

        return reward

    def reset(self):
        pass



class RandomAgent(DyadSliderAgent):

    def __init__(self, action_space):
        self.action_space = action_space

    def action_policy(self, observation):
        return self.action_space.sample()


class FixedAgent(DyadSliderAgent):

    def action_policy(self, observation):
        error, error_dot, force_interaction, force_interaction_dot = observation

        if error > 0:
            action = -1
        elif error < 0:
            action = 1
        else:
            action = 0

        return action

    def action_to_force(self, action):
        if action > 0:
            force = self.force_max
        elif action < 0:
            force = self.force_min
        else:
            force = 0

        return force


class PIDAgent(DyadSliderAgent):

    def __init__(self, kP, kI, kD, **kwargs):

        self.kP = kP
        self.kI = kI
        self.kD = kD

        self.reset()

        super().__init__(**kwargs)

    def action_policy(self, observation):
        error, error_dot, force_interaction, force_interaction_dot = observation

        self.error_sum += error
        action = (self.kP * error) + (self.kI * self.error_sum) + (self.kD * error_dot)

        return action


    def reset(self):
        self.error_sum = 0.0
