#!/usr/bin/env python3

import gym
import numpy as np

from dyad_slider_asym_effort_env import DyadSliderAsymEffortEnv
from dyadslideragent import PIDAgent

np.random.seed()

def sos_gen():
    amplitudes = [0.4, 0.3, 0.2, 0.1]
    frequencies = [0.4, 0.3, 0.2, 0.1]

    a = np.random.permutation(amplitudes)
    f = np.random.permutation(frequencies)
    p = np.random.random(size=a.size) * 2.0 * np.pi

    def sos_fn(t):
        result = 0
        for i in range(len(amplitudes)):
            result += (a[i] * np.sin(f[i] * np.pi * (t + p[i])))
        return result

    return sos_fn


env = DyadSliderAsymEffortEnv(reference_generator=sos_gen)

agent1 = PIDAgent(-1, 0, 0)
agent2 = PIDAgent(-2, 0, 0, perspective = 1)

state = env.reset()
for _ in range(1000):
    env.render()

    state, reward, done = env.step([agent1.get_force(state), agent2.get_force(state)])
env.close()
