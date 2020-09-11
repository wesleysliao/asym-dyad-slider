#!/usr/bin/env python3

import gym
from gym import error, spaces, utils
from gym.utils import seeding


import numpy as np


from dynamicobject import DynamicObject, Damping, CompressionSpring, TensionSpring, BindPosition, PositionLimits

def fixed_sos_gen():
    return lambda t: ((0.5 * np.sin(0.3 * np.pi *t)) + (0.3 * np.sin(0.5 * np.pi * t)) + (0.2 * np.sin(0.1 * np.pi * t)))

def zero_ref_gen():
    return lambda x: 0


class DyadSliderAsymEffortEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 60,
    }

    def __init__(self,
                 simulation_freq_Hz = 960,
                 action_freq_Hz = 60,

                 episode_length_s = 20.0,

                 n_agents = 2,
                 agent_force_min = -10.0,
                 agent_force_max = 10.0,

                 force_net_limits = np.array([-np.inf, np.inf]),
                 force_interaction_limits = np.array([-np.inf, np.inf]),

                 slider_mass = 1.0,
                 slider_damping = 0.5,
                 slider_limits = np.array([-1.0, 1.0]),

                 handle_mass = 0.1,
                 handle_damping = 0.1,
                 handle_spring_const = 40.0,
                 handle_rest_length = 0.1,

                 agent1_push_multiplier = 1.0,
                 agent1_pull_multiplier = 1.0,
                 agent1_handle_tracking_ratio = 1.0,
                 agent2_push_multiplier = 1.0,
                 agent2_pull_multiplier = 1.0,
                 agent2_handle_tracking_ratio = 1.0,

                 reference_generator = fixed_sos_gen,
    ):

        self.simulation_freq_Hz = simulation_freq_Hz
        self.simulation_timestep_s = 1.0 / simulation_freq_Hz
        self.action_timestep_s = 1.0 / action_freq_Hz
        self.simsteps_per_action = int(simulation_freq_Hz / action_freq_Hz)
        self.max_episode_steps = int(action_freq_Hz * episode_length_s)
        self.episode_length_s = episode_length_s


        self.n_agents = n_agents

        self.agent_force_min = agent_force_min
        self.agent_force_max = agent_force_max

        if np.isscalar(agent_force_min):
            self.action_space = spaces.Box(low = agent_force_min,
                                           high = agent_force_max,
                                           shape = (n_agents,),
                                           dtype = np.float32)
        else:
            self.action_space = spaces.Box(low = agent_force_min,
                                           high = agent_force_max,
                                           dtype = np.float32)

        self.slider_mass = slider_mass
        self.slider_damping = slider_damping
        self.slider_limits = slider_limits
        self.slider_range = slider_limits[1] - slider_limits[0]

        self.handle_mass = handle_mass
        self.handle_damping = handle_damping
        self.handle_spring_const = handle_spring_const
        self.handle_rest_length = handle_rest_length

        self.agent1_push_multiplier = agent1_push_multiplier
        self.agent1_pull_multiplier = agent1_pull_multiplier
        self.agent1_handle_tracking_ratio = agent1_handle_tracking_ratio
        self.agent2_push_multiplier = agent2_push_multiplier
        self.agent2_pull_multiplier = agent2_pull_multiplier
        self.agent2_handle_tracking_ratio = agent2_handle_tracking_ratio


        self.dynamic_objects = []
        self.constraints = []

        self.slider = DynamicObject(slider_mass)
        self.dynamic_objects.append(self.slider)

        self.agent1_handle = DynamicObject(handle_mass)
        self.dynamic_objects.append(self.agent1_handle)

        self.agent1_handle_copy = DynamicObject(0.0)
        self.dynamic_objects.append(self.agent1_handle_copy)

        self.agent1_slider_copy = DynamicObject(0.0)
        self.dynamic_objects.append(self.agent1_slider_copy)

        self.agent2_handle = DynamicObject(handle_mass)
        self.dynamic_objects.append(self.agent2_handle)

        self.agent2_handle_copy = DynamicObject(0.0)
        self.dynamic_objects.append(self.agent2_handle_copy)

        self.agent2_slider_copy = DynamicObject(0.0)
        self.dynamic_objects.append(self.agent2_slider_copy)

        self.constraints.append(BindPosition(self.agent1_slider_copy, self.slider,
                                             proportion=self.agent1_handle_tracking_ratio))

        self.constraints.append(BindPosition(self.agent2_slider_copy, self.slider,
                                             proportion=self.agent2_handle_tracking_ratio))

        self.constraints.append(BindPosition(self.agent1_handle_copy, self.agent1_handle))
        self.constraints.append(BindPosition(self.agent2_handle_copy, self.agent2_handle))

        self.constraints.append(CompressionSpring(self.agent1_handle, self.agent1_slider_copy,
                                                  self.agent1_push_multiplier * self.handle_spring_const,
                                                  -self.handle_rest_length))
        self.constraints.append(TensionSpring(self.agent1_handle, self.agent1_slider_copy,
                                              self.agent1_pull_multiplier * self.handle_spring_const,
                                              -self.handle_rest_length))

        if(n_agents > 1):
            self.constraints.append(CompressionSpring(self.agent2_handle, self.agent2_slider_copy,
                                                    self.agent2_push_multiplier * self.handle_spring_const,
                                                    self.handle_rest_length))
            self.constraints.append(TensionSpring(self.agent2_handle, self.agent2_slider_copy,
                                                self.agent2_pull_multiplier * self.handle_spring_const,
                                                self.handle_rest_length))


        self.constraints.append(CompressionSpring(self.slider, self.agent1_handle_copy,
                                                  self.handle_spring_const,
                                                  -self.handle_rest_length))
        self.constraints.append(TensionSpring(self.slider, self.agent1_handle_copy,
                                              self.handle_spring_const,
                                              -self.handle_rest_length))

        if(n_agents > 1):
            self.constraints.append(CompressionSpring(self.slider, self.agent2_handle_copy,
                                                    self.handle_spring_const, self.handle_rest_length))
            self.constraints.append(TensionSpring(self.slider, self.agent2_handle_copy,
                                                self.handle_spring_const, -self.handle_rest_length))


        self.constraints.append(Damping(self.slider_damping, self.slider))
        self.constraints.append(PositionLimits(self.slider, self.slider_limits[1], self.slider_limits[0]))
        self.constraints.append(Damping(self.handle_damping, self.agent1_handle))
        self.constraints.append(PositionLimits(self.agent1_handle, self.slider_limits[1], self.slider_limits[0]))
        self.constraints.append(Damping(self.handle_damping, self.agent2_handle))
        self.constraints.append(PositionLimits(self.agent2_handle, self.slider_limits[1], self.slider_limits[0]))


        self.force_net_limits = force_net_limits
        self.force_interaction_limits = force_interaction_limits

        self.reference_generator = reference_generator


        self.viewer = None
        self.reset()



    def step(self, action):

        done = False

        r_1 = self.reference_trajectory_fn(self.t + self.action_timestep_s)
        self.r_dot = (r_1 - self.r) / self.action_timestep_s

        action = np.clip(action, self.agent_force_min, self.agent_force_max)

        p1_force, p2_force = action
        p2_force *= -1.0

        for i in range(self.simsteps_per_action):

            for constraint in self.constraints:
                constraint.apply()

            if ((p1_force <= 0 and self.agent1_handle.force <= 0)
                or (p1_force >= 0 and self.agent1_handle.force >= 0)):
                agent1_interaction_force = min(p1_force, self.agent1_handle.force)
            else:
                agent1_interaction_force = 0.0
            self.agent1_interaction_force_dot = (agent1_interaction_force - self.agent1_interaction_force) / self.simulation_timestep_s
            self.agent1_interaction_force = agent1_interaction_force


            self.agent1_handle.add_force(p1_force)
            if(self.n_agents > 1):

                if ((p2_force <= 0 and self.agent2_handle.force <= 0)
                    or (p2_force >= 0 and self.agent2_handle.force >= 0)):
                    agent2_interaction_force = min(p2_force, self.agent2_handle.force)
                else:
                    agent2_interaction_force = 0.0

                self.agent2_handle.add_force(p2_force)
                self.agent2_interaction_force_dot = (agent2_interaction_force - self.agent2_interaction_force) / self.simulation_timestep_s
                self.agent2_interaction_force = agent2_interaction_force



            for dyn_obj in self.dynamic_objects:
                dyn_obj.step(self.simulation_timestep_s)

        self.t += self.action_timestep_s
        self.r = r_1


        reward = self.action_timestep_s * (1.0 - (abs(self.slider.state[0] - r_1) / self.slider_range) )

        if (self.t >= self.episode_length_s):
            done = True

        return self.observe(), reward, done


    def reset(self):
        self.t = 0.0
        self.error = 0.0

        self.reference_trajectory_fn = self.reference_generator()
        self.r = self.reference_trajectory_fn(self.t)
        self.r_dot = 0.0

        if self.viewer:
            self.viewer.close()
            self.viewer = None

        self.slider.state = [0.0, 0.0, 0.0]
        self.agent1_slider_copy.state = [0.0, 0.0, 0.0]
        self.agent2_slider_copy.state = [0.0, 0.0, 0.0]
        self.agent1_handle.state = [-self.handle_rest_length, 0.0, 0.0]
        self.agent1_handle_copy.state = [-self.handle_rest_length, 0.0, 0.0]
        self.agent2_handle.state = [self.handle_rest_length, 0.0, 0.0]
        self.agent2_handle_copy.state = [self.handle_rest_length, 0.0, 0.0]

        self.agent1_interaction_force = 0.0
        self.agent1_interaction_force_dot = 0.0
        self.agent2_interaction_force = 0.0
        self.agent2_interaction_force_dot = 0.0


        return self.observe()

    def observe(self):
        x = self.slider.state[0]
        x_dot = self.slider.state[1]

        state = np.array([x, x_dot, self.r, self.r_dot,
                          self.agent1_interaction_force,
                          self.agent1_interaction_force_dot,
                          self.agent2_interaction_force,
                          self.agent2_interaction_force_dot])
        return state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400


        world_height = self.slider_range
        scale_y = screen_height / world_height

        scale_x = 1.0 * (screen_width / 2)

        egg_x = screen_width / 2
        egg_width = 20.0
        egg_height = 30.0

        handle_width = 10
        handle_height = 15

        reference_width = 2.0
        reference_x_resolution = int(20 * self.episode_length_s)

        reference_points = np.zeros((reference_x_resolution, 2))
        reference_scale = np.linspace(0, self.episode_length_s, reference_x_resolution)
        reference_points[:, 0] = (scale_x * reference_scale) + (screen_width / 2)
        reference_points[:, 1] = scale_y * self.reference_trajectory_fn(reference_scale)


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -egg_width / 2, egg_width / 2, egg_height / 2, -egg_height / 2
            egg = rendering.FilledPolygon([(l,0), (0,t), (r,0), (0,b)])
            self.egg_transform = rendering.Transform()
            egg.add_attr(self.egg_transform)
            self.viewer.add_geom(egg)

            l,r,t,b = -handle_width / 2, handle_width / 2, handle_height / 2, -handle_height / 2
            a1_handle = rendering.FilledPolygon([(l,0), (0,t), (r,0), (0,b)])
            self.a1_handle_transform = rendering.Transform()
            a1_handle.add_attr(self.a1_handle_transform)
            self.viewer.add_geom(a1_handle)

            l,r,t,b = -handle_width / 2, handle_width / 2, handle_height / 2, -handle_height / 2
            a2_handle = rendering.FilledPolygon([(l,0), (0,t), (r,0), (0,b)])
            self.a2_handle_transform = rendering.Transform()
            a2_handle.add_attr(self.a2_handle_transform)
            self.viewer.add_geom(a2_handle)

            reference = rendering.PolyLine(reference_points, False)
            self.reference_transform = rendering.Transform()
            reference.add_attr(self.reference_transform)
            self.viewer.add_geom(reference)



        egg_y = (self.slider.state[0] * scale_y) + (screen_height / 2)
        self.egg_transform.set_translation(egg_x, egg_y)

        a1_handle_y = (self.agent1_handle.state[0] * scale_y) + (screen_height / 2)
        self.a1_handle_transform.set_translation(egg_x, a1_handle_y)

        a2_handle_y = (self.agent2_handle.state[0] * scale_y) + (screen_height / 2)
        self.a2_handle_transform.set_translation(egg_x, a2_handle_y)

        self.reference_transform.set_translation(-self.t * scale_x, screen_height / 2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
