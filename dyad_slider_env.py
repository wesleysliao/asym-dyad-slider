import gym
from gym import error, spaces, utils
from gym.utils import seeding


import numpy as np
from scipy.integrate import RK45


def zero_ref_gen():
    return lambda x: 0

class DyadSliderEnv(gym.Env):
    """
    Description:
        Two agents apply forces on an object to follow a target which moves in one dimension.
    Source:
    Observation:
        Observations of the state depend on the agent

    Actions:
        Applied force

    Reward:
        Reward is tracking error

    Starting State:

    Episode Termination:
        Episode length is greater than
        Force limits are exceeded
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50,
    }

    def __init__(self,
                 simulation_freq_Hz = 500,
                 action_freq_Hz = 50,

                 episode_length_s = 20.0,

                 n_agents = 2,
                 agent_force_min = -1.0,
                 agent_force_max = 1.0,

                 force_net_limits = np.array([-np.inf, np.inf]),
                 force_interaction_limits = np.array([-np.inf, np.inf]),

                 slider_mass = 1.0,
                 slider_friction_coeff = 0.0,
                 slider_limits = np.array([-1.0, 1.0]),

                 reference_generator = zero_ref_gen,

                 integration = "euler", # "rk45"
    ):

        self.simulation_freq_Hz = simulation_freq_Hz
        self.simulation_timestep_s = 1.0 / simulation_freq_Hz
        self.action_timestep_s = 1.0 / action_freq_Hz
        self.simsteps_per_action = int(simulation_freq_Hz / action_freq_Hz)
        self.max_episode_steps = int(action_freq_Hz * episode_length_s)
        self.episode_length_s = episode_length_s

        self.integration = integration

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
        self.slider_friction_coeff = slider_friction_coeff
        self.slider_limits = slider_limits
        self.slider_range = slider_limits[1] - slider_limits[0]

        self.force_net_limits = force_net_limits
        self.force_interaction_limits = force_interaction_limits

        self.reference_generator = reference_generator


        self.viewer = None
        self.reset()
       


    def step(self, action):

        done = False

        x_0, x_dot_0, r_0, r_dot_0, \
        force_net_0, force_net_dot_0, \
        force_interaction_0, force_interaction_dot_0 = self.state

        action = np.clip(action, self.agent_force_min, self.agent_force_max)

        p1_force, p2_force = action
        if (p1_force <= 0 and p2_force <= 0 or p1_force >= 0 and p2_force >= 0):
            force_interaction_1 = min(p1_force, p2_force)
        else:
            force_interaction_1 = 0.0

        p2_force *= -1.0
        force_net_1 = p1_force + p2_force

        force_net_dot_1 = (force_net_1 - force_net_0) / self.action_timestep_s
        force_interaction_dot_1 = (force_interaction_1 - force_interaction_0) / self.action_timestep_s

        r_1 = self.reference_trajectory_fn(self.t + self.action_timestep_s)
        r_dot_1 = (r_1 - r_0) / self.action_timestep_s

        self.state[2:] = np.array([r_1, r_dot_1,
                                   force_net_1, force_net_dot_1,
                                   force_interaction_1, force_interaction_dot_1])

        if self.check_force_limits(self.state):
            done = True

        elif self.integration == "euler":
            x_1, x_dot_1 = x_0, x_dot_0
            for i in range(self.simsteps_per_action):
                velocity, acceleration = self.slider_dynamics(None, [x_1, x_dot_1])

                x_dot_1 += (acceleration * self.simulation_timestep_s)
                x_1 += (velocity * self.simulation_timestep_s)

                if self.check_x_limits(self.state):
                    done = True
                    break

            self.state[:2] = np.array([x_1, x_dot_1])
            self.t += self.action_timestep_s

        elif self.integration == "rk45":
            while(self.rk45.t < self.t + self.action_timestep_s and self.rk45.t < self.episode_length_s):
                self.rk45.step()

                self.state[:2] = self.rk45.y
                if self.check_x_limits(self.state):
                    done = True
                    break

            self.t = self.rk45.t

        reward = self.action_timestep_s * (1.0 - (abs(self.state[0] - r_1) / self.slider_range) )

        if (self.t >= self.episode_length_s):
            done = True

        return self.observe(self.state), reward, done


    def check_x_limits(self, state):
        return ((state[0] < self.slider_limits[0]) or (state[0] > self.slider_limits[1]))

    def check_force_limits(self, state):
        return ((state[4] < self.force_net_limits[0]) or (state[4] > self.force_net_limits[1])
                or (state[6] < self.force_interaction_limits[0]) or (state[6] > self.force_interaction_limits[1]))


    def slider_dynamics(self, t, y):
        x, x_dot = y
        force_net = self.state[4]

        acceleration = (force_net - (self.slider_friction_coeff * x_dot)) / self.slider_mass
        velocity = x_dot

        return [velocity, acceleration]


    def reset(self):
        self.t = 0.0
        self.error = 0.0

        #state = x, x_dot, r, r_dot, force_net, force_net_dot, force_interaction, force_interaction_dot
        self.state = np.zeros((8,), dtype=np.float32)

        if self.integration == "rk45":
            self.rk45 = RK45(self.slider_dynamics, 0.0, self.state[:2], self.episode_length_s, max_step=self.action_timestep_s)

        self.reference_trajectory_fn = self.reference_generator()

        if self.viewer:
            self.viewer.close()
            self.viewer = None

        return self.observe(self.state)

    def observe(self, state):
        x, x_dot, r, r_dot, \
        force_net, force_net_dot, \
        force_interaction, force_interaction_dot = state

        return np.array([x, x_dot, r, r_dot,
                         force_interaction, force_interaction_dot])

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400


        world_height = self.slider_range
        scale_y = screen_height / world_height

        scale_x = 1.0 * (screen_width / 2)

        egg_x = screen_width / 2
        egg_width = 20.0
        egg_height = 30.0

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

            reference = rendering.PolyLine(reference_points, False)
            self.reference_transform = rendering.Transform()
            reference.add_attr(self.reference_transform)
            self.viewer.add_geom(reference)

        if self.state is None: return None


        x, x_dot, r, r_dot, \
        force_net, force_net_dot, \
        force_interaction, force_interaction_dot = self.state

        egg_y = (x * scale_y) + (screen_height / 2)
        self.egg_transform.set_translation(egg_x, egg_y)

        self.reference_transform.set_translation(-self.t * scale_x, screen_height / 2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
