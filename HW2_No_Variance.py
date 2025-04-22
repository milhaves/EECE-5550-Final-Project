import gymnasium
import numpy as np
import gym_neu_racing
from gymnasium import spaces
from gym_neu_racing.envs.wrappers import StateFeedbackWrapper
import matplotlib.pyplot as plt
from typing import Callable
import matplotlib.cm as cmx
import matplotlib.colors as colors
from gym_neu_racing import motion_models
import cvxpy as cp

# Create an instance of the mobile robot simulator we'll use this semester
env = gymnasium.make("gym_neu_racing/NEURacing-v0")

# Tell the simulator to directly provide the current state vector (no sensors yet)
env = StateFeedbackWrapper(env)

class MPPIRacetrack:
    def __init__(
        self,
        static_map,
        motion_model=motion_models.Unicycle(),
        n = 200,
        lamda = 1,
        T = 2,
        timestep = 0.1,
        control_inputs = 2,
        nominal_control = np.zeros((1,2)),
        states = 3,
        v_min=0,
        v_max=1,
        w_min=-2 * np.pi,
        w_max=2 * np.pi,
        square_dimension = 5
    ):
        """ Your implementation here """
        self.motion_model = motion_model
        self.static_map = static_map
        self.motion_model = motion_model
        self.n = n
        self.lamda = lamda
        self.T = T
        self.timestep = timestep
        self.control_inputs = control_inputs
        self.nominal_control = nominal_control
        self.states = states
        self.action_space = spaces.Box(
            np.array([v_min, w_min]),
            np.array([v_max, w_max]),
            shape=(2,),
            dtype=float,
        )
        self.square_dimension = square_dimension

        self.turn1 = np.array([-2.5,-0.75])
        self.turn2 = np.array([-2,3.5])
        self.turn3 = np.array([1.25,4])
        self.turn4 = np.array([3.5,1.75])
        self.turn5 = np.array([2.25,-1.5])
        self.turn6 = np.array([0,-2])

        self.waypoints = [self.turn1, self.turn2, self.turn3, self.turn4, self.turn5, self.turn6]
        self.waypoint_index = 0

        # print(self.bottom_goal)

        # raise NotImplementedError

    def plot_rollouts(self, rollout_states, initial_state: np.ndarray, goal_pos:np.ndarray):
        for i in range(self.n):
          plt.plot(rollout_states[i,:,0],rollout_states[i,:,1],color='grey', linestyle='dashed',linewidth = 0.5)

        # plt.plot(rollout_states[0,:,0],rollout_states[0,:,1],color='grey', linestyle='dashed',linewidth = 0.5)

        # print(rollout_states[0,:,0])
        # print(rollout_states[0,:,1])

        plt.scatter(goal_pos[0],goal_pos[1],color='red')
        plt.scatter(initial_state[0],initial_state[1],color='green')
        plt.title('Sample Rollouts')
        plt.xlabel('X pos')
        plt.ylabel('Y pos')
        plt.axis('equal')
        plt.show()

    def compute_rollouts(self, initial_state: np.ndarray, goal_pos: np.ndarray):
        # delta = np.random.normal(0,2.5,(self.n,(self.T/self.timestep),self.control_inputs))
        delta_v = np.random.normal(0.9,0.25,(self.n,int(self.T/self.timestep)))
        # delta_w = np.random.normal(0,2.5,(self.n,int(self.T/self.timestep)))
        # delta_v = np.zeros((self.n, int(self.T / self.timestep))) + 1.0
        delta_w = np.random.normal(0,2.5,(self.n,int(self.T/self.timestep)))
        # print(delta_v)
        # print(delta_w)
        action = np.zeros(2)

        # rollout_states = np.zeros([self.n,(self.T/self.timestep),self.states])
        rollout_states = np.zeros([self.n,int(self.T/self.timestep),self.states])
        rollout_states[:,0,0] = initial_state[0]
        rollout_states[:,0,1] = initial_state[1]
        rollout_states[:,0,2] = initial_state[2]
        # print(rollout_states[:,0,:])

        for i in range(0,self.n):
          # for j in range(1,(self.T/self.timestep)):
          for j in range(1,int(self.T/self.timestep)):
            previous_state = rollout_states[i,j-1,:]
            action[0] = self.nominal_control[0,0] + delta_v[i,j-1]
            action[1] = self.nominal_control[0,1] + delta_w[i,j-1]
            rollout_states[i,j,:] = self.motion_model.step(previous_state,action)

        return rollout_states, delta_v, delta_w

    def score_rollouts(self, rollout_states, delta_v, delta_w, initial_state: np.array, goal_pos: np.ndarray):
        # print(goal_pos)
        rollout_scores = np.zeros(self.n)
        for i in range(self.n):
          # x_diff_goal = goal_pos[0] - rollout_states[i,-1,0]
          # y_diff_goal = goal_pos[1] - rollout_states[i,-1,1]
          # goal_dist = np.sqrt(x_diff_goal**2+y_diff_goal**2)
          # rollout_scores[i] = rollout_scores[i] + goal_dist
          end_pos = np.array([rollout_states[i, -1, 0], rollout_states[i, -1, 1]])
          rollout_scores[i] += np.linalg.norm(end_pos - goal_pos)

          for j in range(int(self.T/self.timestep)):
            current_pos = np.zeros((1,2))
            current_pos[0,0] = rollout_states[i,j,0]
            current_pos[0,1] = rollout_states[i,j,1]
            # _, inObstacle = self.static_map.world_coordinates_to_map_indices(current_pos)
            # if inObstacle == True:
            #   rollout_scores[i] = rollout_scores[i] + 600
            #   break
            # print("Current_pos: ",current_pos)
            map_indices, _ = self.static_map.world_coordinates_to_map_indices(np.array([current_pos[0,0], current_pos[0,1]]))
            if self.static_map.static_map[map_indices[0],map_indices[1]]:
              rollout_scores[i] += 100
              break

        # print(rollout_scores)

        return rollout_scores

    def compute_control(self, rollout_scores, delta_v, delta_w):
        weighted_control = np.zeros((2,int(self.T/self.timestep)))
        sum_weight = 0
        sum_weight_delta_v = 0
        sum_weight_delta_w = 0
        for i in range(self.n):
          sum_weight = sum_weight + np.exp((-1/self.lamda)*rollout_scores[i])
          sum_weight_delta_v = sum_weight_delta_v + np.exp((-1/self.lamda)*rollout_scores[i])*delta_v[i,:]
          sum_weight_delta_w = sum_weight_delta_w + np.exp((-1/self.lamda)*rollout_scores[i])*delta_w[i,:]

        weighted_control[0] = self.nominal_control[0,0] + (sum_weight_delta_v/sum_weight)
        weighted_control[1] = self.nominal_control[0,1] + (sum_weight_delta_w/sum_weight)

        # print(weighted_control)

        return weighted_control

    def get_action(self, initial_state: np.array) -> np.array:
        """ Your implementation here """

        # goal_pos = np.zeros((1,2))

        goal_pos = self.waypoints[self.waypoint_index]

        # print(initial_state)
        # print(self.bottom_goal)
        # print(self.right_goal)

        if np.linalg.norm(initial_state[:2] - goal_pos) < 0.5:
          self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
          goal_pos = self.waypoints[self.waypoint_index]

        print(initial_state)
        print(goal_pos)
        # print(self.waypoint_index)

        rollout_states, delta_v, delta_w = self.compute_rollouts(initial_state,goal_pos)

        # self.plot_rollouts(rollout_states,initial_state,goal_pos)

        rollout_scores = self.score_rollouts(rollout_states,delta_v,delta_w,initial_state,goal_pos)

        all_actions = self.compute_control(rollout_scores, delta_v, delta_w)

        action = np.clip(all_actions[:,0], self.action_space.low, self.action_space.high)

        action = all_actions[:,0]

        # raise NotImplementedError
        return action

def run_planner_on_racetrack(
    env: gymnasium.Env,
    planner_class=MPPIRacetrack,
    seed: int = 0,
    num_laps: int = 3,
) -> int:

    np.random.seed(seed)
    obs, _ = env.reset()
    env.unwrapped.laps_left = num_laps

    # Create an instance of your planner
    planner = planner_class(static_map=env.unwrapped.map)

    # Draw a map of the environment with the finish line + initial position
    ax = env.unwrapped.map.draw_map(show=False)
    ax.plot(
        env.unwrapped.finish_line[:, 0],
        env.unwrapped.finish_line[:, 1],
        "g",
        lw=3,
    )
    ax.plot(obs[0], obs[1], "rx")

    # turn1 = np.array([-2.75,-1.5])
    # turn2 = np.array([-2.25,3.25])
    # turn3 = np.array([2.5,3.25])
    # turn4 = np.array([1.75,-1.75])

    turn1 = np.array([-2.5,-0.75])
    turn2 = np.array([-2,3])
    turn3 = np.array([1.25,4])
    turn4 = np.array([3.5,1.75])
    turn5 = np.array([2.25,-1.5])
    turn6 = np.array([0,-2])

    ax.plot(turn1[0],turn1[1], "kx")
    ax.plot(turn2[0],turn2[1], "kx")
    ax.plot(turn3[0],turn3[1], "kx")
    ax.plot(turn4[0],turn4[1], "kx")
    ax.plot(turn5[0],turn5[1], "kx")
    ax.plot(turn6[0],turn6[1], "kx")

    # Run the environment for num_timesteps, unless the robot hits an obstacle
    # or successfully completes the number of laps needed
    num_timesteps = 500
    success = False
    for t in range(num_timesteps):
        action = planner.get_action(obs)
        # print(action)
        obs, _, terminated, _, _ = env.step(action)

        ax.plot(obs[0], obs[1], "bx")

        # print(obs)
        print(t)
        print('-----------------')
        if terminated or t==499:
          ax.plot(obs[0], obs[1], "rx")

        if terminated:
            success = True
            break

    num_timesteps_used = t

    plt.title("Unicycle on Racetrack (MPPI)")
    plt.show()

    if success:
        return num_timesteps_used
    else:
        return -1


seed = 0
num_laps = 3
planner_class = MPPIRacetrack
num_timesteps_used = run_planner_on_racetrack(
    env, planner_class=planner_class, seed=seed, num_laps=num_laps
)
print(f"num timesteps used: {num_timesteps_used}")