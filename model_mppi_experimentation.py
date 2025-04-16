import numpy as np
import scipy as sp
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import njit

##### TWIP-R Motion Model Params (taken from "Dynamic Modeling of a Two-wheeled Inverted Pendulum Balancing Mobile Robot") #####
# m_p = 40 # pendulum mass in kg
# l_p = 0.69 # pendulum length in m
# r_w = 0.075 # wheel radius in m
# W_b = 0.215 # wheelbase in m
W_b = 0.59 # wheelbase in m
l_p = 0.14 # pendulum length in m
r_w = 0.2 # wheel radius in m
m_p = 41 # pendulum mass in kg
m_b = 4 # base mass in kg
J_inertia = 0.4
K_inertia = 0.2
I1 = 0
I2 = 0
I3 = 0
C_alpha = 0 # coefficient of viscous friction on the wheel axis, not explicitly listed

##### MPPI Params (taken from "Simultaneous Tracking and Balancing Control of Two-Wheeled Inverted Pendulum with Roll-joint using Dynamic Variance MPPI") #####
n = 2048 # number of rollouts per input
T = 1.1 # time horizon in sec
timestep = 0.01 # time step in sec
lamda = 1 # I know it's spelled wrong, I just need to make it distinct from the lambda that already exists
K = 1
mu = 0

@njit
def motion_model(prev_state, control_input, dt):
    # x_m, y_m, v_m, phi_m, phi_m_dot, alpha, alpha_dot, beta, beta_dot = prev_state
    # v_m_dot, phi_m_ddot, beta_ddot = control_input

    # print(prev_state)

    x_m = prev_state[0]
    y_m = prev_state[1]
    v_m = prev_state[2]
    phi_m = prev_state[3]
    phi_m_dot = prev_state[4]
    alpha = prev_state[5]
    alpha_dot = prev_state[6]
    beta = prev_state[7]
    beta_dot = prev_state[8]

    v_m_dot = control_input[0]
    phi_m_ddot = control_input[1]
    beta_ddot = control_input[2]

    ### Equations from M*v + h = B*tau ###
    m_11 = m_p + 2*m_b + 2*J_inertia/r_w**2
    m_12 = m_p*l_p*np.cos(alpha)
    m_13 = 0
    m_14 = 0
    m_22 = I2 + m_p*l_p**2
    m_23 = 0

    c_12 = -m_p*l_p*alpha_dot*np.sin(alpha)
    c_13 = -m_p*l_p*phi_m_dot*np.sin(alpha)
    c_23 = (I3 - I1 - m_p*l_p**2)*phi_m_dot*np.sin(alpha)*np.cos(alpha)
    # c_31 = m_p*l_p*phi_m_dot*np.sin(alpha)
    # c_32 = -(I3 - I1 - m_p*l_p**2)*phi_m_dot*np.sin(alpha)*np.cos(alpha)
    # c_33 = -(I3 - I1 - m_p*l_p**2)*alpha_dot*np.sin(alpha)*np.cos(alpha)

    # d_11 = 2*C_alpha/r_w**2
    d_12 = -2*C_alpha/r_w**2
    d_21 = d_12
    d_22 = 2*C_alpha
    # d_33 = (W_b**2/(2*r_w**2))*C_alpha

    ### Decoupling System Using PFL ###
    a_p = v_m_dot
    a_y = phi_m_ddot
    a_r = beta_ddot

    b_1 = -(r_w*m_11 + m_12)/(r_w*m_12 + m_22)
    b_2 = -(r_w*m_13 + m_23)/(r_w*m_12 + m_22)
    b_3 = r_w*m_14

    h_1 = alpha_dot*c_12 + phi_m_dot*c_13 + v_m*d_12
    h_2 = phi_m_dot*c_23 + v_m*d_21 + alpha_dot*d_22 - m_p*l_p*np.sin(alpha)

    # Position System
    d_dt_x_m = v_m*np.cos(phi_m)
    d_dt_y_m = v_m*np.sin(phi_m)

    # Pitch System
    d_dt_alpha = alpha_dot
    d_dt_alpha_dot = b_1*a_p + b_2*a_y + b_3*a_r + r_w*h_1 +h_2
    d_dt_v_m = v_m_dot

    # Roll System
    d_dt_beta = beta_dot
    d_dt_beta_dot = a_r

    # Yaw System
    d_dt_phi_m = phi_m_dot
    d_dt_phi_m_dot = a_y

    ### Update State ###
    v_m_next = v_m + d_dt_v_m*dt
    x_m_next = x_m + d_dt_x_m*dt
    y_m_next = y_m + d_dt_y_m*dt

    alpha_dot_next = alpha_dot + d_dt_alpha_dot*dt
    alpha_next = alpha + d_dt_alpha*dt

    phi_m_dot_next = phi_m_dot + d_dt_phi_m_dot*dt
    phi_m_next = phi_m + d_dt_phi_m*dt

    beta_dot_next = beta_dot + d_dt_beta_dot*dt
    beta_next = beta + d_dt_beta*dt

    next_state = np.array([x_m_next, y_m_next, v_m_next, phi_m_next, phi_m_dot_next, alpha_next, alpha_dot_next, beta_next, beta_dot_next])

    return next_state

def compute_rollouts(initial_state, A):
    epsilon_p = np.random.normal(0, A, (n, int(T/timestep)))
    epsilon_r = np.random.normal(0, A, (n, int(T/timestep)))
    epsilon_y = np.random.normal(0, A, (n, int(T/timestep)))

    rollout_states_p = np.zeros((n, int(T/timestep), 9))
    rollout_states_p[:,0,0] = initial_state[0]
    rollout_states_p[:,0,1] = initial_state[1]
    rollout_states_p[:,0,2] = initial_state[5]
    rollout_states_p[:,0,3] = initial_state[6]
    rollout_states_p[:,0,4] = initial_state[3]

    rollout_states_r = np.zeros((n, int(T/timestep), 9))
    rollout_states_r[:, 0, 0] = initial_state[5]
    rollout_states_r[:, 0, 1] = initial_state[6]
    rollout_states_r[:, 0, 2] = initial_state[7]
    rollout_states_r[:, 0, 3] = initial_state[8]

    rollout_states_y = np.zeros((n, int(T/timestep), 9))
    rollout_states_y[:, 0, 0] = initial_state[0]
    rollout_states_y[:, 0, 1] = initial_state[1]
    rollout_states_y[:, 0, 2] = initial_state[5]
    rollout_states_y[:, 0, 3] = initial_state[6]
    rollout_states_y[:, 0, 4] = initial_state[3]
    rollout_states_y[:, 0, 5] = initial_state[4]

    for i in range(n):
        for j in range(1,int(T/timestep)):
            action_p = epsilon_p[i,j-1]
            action_r = epsilon_r[i,j-1]
            action_y = epsilon_y[i,j-1]

            rollout_states_p[i,j] = motion_model(rollout_states_p[i,j-1], np.array([action_p, 0, 0]), timestep)
            rollout_states_r[i,j] = motion_model(rollout_states_r[i,j-1], np.array([0, action_r, 0]), timestep)
            rollout_states_y[i,j] = motion_model(rollout_states_y[i,j-1], np.array([0, 0, action_y]), timestep)

    return rollout_states_p, rollout_states_r, rollout_states_y, epsilon_p, epsilon_r, epsilon_y

def score_rollouts(rollout_states_p, rollout_states_r, rollout_states_y, goal_state):
    scores_p = np.zeros(n)
    scores_r = np.zeros(n)
    scores_y = np.zeros(n)

    for i in range(n):
        ### Scoring pitch rollouts based on deviations from goal state ###
        x_diff_p = rollout_states_p[i,-1,0] - goal_state[0]
        y_diff_p = rollout_states_p[i,-1,1] - goal_state[1]
        goal_dist_p = np.sqrt(x_diff_p**2 + y_diff_p**2)
        alpha_diff_p = np.abs(rollout_states_p[i,-1,2] - goal_state[5])
        alpha_dot_diff_p = np.abs(rollout_states_p[i,-1,3] - goal_state[6])
        v_m_diff_p = np.abs(rollout_states_p[i,-1,4] - goal_state[3])
        scores_p[i] = goal_dist_p + alpha_diff_p + alpha_dot_diff_p + v_m_diff_p

        ### Scoring roll rollouts based on deviations from goal state ###
        alpha_diff_r = np.abs(rollout_states_r[i,-1,0] - goal_state[5])
        alpha_dot_diff_r = np.abs(rollout_states_r[i,-1,1] - goal_state[6])
        beta_diff_r = np.abs(rollout_states_r[i,-1,2] - goal_state[7])
        beta_dot_diff_r = np.abs(rollout_states_r[i,-1,3] - goal_state[8])
        scores_r[i] = alpha_diff_r + alpha_dot_diff_r + beta_diff_r + beta_dot_diff_r

        ### Scoring pitch rollouts based on deviations from goal state ###
        x_diff_y = rollout_states_y[i,-1,0] - goal_state[0]
        y_diff_y = rollout_states_y[i,-1,1] - goal_state[1]
        goal_dist_y = np.sqrt(x_diff_y**2 + y_diff_y**2)
        alpha_diff_y = np.abs(rollout_states_y[i,-1,2] - goal_state[5])
        alpha_dot_diff_y = np.abs(rollout_states_y[i,-1,3] - goal_state[6])
        phi_m_diff_y = np.abs(rollout_states_y[i,-1,4] - goal_state[3])
        phi_m_dot_diff_y = np.abs(rollout_states_y[i,-1,5] - goal_state[4])
        scores_y[i] = goal_dist_y + alpha_diff_y + alpha_dot_diff_y + phi_m_diff_y + phi_m_dot_diff_y

    return scores_p, scores_r, scores_y

def compute_control(scores_p, scores_r, scores_y, epsilon_p, epsilon_r, epsilon_y):
    weighted_control = np.zeros((3,int(T/timestep)))
    sum_weight_p = 0
    sum_weight_r = 0
    sum_weight_y = 0
    sum_weight_epsilon_p = 0
    sum_weight_epsilon_r = 0
    sum_weight_epsilon_y = 0

    for i in range(n):
        sum_weight_p += np.exp((-1/lamda)*scores_p[i])
        sum_weight_r += np.exp((-1/lamda)*scores_r[i])
        sum_weight_y += np.exp((-1/lamda)*scores_y[i])

        sum_weight_epsilon_p += np.exp((-1/lamda)*scores_p[i])*epsilon_p[i]
        sum_weight_epsilon_r += np.exp((-1/lamda)*scores_r[i])*epsilon_r[i]
        sum_weight_epsilon_y += np.exp((-1/lamda)*scores_y[i])*epsilon_y[i]

    weighted_control[0,:] = sum_weight_epsilon_p/sum_weight_p
    weighted_control[1,:] = sum_weight_epsilon_r/sum_weight_r
    weighted_control[2,:] = sum_weight_epsilon_y/sum_weight_y

    return weighted_control

def get_action(initial_state, goal_state):
    rollout_states_p, rollout_states_r, rollout_states_y, epsilon_p, epsilon_r, epsilon_y = compute_rollouts(initial_state, 2)

    scores_p, scores_r, scores_y = score_rollouts(rollout_states_p, rollout_states_r, rollout_states_y, goal_state)

    all_actions = compute_control(scores_p, scores_r, scores_y, epsilon_p, epsilon_r, epsilon_y)

    action = all_actions[:,0]

    return action

def plot_path(recorded_states, reference_states, time, title):
    plt.figure(figsize=(10, 5))
    plt.plot(recorded_states[0, :], recorded_states[1, :], label='Recorded Path')
    plt.plot(reference_states[0, :], reference_states[1, :], label='Reference Path', linestyle='--')
    plt.title(title)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid()
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(time, recorded_states[5, :], label='Recorded Alpha')
    plt.plot(time, reference_states[5, :], label='Reference Alpha', linestyle='--')
    plt.plot(time, recorded_states[7, :], label='Recorded Beta')
    plt.plot(time, reference_states[7, :], label='Reference Beta', linestyle='--')
    plt.plot(time, recorded_states[3, :], label='Recorded Phi_m ')
    plt.plot(time, reference_states[3, :], label='Reference Phi_m ', linestyle='--')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid()
    # plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(time, recorded_states[2, :], label='Recorded V_m')
    plt.plot(time, reference_states[2, :], label='Reference V_m', linestyle='--')
    plt.plot(time, recorded_states[6, :], label='Recorded Alpha Dot')
    plt.plot(time, reference_states[6, :], label='Reference Alpha Dot', linestyle='--')
    plt.plot(time, recorded_states[8, :], label='Recorded Beta Dot')
    plt.plot(time, reference_states[8, :], label='Reference Beta Dot', linestyle='--')
    plt.plot(time, recorded_states[4, :], label='Recorded Phi_m Dot')
    plt.plot(time, reference_states[4, :], label='Reference Phi_m Dot', linestyle='--')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Linear Velocity (m/s) / Angular Velocity (rad/s)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Initial state: [x_m, y_m, v_m, phi_m, phi_m_dot, alpha, alpha_dot, beta, beta_dot]
    initial_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Time vector
    time_horizon = np.arange(0, 10 + timestep, timestep)

    x_ref = np.zeros(time_horizon.shape)
    y_ref = np.zeros(time_horizon.shape)

    alpha_ref = np.zeros(time_horizon.shape)
    beta_ref = np.zeros(time_horizon.shape)
    phi_m_ref = np.zeros(time_horizon.shape)

    v_m_ref = np.zeros(time_horizon.shape)
    alpha_dot_ref = np.zeros(time_horizon.shape)
    phi_m_dot_ref = np.zeros(time_horizon.shape)
    beta_dot_ref = np.zeros(time_horizon.shape)

    for i in range(len(time_horizon)):
        x_ref[i] = 5.25*time_horizon[i]
        y_ref[i] = 1 - np.cos((2*np.pi/15)*x_ref[i])

        alpha_ref[i] = 0
        # beta_ref[i] = # beta_ref needs to be calculated based on phi_m_dot_ref

        if i == 0:
            phi_m_ref[i] = (np.arctan2(y_ref[i], x_ref[i]))
        else:
            phi_m_ref[i] = (np.arctan2(y_ref[i] - y_ref[i-1], x_ref[i] - x_ref[i-1]))

        # phi_m_ref[i] = np.arctan2(y_ref[i], x_ref[i])

        if time_horizon[i] < 3:
            v_m_ref[i] = (4/3)*time_horizon[i]
        else:
            v_m_ref[i] = 4

        alpha_dot_ref[i] = 0

        if time_horizon[i] == 0:
            phi_m_dot_ref[i] = 0
        else:
            phi_m_dot_ref[i] = (phi_m_ref[i] - phi_m_ref[i-1])/timestep

        beta_ref[i] = ((-(v_m_ref[i]*phi_m_dot_ref[i])*(l_p*m_p + (m_b + m_p)*r_w))/(m_p*l_p*-9.81))

        if time_horizon[i] == 0:
            beta_dot_ref[i] = 0
        else:
            beta_dot_ref[i] = (beta_ref[i] - beta_ref[i-1])/timestep

    # Reference state: [x_m, y_m, v_m, phi_m, phi_m_dot, alpha, alpha_dot, beta, beta_dot]
    # reference_state = [x_ref, y_ref, v_m_ref, phi_m_ref, phi_m_dot_ref, alpha_ref, alpha_dot_ref, beta_ref, beta_dot_ref]

    recorded_states = np.zeros((9, len(time_horizon)))

    # for i in range(len(time_horizon)):
    #     # Get the action using MPPI
    #     action = get_action(initial_state, np.array([x_ref[i], y_ref[i], v_m_ref[i], phi_m_ref[i], phi_m_dot_ref[i], alpha_ref[i], alpha_dot_ref[i], beta_ref[i], beta_dot_ref[i]]))
    #
    #     # Update the state using the motion model
    #     new_state = motion_model(initial_state, action, timestep)
    #
    #     initial_state = new_state.copy()
    #
    #     # Store the recorded states
    #     recorded_states[:, i] = new_state
    #
    #     print(i)

    reference_states = np.vstack([
        x_ref,
        y_ref,
        v_m_ref,
        phi_m_ref,
        phi_m_dot_ref,
        alpha_ref,
        alpha_dot_ref,
        beta_ref,
        beta_dot_ref
    ])

    plot_path(recorded_states, reference_states, time_horizon, 'S-Curve Trajectory with Fixed Variance = 2')