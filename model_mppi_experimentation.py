import numpy as np
import scipy as sp
from scipy.integrate import cumulative_trapezoid as cum_traps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import njit
from numba import jit

##### TWIP-R Motion Model Params (taken from "Dynamic Modeling of a Two-wheeled Inverted Pendulum Balancing Mobile Robot") #####
# m_p = 40 # pendulum mass in kg
# l_p = 0.69 # pendulum length in m
# r_w = 0.075 # wheel radius in m
# W_b = 0.215 # wheelbase in m
d = 0.59 # wheelbase in m
l = 0.14 # pendulum length in m
R = 0.2 # wheel radius in m
m_W = 41 # pendulum mass in kg
m_B = 4 # base mass in kg
J_inertia = 0.4
K_inertia = 0.2
I_1 = 0
I_2 = 0
I_3 = 0
C_alpha = 0 # coefficient of viscous friction on the wheel axis, not explicitly listed

##### MPPI Params (taken from "Simultaneous Tracking and Balancing Control of Two-Wheeled Inverted Pendulum with Roll-joint using Dynamic Variance MPPI") #####
# n = 2048 # number of rollouts per input
n = 50
T = 1.1 # time horizon in sec
timestep = 0.01 # time step in sec
lamda = 1 # I know it's spelled wrong, I just need to make it distinct from the lambda that already exists
K = 1
mu = 0

# @njit
# @jit(nopython=True)
def motion_model(prev_state, control_input, dt):
    # x_m, y_m, v_m, phi_m, phi_m_dot, alpha, alpha_dot, ~~beta~~, ~~beta_dot~~ = prev_state
    # v_m_dot, phi_m_ddot, ~~beta_ddot~~ = control_input

    # print(prev_state)
    # print("prev_state dtype:", prev_state.dtype)
    # print("control_input dtype:", control_input.dtype)

    x_m = prev_state[0]
    y_m = prev_state[1]
    v_m = prev_state[2]
    phi_m = prev_state[3]
    phi_m_dot = prev_state[4]
    alpha = prev_state[5]
    alpha_dot = prev_state[6]
    # beta = prev_state[7]
    # beta_dot = prev_state[8]

    v_m_dot = control_input[0]
    phi_m_ddot = control_input[2]
    # beta_ddot = control_input[2]

    ############### Disregard all of this ###############
    ### Equations from M*v + h = B*tau ###
    # m_11 = m_p + 2*m_b + 2*J_inertia/r_w**2
    # m_12 = m_p*l_p*np.cos(alpha)
    # # m_13 = 0
    # m_13 = 1
    # # m_14 = 0
    # m_14 = 1
    # m_22 = I2 + m_p*l_p**2
    # # m_23 = 0
    # m_23 = 1
    #
    # c_12 = -m_p*l_p*alpha_dot*np.sin(alpha)
    # c_13 = -m_p*l_p*phi_m_dot*np.sin(alpha)
    # c_23 = (I3 - I1 - m_p*l_p**2)*phi_m_dot*np.sin(alpha)*np.cos(alpha)
    # # c_31 = m_p*l_p*phi_m_dot*np.sin(alpha)
    # # c_32 = -(I3 - I1 - m_p*l_p**2)*phi_m_dot*np.sin(alpha)*np.cos(alpha)
    # # c_33 = -(I3 - I1 - m_p*l_p**2)*alpha_dot*np.sin(alpha)*np.cos(alpha)
    #
    # # d_11 = 2*C_alpha/r_w**2
    # d_12 = -2*C_alpha/r_w**2
    # d_21 = d_12
    # d_22 = 2*C_alpha
    # # d_33 = (W_b**2/(2*r_w**2))*C_alpha

    ############### This is actually the right stuff ###############
    m_11 = m_W + 2*m_B + 2*J_inertia/R**2
    m_12 = m_W*l*np.cos(alpha)
    m_13 = 0
    m_21 = m_12
    m_22 = I_2 + m_W*l**2
    m_23 = 0
    m_31 = 0
    m_32 = 0
    m_33 = I_3 + 2*K + (m_W + J_inertia/R**2)*d**2/2 - (I_3 - I_1 - m_B*l**2)*np.sin(alpha)**2
    M_matrix = np.array([[m_11, m_12, m_13],[m_21, m_22, m_23],[m_31, m_32, m_33]], dtype=np.float64)

    c_11 = 0
    c_12 = -m_B*l*alpha_dot*np.sin(alpha)
    c_13 = -m_B*l*phi_m_dot*np.sin(alpha)
    c_21 = 0
    c_22 = 0
    c_23 = (I_3 - I_1 - m_B*l**2)*phi_m_dot*np.sin(alpha)*np.cos(alpha)
    c_31 = m_B*l*phi_m_dot*np.sin(alpha)
    c_32 = -(I_3 - I_1 - m_B*l**2)*phi_m_dot*np.sin(alpha)*np.cos(alpha)
    c_33 = -(I_3 - I_1 - m_B*l**2)*alpha_dot*np.sin(alpha)*np.cos(alpha)
    C_matrix = np.array([[c_11, c_12, c_13],[c_21, c_22, c_23],[c_31, c_32, c_33]], dtype=np.float64)

    d_11 = 2*C_alpha/R**2
    d_12 = -2*C_alpha/R
    d_13 = 0
    d_21 = d_12
    d_22 = 2*C_alpha
    d_23 = 0
    d_31 = 0
    d_32 = 0
    d_33 = (d**2/(2*R**2))*C_alpha
    D_matrix = np.array([[d_11, d_12, d_13],[d_21, d_22, d_23],[d_31, d_32, d_33]], dtype=np.float64)
    # print("D_matrix:", D_matrix, D_matrix.dtype)

    B_matrix = np.array([[1/R,1/R],[-1,-1],[-d/(2*R),d/(2*R)]], dtype=np.float64)
    # print("B_matrix:", B_matrix, B_matrix.dtype)

    # G_matrix = np.array([[0],[-m_B*l*9.81*np.sin(alpha)],[0]], dtype=np.float64)
    G_matrix = np.empty((3,1), dtype=np.float64)
    G_matrix[0, 0] = 0.0
    G_matrix[1, 0] = np.float64(-m_B*l*9.81*np.sin(alpha))
    G_matrix[2, 0] = 0.0
    # print("G_matrix:", G_matrix, G_matrix.dtype)

    # q_dot = np.array([[v_m],[alpha_dot],[phi_m_ddot]], dtype=np.float64)
    q_dot = np.empty((3, 1), dtype=np.float64)
    q_dot[0, 0] = np.float64(v_m)
    q_dot[1, 0] = np.float64(alpha_dot)
    q_dot[2, 0] = np.float64(phi_m_ddot)

    # print("M_matrix: ", M_matrix)
    # print("C_matrix: ", C_matrix)
    # print("D_matrix: ", D_matrix)
    # print("B_matrix: ", B_matrix)
    # print("G_matrix: ", G_matrix)
    # print("q_dot: ", q_dot)

    # h = C_matrix@q_dot + D_matrix@q_dot + G_matrix
    h = np.dot(C_matrix, q_dot) + np.dot(D_matrix, q_dot) + G_matrix

    # S = np.array([[1,0,0],[0,0,1]], dtype=np.float64)
    # v = np.array([v_m_dot, phi_m_ddot], dtype=np.float64)
    S = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    v = np.array([np.float64(v_m_dot), np.float64(phi_m_ddot)], dtype=np.float64)
    # v = np.empty((2, 1), dtype=np.float64)
    # v[0, 0] = v_m_dot
    # v[1, 0] = phi_m_ddot

    # S = np.array(S, dtype=np.float64, order='C')
    # S = np.ascontiguousarray(S, dtype=np.float64)

    # M_matrix = np.array(M_matrix, dtype=np.float64, order='C')
    # M_matrix = np.ascontiguousarray(M_matrix, dtype=np.float64)

    # B_matrix = np.array(B_matrix, dtype=np.float64, order='C')
    # B_matrix = np.ascontiguousarray(B_matrix, dtype=np.float64)

    M_inv = np.linalg.inv(M_matrix)
    # Phi = S@M_inv@B_matrix
    # eta = S@M_inv@h
    Phi = np.dot(S, np.dot(M_inv, B_matrix))
    eta = np.dot(S, np.dot(M_inv, h))

    # tau = np.linalg.solve(Phi, v + eta)
    tau = np.linalg.solve(Phi, v + eta.flatten())

    theta_index = 1
    # e_theta = np.zeros(3)
    # e_theta[theta_index] = 1.0
    e_theta = np.empty((3, 1), dtype=np.float64)
    e_theta[0, 0] = np.float64(0.0)
    e_theta[1, 0] = np.float64(1.0)
    e_theta[2, 0] = np.float64(0.0)
    # alpha_ddot = e_theta@M_inv@(B_matrix@tau - h)
    # alpha_ddot = alpha_ddot[0]
    # alpha_ddot = np.dot(e_theta, np.dot(M_inv, np.dot(B_matrix, tau) - h.flatten()))

    # print("B_matrix shape:", B_matrix.shape)  # (3, 2)
    # print("tau shape:", tau.shape)  # (2,)
    # print("B_matrix @ tau shape:", (B_matrix @ tau).shape)  # Should be (3,)
    # print("h.flatten() shape:", h.flatten().shape)  # Should be (3,)
    # print("B_matrix @ tau - h.flatten() shape:", (B_matrix @ tau - h.flatten()).shape)
    #
    # dot1 = B_matrix @ tau
    # dot2 = dot1 - h.flatten()
    # dot3 = M_inv @ dot2
    # print("M_inv shape:", M_inv.shape)  # Should be (3, 3)
    # print("M_inv @ (...) shape:", dot3.shape)  # Should be (3,)
    # print("e_theta shape:", e_theta.shape)  # Should be (3,)

    # alpha_ddot = float(np.dot(e_theta.reshape(1, 3), np.dot(M_inv, np.dot(B_matrix, tau) - h.flatten())))
    alpha_ddot = np.dot(e_theta.reshape(1, 3), np.dot(M_inv, np.dot(B_matrix, tau) - h.flatten())).item()

    # print("alpha_ddot: ", alpha_ddot)

    ############### Disregard this too ###############
    ### Decoupling System Using PFL ###
    # a_p = v_m_dot
    # a_y = phi_m_ddot
    # a_r = beta_ddot

    # b_1 = -(r_w*m_11 + m_12)/(r_w*m_12 + m_22)
    # b_2 = -(r_w*m_13 + m_23)/(r_w*m_12 + m_22)
    # b_3 = r_w*m_14

    # h_1 = alpha_dot*c_12 + phi_m_dot*c_13 + v_m*d_12
    # h_2 = phi_m_dot*c_23 + v_m*d_21 + alpha_dot*d_22 - m_p*l_p*np.sin(alpha)

    ############### This is correct ###############

    # Position System
    d_dt_x_m = v_m*np.cos(phi_m)
    d_dt_y_m = v_m*np.sin(phi_m)

    # Pitch System
    d_dt_alpha = alpha_dot
    d_dt_alpha_dot = alpha_ddot
    d_dt_v_m = v_m_dot

    # Roll System
    # d_dt_beta = beta_dot
    # d_dt_beta_dot = a_r

    # Yaw System
    d_dt_phi_m = phi_m_dot
    d_dt_phi_m_dot = phi_m_ddot

    ### Update State ###
    v_m_next = v_m + d_dt_v_m*dt
    x_m_next = x_m + d_dt_x_m*dt
    y_m_next = y_m + d_dt_y_m*dt

    alpha_dot_next = alpha_dot + d_dt_alpha_dot*dt
    alpha_next = alpha + d_dt_alpha*dt

    phi_m_dot_next = phi_m_dot + d_dt_phi_m_dot*dt
    phi_m_next = phi_m + d_dt_phi_m*dt

    # beta_dot_next = beta_dot + d_dt_beta_dot*dt
    # beta_next = beta + d_dt_beta*dt

    # next_state = np.array([x_m_next, y_m_next, v_m_next, phi_m_next, phi_m_dot_next, alpha_next, alpha_dot_next, beta_next, beta_dot_next])
    # next_state = np.array([x_m_next, y_m_next, v_m_next, phi_m_next, phi_m_dot_next, alpha_next, alpha_dot_next], dtype=np.float64)

    # print("x_m_next:", x_m_next, type(x_m_next))
    # print("phi_m_dot_next:", phi_m_dot_next, type(phi_m_dot_next))
    # print("alpha_dot_next:", alpha_dot_next, type(alpha_dot_next))

    # next_state = np.array([
    #     np.float64(x_m_next),
    #     np.float64(y_m_next),
    #     np.float64(v_m_next),
    #     np.float64(phi_m_next),
    #     np.float64(phi_m_dot_next),
    #     np.float64(alpha_next),
    #     np.float64(alpha_dot_next)
    # ], dtype=np.float64)

    next_state = np.empty(7, dtype=np.float64)
    next_state[0] = np.float64(x_m_next)
    next_state[1] = np.float64(y_m_next)
    next_state[2] = np.float64(v_m_next)
    next_state[3] = np.float64(phi_m_next)
    next_state[4] = np.float64(phi_m_dot_next)
    next_state[5] = np.float64(alpha_next)
    next_state[6] = np.float64(alpha_dot_next)

    # print("motion_model result type:", type(next_state), next_state.dtype, next_state.shape)

    # return np.asarray(next_state, dtype=np.float64)

    return next_state

# @njit
def compute_rollouts(initial_state, A):
    epsilon_p = np.random.normal(0, A, (n, int(T/timestep)))
    # epsilon_r = np.random.normal(0, A, (n, int(T/timestep)))
    epsilon_y = np.random.normal(0, A, (n, int(T/timestep)))

    initial_state = np.asarray(initial_state, dtype=np.float64)

    # print("initial_state type:", type(initial_state), initial_state.dtype)

    # rollout_states_p = np.zeros((n, int(T/timestep), 7), dtype=np.float64)
    # rollout_states_p[:,0,0] = initial_state[0]
    # rollout_states_p[:,0,1] = initial_state[1]
    # rollout_states_p[:,0,2] = initial_state[2]
    # rollout_states_p[:,0,5] = initial_state[5]
    # rollout_states_p[:,0,6] = initial_state[6]

    rollout_states = np.zeros((n, int(T/timestep), 7), dtype=np.float64)
    rollout_states[:, 0, 0] = initial_state[0]
    rollout_states[:, 0, 1] = initial_state[1]
    rollout_states[:, 0, 2] = initial_state[2]
    rollout_states[:, 0, 3] = initial_state[3]
    rollout_states[:, 0, 4] = initial_state[4]
    rollout_states[:, 0, 5] = initial_state[5]
    rollout_states[:, 0, 6] = initial_state[6]

    # print("rollout_states_p type:", type(rollout_states_p), rollout_states_p.dtype)

    # rollout_states_r = np.zeros((n, int(T/timestep), 9))
    # rollout_states_r[:, 0, 0] = initial_state[5]
    # rollout_states_r[:, 0, 1] = initial_state[6]
    # rollout_states_r[:, 0, 2] = initial_state[7]
    # rollout_states_r[:, 0, 3] = initial_state[8]

    # rollout_states_y = np.zeros((n, int(T/timestep), 7), dtype=np.float64)
    # rollout_states_y[:, 0, 0] = initial_state[0]
    # rollout_states_y[:, 0, 1] = initial_state[1]
    # rollout_states_y[:, 0, 3] = initial_state[3]
    # rollout_states_y[:, 0, 4] = initial_state[4]
    # rollout_states_y[:, 0, 5] = initial_state[5]
    # rollout_states_y[:, 0, 6] = initial_state[6]

    # print("rollout_states_y type:", type(rollout_states_y), rollout_states_y.dtype)

    for i in range(n):
        for j in range(1,int(T/timestep)):
            action_p = epsilon_p[i,j-1]
            # print("action_p:", action_p, type(action_p))
            # print(action_p)
            # action_r = epsilon_r[i,j-1]
            action_y = epsilon_y[i,j-1]

            # print("rollout_states_p.dtype:", rollout_states_p.dtype)
            # print("rollout_states_p.shape:", rollout_states_p.shape)
            # print("rolloutstates_p[i,j-1].dtype:", np.asarray(rollout_states_p[i,j-1], dtype=np.float64).dtype)
            # print(type(np.asarray(rollout_states_p[i,j-1], dtype=np.float64)))
            # print(type(action_p))
            # print(rollout_states_p[i, j - 1].shape)

            # print("motion_model.dtype:", motion_model(np.asarray(rollout_states_p[i,j-1], dtype=np.float64), np.array([action_p, 0.0, 0.0], dtype=np.float64), timestep).dtype)
            # print("motion_model shape", motion_model(np.asarray(rollout_states_p[i,j-1], dtype=np.float64), np.array([action_p, 0.0, 0.0], dtype=np.float64), timestep).shape)

            # rollout_states_p[i,j] = np.asarray(motion_model(np.asarray(rollout_states_p[i,j-1], dtype=np.float64), np.array([np.float64(action_p), np.float64(0.0), np.float64(0.0)], dtype=np.float64), timestep))
            # state = np.array(rollout_states_p[i, j - 1], dtype=np.float64)
            # action = np.array([action_p, 0.0, 0.0], dtype=np.float64)
            # rollout_states_p[i, j, :] = motion_model(state, action, timestep)
            # # rollout_states_r[i,j] = motion_model(rollout_states_r[i,j-1], np.array([0, action_r, 0]), timestep)
            # rollout_states_y[i,j,:] = motion_model(rollout_states_y[i,j-1].astype(np.float64), np.array([0.0, 0.0, action_y], dtype=np.float64), timestep)
            state = rollout_states[i,j-1,:]
            action = np.array([action_p, 0, action_y])
            rollout_states[i,j,:] = motion_model(state, action, timestep)

    # return rollout_states_p, rollout_states_r, rollout_states_y, epsilon_p, epsilon_r, epsilon_y
    # return rollout_states_p, rollout_states_y, epsilon_p, epsilon_y
    return rollout_states, epsilon_p, epsilon_y

def score_rollouts(rollout_states, goal_state):
    # scores_p = np.zeros(n)
    # # scores_r = np.zeros(n)
    # scores_y = np.zeros(n)
    scores = np.zeros(n)

    for i in range(n):
        x_diff = np.abs(rollout_states[i,-1,0] - goal_state[0])
        y_diff = np.abs(rollout_states[i,-1,1] - goal_state[1])
        v_m_diff = np.abs(rollout_states[i,-1,2] - goal_state[2])
        phi_m_diff = np.abs(rollout_states[i,-1,3] - goal_state[3])
        phi_m_dot_diff = np.abs(rollout_states[i,-1,4] - goal_state[4])
        alpha_diff = np.abs(rollout_states[i,-1,5] - goal_state[5])
        alpha_dot_diff = np.abs(rollout_states[i,-1,6] - goal_state[6])
        scores[i] = x_diff + y_diff + v_m_diff + phi_m_diff + phi_m_dot_diff + alpha_diff + alpha_dot_diff

    print("Min Score:", np.min(scores))
    print("Max Score:", np.max(scores))
    print("Scores: ", scores)

    # for i in range(n):
    #     ### Scoring pitch rollouts based on deviations from goal state ###
    #     x_diff_p = rollout_states_p[i,-1,0] - goal_state[0]
    #     y_diff_p = rollout_states_p[i,-1,1] - goal_state[1]
    #     goal_dist_p = np.sqrt(x_diff_p**2 + y_diff_p**2)
    #     alpha_diff_p = np.abs(rollout_states_p[i,-1,2] - goal_state[5])
    #     alpha_dot_diff_p = np.abs(rollout_states_p[i,-1,3] - goal_state[6])
    #     v_m_diff_p = np.abs(rollout_states_p[i,-1,4] - goal_state[3])
    #     scores_p[i] = goal_dist_p + alpha_diff_p + alpha_dot_diff_p + v_m_diff_p
    #
    #     ### Scoring roll rollouts based on deviations from goal state ###
    #     alpha_diff_r = np.abs(rollout_states_r[i,-1,0] - goal_state[5])
    #     alpha_dot_diff_r = np.abs(rollout_states_r[i,-1,1] - goal_state[6])
    #     beta_diff_r = np.abs(rollout_states_r[i,-1,2] - goal_state[7])
    #     beta_dot_diff_r = np.abs(rollout_states_r[i,-1,3] - goal_state[8])
    #     scores_r[i] = alpha_diff_r + alpha_dot_diff_r + beta_diff_r + beta_dot_diff_r
    #
    #     ### Scoring pitch rollouts based on deviations from goal state ###
    #     x_diff_y = rollout_states_y[i,-1,0] - goal_state[0]
    #     y_diff_y = rollout_states_y[i,-1,1] - goal_state[1]
    #     goal_dist_y = np.sqrt(x_diff_y**2 + y_diff_y**2)
    #     alpha_diff_y = np.abs(rollout_states_y[i,-1,2] - goal_state[5])
    #     alpha_dot_diff_y = np.abs(rollout_states_y[i,-1,3] - goal_state[6])
    #     phi_m_diff_y = np.abs(rollout_states_y[i,-1,4] - goal_state[3])
    #     phi_m_dot_diff_y = np.abs(rollout_states_y[i,-1,5] - goal_state[4])
    #     scores_y[i] = goal_dist_y + alpha_diff_y + alpha_dot_diff_y + phi_m_diff_y + phi_m_dot_diff_y

    # Cost function: q_i = (state_ref_i - state_i).T*Q*(state_ref_i - state_i) + prev_action_i*R*prev_action_i

    # Q_p = np.array([[1, 0, 0, 0, 0],[0, 10, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 1]], dtype=np.float64)
    # # Q_r = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    # Q_y = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], dtype=np.float64)
    # R = 1

    # for i in range(n):
    #     ### Pitch Cost ###
    #     pitch_state = np.array([rollout_states_p[i,-1,0], rollout_states_p[i,-1,1], rollout_states_p[i,-1,2], rollout_states_p[i,-1,5], rollout_states_p[i,-1,6]], dtype=np.float64)
    #     pitch_ref = np.array([goal_state[0], goal_state[1], goal_state[2], goal_state[5], goal_state[6]], dtype=np.float64)
    #     scores_p[i] = (pitch_ref - pitch_state).T @ Q_p @ (pitch_ref - pitch_state) + prev_action[0]*R*prev_action[0]
    #
    #     ### Roll Cost ###
    #     # roll_state = np.array([rollout_states_r[i, -1, 5], rollout_states_r[i, -1, 6], rollout_states_r[i, -1, 7], rollout_states_r[i, -1, 8]])
    #     # roll_ref = np.array([goal_state[5], goal_state[6], goal_state[7], goal_state[8]])
    #     # scores_r[i] = (roll_ref - roll_state).T @ Q_r @ (roll_ref - roll_state) + prev_action[1]*R*prev_action[1]
    #
    #     ### Yaw Cost ###
    #     yaw_state = np.array([rollout_states_y[i, -1, 0], rollout_states_y[i, -1, 1], rollout_states_y[i, -1, 3], rollout_states_y[i, -1, 4], rollout_states_y[i, -1, 5], rollout_states_y[i, -1, 6]], dtype=np.float64)
    #     yaw_ref = np.array([goal_state[0], goal_state[1], goal_state[3], goal_state[4], goal_state[5], goal_state[6]], dtype=np.float64)
    #     scores_y[i] = (yaw_ref - yaw_state).T @ Q_y @ (yaw_ref - yaw_state) + prev_action[2]*R*prev_action[2]



    # return scores_p, scores_r, scores_y
    return scores

def compute_control(scores, epsilon_p, epsilon_y):
    weighted_control = np.zeros((3,int(T/timestep)))
    # sum_weight_p = 0
    # # sum_weight_r = 0
    # sum_weight_y = 0
    # sum_weight_epsilon_p = 0
    # # sum_weight_epsilon_r = 0
    # sum_weight_epsilon_y = 0
    #
    # for i in range(n):
    #     sum_weight_p += np.exp((-1/lamda)*scores_p[i])
    #     # sum_weight_r += np.exp((-1/lamda)*scores_r[i])
    #     sum_weight_y += np.exp((-1/lamda)*scores_y[i])
    #
    #     sum_weight_epsilon_p += np.exp((-1/lamda)*scores_p[i])*epsilon_p[i]
    #     # sum_weight_epsilon_r += np.exp((-1/lamda)*scores_r[i])*epsilon_r[i]
    #     sum_weight_epsilon_y += np.exp((-1/lamda)*scores_y[i])*epsilon_y[i]
    #
    # # weighted_control[0,:] = sum_weight_epsilon_p/sum_weight_p
    # # # weighted_control[1,:] = sum_weight_epsilon_r/sum_weight_r
    # # weighted_control[2,:] = sum_weight_epsilon_y/sum_weight_y
    #
    # if sum_weight_p > 1e-8:
    #     weighted_control[0, :] = sum_weight_epsilon_p / sum_weight_p
    # else:
    #     weighted_control[0, :] = 0  # Or some default
    #
    # if sum_weight_y > 1e-8:
    #     weighted_control[2, :] = sum_weight_epsilon_y / sum_weight_y
    # else:
    #     weighted_control[2, :] = 0

    sum_weight = 0
    sum_weight_v_m_dot = 0
    sum_weight_phi_m_ddot = 0

    for i in range(n):
        weight = np.exp((-1/lamda)*scores[i])
        sum_weight += weight
        sum_weight_v_m_dot += weight * epsilon_p[i,:]
        sum_weight_phi_m_ddot += weight * epsilon_y[i,:]

    if sum_weight > 1e-8:
        weighted_control[0,:] = sum_weight_v_m_dot/sum_weight
        weighted_control[1,:] = 0
        weighted_control[2,:] = sum_weight_phi_m_ddot/sum_weight
    else:
        weighted_control[0,:] = 0
        weighted_control[1,:] = 0
        weighted_control[2,:] = 0

    return weighted_control

def get_action(initial_state, goal_state, prev_action):
    # rollout_states_p, rollout_states_r, rollout_states_y, epsilon_p, epsilon_r, epsilon_y = compute_rollouts(initial_state, 2)
    rollout_states, epsilon_p, epsilon_y = compute_rollouts(initial_state, 2)

    # scores_p, scores_r, scores_y = score_rollouts(rollout_states_p, rollout_states_r, rollout_states_y, goal_state, prev_action)
    scores = score_rollouts(rollout_states, goal_state)

    # all_actions = compute_control(scores_p, scores_r, scores_y, epsilon_p, epsilon_r, epsilon_y)
    all_actions = compute_control(scores, epsilon_p, epsilon_y)

    action = all_actions[:,0]

    return action

def plot_path(recorded_states, reference_states, time, sim_time, title):

    for i in range(recorded_states.shape[0]-1):
        for j in range(recorded_states.shape[1]-1):
            if recorded_states[i, j] == np.nan:
                recorded_states[i, j] = 0

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
    plt.scatter(sim_time, recorded_states[5, :], label='Recorded Alpha')
    plt.plot(time, reference_states[5, :], label='Reference Alpha', linestyle='--')
    # plt.plot(time, recorded_states[7, :], label='Recorded Beta')
    # plt.plot(time, reference_states[7, :], label='Reference Beta', linestyle='--')
    plt.scatter(sim_time, recorded_states[3, :], label='Recorded Phi_m ')
    plt.plot(time, reference_states[3, :], label='Reference Phi_m ', linestyle='--')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid()
    # plt.show()

    plt.figure(figsize=(10, 7))
    plt.scatter(sim_time, recorded_states[2, :], label='Recorded V_m')
    plt.plot(time, reference_states[2, :], label='Reference V_m', linestyle='--')
    plt.scatter(sim_time, recorded_states[6, :], label='Recorded Alpha Dot')
    plt.plot(time, reference_states[6, :], label='Reference Alpha Dot', linestyle='--')
    # plt.plot(time, recorded_states[8, :], label='Recorded Beta Dot')
    # plt.plot(time, reference_states[8, :], label='Reference Beta Dot', linestyle='--')
    plt.scatter(sim_time, recorded_states[4, :], label='Recorded Phi_m Dot')
    plt.plot(time, reference_states[4, :], label='Reference Phi_m Dot', linestyle='--')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Linear Velocity (m/s) / Angular Velocity (rad/s)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Initial state: [x_m, y_m, v_m, phi_m, phi_m_dot, alpha, alpha_dot, ~~beta~~, ~~beta_dot~~]
    # initial_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    initial_state = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float64)

    # Time vector
    time_horizon = np.arange(0, 10 + timestep, timestep)
    # print(len(time_horizon))

    x_ref = np.zeros(time_horizon.shape)
    y_ref = np.zeros(time_horizon.shape)

    # x_ref, y_ref = mimicking_the_reference_trajectory()

    alpha_ref = np.zeros(time_horizon.shape)
    # beta_ref = np.zeros(time_horizon.shape)
    phi_m_ref = np.zeros(time_horizon.shape)

    v_m_ref = np.zeros(time_horizon.shape)
    alpha_dot_ref = np.zeros(time_horizon.shape)
    phi_m_dot_ref = np.zeros(time_horizon.shape)
    # beta_dot_ref = np.zeros(time_horizon.shape)

    for i in range(len(time_horizon)):
        # x_ref[i] = 5.25*time_horizon[i]
        # y_ref[i] = 1 - np.cos((2*np.pi/15)*x_ref[i])

        alpha_ref[i] = 0
        # beta_ref[i] = # beta_ref needs to be calculated based on phi_m_dot_ref

        # if i == 0:
        #     phi_m_ref[i] = (np.arctan2(y_ref[i], x_ref[i]))
        # else:
        #     phi_m_ref[i] = (np.arctan2(y_ref[i] - y_ref[i-1], x_ref[i] - x_ref[i-1]))

        # phi_m_ref[i] = np.arctan2(y_ref[i], x_ref[i])

        phi_m_ref[i] = 0.5*np.sin(np.pi*time_horizon[i]/4)*(1 - np.exp(-time_horizon[i]))

        if time_horizon[i] < 3:
            v_m_ref[i] = (1/3)*time_horizon[i]
        else:
            v_m_ref[i] = 1

        if time_horizon[i] == 0:
            x_ref[i] = v_m_ref[i]*np.cos(phi_m_ref[i])*timestep
            y_ref[i] = v_m_ref[i]*np.sin(phi_m_ref[i])*timestep
        else:
            x_ref[i] = x_ref[i-1] + v_m_ref[i]*np.cos(phi_m_ref[i])*timestep
            y_ref[i] = y_ref[i-1] + v_m_ref[i]*np.sin(phi_m_ref[i])*timestep

        alpha_dot_ref[i] = 0

        if time_horizon[i] == 0:
            phi_m_dot_ref[i] = 0
        else:
            phi_m_dot_ref[i] = (phi_m_ref[i] - phi_m_ref[i-1])/timestep

        # beta_ref[i] = ((-(v_m_ref[i]*phi_m_dot_ref[i])*(l_p*m_p + (m_b + m_p)*r_w))/(m_p*l_p*-9.81))

        # if time_horizon[i] == 0:
        #     beta_dot_ref[i] = 0
        # else:
        #     beta_dot_ref[i] = (beta_ref[i] - beta_ref[i-1])/timestep

    # Reference state: [x_m, y_m, v_m, phi_m, phi_m_dot, alpha, alpha_dot, beta, beta_dot]
    # reference_state = [x_ref, y_ref, v_m_ref, phi_m_ref, phi_m_dot_ref, alpha_ref, alpha_dot_ref, beta_ref, beta_dot_ref]

    # sim_time = np.zeros((len(time_horizon) - 110))
    sim_time = np.arange(0, 10 + timestep - 1.1, timestep)
    recorded_states = np.zeros((7, len(sim_time)))
    recorded_actions = np.zeros((3, len(sim_time)))

    for i in range(len(sim_time)):
        prev_action = np.zeros(3)
        if time_horizon[i] != 0:
            prev_action = recorded_actions[:, i-1]

        # Get the action using MPPI
        action = get_action(initial_state, np.array([x_ref[i+110], y_ref[i+110], v_m_ref[i+110], phi_m_ref[i+110], phi_m_dot_ref[i+110], alpha_ref[i+110], alpha_dot_ref[i+110]], dtype=np.float64), prev_action)

        print("Goal State", np.array([x_ref[i+110], y_ref[i+110], v_m_ref[i+110], phi_m_ref[i+110], phi_m_dot_ref[i+110], alpha_ref[i+110], alpha_dot_ref[i+110]], dtype=np.float64))
        print("Action: ", action)

        # Update the state using the motion model
        new_state = motion_model(initial_state, action, timestep)
        print("New State: ", new_state)

        initial_state = new_state.copy()

        # Store the recorded states
        recorded_states[:, i] = new_state
        recorded_actions[:, i] = action

        print("Iteration: ", i)

    # reference_states = np.vstack([
    #     x_ref,
    #     y_ref,
    #     v_m_ref,
    #     phi_m_ref,
    #     phi_m_dot_ref,
    #     alpha_ref,
    #     alpha_dot_ref,
    #     beta_ref,
    #     beta_dot_ref
    # ])

    reference_states = np.vstack([
        x_ref,
        y_ref,
        v_m_ref,
        phi_m_ref,
        phi_m_dot_ref,
        alpha_ref,
        alpha_dot_ref,
    ])

    plot_path(recorded_states, reference_states, time_horizon, sim_time, 'S-Curve Trajectory with Fixed Variance = 2')