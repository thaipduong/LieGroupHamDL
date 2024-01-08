# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control, RSS 2021
# Thai Duong, Nikolay Atanasov

# code structure follows the style of HNN by Greydanus et al. and SymODEM by Zhong et al.
# https://github.com/greydanus/hamiltonian-nn
# https://github.com/Physics-aware-AI/Symplectic-ODENet

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import pickle
from se3hamneuralode import compute_rotation_matrix_from_quaternion, to_pickle, from_pickle, UnstructuredSE3NODE, SE3HamNODE, UnstructuredSE3HamNODE, SE3HamNODEGT
solve_ivp = scipy.integrate.solve_ivp
from scipy.spatial.transform import Rotation
from torchdiffeq import odeint_adjoint as odeint
from controller_utils import *

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
run = 1
def get_model_unstructuredHam():
    model = UnstructuredSE3HamNODE(device=device, pretrain=False).to(device)
    path = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p-unstructuredHam.tar'
    model.load_state_dict(torch.load(path, map_location=device))
    path_pickle4 = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p-unstructuredHam-stats-pickle4.pkl'
    stats = from_pickle(path_pickle4)
    # path = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p-unstructuredHam-stats.pkl'
    # stats = from_pickle(path)
    # path_pickle4 = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p-unstructuredHam-stats-pickle4.pkl'
    # to_pickle(stats, path_pickle4, protocol=pickle.DEFAULT_PROTOCOL)
    return model, stats

def get_model_unstructured():
    model = UnstructuredSE3NODE(device=device).to(device)
    path = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p-unstructured.tar'
    model.load_state_dict(torch.load(path, map_location=device))
    path_pickle4 = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p-unstructured-stats-pickle4.pkl'
    stats = from_pickle(path_pickle4)
    # path = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p-unstructured-stats.pkl'
    # stats = from_pickle(path)
    # path_pickle4 = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p-unstructured-stats-pickle4.pkl'
    # to_pickle(stats, path_pickle4, protocol=pickle.DEFAULT_PROTOCOL)
    return model, stats

def get_model():
    model = SE3HamNODE(device=device, pretrain=False, turnon_dissipation=False).to(device)
    path = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p.tar'
    model.load_state_dict(torch.load(path, map_location=device))
    path_pickle4 = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p-stats-pickle4.pkl'
    stats = from_pickle(path_pickle4)
    # path = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p-stats.pkl'
    # stats = from_pickle(path)
    # path_pickle4 = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p-stats-pickle4.pkl'
    # to_pickle(stats, path_pickle4, protocol=pickle.DEFAULT_PROTOCOL)
    return model, stats

def get_gtmodel():
    model = SE3HamNODEGT(device=device).to(device)
    # path = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p-stats.pkl'
    # stats = from_pickle(path)
    # path_pickle4 = './data/run' + str(run) + '/quadrotor-se3ham-rk4-5p-stats-pickle4.pkl'
    # to_pickle(stats, path_pickle4, protocol=pickle.DEFAULT_PROTOCOL)
    return model, None

def get_init_state(rand = True):
    seed = 0
    np_random, seed = gym.utils.seeding.np_random(seed)
    rand_ = np_random.uniform(
        low=[-2, -2, 0, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, -0.5, -0.5, -0.5],
        high=[2, 2, 2, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    x = rand_[0:3] if rand else np.array([0.0, 0.0, 0.0])  #np.array([0.0, 0.0, 1.0]) #
    x_dot = rand_[3:6] if rand else np.array([0.0, 0.0, 0.0])  #
    # Uniformly generate quaternion using http://planning.cs.uiuc.edu/node198.html
    u1, u2, u3 = rand_[6], rand_[7], rand_[8]
    quat = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                          np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)])
    omega =   rand_[9:12] if rand else np.array([0.0, 0.0, 0.0]) #
    r = Rotation.from_quat(quat)
    R = r.as_matrix()
    x_dot_bodyframe = np.matmul(np.transpose(R), x_dot)
    obs = np.hstack((x, R.flatten(), x_dot_bodyframe, omega))
    u0 = [0.0, 0.0, 0.0, 0.0]
    # State orders: x (3), R (9), linear vel (3), angular vel (3), control (4)
    y0_u = np.concatenate((obs, np.array(u0)))
    return y0_u


if __name__ == "__main__":
    # Figure and font size
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 4
    # Load trained model
    gt_model, _ = get_gtmodel()
    model, stats = get_model()
    model_unstructured, stats_unstructured = get_model_unstructured()
    model_unstructuredHam, stats_unstructuredHam = get_model_unstructuredHam()


    # # Load train/test data
    # train_x_hat = stats['train_x_hat']
    # test_x_hat = stats['test_x_hat']
    # train_x = stats['train_x']
    # test_x = stats['test_x']
    # t_eval = stats['t_eval']
    # print("Loaded data!")

    # Plot loss
    fig = plt.figure(figsize=figsize, linewidth=5)
    ax = fig.add_subplot(111)
    train_loss = stats['train_loss']
    test_loss = stats['test_loss']
    train_loss_unstructured = stats_unstructured['train_loss']
    test_loss_unstructured = stats_unstructured['test_loss']
    train_loss_unstructuredHam= stats_unstructuredHam['train_loss']
    test_loss_unstructuredHam = stats_unstructuredHam['test_loss']
    ax.plot(train_loss_unstructured[0:5000], 'b', linewidth=line_width, label='black-box')
    ax.plot(train_loss_unstructuredHam[0:5000], 'g', linewidth=line_width, label='unstructured Hamiltonian')
    ax.plot(train_loss[0:5000], 'r', linewidth=line_width, label='structured Hamiltonian')
    #ax.plot(test_loss[0:], 'r', linewidth=line_width, label='test loss')
    plt.xlabel("iterations", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.yscale('log')
    #plt.ylim(1e-7, 100)
    plt.legend(fontsize=fontsize, loc = 3)
    plt.grid()
    plt.savefig('./png/journal/loss_log_comparison_quadrotor.pdf', bbox_inches='tight')
    plt.show()


# Initial condition from gym
import gym
import envs
# time intervals
time_step = 500 ; n_eval = 500
t_span = [0,time_step*0.01]
t_eval = torch.linspace(t_span[0], t_span[1], n_eval)
t_eval_gt = torch.linspace(t_span[0]-0.1, t_span[1]+0.1, n_eval)

# init state
y0_u = get_init_state()
y0_u = torch.tensor(y0_u, requires_grad=True, device=device, dtype=torch.float32).view(1, 22)
y0_u = torch.cat((y0_u, y0_u), dim = 0)







# Roll out our dynamics model from the initial state
y_gt = odeint(gt_model, y0_u, t_eval, method='rk4')
y_structured = odeint(model, y0_u, t_eval, method='rk4')
y_unstructured = odeint(model_unstructured, y0_u, t_eval, method='rk4')
y_unstructuredHam = odeint(model_unstructuredHam, y0_u, t_eval, method='rk4')


all_ys = [y_structured, y_unstructured, y_unstructuredHam]
total_energy_learned = []
m = 0.027
g = 9.8
J = np.array([2.3951, 2.3951, 3.2347])*1e-5
J_inv = 1/np.array(J)
# State orders: x (3), R (9), linear vel (3), angular vel (3), control (4)
for y_i in all_ys:
    y = y_i.detach().cpu().numpy()
    y = y[:,0,:]
    total_energy_learned_i = []
    for i in range(len(y)):
        energy = m*g*y[i,2] + 0.5*m*(y[i,12]**2 + y[i,13]**2 + y[i,14]**2) + \
                 0.5*(J[0]*y[i,15]**2 + J[1]*y[i,16]**2 + J[2]*y[i,17]**2)
        total_energy_learned_i.append(energy)
    total_energy_learned.append(total_energy_learned_i)

learned_unstructuredHam = model_unstructuredHam.H_net(y_unstructured[:,0,:18])
learned_unstructuredHam = learned_unstructuredHam[:,0]
learned_unstructuredHam = learned_unstructuredHam.detach().cpu().numpy()

y = y_structured.detach().cpu().numpy()
y = y[:,0,:]
pose = torch.tensor(y[:,0:12], requires_grad=True, dtype=torch.float32).to(device)
x, R = torch.split(pose, [3, 9], dim=1)

# Get the output of the neural networks
offset = 0#0.19576
scaleM = 1.#1.33
scaleJ = 1.#150
# offset = 0.05#0.19576
# scaleM = 1.33
# scaleJ = 150
g_q = model.g_net(pose)
V_q = model.V_net(pose)/scaleM + offset
M_q_inv1 = scaleM*model.M_net1(x)
M_q_inv2 = scaleJ*model.M_net2(R)
V_q = V_q.detach().cpu().numpy()
M_q_inv1 = M_q_inv1.detach().cpu().numpy()
M_q_inv2 = M_q_inv2.detach().cpu().numpy()

# Calculate total energy from the learned dynamics
learned_Ham = []

for i in range(len(M_q_inv1)):
    m1 = np.linalg.inv(M_q_inv1[i,:,:])
    m2 = np.linalg.inv(M_q_inv2[i,:,:])
    V = V_q[i, 0]

    energy = 0.5*np.matmul(y[i,12:15].T, np.matmul(m1, y[i,12:15]))
    energy = energy + 0.5*np.matmul(y[i,15:18].T, np.matmul(m2, y[i,15:18]))
    energy = energy + V_q[i,0]
    learned_Ham.append(energy)
learned_Ham = np.array(learned_Ham)

# fig = plt.figure(figsize=figsize)
# plt.plot(t_eval_gt, total_energy_learned[0][0]*np.ones(time_step), 'k', linestyle="dotted", linewidth=3, label='ground truth')
# plt.plot(t_eval, total_energy_learned[1], 'b', linewidth=4, label='black-box')
# plt.plot(t_eval, total_energy_learned[2], 'g', linewidth=4, label='unstructured Hamiltonian')
# plt.plot(t_eval, total_energy_learned[0], 'r', linewidth=4, label='structured Hamiltonian')
# plt.xlabel("$t$", fontsize=24)
# #plt.yscale('log')
# #plt.ylim(4, 10)
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.legend(fontsize=24)
# plt.savefig('./png/journal/hamiltonian_comparison_quadrotor.pdf', bbox_inches='tight')
# plt.show()

fig = plt.figure(figsize=figsize)
#plt.plot(t_eval_gt, total_energy_learned[0][0]*np.ones(time_step), 'k', linestyle="dotted", linewidth=3, label='ground truth')
plt.plot(t_eval, total_energy_learned[1], 'b', linewidth=4, label='black-box')
plt.plot(t_eval, learned_unstructuredHam, 'g', linewidth=4, label='unstructured Hamiltonian')
plt.plot(t_eval, learned_Ham, 'r', linewidth=4, label='structured Hamiltonian')
plt.xlabel("$t$", fontsize=24)
plt.yscale('log')
#plt.ylim(4, 10)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)
plt.grid()
plt.savefig('./png/journal/hamiltonian_comparison_quadrotor_learned.pdf', bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=figsize)
#plt.plot(t_eval_gt, total_energy_learned[0][0]*np.ones(time_step), 'k', linestyle="dotted", linewidth=3, label='ground truth')
plt.plot(t_eval, learned_Ham, 'r', linewidth=4)
plt.xlabel("$t$", fontsize=32)
#plt.yscale('log')
plt.ylim(0.443, 0.444)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
#plt.legend(fontsize=24)
plt.grid()
plt.savefig('./png/journal/hamiltonian.pdf', bbox_inches='tight')
plt.show()
############################## PLOT SE(3) CONSTRAINTS ROLLED OUT FROM OUR DYNAMICS ###############################

det = []
RRT_I_dist = []
angle_hat = []
angle_dot_hat = []
for y_i in all_ys:
    y_i = y_i.detach().cpu().numpy()
    det_i = []
    RRT_I_dist_i = []
    angle_hat_i = []
    angle_dot_hat_i = []
    for i in range(len(y_i)):
        R_hat = y_i[i, 0, 3:12]
        R_hat = R_hat.reshape(3, 3)
        R_det = np.linalg.det(R_hat)
        det_i.append(np.abs(R_det - 1))
        R_RT = np.matmul(R_hat, R_hat.transpose())
        RRT_I = np.linalg.norm(R_RT - np.diag([1.0, 1.0, 1.0]))
        RRT_I_dist_i.append(RRT_I)
        r = Rotation.from_matrix(R_hat)
        angle_hat_i.append(r.as_euler('zyx')[0])
        angle_dot_hat_i.append(y_i[i,0, 11])
    RRT_I_dist.append(RRT_I_dist_i)
    det.append(det_i)
    angle_hat.append(angle_hat_i)
    angle_dot_hat.append(angle_dot_hat_i)


figsize = (12, 7.8)
fontsize = 24
fontsize_ticks = 32
line_width = 4
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
ax.plot(t_eval, det[1], 'b', linewidth=line_width, label=r'$|det(R) - 1|$ - black-box')
ax.plot(t_eval, RRT_I_dist[1], 'b', linestyle = 'dashed', linewidth=line_width, label=r'$\Vert R R^\top - I\Vert$ - black-box')

ax.plot(t_eval, det[2], 'g', linewidth=line_width, label=r'$|det(R) - 1|$ - unstructured Hamiltonian')
ax.plot(t_eval, RRT_I_dist[2], 'g', linestyle = 'dashed', linewidth=line_width, label=r'$\Vert R R^\top - I\Vert$ - unstructured Hamiltonian')

ax.plot(t_eval, det[0], 'r', linewidth=line_width, label=r'$|det(R) - 1|$ - structured Hamiltonian')
ax.plot(t_eval, RRT_I_dist[0], 'r', linestyle = 'dashed', linewidth=line_width, label=r'$\Vert R R^\top - I\Vert$ - structured Hamiltonian')


plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
#plt.ylim(1e-5, 2000000)
plt.yscale('log')
plt.legend(fontsize=fontsize)
plt.grid()
plt.savefig('./png/journal/SO3_constraints_comparison_quadrotor.pdf', bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)
ax.plot(t_eval, det[0], 'r', linewidth=line_width, label=r'$|det(R) - 1|$')
ax.plot(t_eval, RRT_I_dist[0], 'b', linewidth=line_width, label=r'$\Vert R R^\top - I\Vert$')


plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
#plt.ylim(1e-5, 2000000)
plt.yscale('log')
plt.legend(fontsize=fontsize)
plt.grid()
plt.savefig('./png/journal/SO3_constraints.pdf', bbox_inches='tight')
plt.show()

#######################################################
