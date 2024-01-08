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
        low=[-2, -2, 0, -1.5, -1.5, -1.5, 0.0, 0.0, 0.0, -0.5, -0.5, -0.5],
        high=[2, 2, 2, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
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

    # # Plot loss
    # fig = plt.figure(figsize=figsize, linewidth=5)
    # ax = fig.add_subplot(111)
    # train_loss = stats['train_loss']
    # test_loss = stats['test_loss']
    # train_loss_unstructured = stats_unstructured['train_loss']
    # test_loss_unstructured = stats_unstructured['test_loss']
    # train_loss_unstructuredHam= stats_unstructuredHam['train_loss']
    # test_loss_unstructuredHam = stats_unstructuredHam['test_loss']
    # ax.plot(train_loss_unstructured[0:5000], 'b', linewidth=line_width, label='black-box')
    # ax.plot(train_loss_unstructuredHam[0:5000], 'g', linewidth=line_width, label='unstructured Hamiltonian')
    # ax.plot(train_loss[0:5000], 'r', linewidth=line_width, label='structured Hamiltonian')
    # #ax.plot(test_loss[0:], 'r', linewidth=line_width, label='test loss')
    # plt.xlabel("iterations", fontsize=fontsize_ticks)
    # plt.xticks(fontsize=fontsize_ticks)
    # plt.yticks(fontsize=fontsize_ticks)
    # plt.yscale('log')
    # #plt.ylim(1e-7, 100)
    # plt.legend(fontsize=fontsize, loc = 3)
    # plt.grid()
    # plt.savefig('./png/journal/loss_log_comparison_quadrotor.pdf', bbox_inches='tight')
    # plt.show()


# Initial condition from gym
import gym
import envs
# time intervals
time_step = 50 ; n_eval = 50
t_span = [0,time_step*0.01]
t_eval = torch.linspace(t_span[0], t_span[1], n_eval)
t_span_unstructuredHam = [0,time_step*0.3]
t_eval_unstructuredHam = torch.linspace(t_span_unstructuredHam[0], t_span_unstructuredHam[1], n_eval)
t_eval_gt = torch.linspace(t_span[0]-0.1, t_span[1]+0.1, n_eval)

# init state
y0_u = get_init_state()
y0_u = torch.tensor(y0_u, requires_grad=True, device=device, dtype=torch.float32).view(1, 22)
y0_u = torch.cat((y0_u, y0_u), dim = 0)







# Roll out our dynamics model from the initial state
y_gt = odeint(gt_model, y0_u, t_eval, method='rk4')
y_structured = odeint(model, y0_u, t_eval, method='rk4')
y_unstructured = odeint(model_unstructured, y0_u, t_eval, method='rk4')
y_unstructuredHam = odeint(model_unstructuredHam, y0_u, t_eval_unstructuredHam, method='rk4')


#######################################################

y_gt = y_gt.detach().cpu().numpy()
y_structured = y_structured.detach().cpu().numpy()
y_unstructured = y_unstructured.detach().cpu().numpy()
y_unstructuredHam = y_unstructuredHam.detach().cpu().numpy()
quadplot_comparison(y_gt[:, 0, :], y_structured[:, 0, :], y_unstructuredHam[:, 0, :], y_unstructured[:, 0, :])