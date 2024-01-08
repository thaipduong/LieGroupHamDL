# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control, RSS 2021
# Thai Duong, Nikolay Atanasov

# code structure follows the style of HNN by Greydanus et al. and SymODEM by Zhong et al.
# https://github.com/greydanus/hamiltonian-nn
# https://github.com/Physics-aware-AI/Symplectic-ODENet

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from se3hamneuralode import compute_rotation_matrix_from_quaternion, from_pickle, UnstructuredSO3NODE, SO3HamNODE, UnstructuredSO3HamNODE
solve_ivp = scipy.integrate.solve_ivp
from scipy.spatial.transform import Rotation
from torchdiffeq import odeint_adjoint as odeint


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')
run = 2
def get_model_unstructuredHam():
    model = UnstructuredSO3HamNODE(device=device, u_dim=1).to(device)
    path = './data/run' + str(run) + '/pendulum-so3ham-rk4-5p-unstructuredHam-5000.tar'
    model.load_state_dict(torch.load(path, map_location=device))
    path = './data/run' + str(run) + '/pendulum-so3ham-rk4-5p-unstructuredHam-stats.pkl'
    stats = from_pickle(path)
    return model, stats

def get_model_unstructured():
    model = UnstructuredSO3NODE(device=device, u_dim=1).to(device)
    path = './data/run' + str(run) + '/pendulum-so3ham-rk4-5p-unstructured-5000.tar'
    model.load_state_dict(torch.load(path, map_location=device))
    path = './data/run' + str(run) + '/pendulum-so3ham-rk4-5p-unstructured-stats.pkl'
    stats = from_pickle(path)
    return model, stats

def get_model():
    model = SO3HamNODE(device=device, u_dim=1).to(device)
    path = './data/run' + str(run) + '/pendulum-so3ham-rk4-5p-5000.tar'
    model.load_state_dict(torch.load(path, map_location=device))
    path = './data/run' + str(run) + '/pendulum-so3ham-rk4-5p-stats.pkl'
    stats = from_pickle(path)
    return model, stats

if __name__ == "__main__":
    # Figure and font size
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 4
    # Load trained model
    model, stats = get_model()
    model_unstructured, stats_unstructured = get_model_unstructured()
    model_unstructuredHam, stats_unstructuredHam = get_model_unstructuredHam()


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
    plt.ylim(1e-7, 100)
    plt.legend(fontsize=fontsize, loc = 1)
    plt.grid()
    plt.savefig('./png/loss_log_comparison_pendulum.png', bbox_inches='tight')
    plt.show()


# Initial condition from gym
import gym
import envs
# time intervals
time_step = 1000 ; n_eval = 1000
t_span = [0,time_step*0.05]
t_eval = torch.linspace(t_span[0], t_span[1], n_eval)
t_eval_gt = torch.linspace(t_span[0]-0.5, t_span[1]+0.5, n_eval)
# init angle
init_angle = np.pi/2
u0 = 0.0

######################################### PLOT GROUND-TRUTH ENERGY #########################################

env = gym.make('MyPendulum-v1')
# record video
env = gym.wrappers.Monitor(env, './videos/' + 'pendulum' + '/', force=True)
env.reset(ori_rep='angle')
env.env.state = np.array([init_angle, u0], dtype=np.float32)
obs = env.env.get_obs()
obs_list = []
for _ in range(time_step):
    obs_list.append(obs)
    obs, _, _, _ = env.step([u0])

true_ivp = np.stack(obs_list, 1)
true_ivp = np.concatenate((true_ivp, np.zeros((1, time_step))), axis=0)
y0_u = np.asarray([np.cos(init_angle), np.sin(init_angle), 0, u0])
true_ivp = np.concatenate((true_ivp, np.zeros((1, time_step))), axis=0)

E_true = true_ivp.T[:, 2]**2 / 6 + 5 * (1 - true_ivp.T[:, 0])
env.close()


angle = []
angle_dot = []

for i in range(len(true_ivp.T)):
    cosy = true_ivp.T[i, 0]
    siny = true_ivp.T[i, 1]
    R = np.array([[cosy, -siny, 0],
                 [siny, cosy, 0],
                 [0, 0, 1]])
    r = Rotation.from_matrix(R)
    a = r.as_euler('zyx')
    angle.append(r.as_euler('zyx')[0])
    angle_dot.append(true_ivp.T[i, 2])


############################## PLOT TOTAL ENERGY ROLLED OUT FROM OUR DYNAMICS ###############################

# Get initial state from the pendulum environment
env = gym.make('MyPendulum-v1')
# record video
env = gym.wrappers.Monitor(env, './videos/' + 'single-embed' + '/', force=True) # , video_callable=lambda x: True, force=True
env.reset(ori_rep='rotmat')
env.env.state = np.array([init_angle, u0], dtype=np.float32)
obs = env.env.get_obs()
env.close()
y0_u = np.concatenate((obs, np.array([u0])))
y0_u = torch.tensor(y0_u, requires_grad=True, device=device, dtype=torch.float32).view(1, 13)

# Roll out our dynamics model from the initial state
y = odeint(model, y0_u, t_eval, method='rk4')
y_unstructured = odeint(model_unstructured, y0_u, t_eval, method='rk4')
y_unstructuredHam = odeint(model_unstructuredHam, y0_u, t_eval, method='rk4')


all_ys = [y, y_unstructured, y_unstructuredHam]
total_energy_learned = []
for y_i in all_ys:
    y = y_i.detach().cpu().numpy()
    cos_y = y[:,0,0]
    sin_y = y[:,0,3]
    y_dot = y[:,0,11]
    y = y[:,0,:]
    # Determine the scaling factor beta and the potential_energy_offset from analyze_pend_SO3.py's results.
    # This should be changed according to analyze_pend_SO3.py if we have new results.
    scaling_factor = 2.32
    potential_energy_offset = 5.1
    total_energy_learned_i = []
    for i in range(len(y)):
        energy = (y[i,11]**2)/6
        energy = energy + 5 * (1 - y[i,0])
        total_energy_learned_i.append(energy)
    total_energy_learned.append(total_energy_learned_i)

fig = plt.figure(figsize=figsize)
plt.plot(t_eval_gt, 5*np.ones(time_step), 'k', linestyle="dotted", linewidth=3, label='ground truth')
plt.plot(t_eval, total_energy_learned[1], 'b', linewidth=4, label='black-box')
plt.plot(t_eval, total_energy_learned[2], 'g', linewidth=4, label='unstructured Hamiltonian')
plt.plot(t_eval, total_energy_learned[0], 'r', linewidth=4, label='structured Hamiltonian')
plt.xlabel("$t$", fontsize=24)
#plt.yscale('log')
plt.ylim(4, 10)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)
plt.grid()
plt.savefig('./png/hamiltonian_comparison.png', bbox_inches='tight')
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
        R_hat = y_i[i, 0, 0:9]
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
plt.ylim(1e-5, 3000000)
plt.yscale('log')
plt.legend(fontsize=fontsize, loc = "center right")
plt.grid()
plt.savefig('./png/SO3_constraints_comparison.png', bbox_inches='tight')
plt.show()

############################## PLOT PHASE PORTRAIT ROLLED OUT FROM OUR DYNAMICS ###############################

fig = plt.figure(figsize=figsize, linewidth=5)
ax = fig.add_subplot(111)


ax.plot(angle[:100], angle_dot[:100], 'gray', linewidth=line_width*2, label='Ground truth')
ax.plot(angle_hat[1][:100],angle_dot_hat[1][:100], 'b',linestyle = "dotted", linewidth=line_width, label='black-box')
ax.plot(angle_hat[2][:100],angle_dot_hat[2][:100], 'g',linestyle = "dotted", linewidth=line_width, label='unstructured Hamiltonian')
ax.plot(angle_hat[0][:100],angle_dot_hat[0][:100], 'r',linestyle = "dotted", linewidth=line_width, label='structured Hamiltonian')

plt.xlabel("pendulum angle", fontsize=fontsize_ticks)
plt.ylabel("angular velocity", fontsize=fontsize_ticks)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.legend(fontsize=fontsize, loc = 1)
plt.xlim(-2, 3.5)
plt.ylim(-7.0, 9.0)
plt.grid()
plt.savefig('./png/phase_portrait_comparison.png', bbox_inches='tight')
plt.show()

