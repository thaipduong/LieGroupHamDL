import torch, argparse
import numpy as np
import os, sys
from torchdiffeq import odeint_adjoint as odeint
from se3hamneuralode import to_pickle
import time
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))+'/data/run1'
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--learn_rate', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=0, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=100, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='quadrotor', type=str, help='only one option right now')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_points', type=int, default=10,
                        help='number of evaluation points by the ODE solver, including the initial point')
    parser.add_argument('--solver', default='rk4', type=str, help='type of ODE Solver for Neural ODE')
    parser.add_argument('--float', default=32, type=int, help='number of gradient steps')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


def train(args):

    if args.float == 64:
        float_type = torch.float64
        torch.set_default_dtype(torch.float64)
    else:
        float_type = torch.float32
        torch.set_default_dtype(torch.float32)
    from se3hamneuralode import SE3HamNODEPX4

    device = 'cpu'

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the model
    if args.verbose:
        print("Start training with num of points = {} and solver {}.".format(args.num_points, args.solver))
    model = SE3HamNODEPX4(device=device, pretrain = False).to(device)
    # Load saved params if needed
    path = '{}/quadrotor-se3ham-rk4-10p3-2500.tar'.format(args.save_dir)
    model.load_state_dict(torch.load(path, map_location=device))

    pose = torch.tensor([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.], requires_grad=True, dtype=float_type).to(device)
    pose = pose.view(1, 12)
    x_tensor, R_tensor = torch.split(pose, [3, 9], dim=1)

    ts_gnet = torch.jit.trace(model.g_net, pose)
    ts_gnet.save("./saved_ts/ts_gnet_simdrone_friction.pt")
    ts_Dvnet = torch.jit.trace(model.Dv_net, x_tensor)
    ts_Dvnet.save("./saved_ts/ts_Dvnet_simdrone_friction.pt")
    ts_Dwnet = torch.jit.trace(model.Dw_net, R_tensor)
    ts_Dwnet.save("./saved_ts/ts_Dwnet_simdrone_friction.pt")

    t_eval = [0., 0.004, 0.008, 0.012, 0.016]
    pose = torch.tensor([[0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.], \
                         [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
                         [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
                         [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
                         [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]], requires_grad=True, dtype=float_type).to(
        device)
    pose = pose.view(5, 12)
    x_tensor, R_tensor = torch.split(pose, [3, 9], dim=1)
    g_q = model.g_net(pose)

    # Plot g_v(q)
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 4
    fig = plt.figure(figsize=figsize)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 0, 0], 'c--', linewidth=line_width,
             label=r'Other $g_{v}(q)$')
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 1, 0], 'c--', linewidth=line_width)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 2, 0], 'b--', linewidth=line_width,
             label=r'$g_{v}(q)[2,0]$')
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.savefig('./png/g_v_x.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # Plot g_omega(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 1], 'r--', linewidth=line_width,
             label=r'$g_{\omega}(q)[0,1]$')
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 2], 'g--', linewidth=line_width,
             label=r'$g_{\omega}(q)[1,2]$')
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 3], 'b--', linewidth=line_width,
             label=r'$g_{\omega}(q)[2,3]$')
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 0], 'c--', linewidth=2,
             label=r'Other $g_{\omega}(q)$')
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 2], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 3], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 0], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 1], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 3], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 0], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 1], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 2], 'c--', linewidth=2)
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.savefig('./png/g_w_x.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


    Dv_net = model.Dv_net(x_tensor)
    Dw_net = model.Dw_net(R_tensor)
    fig = plt.figure(figsize=figsize)
    plt.plot(t_eval, Dv_net.detach().cpu().numpy()[:, 2, 2], 'r--', linewidth=line_width,
             label=r'$D_{v}(q)[0,0]$')
    plt.plot(t_eval, Dv_net.detach().cpu().numpy()[:, 0, 0], 'g--', linewidth=line_width,
             label=r'$D_{v}(q)[1,1]$')
    plt.plot(t_eval, Dv_net.detach().cpu().numpy()[:, 1, 1], 'b--', linewidth=line_width,
             label=r'$D_{v}(q)[2,2]$')
    plt.plot(t_eval, Dv_net.detach().cpu().numpy()[:, 0, 1], 'c--', linewidth=line_width,
             label=r'Other $D_{v}(q)[i,j]$')
    plt.plot(t_eval, Dv_net.detach().cpu().numpy()[:, 0, 2], 'c--', linewidth=line_width)
    plt.plot(t_eval, Dv_net.detach().cpu().numpy()[:, 1, 0], 'c--', linewidth=line_width)

    plt.plot(t_eval, Dv_net.detach().cpu().numpy()[:, 1, 2], 'c--', linewidth=line_width)
    plt.plot(t_eval, Dv_net.detach().cpu().numpy()[:, 2, 0], 'c--', linewidth=line_width)
    plt.plot(t_eval, Dv_net.detach().cpu().numpy()[:, 2, 1], 'c--', linewidth=line_width)
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    # plt.savefig('./png/M1_x_all.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # Plot M2^-1(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(t_eval, Dw_net.detach().cpu().numpy()[:, 0, 0], 'r--', linewidth=line_width,
             label=r'$D_{w}(q)[0, 0]$')
    plt.plot(t_eval, Dw_net.detach().cpu().numpy()[:, 1, 1], 'g--', linewidth=line_width,
             label=r'$D_{w}(q)[1, 1]$')
    plt.plot(t_eval, Dw_net.detach().cpu().numpy()[:, 2, 2], 'b--', linewidth=line_width,
             label=r'$D_{w}(q)[2,2]$')
    plt.plot(t_eval, Dw_net.detach().cpu().numpy()[:, 0, 1], 'c--', linewidth=line_width,
             label=r'Other $D_{w}(q)[i,j]$')
    plt.plot(t_eval, Dw_net.detach().cpu().numpy()[:, 0, 2], 'c--', linewidth=line_width)
    plt.plot(t_eval, Dw_net.detach().cpu().numpy()[:, 1, 0], 'c--', linewidth=line_width)
    plt.plot(t_eval, Dw_net.detach().cpu().numpy()[:, 1, 2], 'c--', linewidth=line_width)
    plt.plot(t_eval, Dw_net.detach().cpu().numpy()[:, 2, 0], 'c--', linewidth=line_width)
    plt.plot(t_eval, Dw_net.detach().cpu().numpy()[:, 2, 1], 'c--', linewidth=line_width)
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize, loc='lower right')
    # plt.savefig('./png/M2_x_all.png', bbox_inches='tight')
    plt.show()


    # ts_Vnet = torch.jit.trace(model.V_net, pose)
    # ts_Vnet.save("ts_Vnet.pt")
    # Query the values of masses M1, M2, potential energy V, input coeff g.
    # g_q = self.model.g_net(pose)
    # V_q = self.model.V_net(pose)
    # M1 = self.model.M_net1(x_tensor)
    # M2 = self.model.M_net2(R_tensor)

if __name__ == "__main__":
    args = get_args()
    train(args)
