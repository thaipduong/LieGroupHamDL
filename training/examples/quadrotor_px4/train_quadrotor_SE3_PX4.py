# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control, RSS 2021
# Thai Duong, Nikolay Atanasov

# code structure follows the style of HNN by Greydanus et al. and SymODEM by Zhong et al.
# https://github.com/greydanus/hamiltonian-nn
# https://github.com/Physics-aware-AI/Symplectic-ODENet

import torch, argparse
import numpy as np
import os, sys
from torchdiffeq import odeint_adjoint as odeint
from se3hamneuralode import to_pickle
import time
from sklearn.model_selection import train_test_split
import torchode
import matplotlib.pyplot as plt
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))+'/data/run1'
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=2500, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=100, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='quadrotor', type=str, help='only one option right now')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_points', type=int, default=10,
                        help='number of evaluation points by the ODE solver, including the initial point')
    parser.add_argument('--solver', default='rk4', type=str, help='type of ODE Solver for Neural ODE')
    parser.add_argument('--float', default=64, type=int, help='number of gradient steps')
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

    from se3hamneuralode import pose_L2_geodesic_loss


    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the model
    if args.verbose:
        print("Start training with num of points = {} and solver {}.".format(args.num_points, args.solver))
    model = SE3HamNODEPX4(device=device, pretrain = False).to(device)
    # Save/load pre-init model
    path = './data/quadrotor-se3ham-rk4-pre-init.tar'
    model.load_state_dict(torch.load(path, map_location=device))
    #torch.save(model.state_dict(), path)
    num_parm = get_model_parm_nums(model)
    print('Model contains {} parameters'.format(num_parm))
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0.0)

    # Load dataset collected from PX4 simulator
    gazebo_data = np.load("./data/se3hamdl_dataset_frd_thrust_torque.npz")
    gz_data =  gazebo_data["dataset"]
    gz_time = gazebo_data["time_span"]
    gz_raw_time = gazebo_data["raw_time"]

    # Break trajectories in the dataset into small sequences.
    gz_data_cat = None
    gz_time_cat = None
    data_point_count = len(gz_data) - args.num_points + 1
    for i in range(args.num_points):
        if gz_data_cat is None:
            gz_data_cat = gz_data[i:i+data_point_count, None, :]
            gz_time_cat = gz_time[i:i+data_point_count, None, 0]
        else:
            gz_data_cat = np.concatenate((gz_data_cat, gz_data[i:i+data_point_count, None, :]), axis=1)
            gz_time_cat = np.concatenate((gz_time_cat, gz_time[i:i+data_point_count, None, 0]), axis=1)
    gz_time_cat_offset = gz_time_cat - gz_time_cat[:, None, 0]
    gz_data_cat[:,:,0:3] = gz_data_cat[:,:,0:3] - gz_data_cat[:, None, 0, 0:3]
    gz_omega_cat = np.linalg.norm(gz_data_cat[:, 0, 19:22], axis = 1)
    # Plot the original data sequences
    plt.plot(gz_omega_cat)
    plt.show()

    ## Since the data sequences around hovering are dominating, we replicate the data sequences with high angular velocity
    ## to balance out the dataset.
    threshold = 0.15
    gz_data_cat_high_omega = gz_data_cat[gz_omega_cat>=threshold, :, :]
    gz_data_cat_low_omega = gz_data_cat[gz_omega_cat < threshold, :, :]
    gz_time_cat_offset_high_omega = gz_time_cat_offset[gz_omega_cat>=threshold]
    gz_time_cat_offset_low_omega = gz_time_cat_offset[gz_omega_cat<threshold]

    ### Ratio of the replicated data sequences
    ratio = int(0.25*len(gz_data_cat_low_omega)/len(gz_data_cat_high_omega))
    for i in range(2,ratio):
        gz_data_cat = np.concatenate((gz_data_cat, gz_data_cat_high_omega), axis = 0)
        gz_time_cat_offset = np.concatenate((gz_time_cat_offset, gz_time_cat_offset_high_omega), axis = 0)
    gz_omega_cat = np.linalg.norm(gz_data_cat[:, 0, 19:22], axis = 1)
    # Plot the processed data sequences
    plt.figure()
    plt.plot(gz_omega_cat)
    plt.show()

    # Split into train and test set
    X_train, X_test, t_train, t_test = train_test_split(
        gz_data_cat, gz_time_cat_offset, test_size=0.1, random_state=1)

    # gz_omega_cat = np.linalg.norm(X_train[:, 0, 15:18], axis = 1)
    # plt.plot(gz_omega_cat)
    # plt.show()

    # Sampling the dataset due to limited GPU memory for training
    sampling_freq = 6
    gz_train_cat = X_train[::sampling_freq]
    t_train_cat = t_train[::sampling_freq]
    gz_test_cat = X_test[::sampling_freq]
    t_test_cat = t_test[::sampling_freq]
    train_x_cat = torch.tensor(gz_train_cat, requires_grad=True, dtype=float_type).to(device)
    test_x_cat = torch.tensor(gz_test_cat, requires_grad=True, dtype=float_type).to(device)

    # Create time tensor for torchode
    t_eval_train = None
    t_eval_test = None
    for i in range(args.num_points - 1):
        if t_eval_train is None:
            t_eval_train = t_train_cat[None, :, i:i + 2] - t_train_cat[None, :, None, i]
            t_eval_test = t_test_cat[None, :, i:i + 2] - t_test_cat[None, :, None, i]
        else:
            t_eval_train = np.concatenate((t_eval_train, t_train_cat[None, :, i:i + 2] - t_train_cat[None, :, None, i]), axis=0)
            t_eval_test = np.concatenate((t_eval_test, t_test_cat[None, :, i:i + 2] - t_test_cat[None, :, None, i]), axis=0)

    t_eval_train = torch.tensor(t_eval_train, requires_grad=True, dtype=float_type).to(device)/1e6
    t_eval_test = torch.tensor(t_eval_test, requires_grad=True, dtype=float_type).to(device)/1e6


    # Set up torchode
    term = torchode.ODETerm(model)
    step_method = torchode.Dopri5(term=term)
    step_size_controller = torchode.IntegralController(atol=1e-6, rtol=1e-3, term=term)
    adjoint = torchode.AutoDiffAdjoint(step_method, step_size_controller).to(device)
    jit_solver = torch.compile(adjoint)

    # Training stats
    stats = {'train_loss': [], 'test_loss': [], 'forward_time': [], 'backward_time': [], 'nfe': [], 'train_x_loss': [],\
             'test_x_loss':[], 'train_v_loss': [], 'test_v_loss': [], 'train_w_loss': [], 'test_w_loss': [], 'train_geo_loss':[], 'test_geo_loss':[],\
             'train_l1_gq_loss':[], 'test_l1_gq_loss':[] }
    start = time.time()
    # Start training
    for step in range(0,args.total_steps + 1):
        #print(step)
        train_loss = 0
        test_loss = 0
        train_x_loss = 0
        train_v_loss = 0
        train_w_loss = 0
        train_geo_loss = 0
        test_x_loss = 0
        test_v_loss = 0
        test_w_loss = 0
        test_geo_loss = 0

        # Predict states
        for i in range(args.num_points-1):
            if i == 0:
                problem = torchode.InitialValueProblem(y0=train_x_cat[:, i, :], t_eval=t_eval_train[i])
                sol = adjoint.solve(problem)
                train_x_hat = sol.ys
            else:
                train_x_hat[:, i, 18:] = train_x_cat[:, i, 18:]
                problem = torchode.InitialValueProblem(y0=train_x_hat[:, i, :], t_eval=t_eval_train[i])
                sol = adjoint.solve(problem)
                next_sol = sol.ys[:,1:,:]
                train_x_hat = torch.cat((train_x_hat, next_sol), dim=1)

        target = train_x_cat[:, 1:, :].permute(1,0,2)
        target_hat = train_x_hat[:, 1:, :].permute(1,0,2)

        # Calculate loss
        train_loss_mini, x_loss_mini, v_loss_mini, w_loss_mini, geo_loss_mini = \
            pose_L2_geodesic_loss(target, target_hat, split=[model.xdim, model.Rdim, model.twistdim, model.udim])
        train_loss = train_loss + train_loss_mini
        train_x_loss = train_x_loss + x_loss_mini
        train_v_loss = train_v_loss + v_loss_mini
        train_w_loss = train_w_loss + w_loss_mini
        train_geo_loss = train_geo_loss + geo_loss_mini
        batch_size = train_x_cat.shape[0]

        # Query the neural networks to add L1 regularization
        Gq_output = model.g_net(train_x_cat[:, 0, :model.posedim]).reshape((batch_size, 24))
        Dv_output = model.Dv_net(train_x_cat[:, 0, :model.xdim]).reshape((batch_size, 9))
        Dw_output = model.Dw_net(train_x_cat[:, 0, model.xdim:model.posedim]).reshape((batch_size, 9))
        train_l1_gq_loss = Gq_output.abs().mean()
        train_l1_Dvq_loss = Dv_output.abs().mean()
        train_l1_Dwq_loss = Dw_output.abs().mean()

        # Loss function for training
        l1_coeff = 0.02
        train_loss_mini = train_loss_mini + l1_coeff*(train_l1_gq_loss + train_l1_Dvq_loss + train_l1_Dwq_loss)

        # Calculate loss for test data
        for i in range(args.num_points - 1):
            if i == 0:
                problem = torchode.InitialValueProblem(y0=test_x_cat[:, i, :], t_eval=t_eval_test[i])
                sol = adjoint.solve(problem)
                test_x_hat = sol.ys
            else:
                test_x_hat[:, i, 18:] = test_x_cat[:, i, 18:]
                problem = torchode.InitialValueProblem(y0=test_x_hat[:, i, :], t_eval=t_eval_test[i])
                sol = adjoint.solve(problem)
                next_sol = sol.ys[:, 1:, :]
                test_x_hat = torch.cat((test_x_hat, next_sol), dim=1)

        target = test_x_cat[:, 1:, :].permute(1,0,2)
        target_hat = test_x_hat[:, 1:, :].permute(1,0,2)
        test_loss_mini, x_loss_mini, v_loss_mini, w_loss_mini, geo_loss_mini = \
            pose_L2_geodesic_loss(target, target_hat, split=[model.xdim, model.Rdim, model.twistdim, model.udim])
        test_loss = test_loss + test_loss_mini
        test_x_loss = test_x_loss + x_loss_mini
        test_v_loss = test_v_loss + v_loss_mini
        test_w_loss = test_w_loss + w_loss_mini
        test_geo_loss = test_geo_loss + geo_loss_mini
        batch_size = test_x_cat.shape[0]
        Gq_output = model.g_net(test_x_cat[:, 0, :model.posedim]).reshape((batch_size, 24)) # Gq values for the input at t = 0
        Dv_output = model.Dv_net(test_x_cat[:, 0, :model.xdim]).reshape((batch_size, 9))
        Dw_output = model.Dw_net(test_x_cat[:, 0, model.xdim:model.posedim]).reshape((batch_size, 9))

        test_l1_gq_loss = Gq_output.abs().mean()
        test_l1_Dvq_loss = Dv_output.abs().mean()
        test_l1_Dwq_loss = Dw_output.abs().mean()

        # Gradient descent
        t = time.time()
        if step > 0:
            log_train_loss_mini = torch.log(train_loss_mini)
            train_loss_mini.backward(retain_graph=True)
            optim.step()
            optim.zero_grad()
        backward_time = time.time() - t

        # Logging stats
        stats['train_loss'].append(train_loss.item())
        stats['test_loss'].append(test_loss.item())
        stats['train_x_loss'].append(train_x_loss.item())
        stats['test_x_loss'].append(test_x_loss.item())
        stats['train_v_loss'].append(train_v_loss.item())
        stats['test_v_loss'].append(test_v_loss.item())
        stats['train_w_loss'].append(train_w_loss.item())
        stats['test_w_loss'].append(test_w_loss.item())
        stats['train_geo_loss'].append(train_geo_loss.item())
        stats['test_geo_loss'].append(test_geo_loss.item())
        stats['train_l1_gq_loss'].append(train_l1_gq_loss.item())
        stats['test_l1_gq_loss'].append(test_l1_gq_loss.item())

        stats['backward_time'].append(backward_time)
        stats['nfe'].append(model.nfe)
        if step % args.print_every == 0:
            current = time.time()
            print("time per {} loops: {}".format(args.print_every, current - start))
            start = current
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, train_loss.item(), test_loss.item()))
            print("step {}, train_x_loss {:.4e}, test_x_loss {:.4e}".format(step, train_x_loss.item(),
                                                                            test_x_loss.item()))
            print("step {}, train_v_loss {:.4e}, test_v_loss {:.4e}".format(step, train_v_loss.item(),
                                                                            test_v_loss.item()))
            print("step {}, train_w_loss {:.4e}, test_w_loss {:.4e}".format(step, train_w_loss.item(),
                                                                            test_w_loss.item()))
            print("step {}, train_geo_loss {:.4e}, test_geo_loss {:.4e}".format(step, train_geo_loss.item(),
                                                                                test_geo_loss.item()))
            print("step {}, train_l1_gq_loss {:.4e}, test_l1_gq_loss {:.4e}".format(step, train_l1_gq_loss.item(),
                                                                                test_l1_gq_loss.item()))
            print("step {}, nfe {:.4e}".format(step, model.nfe))
            # # Uncomment this to save model every args.print_every steps
            os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
            label = '-se3ham'
            path = '{}/{}{}-{}-{}p3-{}.tar'.format(args.save_dir, args.name, label, args.solver, args.num_points, step)
            torch.save(model.state_dict(), path)

    stats['train_x_cat'] = train_x_cat.detach().cpu().numpy()
    stats['test_x_cat'] = test_x_cat.detach().cpu().numpy()
    stats['train_x_hat'] = train_x_hat.detach().cpu().numpy()
    stats['test_x_hat'] = test_x_hat.detach().cpu().numpy()
    stats['t_eval_train'] = t_eval_train.detach().cpu().numpy()
    stats['t_eval_test'] = t_eval_test.detach().cpu().numpy()
    stats['t_train_cat'] = t_train_cat
    stats['t_test_cat'] = t_test_cat
    return model, stats


if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # Save model
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-se3ham'
    path = '{}/{}{}-{}-{}p.tar'.format(args.save_dir, args.name, label, args.solver, args.num_points)
    torch.save(model.state_dict(), path)
    path = '{}/{}{}-{}-{}p-stats.pkl'.format(args.save_dir, args.name, label, args.solver, args.num_points)
    print("Saved file: ", path)
    to_pickle(stats, path)
