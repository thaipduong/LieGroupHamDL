# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control, RSS 2021
# Thai Duong, Nikolay Atanasov

# code structure follows the style of HNN by Greydanus et al. and SymODEM by Zhong et al.
# https://github.com/greydanus/hamiltonian-nn
# https://github.com/Physics-aware-AI/Symplectic-ODENet

import torch
import numpy as np

from se3hamneuralode import MLP, PSD, MatrixNet
from se3hamneuralode import compute_rotation_matrix_from_quaternion
from .utils import L2_loss



class UnstructuredSE3NODE(torch.nn.Module):
    def __init__(self, device=None, f_net = None, udim = 4):
        super(UnstructuredSE3NODE, self).__init__()
        init_gain = 0.001
        self.xdim = 3
        self.Rdim = 9
        self.linveldim = 3
        self.angveldim = 3
        self.posedim = self.xdim + self.Rdim #3 for position + 12 for rotmat
        self.twistdim = self.linveldim + self.angveldim #3 for linear vel + 3 for ang vel
        self.udim = udim
        if f_net is None:
            self.f_net = MLP(self.posedim + self.twistdim, 1000, self.posedim + self.twistdim,
                             init_gain=init_gain).to(device)
        else:
            self.f_net = f_net
        self.device = device
        self.nfe = 0


    def forward(self, t, x):
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            zero_vec = torch.zeros(bs, self.udim, dtype=torch.float32, device=self.device)

            q_dq, u = torch.split(x, [self.posedim + self.twistdim, self.udim], dim=1)
            dq_ddq = self.f_net(q_dq)

            return torch.cat((dq_ddq, zero_vec), dim=1)