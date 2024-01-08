# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control, RSS 2021
# Thai Duong, Nikolay Atanasov

# code structure follows the style of HNN by Greydanus et al. and SymODEM by Zhong et al.
# https://github.com/greydanus/hamiltonian-nn
# https://github.com/Physics-aware-AI/Symplectic-ODENet
import torch
from se3hamneuralode import MLP, PSD, MatrixNet


class UnstructuredSO3NODE(torch.nn.Module):
    '''
    Architecture for input (q, q_dot, u),
    where q represent quaternion, a tensor of size (bs, n),
    q and q_dot are tensors of size (bs, n), and
    u is a tensor of size (bs, 1).
    '''

    def __init__(self, f_net = None, device=None, u_dim=3, init_gain=1):
        super(UnstructuredSO3NODE, self).__init__()
        self.rotmatdim = 9
        self.angveldim = 3
        self.u_dim = u_dim
        if f_net is None:
            self.f_net = MLP(self.rotmatdim + self.angveldim, 500, self.rotmatdim + self.angveldim, init_gain=init_gain).to(device)
        else:
            self.f_net = f_net

        self.device = device
        self.nfe = 0

    def forward(self, t, x):
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            zero_vec = torch.zeros(bs, self.u_dim, dtype=torch.float32, device=self.device)

            q_dq, u = torch.split(x, [self.rotmatdim + self.angveldim, self.u_dim], dim=1)
            dq_ddq = self.f_net(q_dq)

            return torch.cat((dq_ddq, zero_vec), dim=1)
