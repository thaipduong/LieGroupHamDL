# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control, RSS 2021
# Thai Duong, Nikolay Atanasov

# code structure follows the style of HNN by Greydanus et al. and SymODEM by Zhong et al.
# https://github.com/greydanus/hamiltonian-nn
# https://github.com/Physics-aware-AI/Symplectic-ODENet

import torch
import numpy as np


class SE3HamNODEGT(torch.nn.Module):
    def __init__(self, device=None, udim = 4):
        super(SE3HamNODEGT, self).__init__()
        init_gain = 0.001
        self.xdim = 3
        self.Rdim = 9
        self.linveldim = 3
        self.angveldim = 3
        self.posedim = self.xdim + self.Rdim #3 for position + 12 for rotmat
        self.twistdim = self.linveldim + self.angveldim #3 for linear vel + 3 for ang vel
        self.udim = udim
        self.device = device
        self.nfe = 0


    def forward(self, t, input):
        with torch.enable_grad():
            self.nfe += 1
            bs = input.shape[0]
            q, q_dot, u = torch.split(input, [self.posedim, self.twistdim, self.udim], dim=1)
            x, R = torch.split(q, [self.xdim, self.Rdim], dim=1)
            q_dot_v, q_dot_w = torch.split(q_dot, [self.linveldim, self.angveldim], dim=1)

            m = 1/0.027
            m_guess = m * torch.eye(3, requires_grad=True, dtype=torch.float32)
            m_guess = m_guess.reshape((1, 3, 3))
            M_q_inv1 = m_guess.repeat(bs, 1, 1).to(self.device)
            #M_q_inv1 = self.M_net1(x)
            # np.array([2.3951, 2.3951, 3.2347])*1e-5
            J = np.diag([2.3951e-5, 2.3951e-5, 3.2347e-5])
            J_inv = np.linalg.inv(J)
            # inertia_guess = 71429 * torch.eye(3, requires_grad=True)
            # inertia_guess[2:2] = 46083
            inertia_guess = torch.tensor(J_inv, requires_grad=True, dtype=torch.float32)
            inertia_guess = inertia_guess.reshape((1, 3, 3))
            M_q_inv2 = inertia_guess.repeat(bs, 1, 1).to(self.device)


            q_dot_aug_v = torch.unsqueeze(q_dot_v, dim=2)
            q_dot_aug_w = torch.unsqueeze(q_dot_w, dim=2)
            pv = torch.squeeze(torch.matmul(torch.inverse(M_q_inv1), q_dot_aug_v), dim=2)
            pw = torch.squeeze(torch.matmul(torch.inverse(M_q_inv2), q_dot_aug_w), dim=2)
            q_p = torch.cat((q, pv, pw), dim=1)
            q, pv, pw = torch.split(q_p, [self.posedim, self.linveldim, self.angveldim], dim=1)
            x, R = torch.split(q, [self.xdim, self.Rdim], dim=1)

            V_q = 0.027 * 9.81 * q[:, 2]
            # Calculate the Hamiltonian
            p_aug_v = torch.unsqueeze(pv, dim=2)
            p_aug_w = torch.unsqueeze(pw, dim=2)
            H = torch.squeeze(torch.matmul(torch.transpose(p_aug_v, 1, 2), torch.matmul(M_q_inv1, p_aug_v))) / 2.0 + \
                torch.squeeze(torch.matmul(torch.transpose(p_aug_w, 1, 2), torch.matmul(M_q_inv2, p_aug_w))) / 2.0 + \
                torch.squeeze(V_q)

            # Calculate the partial derivative using autograd
            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
            # Order: position (3), rotmat (9), lin vel (3) in body frame, ang vel (3) in body frame
            dHdx, dHdR, dHdpv, dHdpw = torch.split(dH, [self.xdim, self.Rdim, self.linveldim, self.angveldim], dim=1)

            # Calculate g*u
            f = np.array([[0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
            f = torch.tensor(f, dtype=torch.float32).to(self.device)
            f = f.reshape((1, 6, 4))
            g_q = f.repeat(bs, 1, 1).to(self.device)
            F = torch.squeeze(torch.matmul(g_q, torch.unsqueeze(u, dim=2)))

            # Hamilton's equation on SE(3) manifold for (q,p)
            Rmat = R.view(-1, 3, 3)
            dx = torch.squeeze(torch.matmul(Rmat, torch.unsqueeze(dHdpv, dim=2)))
            dR03 = torch.cross(Rmat[:, 0, :], dHdpw)
            dR36 = torch.cross(Rmat[:, 1, :], dHdpw)
            dR69 = torch.cross(Rmat[:, 2, :], dHdpw)
            dR = torch.cat((dR03, dR36, dR69), dim=1)
            dpv = torch.cross(pv, dHdpw) \
                  - torch.squeeze(torch.matmul(torch.transpose(Rmat, 1, 2), torch.unsqueeze(dHdx, dim=2))) \
                  + F[:, 0:3]
            dpw = torch.cross(pw, dHdpw) \
                  + torch.cross(pv, dHdpv) \
                  + torch.cross(Rmat[:, 0, :], dHdR[:, 0:3]) \
                  + torch.cross(Rmat[:, 1, :], dHdR[:, 3:6]) \
                  + torch.cross(Rmat[:, 2, :], dHdR[:, 6:9]) \
                  + F[:,3:6]

            # Hamilton's equation on SE(3) manifold for twist xi
            # dM_inv_dt1 = torch.zeros_like(M_q_inv1)
            # for row_ind in range(self.linveldim):
            #     for col_ind in range(self.linveldim):
            #         dM_inv1 = \
            #             torch.autograd.grad(M_q_inv1[:, row_ind, col_ind].sum(), x, create_graph=True)[0]
            #         dM_inv_dt1[:, row_ind, col_ind] = (dM_inv1 * dx).sum(-1)
            dv = torch.squeeze(torch.matmul(M_q_inv1, torch.unsqueeze(dpv, dim=2)), dim=2) #\
                  #+ torch.squeeze(torch.matmul(dM_inv_dt1, torch.unsqueeze(pv, dim=2)), dim=2)

            # dM_inv_dt2 = torch.zeros_like(M_q_inv2)
            # for row_ind in range(self.angveldim):
            #     for col_ind in range(self.angveldim):
            #         dM_inv2 = \
            #             torch.autograd.grad(M_q_inv2[:, row_ind, col_ind].sum(), R, create_graph=True)[0]
            #         dM_inv_dt2[:, row_ind, col_ind] = (dM_inv2 * dR).sum(-1)
            dw = torch.squeeze(torch.matmul(M_q_inv2, torch.unsqueeze(dpw, dim=2)), dim=2) #\
                  #+ torch.squeeze(torch.matmul(dM_inv_dt2, torch.unsqueeze(pw, dim=2)), dim=2)

            batch_size = input.shape[0]
            zero_vec = torch.zeros(batch_size, self.udim, dtype=torch.float32, device=self.device)
            return torch.cat((dx, dR, dv, dw, zero_vec), dim=1)