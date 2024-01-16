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



class SE3HamNODEPX4(torch.nn.Module):
    def __init__(self, device=None, pretrain = True, Dv_net  = None, Dw_net = None, g_net = None, udim = 4):
        super(SE3HamNODEPX4, self).__init__()
        self.xdim = 3
        self.Rdim = 9
        self.linveldim = 3
        self.angveldim = 3
        self.posedim = self.xdim + self.Rdim #3 for position + 12 for rotmat
        self.twistdim = self.linveldim + self.angveldim #3 for linear vel + 3 for ang vel
        self.udim = udim
        if Dv_net is None:
            self.Dv_net = PSD(self.xdim, 20, self.linveldim, init_gain=.000001, epsilon = 0.0).to(device)
        else:
            self.Dv_net = Dv_net
        if Dw_net is None:
            self.Dw_net = PSD(self.Rdim, 20, self.twistdim - self.linveldim, init_gain=.000001, epsilon = 0.0).to(device)
        else:
            self.Dw_net = Dw_net
        if g_net is None:
            self.g_net = MatrixNet(self.posedim, 20, self.twistdim*self.udim, shape=(self.twistdim,self.udim), init_gain=0.000001).to(device)
        else:
            self.g_net = g_net
        self.device = device
        self.nfe = 0
        if pretrain:
            self.pretrain()

    def pretrain(self):
        # Pretrain M_net2
        batch = 250000
        # Uniformly generate quaternion using http://planning.cs.uiuc.edu/node198.html
        rand_ = np.random.uniform(size=(batch, 3))
        u1, u2, u3 = rand_[:, 0], rand_[:, 1], rand_[:, 2]
        quat = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                         np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)])
        x_tensor = torch.tensor(0.1 * np.random.uniform(low = -1.0, high=1.0,size=(batch, 3))).to(self.device)
        q_tensor = torch.tensor(quat.transpose(), dtype=torch.float64).view(batch, 4).to(self.device)
        R_tensor = compute_rotation_matrix_from_quaternion(q_tensor)
        R_tensor = R_tensor.view(-1, 9)
        q_tensor = torch.cat((x_tensor, R_tensor), dim=1)
        g_net = self.g_net(q_tensor)
        Dv_net = self.Dv_net(x_tensor)
        Dw_net = self.Dw_net(R_tensor)
        # Train M_net2 to output identity matrix
        g_guess = torch.zeros((batch, 6, 4)).to(self.device)
        D_guess = torch.zeros((batch, 3, 3)).to(self.device)
        optim_g = torch.optim.Adam(self.g_net.parameters(), 1e-3, weight_decay=0.0)
        optim_Dv = torch.optim.Adam(self.Dv_net.parameters(), 1e-3, weight_decay=0.0)
        optim_Dw = torch.optim.Adam(self.Dw_net.parameters(), 1e-3, weight_decay=0.0)
        loss = (L2_loss(g_net, g_guess) + L2_loss(D_guess, Dv_net) + L2_loss(D_guess, Dw_net))
        print("Start pretraining Mnet2!", loss.detach().cpu().numpy())
        step = 1
        while loss > 1e-9:
            loss.backward()
            optim_g.step()
            optim_Dv.step()
            optim_Dw.step()
            optim_g.zero_grad()
            optim_Dv.zero_grad()
            optim_Dw.zero_grad()
            if step % 10 == 0:
                print("step", step, loss.detach().cpu().numpy())
            g_net = self.g_net(q_tensor)
            Dv_net = self.Dv_net(x_tensor)
            Dw_net = self.Dw_net(R_tensor)
            loss =  (L2_loss(g_net, g_guess) + L2_loss(D_guess, Dv_net) + L2_loss(D_guess, Dw_net))
            step = step + 1
        print("Pretraining g_net done!", loss.detach().cpu().numpy())
        # Delete data and cache to save memory
        del q_tensor
        torch.cuda.empty_cache()

    def forward(self, t, input):
        with torch.enable_grad():
            self.nfe += 1
            q, q_dot, u = torch.split(input, [self.posedim, self.twistdim, self.udim], dim=1)
            x, R = torch.split(q, [self.xdim, self.Rdim], dim=1)
            q_dot_v, q_dot_w = torch.split(q_dot, [self.linveldim, self.angveldim], dim=1)

            batch_size = input.shape[0]
            # As pointed out in the paper, if we scale the network's output by a scaling factor, the dynamics do not
            # change. So to simplify the training process, we can assume a fixed mass, and learn a scaled version of
            # the other neural networks.
            m = 1.3
            m_inv = 1 / m
            m_guess = m_inv * torch.eye(3, requires_grad=True, dtype=torch.float64)
            m_guess = m_guess.reshape((1, 3, 3))
            M_q_inv1 = m_guess.repeat(batch_size, 1, 1).to(self.device)
            J = np.diag([0.012, 0.012, 0.02])
            J_inv = np.linalg.inv(J)
            inertia_guess = torch.tensor(J_inv, requires_grad=True, dtype=torch.float64)
            inertia_guess = inertia_guess.reshape((1, 3, 3))
            M_q_inv2 = inertia_guess.repeat(batch_size, 1, 1).to(self.device)

            q_dot_aug_v = torch.unsqueeze(q_dot_v, dim=2)
            q_dot_aug_w = torch.unsqueeze(q_dot_w, dim=2)
            pv = torch.squeeze(torch.matmul(torch.inverse(M_q_inv1), q_dot_aug_v), dim=2)
            pw = torch.squeeze(torch.matmul(torch.inverse(M_q_inv2), q_dot_aug_w), dim=2)
            q_p = torch.cat((q, pv, pw), dim=1)
            q, pv, pw = torch.split(q_p, [self.posedim, self.linveldim, self.angveldim], dim=1)
            x, R = torch.split(q, [self.xdim, self.Rdim], dim=1)
            V_q = m * 9.81 * q[:, 2]

            # Query the neural networks that learned a scaled version of the ground-truth values.
            g_q = self.g_net(q)
            Dv_net = self.Dv_net(x)
            Dw_net = self.Dw_net(R)


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
                  - torch.squeeze(torch.matmul(Dv_net,  torch.unsqueeze(dHdpv, dim = 2))) \
                  + F[:, 0:3]
            dpw = torch.cross(pw, dHdpw) \
                  + torch.cross(pv, dHdpv) \
                  + torch.cross(Rmat[:, 0, :], dHdR[:, 0:3]) \
                  + torch.cross(Rmat[:, 1, :], dHdR[:, 3:6]) \
                  + torch.cross(Rmat[:, 2, :], dHdR[:, 6:9]) \
                  - torch.squeeze(torch.matmul(Dw_net,  torch.unsqueeze(dHdpw, dim = 2))) \
                  + F[:,3:6]

            # Hamilton's equation on SE(3) manifold for twist xi
            dv = torch.squeeze(torch.matmul(M_q_inv1, torch.unsqueeze(dpv, dim=2)), dim=2)
            dw = torch.squeeze(torch.matmul(M_q_inv2, torch.unsqueeze(dpw, dim=2)), dim=2)

            zero_vec = torch.zeros(batch_size, self.udim, dtype=torch.float32, device=self.device)
            return torch.cat((dx, dR, dv, dw, zero_vec), dim=1)