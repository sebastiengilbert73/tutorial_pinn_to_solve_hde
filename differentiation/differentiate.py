import torch

def first_derivative(neural_net, x):  # x.shape = (B, N_d)
    #x.requires_grad = True
    u = neural_net(x)
    du_dx0__du_dx1__du_dxn = torch.autograd.grad(u, x, grad_outputs=u.data.new(u.shape).fill_(1), create_graph=True)[0]  # (B, N_d)
    return du_dx0__du_dx1__du_dxn

def second_derivative(neural_net, x, first_derivative_ndx):  # x.shape = (B, N_d)
    du_dx0__du_dx1__du_dxn = first_derivative(neural_net, x)  # (B, N_d)
    du_dxi = du_dx0__du_dx1__du_dxn[:, first_derivative_ndx]  # (B)
    d2u_dxidx0__d2u_dxidx1__d2u_dxidxn = torch.autograd.grad(du_dxi, x, grad_outputs=du_dxi.data.new(du_dxi.shape).fill_(1), create_graph=True)[0]  # (B, N_d)
    return d2u_dxidx0__d2u_dxidx1__d2u_dxidxn