import torch
from torchmin import minimize

def nesterovs_minimizer(x,gradient, eta, M,max_iters = 1500):
    d = x.shape[-1]
    A = 0
    y = x
    tau = 1
    k = 0
    mu = 1/eta - M
    L = 1/eta + M
    while torch.max(torch.sum(gradient(x)**2,dim=-1)) > (M*d)**2 and k < max_iters:
        a = (tau + (tau**2 + 4 * tau * L * A)**.5)/(2*L)
        Anext = A + a
        tx = A/Anext * y + a/Anext * x
        tauNext = tau + a * mu
        grad_tx = gradient(tx)
        yNext = tx - grad_tx/(mu + L)
        xNext = (tau * x + a * mu * tx - a * grad_tx)/tauNext
        k+=1
        A = Anext
        x = xNext
        y = yNext
        tau = tauNext
    return y, k

def gradient_descent(x0,gradient, threshold, al=1e-4):
    with torch.no_grad():
        xnew = x0
        k = 0
        while torch.max(torch.sum(gradient(xnew)**2,dim=-1)**.5) > threshold and k <3000:
            k+=1
            xnew = xnew - al * gradient(xnew)
        return xnew




























def newton_conjugate_gradient_ADDDS(x0,
                              potential_func,
                              max_iters=50,
                              regularization_objective_value=None,
                              regularization_lambda=0.01,
                              grad_potential_func=None,

                              vals_list=None,
                              grads_list=None
                              ):
    torch.autograd.set_detect_anomaly(True)

    def objective_to_minimize(x_opt_var):
        val = potential_func(x_opt_var)

        if grad_potential_func is not None and regularization_objective_value is not None and regularization_lambda > 0:
            grad = grad_potential_func(x_opt_var)
            grad_abs_sum = grad.abs().sum()


            if vals_list is not None:
                vals_list.append(val.item())
            if grads_list is not None:
                grads_list.append(grad_abs_sum.item())

            val = val + regularization_lambda * grad_abs_sum
        return val

    return minimize(
        objective_to_minimize, x0,
        method='cg',

        max_iter=max(1, int(max_iters)),
        disp=0
    ).x


def newton_conjugate_gradient(x0, potential, max_iters=50, grad_potential_func=None):

    torch.autograd.set_detect_anomaly(True)


    return minimize(
        potential, x0,
        method='newton-cg',
        options=dict(line_search='strong-wolfe'),
        max_iter=max_iters,
        disp=0
    ).x




def newton_conjugate_gradient_Hessian(x0,
                                      potential_func,
                                      max_iters=50,
                                      regularization_lambda=0.01):
    torch.autograd.set_detect_anomaly(True)

    def objective_to_minimize(x_opt_var):

        val = potential_func(x_opt_var)


        if regularization_lambda > 0:


            hessian_mat = torch.autograd.functional.hessian(potential_func, x_opt_var, create_graph=True)


            hessian_norm = torch.linalg.norm(hessian_mat, ord='fro')


            val = val + regularization_lambda * hessian_norm

        return val


    from torchmin import minimize
    return minimize(
        objective_to_minimize, x0,
        method='cg',
        max_iter=max(1, int(max_iters)),
        disp=0
    ).x

