from Models.CoherenceNets.MethodeB import MethodeB
from utils.distance_metrics import VectorDistance

theta_method='Optim'


def coherence_loss(pred, flow, param_model='Quadratic', v_distance='l1'):
    """Compute coherence loss for a segment based on the given flow.

    Parameters
    ----------
    pred torch.tensor (b, L, I, J) : Mask proba predictions
    flow torch.tensor (b, 2, I, J) : Optical flow map
    param_model str : Parametric model used for regression ( Affine, Quadratic). Default : Quadratic
    v_distance str : name of the vector distance to use for computation (l1, l2, squared). Default : l1
    Returns
    -------
    coherence_losses (b): coherence loss for each sample of the batch
    """
    batch = {'Flow' : flow, 'Pred' : pred}
    vdist = VectorDistance(v_distance)
    I, J = pred.shape[2:]
    XoT = MethodeB.init_Xo(I, J, param_model).T
    theta = MethodeB.ComputeThetaOptim(batch, XoT, vdist).detach()
    coherence_losses = MethodeB.CoherenceLoss(theta, batch['Pred'], batch['Flow'], XoT, vdist)
    return coherence_losses



if __name__ = '__main__' :
    import torch
    pred = torch.rand(5, 3, 20, 50, requires_grad=True)
    flow =  torch.rand(5, 2, 20, 50)

    coherence_loss = coherence_loss(pred, flow)


    coherence_loss.mean().backward()

    pred.grad
