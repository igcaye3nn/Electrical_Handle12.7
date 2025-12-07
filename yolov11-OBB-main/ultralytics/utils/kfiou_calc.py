# Copyright (c) SJTU. All rights reserved.
import torch
#from mmdet.models.losses.utils import weighted_loss
from torch import nn
import math

#from ..builder import ROTATED_LOSSES

def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2, 1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma

def kfiou_loss(pred,
               target,
               fun=None,
               beta=1.0 / 9.0,
               eps=1e-7):
    """Kalman filter IoU loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to None.
        beta (float): Defaults to 1.0/9.0.
        eps (float): Defaults to 1e-7.

    Returns:
        loss (torch.Tensor)
    """
    #xy_p = pred[:, :2]
    #xy_t = target[:, :2]
    original_dtype = pred.dtype
    pred = pred.float()
    target = target.float()
    
    xy_p, Sigma_p = xy_wh_r_2_xy_sigma(pred)
    xy_t, Sigma_t = xy_wh_r_2_xy_sigma(target)
    Sigma_p = Sigma_p.float()
    Sigma_t = Sigma_t.float()

    # Smooth-L1 norm
    diff = torch.abs(xy_p - xy_t)
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                          diff - 0.5 * beta).sum(dim=-1)
    Vb_p = 4 * Sigma_p.det().sqrt()
    Vb_t = 4 * Sigma_t.det().sqrt()
    K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
    Sigma = Sigma_p - K.bmm(Sigma_p)
    Vb = 4 * Sigma.det().sqrt()
    Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    if fun == 'ln':
        kf_loss = -torch.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = torch.exp(1 - KFIoU) - 1
    elif fun == 'square':
        kf_loss = 1 - KFIoU.pow(2)
    else:
        kf_loss = 1 - KFIoU

    loss = (0.01 * xy_loss + kf_loss).clamp(1e-7)
    KFIoU_2 =  1 / (1 + torch.log1p(loss))
    loss = loss.to(original_dtype)
    loss = KFIoU_2.to(original_dtype)

    return loss, KFIoU_2

# prompt: define two OBB boxes as torch tensors (pred, target) [3 x [xc, yc, w, h, r]] to call the "kfiou_loss" function
if __name__ == "__main__":
    pred = torch.tensor([
            [100.0, 100.0, 50.0, 30.0, 0.0],
            [200.0, 200.0, 60.0, 40.0, math.pi/4],
            [300.0, 300.0, 70.0, 50.0, math.pi/2]
        ])

    target = torch.tensor([
            [110.0, 100.0, 50.0, 30.0, 0.0],
            [205.0, 195.0, 60.0, 40.0, math.pi/4],
            [300.0, 300.0, 70.0, 50.0, math.pi/2]
        ])

    loss, kfiou2 = kfiou_loss(pred, target, fun=None)
    print(f"{loss}\n{kfiou2}")
    loss, kfiou2 = kfiou_loss(pred, target, fun='ln')
    print(f"{loss}\n{kfiou2}")
    loss, kfiou2 = kfiou_loss(pred, target, fun='exp')
    print(f"{loss}\n{kfiou2}")
