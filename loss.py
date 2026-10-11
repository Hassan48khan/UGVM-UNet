"""
Losses for UGVM-UNet (binary retinal vessel segmentation).

    L = L_seg(out, y)                              BCE + Dice at full resolution
      + w_aux * mean_i L_seg(aux_i, y_i)           deep supervision of every APH / URFI head
      + w_con * L_con                              cross-scale probability consistency
      + w_cl  * L_clDice(out, y)                   optional topology term (off by default)

Auxiliary targets
-----------------
Each auxiliary map is supervised with the ground truth resized by *area*
averaging, i.e. a soft target equal to the vessel fraction inside each coarse
cell. Nearest-neighbour down-sampling would delete 1-pixel vessels. A cell that
is partly vessel gets a target near 0.5, so the head is trained to report high
uncertainty (U = 1 - |2P - 1|) exactly where thin vessels and vessel borders
are - which is what the write/read gates and SAS consume.

Cross-scale consistency
-----------------------
With area targets, the coarse probability should equal the area-pooled fine
probability. L_con = MSE(P_coarse, area_pool(stopgrad(P_fine))) over adjacent
scales. It needs no labels.

Field-of-view mask
------------------
All terms accept an optional `valid` mask (1 inside the fundus FOV) so pixels
outside the FOV do not contribute.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UGVMLoss", "soft_dice_loss", "masked_bce_with_logits", "soft_cldice_loss"]


def _prep_target(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 3:
        target = target.unsqueeze(1)
    return target.float()


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor,
                           valid: Optional[torch.Tensor] = None,
                           pos_weight: Optional[float] = None) -> torch.Tensor:
    """BCE on logits with soft or hard targets, averaged over valid pixels."""
    pw = None if pos_weight is None else logits.new_tensor(pos_weight)
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    if valid is None:
        return loss.mean()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor,
                   valid: Optional[torch.Tensor] = None, eps: float = 1.0) -> torch.Tensor:
    """1 - Dice, computed per sample then averaged. Works with soft targets."""
    if valid is not None:
        prob, target = prob * valid, target * valid
    dims = tuple(range(1, prob.dim()))
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


# ----------------------------- soft clDice --------------------------------- #
def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, 1, 1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def _soft_skeleton(x: torch.Tensor, iters: int) -> torch.Tensor:
    skel = F.relu(x - _soft_open(x))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(prob: torch.Tensor, target: torch.Tensor,
                     valid: Optional[torch.Tensor] = None,
                     iters: int = 3, eps: float = 1.0) -> torch.Tensor:
    """Soft centreline Dice (Shit et al., CVPR 2021).

    `iters` should be at least the largest vessel radius in pixels at the
    training resolution; 3 suits DRIVE / CHASE_DB1 patches, increase for HRF.
    """
    if valid is not None:
        prob, target = prob * valid, target * valid
    skel_p = _soft_skeleton(prob, iters)
    skel_t = _soft_skeleton(target, iters)
    dims = tuple(range(1, prob.dim()))
    tprec = ((skel_p * target).sum(dims) + eps) / (skel_p.sum(dims) + eps)
    tsens = ((skel_t * prob).sum(dims) + eps) / (skel_t.sum(dims) + eps)
    return (1.0 - 2.0 * tprec * tsens / (tprec + tsens)).mean()


# ------------------------------ full loss ---------------------------------- #
class UGVMLoss(nn.Module):
    """Composite loss for UGVM-UNet.

    Args:
        w_bce, w_dice: weights inside L_seg (used for main and auxiliary terms).
        w_aux:         weight of the averaged auxiliary (deep supervision) loss.
        w_con:         weight of the cross-scale consistency loss.
        w_cldice:      weight of soft clDice on the main output (0 disables).
        cldice_iters:  skeletonisation iterations for clDice.
        pos_weight:    optional positive-class weight for BCE (vessels are ~10 %
                       of pixels; values around 2-4 raise sensitivity).

    forward(outputs, target, valid=None) -> (total, parts)
        outputs: logits (B, 1, H, W), or the (logits, aux_list) tuple returned
                 by the model in training mode.
        target:  (B, 1, H, W) or (B, H, W), values in {0, 1}.
        valid:   optional FOV mask, same shape as target.
        parts:   dict of detached scalars for logging.
    """

    def __init__(self, w_bce: float = 1.0, w_dice: float = 1.0, w_aux: float = 0.4,
                 w_con: float = 0.1, w_cldice: float = 0.0, cldice_iters: int = 3,
                 pos_weight: Optional[float] = None):
        super().__init__()
        self.w_bce, self.w_dice = w_bce, w_dice
        self.w_aux, self.w_con, self.w_cldice = w_aux, w_con, w_cldice
        self.cldice_iters = cldice_iters
        self.pos_weight = pos_weight

    def _seg(self, logits, target, valid):
        bce = masked_bce_with_logits(logits, target, valid, self.pos_weight)
        dice = soft_dice_loss(torch.sigmoid(logits), target, valid)
        return self.w_bce * bce + self.w_dice * dice

    @staticmethod
    def _consistency(aux: Sequence[torch.Tensor]) -> torch.Tensor:
        """MSE between coarse probabilities and area-pooled finer probabilities."""
        by_size: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        for a in aux:
            by_size.setdefault(tuple(a.shape[-2:]), []).append(torch.sigmoid(a))
        sizes = sorted(by_size, key=lambda s: s[0] * s[1], reverse=True)  # fine -> coarse
        if len(sizes) < 2:
            return aux[0].new_zeros(())
        probs = [torch.stack(by_size[s], 0).mean(0) for s in sizes]
        loss = 0.0
        for fine, coarse in zip(probs[:-1], probs[1:]):
            ref = F.interpolate(fine.detach(), size=coarse.shape[-2:], mode="area")
            loss = loss + F.mse_loss(coarse, ref)
        return loss / (len(sizes) - 1)

    def forward(self, outputs: Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]],
                target: torch.Tensor, valid: Optional[torch.Tensor] = None):
        if isinstance(outputs, (tuple, list)):
            logits, aux = outputs[0], list(outputs[1])
        else:
            logits, aux = outputs, []
        target = _prep_target(target)
        valid = None if valid is None else _prep_target(valid)

        parts = {}
        total = self._seg(logits, target, valid)
        parts["seg"] = total.detach()

        if aux and self.w_aux > 0:
            cache: Dict[Tuple[int, int], Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
            l_aux = 0.0
            for a in aux:
                size = tuple(a.shape[-2:])
                if size not in cache:
                    t = F.interpolate(target, size=size, mode="area")
                    v = None if valid is None else F.interpolate(valid, size=size, mode="area")
                    cache[size] = (t, v)
                t, v = cache[size]
                l_aux = l_aux + self._seg(a, t, v)
            l_aux = l_aux / len(aux)
            total = total + self.w_aux * l_aux
            parts["aux"] = l_aux.detach()

        if aux and self.w_con > 0:
            l_con = self._consistency(aux)
            total = total + self.w_con * l_con
            parts["con"] = l_con.detach()

        if self.w_cldice > 0:
            l_cl = soft_cldice_loss(torch.sigmoid(logits), target, valid, self.cldice_iters)
            total = total + self.w_cldice * l_cl
            parts["cldice"] = l_cl.detach()

        parts["total"] = total.detach()
        return total, parts


if __name__ == "__main__":
    torch.manual_seed(0)
    y = (torch.rand(2, 1, 64, 64) > 0.9).float()
    out = torch.randn(2, 1, 64, 64, requires_grad=True)
    aux = [torch.randn(2, 1, s, s, requires_grad=True) for s in (16, 16, 8, 4)]
    loss, parts = UGVMLoss(w_cldice=0.1)((out, aux), y)
    loss.backward()
    print({k: round(v.item(), 4) for k, v in parts.items()})
