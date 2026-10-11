"""
UGVM-UNet: Uncertainty-Gated Vision Mamba UNet with Cross-Orientation
Attention for Retinal Vessel Segmentation.

Modules defined here
--------------------
COEMA   Cross-Orientation Efficient Multi-scale Attention. EMA's 1x1 / 3x3
        branches are replaced by horizontal / vertical multi-scale strip
        branches and the cross-spatial step becomes cross-orientation.
APH     Ambiguity Prior Head. A small deep-supervised head that predicts a
        coarse vessel probability P; uncertainty is U = 1 - |2P - 1|.
WRG     Write-Read Gate around SS2D (implemented in GatedSS2D + UGSSBlock):
            write:  h_k = A_k h_{k-1} + (1 - alpha * U_k) * B_k x_k
            read :  y   = (1 + beta * U) * SS2D(x)
        The write gate is applied to the discretised input matrix B, so the
        step size Delta and the read-out matrix C are left untouched.
UGSS    Uncertainty-Gated State Space block = Norm -> COEMA -> APH -> WRG.
SAS     Sparse Ambiguity Scan. A bidirectional state-space scan over only the
        top-K most uncertain tokens, kept in raster order.
URFI    Uncertainty-Routed Feature Interaction (skip fusion) = polarity gate
        -> SAS -> fuse.

All tensors are channels-first (B, C, H, W). The network is binary
(num_classes = 1): the uncertainty definition assumes a single vessel
probability.

Selective scan backend
----------------------
If `mamba_ssm` is installed and the tensors live on CUDA, its compiled kernel
is used. Otherwise a pure-PyTorch reference scan is used (identical maths,
sequential loop, slow for long sequences but fine for CPU tests).
"""

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional fast CUDA kernel
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_cuda
except Exception:  # pragma: no cover - mamba_ssm is optional
    _selective_scan_cuda = None

__all__ = [
    "COEMA",
    "AmbiguityPriorHead",
    "GatedSS2D",
    "UGSSBlock",
    "SparseAmbiguityScan",
    "URFI",
    "UGVMUNet",
    "ugvm_unet_tiny",
    "ugvm_unet_small",
    "ugvm_unet_base",
]


# --------------------------------------------------------------------------- #
# Basic layers
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dimension of a (B, C, H, W) tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """Stochastic depth (per sample)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], *([1] * (x.dim() - 1))).bernoulli_(keep)
        return x * mask / keep


def ambiguity_from_logits(logits: torch.Tensor, smooth: bool = True):
    """Vessel probability P and uncertainty U = 1 - |2P - 1| in [0, 1].

    U is 1 at the decision boundary (P = 0.5) and 0 for confident pixels.
    A 3x3 mean filter removes isolated responses.
    """
    p = torch.sigmoid(logits)
    u = 1.0 - (2.0 * p - 1.0).abs()
    if smooth:
        u = F.avg_pool2d(u, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    return p, u


# --------------------------------------------------------------------------- #
# Selective scan (state-space recurrence)
# --------------------------------------------------------------------------- #
def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    """Pure-PyTorch selective scan.

    u, delta : (b, K*d, L)      input and step size per channel
    A        : (K*d, N)         state matrix (negative)
    B, C     : (b, K, N, L)     input / read-out matrices, shared across the d
                                channels of each of the K scan directions
    D        : (K*d,)           skip term

    Recurrence (same simplified discretisation as Mamba):
        h_i = exp(delta_i * A) * h_{i-1} + delta_i * B_i * u_i
        y_i = <h_i, C_i> + D * u_i
    """
    b, kd, L = u.shape
    K, N = B.shape[1], B.shape[2]
    d = kd // K
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    B = B[:, :, None].expand(b, K, d, N, L).reshape(b, kd, N, L)
    C = C[:, :, None].expand(b, K, d, N, L).reshape(b, kd, N, L)

    h = u.new_zeros(b, kd, N)
    ys = []
    for i in range(L):
        dt = delta[:, :, i].unsqueeze(-1)  # (b, kd, 1)
        h = torch.exp(dt * A) * h + dt * B[..., i] * u[:, :, i].unsqueeze(-1)
        ys.append((h * C[..., i]).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + u * D[None, :, None]
    return y


def selective_scan(u, delta, A, B, C, D, delta_bias):
    if _selective_scan_cuda is not None and u.is_cuda:
        return _selective_scan_cuda(
            u.contiguous(), delta.contiguous(), A.contiguous(), B.contiguous(), C.contiguous(),
            D, z=None, delta_bias=delta_bias, delta_softplus=True, return_last_state=False,
        )
    return selective_scan_ref(u, delta, A, B, C, D, delta_bias=delta_bias, delta_softplus=True)


class SelectiveScanCore(nn.Module):
    """Input-dependent state-space parameters for K scan directions.

    forward(xs, ws):
        xs : (b, K, d, L)  sequences, one per scan direction
        ws : (b, K, 1, L)  optional write-gate weights, multiplied into B
    returns (b, K, d, L)
    """

    def __init__(self, d_inner: int, d_state: int = 16, num_dirs: int = 4,
                 dt_rank: Optional[int] = None, dt_min: float = 1e-3, dt_max: float = 0.1,
                 dt_init_floor: float = 1e-4):
        super().__init__()
        self.d_inner, self.d_state, self.K = d_inner, d_state, num_dirs
        self.dt_rank = dt_rank or math.ceil(d_inner / 16)
        R, N, K = self.dt_rank, d_state, num_dirs

        bound = 1.0 / math.sqrt(d_inner)
        self.x_proj_weight = nn.Parameter(torch.empty(K, R + 2 * N, d_inner).uniform_(-bound, bound))

        dt_std = R ** -0.5
        self.dt_projs_weight = nn.Parameter(torch.empty(K, d_inner, R).uniform_(-dt_std, dt_std))
        # bias such that softplus(bias) is log-uniform in [dt_min, dt_max]
        dt = torch.exp(torch.rand(K, d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        self.dt_projs_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        A = torch.arange(1, N + 1, dtype=torch.float32).log()
        self.A_logs = nn.Parameter(A.repeat(K * d_inner, 1))  # (K*d, N)
        self.Ds = nn.Parameter(torch.ones(K * d_inner))

    def forward(self, xs: torch.Tensor, ws: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, K, d, L = xs.shape
        R, N = self.dt_rank, self.d_state
        out_dtype = xs.dtype
        # the recurrence is run in float32 for numerical stability under AMP
        with torch.autocast(device_type=xs.device.type, enabled=False):
            xs = xs.float()
            x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight.float())
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight.float())
            if ws is not None:
                Bs = Bs * ws.float()  # write gate on the input matrix
            ys = selective_scan(
                xs.reshape(b, K * d, L),
                dts.reshape(b, K * d, L),
                -torch.exp(self.A_logs.float()),
                Bs, Cs,
                self.Ds.float(),
                self.dt_projs_bias.float().reshape(-1),
            )
        return ys.reshape(b, K, d, L).to(out_dtype)


class GatedSS2D(nn.Module):
    """VMamba-style 2D selective scan (4 directions) with an optional write gate.

    forward(x, w):
        x : (B, C, H, W)
        w : (B, 1, H, W) per-token write weight, or None for a plain SS2D
    """

    def __init__(self, dim: int, d_state: int = 16, expand: float = 2.0, d_conv: int = 3):
        super().__init__()
        d_inner = int(expand * dim)
        self.in_proj = nn.Conv2d(dim, 2 * d_inner, 1)
        self.dwconv = nn.Conv2d(d_inner, d_inner, d_conv, padding=d_conv // 2, groups=d_inner)
        self.core = SelectiveScanCore(d_inner, d_state=d_state, num_dirs=4)
        self.out_norm = LayerNorm2d(d_inner)
        self.out_proj = nn.Conv2d(d_inner, dim, 1)

    @staticmethod
    def _unfold(x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, 4, C, H*W): row-major, column-major and their reverses."""
        hw = x.flatten(2)
        wh = x.transpose(2, 3).reshape(x.shape[0], x.shape[1], -1)
        s = torch.stack([hw, wh], dim=1)
        return torch.cat([s, s.flip(-1)], dim=1)

    @staticmethod
    def _fold(ys: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Inverse of `_unfold`, summing the four directions."""
        b, _, c, _ = ys.shape

        def col_to_row(t):
            return t.reshape(b, c, W, H).transpose(2, 3).reshape(b, c, -1)

        y = ys[:, 0] + col_to_row(ys[:, 1]) + ys[:, 2].flip(-1) + col_to_row(ys[:, 3].flip(-1))
        return y.reshape(b, c, H, W)

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        H, W = x.shape[-2:]
        x, z = self.in_proj(x).chunk(2, dim=1)
        x = F.silu(self.dwconv(x))
        ys = self.core(self._unfold(x), None if w is None else self._unfold(w))
        y = self.out_norm(self._fold(ys, H, W)) * F.silu(z)
        return self.out_proj(y)


# --------------------------------------------------------------------------- #
# CO-EMA
# --------------------------------------------------------------------------- #
class COEMA(nn.Module):
    """Cross-Orientation Efficient Multi-scale Attention.

    Channels are split into groups (as in EMA). Each group goes through a
    horizontal branch (depthwise 1xk strips) and a vertical branch (depthwise
    kx1 strips), k in `kernels`. The globally pooled descriptor of one branch
    then re-weights the spatial map of the other:

        w_h = softmax(GAP(GN(Y_h))),  w_v = softmax(GAP(GN(Y_v)))
        M   = w_h . Y_v + w_v . Y_h
        out = X * sigmoid(M)
    """

    def __init__(self, dim: int, groups: int = 8, kernels: Sequence[int] = (7, 11, 21)):
        super().__init__()
        g = max(1, min(groups, dim))
        while dim % g != 0:
            g -= 1
        self.groups = g
        cg = dim // g
        self.h_convs = nn.ModuleList(
            [nn.Conv2d(cg, cg, (1, k), padding=(0, k // 2), groups=cg) for k in kernels])
        self.v_convs = nn.ModuleList(
            [nn.Conv2d(cg, cg, (k, 1), padding=(k // 2, 0), groups=cg) for k in kernels])
        self.h_proj = nn.Conv2d(cg, cg, 1)
        self.v_proj = nn.Conv2d(cg, cg, 1)
        self.h_norm = nn.GroupNorm(cg, cg)
        self.v_norm = nn.GroupNorm(cg, cg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gx = x.reshape(b * self.groups, c // self.groups, h, w)

        yh = sum(conv(gx) for conv in self.h_convs) / len(self.h_convs)
        yv = sum(conv(gx) for conv in self.v_convs) / len(self.v_convs)
        yh = self.h_norm(self.h_proj(yh))
        yv = self.v_norm(self.v_proj(yv))

        wh = torch.softmax(yh.mean(dim=(2, 3)), dim=-1).unsqueeze(1)  # (bg, 1, cg)
        wv = torch.softmax(yv.mean(dim=(2, 3)), dim=-1).unsqueeze(1)
        m = torch.matmul(wh, yv.flatten(2)) + torch.matmul(wv, yh.flatten(2))  # (bg, 1, hw)
        out = gx * torch.sigmoid(m.reshape(b * self.groups, 1, h, w))
        return out.reshape(b, c, h, w)


# --------------------------------------------------------------------------- #
# APH + UGSS block
# --------------------------------------------------------------------------- #
class AmbiguityPriorHead(nn.Module):
    """Predicts a 1-channel vessel logit map from a feature map."""

    def __init__(self, dim: int):
        super().__init__()
        mid = max(dim // 4, 8)
        self.net = nn.Sequential(
            nn.Conv2d(dim, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UGSSBlock(nn.Module):
    """Uncertainty-Gated State Space block.

        z      = COEMA(Norm(x))
        U      = ambiguity(APH(z))            (gradient detached)
        y      = SS2D(z, write = 1 - tanh(a) * U)
        out    = x + DropPath((1 + beta * U) * y)

    `a` and `beta` start at 0, so the block is initially a plain VSS block
    with CO-EMA in front. tanh keeps the write weight inside (0, 2); the sign
    is learned, so the trained value of tanh(a) is itself a result to report
    (positive = uncertain tokens write less, as hypothesised).

    Returns (out, aux_logits). aux_logits is None when both gates are off.
    """

    def __init__(self, dim: int, d_state: int = 16, expand: float = 2.0, drop_path: float = 0.0,
                 use_coema: bool = True, use_write_gate: bool = True, use_read_gate: bool = True,
                 coema_groups: int = 8, coema_kernels: Sequence[int] = (7, 11, 21)):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.coema = COEMA(dim, coema_groups, coema_kernels) if use_coema else nn.Identity()
        self.ss2d = GatedSS2D(dim, d_state=d_state, expand=expand)
        self.drop_path = DropPath(drop_path)

        self.use_write_gate, self.use_read_gate = use_write_gate, use_read_gate
        self.use_gate = use_write_gate or use_read_gate
        if self.use_gate:
            self.aph = AmbiguityPriorHead(dim)
        if use_write_gate:
            self.alpha_raw = nn.Parameter(torch.zeros(1))
        if use_read_gate:
            self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        z = self.coema(self.norm(x))
        if not self.use_gate:
            return x + self.drop_path(self.ss2d(z)), None

        logits = self.aph(z)
        _, u = ambiguity_from_logits(logits.detach())
        w = 1.0 - torch.tanh(self.alpha_raw) * u if self.use_write_gate else None
        y = self.ss2d(z, w)
        if self.use_read_gate:
            y = (1.0 + self.beta * u) * y
        return x + self.drop_path(y), logits


# --------------------------------------------------------------------------- #
# SAS + URFI
# --------------------------------------------------------------------------- #
class SparseAmbiguityScan(nn.Module):
    """Bidirectional state-space scan over the top-K most uncertain tokens.

    The selected tokens keep their raster order, so fragments of a broken thin
    vessel that are far apart in a full scan become close in this sequence.
    Because sequence adjacency no longer implies spatial adjacency, the
    normalised (row, col) position of each token is added as an embedding.

    Returns a (B, C, H, W) residual that is zero at non-selected positions.
    The output projection is zero-initialised, so the module starts as a no-op.
    """

    def __init__(self, dim: int, ratio: float = 0.25, d_state: int = 16, expand: float = 2.0):
        super().__init__()
        assert 0.0 < ratio <= 1.0
        self.ratio = ratio
        d_inner = int(expand * dim)
        self.norm = nn.LayerNorm(dim)
        self.pos = nn.Linear(2, dim)
        self.in_proj = nn.Linear(dim, 2 * d_inner)
        self.dwconv = nn.Conv1d(d_inner, d_inner, 3, padding=1, groups=d_inner)
        self.core = SelectiveScanCore(d_inner, d_state=d_state, num_dirs=2)
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        b, c, h, w = s.shape
        L = h * w
        k = max(1, min(L, int(round(self.ratio * L))))

        idx = u.flatten(1).topk(k, dim=1).indices.sort(dim=1).values  # (b, k), raster order
        idx_c = idx.unsqueeze(-1).expand(-1, -1, c)
        tokens = s.flatten(2).transpose(1, 2).gather(1, idx_c)  # (b, k, c)

        rows = torch.div(idx, w, rounding_mode="floor").to(s.dtype) / max(h - 1, 1)
        cols = (idx % w).to(s.dtype) / max(w - 1, 1)
        coords = torch.stack([rows, cols], dim=-1) * 2.0 - 1.0

        t = self.norm(tokens) + self.pos(coords)
        x, z = self.in_proj(t).chunk(2, dim=-1)
        x = F.silu(self.dwconv(x.transpose(1, 2)))  # (b, d, k)
        ys = self.core(torch.stack([x, x.flip(-1)], dim=1))
        y = ys[:, 0] + ys[:, 1].flip(-1)
        y = self.out_norm(y.transpose(1, 2)) * F.silu(z)
        delta = self.out_proj(y)  # (b, k, c)

        out = delta.new_zeros(b, L, c).scatter(1, idx_c, delta)
        return out.transpose(1, 2).reshape(b, c, h, w)


class URFI(nn.Module):
    """Uncertainty-Routed Feature Interaction (replaces the plain skip).

        P, U = ambiguity(head(D))                       (detached for gating)
        g    = min(1, P + U)                            polarity gate
        E'   = E * (1 - gamma * (1 - g))                gamma = sigmoid(raw) in (0, 1)
        E''  = E' + SAS(E' + D, U)
        F    = Fuse([E'', D])

    g is 1 for confident-vessel and uncertain pixels and small only for
    confident background, so encoder texture there (lesions, background noise)
    is attenuated. gamma is learnable and starts small (~0.12) so that a wrong
    early decoder prediction cannot erase thin-vessel detail from the skip.

    Returns (fused, aux_logits).
    """

    def __init__(self, dim: int, topk_ratio: float = 0.25, d_state: int = 16,
                 use_polarity: bool = True, use_sas: bool = True):
        super().__init__()
        self.use_polarity, self.use_sas = use_polarity, use_sas
        self.use_unc = use_polarity or use_sas
        if self.use_unc:
            self.head = AmbiguityPriorHead(dim)
        if use_polarity:
            self.gamma_raw = nn.Parameter(torch.full((1,), -2.0))
        if use_sas:
            self.sas = SparseAmbiguityScan(dim, ratio=topk_ratio, d_state=d_state)
        self.fuse = nn.Sequential(nn.Conv2d(2 * dim, dim, 1, bias=False), LayerNorm2d(dim))

    def forward(self, e: torch.Tensor, d: torch.Tensor):
        logits = None
        if self.use_unc:
            logits = self.head(d)
            p, u = ambiguity_from_logits(logits.detach())
            if self.use_polarity:
                g = (p + u).clamp(max=1.0)
                e = e * (1.0 - torch.sigmoid(self.gamma_raw) * (1.0 - g))
            if self.use_sas:
                e = e + self.sas(e + d, u)
        return self.fuse(torch.cat([e, d], dim=1)), logits


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class _Down(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.norm = LayerNorm2d(dim_in)
        self.conv = nn.Conv2d(dim_in, dim_out, 2, stride=2)

    def forward(self, x):
        return self.conv(self.norm(x))


class _Up(nn.Module):
    """2x upsampling by pixel shuffle: (B, C_in, H, W) -> (B, C_out, 2H, 2W)."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out * 4, 1, bias=False)
        self.shuffle = nn.PixelShuffle(2)
        self.norm = LayerNorm2d(dim_out)

    def forward(self, x):
        return self.norm(self.shuffle(self.proj(x)))


class UGVMUNet(nn.Module):
    """UGVM-UNet.

    Args:
        in_chans:        input channels (3 for RGB fundus images).
        num_classes:     must be 1 (binary vessel segmentation).
        patch_size:      stem stride. 4 matches VM-UNet / MVM-UNet; 2 keeps more
                         thin-vessel detail at ~4x the token count.
        dims, depths:    encoder widths and number of UGSS blocks per stage.
        dec_depths:      UGSS blocks per decoder stage, fine -> coarse
                         (len(dims) - 1 entries).
        d_state:         state dimension of the selective scan.
        drop_path_rate:  maximum stochastic depth rate.
        topk_ratio:      fraction of tokens scanned by SAS.
        detail_skip:     concatenate a full-resolution conv stem before the
                         final classifier (helps 1-pixel vessels).
        use_*:           ablation switches.

    Forward:
        training mode (or return_aux=True): returns (logits, aux) where aux is
        a list of 1-channel logit maps at several resolutions, to be deeply
        supervised (see loss.py).
        eval mode: returns logits (B, 1, H, W).

    Inputs of any size are accepted (reflect-padded to a multiple of
    `patch_size * 2**(len(dims)-1)` and cropped back). For training, use sizes
    that are already a multiple so the auxiliary maps align with the mask.
    """

    def __init__(self, in_chans: int = 3, num_classes: int = 1, patch_size: int = 4,
                 dims: Sequence[int] = (96, 192, 384, 768), depths: Sequence[int] = (2, 2, 2, 2),
                 dec_depths: Sequence[int] = (2, 2, 2), d_state: int = 16,
                 drop_path_rate: float = 0.1, topk_ratio: float = 0.25, detail_skip: bool = True,
                 use_coema: bool = True, use_write_gate: bool = True, use_read_gate: bool = True,
                 use_polarity: bool = True, use_sas: bool = True):
        super().__init__()
        assert num_classes == 1, "UGVM-UNet is defined for binary segmentation (num_classes=1)."
        assert len(depths) == len(dims) and len(dec_depths) == len(dims) - 1
        self.patch_size = patch_size
        self.num_stages = len(dims)
        self.size_multiple = patch_size * 2 ** (len(dims) - 1)

        blk = dict(d_state=d_state, use_coema=use_coema,
                   use_write_gate=use_write_gate, use_read_gate=use_read_gate)

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], patch_size, stride=patch_size), LayerNorm2d(dims[0]))

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        cur = 0
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            self.enc_stages.append(nn.ModuleList(
                [UGSSBlock(dim, drop_path=dpr[cur + j], **blk) for j in range(depth)]))
            cur += depth
            if i < len(dims) - 1:
                self.downs.append(_Down(dim, dims[i + 1]))

        # decoder stage i works at the resolution / width of encoder stage i
        self.ups = nn.ModuleList([_Up(dims[i + 1], dims[i]) for i in range(len(dims) - 1)])
        self.skips = nn.ModuleList([
            URFI(dims[i], topk_ratio=topk_ratio, d_state=d_state,
                 use_polarity=use_polarity, use_sas=use_sas) for i in range(len(dims) - 1)])
        self.dec_stages = nn.ModuleList([
            nn.ModuleList([UGSSBlock(dims[i], drop_path=0.0, **blk) for _ in range(dec_depths[i])])
            for i in range(len(dims) - 1)])

        head_dim = max(dims[0] // 4, 16)
        self.final_up = nn.Sequential(
            nn.Conv2d(dims[0], head_dim * patch_size ** 2, 1, bias=False),
            nn.PixelShuffle(patch_size))
        self.detail_skip = detail_skip
        detail_dim = 16 if detail_skip else 0
        if detail_skip:
            self.detail = nn.Sequential(
                nn.Conv2d(in_chans, detail_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(detail_dim), nn.ReLU(inplace=True),
                nn.Conv2d(detail_dim, detail_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(detail_dim), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Conv2d(head_dim + detail_dim, head_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_dim), nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, num_classes, 1))

    # ------------------------------------------------------------------ #
    def no_weight_decay_keywords(self):
        """Substrings of parameter names that should not receive weight decay."""
        return {"A_logs", "Ds", "alpha_raw", "beta", "gamma_raw", "norm", "bias"}

    @torch.no_grad()
    def gate_values(self):
        """Current gate strengths, for logging: {name: value}."""
        out = {}
        for name, m in self.named_modules():
            if isinstance(m, UGSSBlock):
                if m.use_write_gate:
                    out[f"{name}.alpha"] = torch.tanh(m.alpha_raw).item()
                if m.use_read_gate:
                    out[f"{name}.beta"] = m.beta.item()
            elif isinstance(m, URFI) and m.use_polarity:
                out[f"{name}.gamma"] = torch.sigmoid(m.gamma_raw).item()
        return out

    def _pad(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        m = self.size_multiple
        ph, pw = (m - h % m) % m, (m - w % m) % m
        if ph == 0 and pw == 0:
            return x, (h, w)
        mode = "reflect" if ph < h and pw < w else "replicate"
        return F.pad(x, (0, pw, 0, ph), mode=mode), (h, w)

    def forward(self, x: torch.Tensor, return_aux: Optional[bool] = None):
        if return_aux is None:
            return_aux = self.training
        x, (h0, w0) = self._pad(x)
        aux: List[torch.Tensor] = []

        def run(blocks, t):
            for b in blocks:
                t, lg = b(t)
                if lg is not None:
                    aux.append(lg)
            return t

        feats = []
        t = self.patch_embed(x)
        for i, stage in enumerate(self.enc_stages):
            t = run(stage, t)
            feats.append(t)
            if i < self.num_stages - 1:
                t = self.downs[i](t)

        t = feats[-1]
        for i in reversed(range(self.num_stages - 1)):
            t = self.ups[i](t)
            t, lg = self.skips[i](feats[i], t)
            if lg is not None:
                aux.append(lg)
            t = run(self.dec_stages[i], t)

        t = self.final_up(t)
        if self.detail_skip:
            t = torch.cat([t, self.detail(x)], dim=1)
        out = self.head(t)[..., :h0, :w0]
        return (out, aux) if return_aux else out


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def ugvm_unet_tiny(**kw):
    """Small model for quick experiments and limited GPU memory."""
    kw.setdefault("dims", (48, 96, 192, 384))
    kw.setdefault("depths", (1, 1, 2, 1))
    kw.setdefault("dec_depths", (1, 1, 1))
    return UGVMUNet(**kw)


def ugvm_unet_small(**kw):
    kw.setdefault("dims", (64, 128, 256, 512))
    kw.setdefault("depths", (2, 2, 2, 2))
    kw.setdefault("dec_depths", (1, 1, 1))
    return UGVMUNet(**kw)


def ugvm_unet_base(**kw):
    """Same widths / depths as VM-UNet, for a like-for-like comparison."""
    return UGVMUNet(**kw)


if __name__ == "__main__":
    model = ugvm_unet_tiny()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.randn(2, 3, 64, 64)
    model.train()
    out, aux = model(x)
    print(f"params: {n:.2f} M | out: {tuple(out.shape)} | aux maps: {len(aux)}")
