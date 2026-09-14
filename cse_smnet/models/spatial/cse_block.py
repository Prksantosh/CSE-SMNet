import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class ConvBNAct(nn.Module):
    """
    Conv2d -> BatchNorm2d -> SiLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        bias=False,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# 1. Multi-Receptive Regular Structural Branch
# ============================================================

class MultiReceptiveRegularBranch(nn.Module):
    """
    Lightweight regular-convolution branch.

    Two depthwise receptive fields:
        3x3 -> fine local structure
        5x5 -> wider structural context

    The two responses are adaptively combined using a channel-wise
    two-branch softmax gate.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 8)

        # --------------------------------------------------------
        # Fine receptive-field branch
        # --------------------------------------------------------
        self.dw3 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # --------------------------------------------------------
        # Larger receptive-field branch
        # --------------------------------------------------------
        self.dw5 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # --------------------------------------------------------
        # Shared channel projection
        # --------------------------------------------------------
        self.project = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # --------------------------------------------------------
        # Adaptive receptive-field selector
        # Produces two channel-wise branch logits
        # --------------------------------------------------------
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.selector = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden,
                channels * 2,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, x):

        f3 = self.dw3(x)
        f5 = self.dw5(x)

        # Joint structural descriptor
        descriptor = self.pool(f3 + f5)

        weights = self.selector(descriptor)

        b, _, _, _ = weights.shape

        weights = weights.view(
            b,
            2,
            x.shape[1],
            1,
            1,
        )

        # Competitive weighting of 3x3 and 5x5 responses
        weights = torch.softmax(
            weights,
            dim=1,
        )

        w3 = weights[:, 0]
        w5 = weights[:, 1]

        out = (
            w3 * f3
            +
            w5 * f5
        )

        out = self.project(out)

        return out


# ============================================================
# 2. Regular-Guided Modulated Deformable Branch
# ============================================================

class GuidedModulatedDeformableBranch(nn.Module):
    """
    Deformable branch whose sampling offsets and modulation masks
    are guided by the regular structural representation.

    Offset / mask predictor input:
        concat(original feature, regular structural feature)

    This implements:

        [DeltaP, M] = O([X || F_r])

        F_d = DCNv2(X; DeltaP, M)
    """

    def __init__(
        self,
        channels,
        kernel_size=3,
    ):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size

        padding = kernel_size // 2

        offset_channels = (
            2
            * kernel_size
            * kernel_size
        )

        mask_channels = (
            kernel_size
            * kernel_size
        )

        # --------------------------------------------------------
        # Structural-guided offset/mask predictor
        # --------------------------------------------------------
        self.guidance = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        self.offset_mask = nn.Conv2d(
            channels,
            offset_channels + mask_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # Start close to normal convolution
        nn.init.zeros_(
            self.offset_mask.weight
        )
        nn.init.zeros_(
            self.offset_mask.bias
        )

        # --------------------------------------------------------
        # Modulated deformable convolution
        # --------------------------------------------------------
        self.deform = DeformConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.norm = nn.BatchNorm2d(
            channels
        )

        self.act = nn.SiLU(
            inplace=True
        )

    def forward(self, x, regular_feature):

        guided_feature = torch.cat(
            [
                x,
                regular_feature,
            ],
            dim=1,
        )

        guided_feature = self.guidance(
            guided_feature
        )

        offset_mask = self.offset_mask(
            guided_feature
        )

        k2 = (
            self.kernel_size
            * self.kernel_size
        )

        offset_channels = 2 * k2

        offset = offset_mask[
            :,
            :offset_channels,
            :,
            :
        ]

        mask = offset_mask[
            :,
            offset_channels:,
            :,
            :
        ]

        # Modulation coefficients in [0,1]
        mask = torch.sigmoid(mask)

        deform_feature = self.deform(
            x,
            offset,
            mask,
        )

        deform_feature = self.norm(
            deform_feature
        )

        deform_feature = self.act(
            deform_feature
        )

        return deform_feature


# ============================================================
# 3. Geometric Residual Complementarity Module
# ============================================================

class GeometricResidualComplementarity(nn.Module):
    """
    Explicitly models the additional information contributed by
    deformable processing relative to regular processing.

        R_g = F_d - F_r

        G_c = sigmoid(phi(|R_g|))

        F_d^c = F_r + G_c * R_g

    Thus the deformable branch contributes primarily where its
    representation differs meaningfully from the regular branch.
    """

    def __init__(
        self,
        channels,
        reduction=8,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            8,
        )

        self.gate = nn.Sequential(

            # ----------------------------------------------------
            # Spatial/channel interaction
            # ----------------------------------------------------
            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=True,
            ),

            nn.Sigmoid(),
        )

    def forward(
        self,
        regular_feature,
        deform_feature,
    ):

        geometric_residual = (
            deform_feature
            -
            regular_feature
        )

        disagreement = torch.abs(
            geometric_residual
        )

        gate = self.gate(
            disagreement
        )

        complementary_feature = (
            regular_feature
            +
            gate * geometric_residual
        )

        return (
            complementary_feature,
            geometric_residual,
            gate,
        )


# ============================================================
# 4. Competitive Spatial-Channel Dual-Branch Fusion
# ============================================================

class CompetitiveDualBranchFusion(nn.Module):
    """
    Produces complementary regular/deformable importance weights.

    The fusion weight considers:

        regular feature
        complementary deformable feature
        branch disagreement |F_d - F_r|

    Two forms of evidence are used:

        1. channel/global branch preference
        2. spatial branch preference

    Final weights are normalized across the two branches:

        A_r + A_d = 1

    and

        F = A_r * F_r + A_d * F_d^c
    """

    def __init__(
        self,
        channels,
        reduction=8,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            8,
        )

        # Input:
        # Fr || Fd_complementary || |Fd-Fr|
        fusion_channels = (
            channels * 3
        )

        # --------------------------------------------------------
        # Channel/global branch selector
        #
        # Produces:
        # B x (2C) x 1 x 1
        # --------------------------------------------------------
        self.global_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.channel_selector = nn.Sequential(
            nn.Conv2d(
                fusion_channels,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                hidden,
                channels * 2,
                kernel_size=1,
                bias=True,
            ),
        )

        # --------------------------------------------------------
        # Spatial branch selector
        #
        # Produces two spatial maps:
        # B x 2 x H x W
        # --------------------------------------------------------
        self.spatial_selector = nn.Sequential(

            nn.Conv2d(
                fusion_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(
                channels,
                2,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

    def forward(
        self,
        regular_feature,
        deform_feature,
        geometric_residual,
    ):

        disagreement = torch.abs(
            geometric_residual
        )

        fusion_input = torch.cat(
            [
                regular_feature,
                deform_feature,
                disagreement,
            ],
            dim=1,
        )

        b, c, h, w = (
            regular_feature.shape
        )

        # ========================================================
        # Channel-wise competitive evidence
        # ========================================================

        channel_descriptor = (
            self.global_pool(
                fusion_input
            )
        )

        channel_logits = (
            self.channel_selector(
                channel_descriptor
            )
        )

        channel_logits = (
            channel_logits.view(
                b,
                2,
                c,
                1,
                1,
            )
        )

        channel_weights = (
            torch.softmax(
                channel_logits,
                dim=1,
            )
        )

        # ========================================================
        # Spatial competitive evidence
        # ========================================================

        spatial_logits = (
            self.spatial_selector(
                fusion_input
            )
        )

        spatial_weights = (
            torch.softmax(
                spatial_logits,
                dim=1,
            )
        )

        spatial_weights = (
            spatial_weights
            .unsqueeze(2)
        )

        # B x 2 x C x H x W
        combined_weights = (
            channel_weights
            *
            spatial_weights
        )

        # Re-normalize across branches
        combined_weights = (
            combined_weights
            /
            (
                combined_weights.sum(
                    dim=1,
                    keepdim=True,
                )
                + 1e-8
            )
        )

        regular_weight = (
            combined_weights[:, 0]
        )

        deform_weight = (
            combined_weights[:, 1]
        )

        fused = (
            regular_weight
            * regular_feature
            +
            deform_weight
            * deform_feature
        )

        return (
            fused,
            regular_weight,
            deform_weight,
        )


# ============================================================
# 5. CSE Block
# ============================================================

class CSEBlock(nn.Module):
    def __init__(
        self,
        channels,
        reduction=8,
        deform_kernel=3,
        residual_scale_init=0.1,
    ):
        super().__init__()

        # --------------------------------------------------------
        # Branch A:
        # Multi-receptive regular structural branch
        # --------------------------------------------------------
        self.regular_branch = (
            MultiReceptiveRegularBranch(
                channels=channels,
                reduction=reduction,
            )
        )

        # --------------------------------------------------------
        # Branch B:
        # Regular-guided modulated deformable branch
        # --------------------------------------------------------
        self.deform_branch = (
            GuidedModulatedDeformableBranch(
                channels=channels,
                kernel_size=deform_kernel,
            )
        )

        # --------------------------------------------------------
        # Explicit cross-branch complementarity
        # --------------------------------------------------------
        self.complementarity = (
            GeometricResidualComplementarity(
                channels=channels,
                reduction=reduction,
            )
        )

        # --------------------------------------------------------
        # Competitive adaptive fusion
        # --------------------------------------------------------
        self.fusion = (
            CompetitiveDualBranchFusion(
                channels=channels,
                reduction=reduction,
            )
        )

        # --------------------------------------------------------
        # Lightweight output refinement
        # --------------------------------------------------------
        self.output_projection = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        # Learnable residual scaling.
        #
        # Small initialization allows the encoder block to start
        # close to identity and progressively learn enhancement.
        self.residual_scale = nn.Parameter(
            torch.tensor(
                float(residual_scale_init)
            )
        )

        self.out_act = nn.SiLU(
            inplace=True
        )

        # --------------------------------------------------------
        # Last-forward diagnostic tensors
        # --------------------------------------------------------
        self.last_regular_weight = None
        self.last_deform_weight = None
        self.last_complementarity_gate = None

    def forward(
        self,
        x,
        return_aux=False,
    ):

        identity = x

        # ========================================================
        # 1. Regular structural representation
        # ========================================================

        regular_feature = (
            self.regular_branch(x)
        )

        # ========================================================
        # 2. Regular-guided deformable representation
        # ========================================================

        deform_feature = (
            self.deform_branch(
                x,
                regular_feature,
            )
        )

        # ========================================================
        # 3. Complementarity enhancement
        # ========================================================

        (
            deform_complementary,
            geometric_residual,
            complementarity_gate,
        ) = self.complementarity(
            regular_feature,
            deform_feature,
        )

        # ========================================================
        # 4. Competitive adaptive fusion
        # ========================================================

        (
            fused,
            regular_weight,
            deform_weight,
        ) = self.fusion(
            regular_feature,
            deform_complementary,
            geometric_residual,
        )

        fused = self.output_projection(
            fused
        )

        # ========================================================
        # 5. Residual output
        # ========================================================

        out = (
            identity
            +
            self.residual_scale
            * fused
        )

        out = self.out_act(
            out
        )

        # Save detached diagnostics only.
        self.last_regular_weight = (
            regular_weight.detach()
        )

        self.last_deform_weight = (
            deform_weight.detach()
        )

        self.last_complementarity_gate = (
            complementarity_gate.detach()
        )

        if not return_aux:
            return out

        aux = {
            "regular_feature": regular_feature,
            "deform_feature": deform_feature,
            "deform_complementary": deform_complementary,
            "geometric_residual": geometric_residual,
            "complementarity_gate": complementarity_gate,
            "regular_weight": regular_weight,
            "deform_weight": deform_weight,
        }

        return out, aux
