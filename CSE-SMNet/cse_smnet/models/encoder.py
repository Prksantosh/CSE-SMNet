import torch.nn as nn

from .spatial.cse_block import CSEBlock


# ============================================================
# CGCDE Encoder Stage
# ============================================================
class EncoderStage(nn.Module):
    """
    Encoder stage:

        Input
          |
          v
        CGCDE
          |
          v
        Downsample Conv
          |
          v
        Output

    CGCDE performs:
        1. Multi-receptive regular structural extraction
        2. Regular-guided deformable feature extraction
        3. Geometric-residual complementarity enhancement
        4. Competitive spatial-channel dual-branch fusion

    Input:
        (B, in_ch, H, W)

    Output:
        (B, out_ch, H/2, W/2)
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        reduction=8,
        deform_kernel=3,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Novel spatial feature enhancement
        # ----------------------------------------------------
        self.cgcde = CSEBlock(
            channels=in_ch,
            reduction=reduction,
            deform_kernel=deform_kernel,
        )

        # ----------------------------------------------------
        # Lightweight spatial downsampling
        # ----------------------------------------------------
        self.down = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_ch
            ),

            nn.SiLU(
                inplace=True
            ),
        )

    @property
    def cse(self):
        """Public CSE-SMNet name; legacy `cgcde` key is retained for checkpoint compatibility."""
        return self.cgcde

    def forward(
        self,
        x,
        return_aux=False,
    ):

        # ----------------------------------------------------
        # CGCDE spatial enhancement
        # ----------------------------------------------------
        if return_aux:

            x, aux = self.cse(
                x,
                return_aux=True,
            )

        else:

            x = self.cse(x)

        # ----------------------------------------------------
        # Downsample
        # ----------------------------------------------------
        x = self.down(x)

        if return_aux:
            return x, aux

        return x