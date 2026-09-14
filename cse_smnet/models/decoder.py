import torch.nn as nn


# ============================================================
# Lightweight Decoder Stage
# ============================================================
class DecoderStage(nn.Module):
    """
    Lightweight decoder stage.

    Pipeline:
        1. Transposed convolution for x2 upsampling
        2. Depthwise convolution for local refinement
        3. Pointwise convolution for channel mixing

    Input:
        (B, in_ch, H, W)

    Output:
        (B, out_ch, 2H, 2W)
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        use_refinement=True,
    ):
        super().__init__()

        # ----------------------------------------------------
        # 1. Learnable spatial upsampling
        # ----------------------------------------------------
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

        # ----------------------------------------------------
        # 2. Lightweight local refinement
        # ----------------------------------------------------
        if use_refinement:

            self.refine = nn.Sequential(

                # Depthwise spatial filtering
                nn.Conv2d(
                    out_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1,
                    groups=out_ch,
                    bias=False,
                ),

                nn.BatchNorm2d(out_ch),

                nn.SiLU(inplace=True),

                # Pointwise channel interaction
                nn.Conv2d(
                    out_ch,
                    out_ch,
                    kernel_size=1,
                    bias=False,
                ),

                nn.BatchNorm2d(out_ch),

                nn.SiLU(inplace=True),
            )

        else:
            self.refine = nn.Identity()

    def forward(self, x):

        x = self.up(x)

        x = self.refine(x)

        return x