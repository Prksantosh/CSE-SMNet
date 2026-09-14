import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionMSELoss(nn.Module):
    """
    Standard pixel-wise MSE reconstruction loss.
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        return self.mse(pred, target)


class SSIMLoss(nn.Module):
    """
    Structural similarity loss.
    Assumes pred/target are already in the same normalized range.
    """
    def __init__(self, window_size=3, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2

    def forward(self, x, y):
        mu_x = F.avg_pool2d(x, self.window_size, 1, self.window_size // 2)
        mu_y = F.avg_pool2d(y, self.window_size, 1, self.window_size // 2)

        sigma_x = F.avg_pool2d(x * x, self.window_size, 1, self.window_size // 2) - mu_x ** 2
        sigma_y = F.avg_pool2d(y * y, self.window_size, 1, self.window_size // 2) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, self.window_size, 1, self.window_size // 2) - mu_x * mu_y

        numerator = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        ssim_map = numerator / (denominator + 1e-8)
        loss = torch.clamp((1.0 - ssim_map) / 2.0, 0.0, 1.0)

        return loss.mean()


class TemporalLoss(nn.Module):
    """
    Temporal consistency loss using motion difference.
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(self, pred, prev_frame, gt):
        pred_motion = pred - prev_frame
        gt_motion = gt - prev_frame
        return self.l1(pred_motion, gt_motion)


class GradientLoss(nn.Module):
    """
    Edge-aware gradient loss to reduce blur.
    Compares horizontal and vertical image gradients.
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        loss_x = torch.abs(pred_dx - target_dx)
        loss_y = torch.abs(pred_dy - target_dy)

        if self.reduction == "mean":
            return loss_x.mean() + loss_y.mean()
        elif self.reduction == "sum":
            return loss_x.sum() + loss_y.sum()
        else:
            return loss_x + loss_y


class CombinedPredictionLoss(nn.Module):
    """
    Combined loss for next-frame prediction.

    Total loss:
        L = lambda_mse * MSE
          + lambda_ssim * SSIM
          + lambda_temp * Temporal
          + lambda_grad * Gradient
    """
    def __init__(
        self,
        lambda_mse=0.60,
        lambda_ssim=0.20,
        lambda_temp=0.10,
        lambda_grad=0.10
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_temp = lambda_temp
        self.lambda_grad = lambda_grad

        self.mse_loss = ReconstructionMSELoss()
        self.ssim_loss = SSIMLoss()
        self.temp_loss = TemporalLoss()
        self.grad_loss = GradientLoss()

    def forward(self, pred, target, prev_frame):
        loss_mse = self.mse_loss(pred, target)
        loss_ssim = self.ssim_loss(pred, target)
        loss_temp = self.temp_loss(pred, prev_frame, target)
        loss_grad = self.grad_loss(pred, target)

        total_loss = (
            self.lambda_mse * loss_mse +
            self.lambda_ssim * loss_ssim +
            self.lambda_temp * loss_temp +
            self.lambda_grad * loss_grad
        )

        return {
            "total_loss": total_loss,
            "mse_loss": loss_mse,
            "ssim_loss": loss_ssim,
            "temp_loss": loss_temp,
            "grad_loss": loss_grad,
        }