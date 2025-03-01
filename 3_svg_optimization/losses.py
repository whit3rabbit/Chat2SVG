import torch
from optim_utils.util_svg import sample_points_by_length_distribution


def laplacian_smoothing_loss(points, num_neighbors=1, weight=1.0):
    n_points = points.size(0)

    avg_neighbors = torch.zeros_like(points)

    for i in range(-num_neighbors, num_neighbors + 1):
        if i == 0:
            continue
        index_shift = (torch.arange(n_points) - i) % n_points
        avg_neighbors += points[index_shift]
    avg_neighbors /= (2 * num_neighbors)

    diff = points - avg_neighbors

    smoothness = torch.norm(diff, p=2)

    return weight * smoothness


def kl_divergence(src_z):
    src_mean = torch.mean(src_z, dim=-1)
    src_std = torch.std(src_z, dim=-1)
    kl_div = 0.5 * torch.sum(src_std**2 + src_mean**2 - 1 - torch.log(src_std**2), dim=-1)
    return kl_div.mean()


def svg_emd_loss(p_pred, p_target, p_target_sub=None, matching=None):

    n, m = len(p_pred), len(p_target)

    if n == 0:
        return 0.

    if p_target_sub is None or matching is None:
        p_target_sub, matching = sample_points_by_length_distribution(
            p_target, n, device=p_pred.device)

    # EMD - Vectorized reordering computation
    indices = torch.arange(n, device=p_pred.device).unsqueeze(0).repeat(n, 1)
    roll_indices = (indices + indices.T) % n
    reordered_ptarget_subs = torch.index_select(
        p_target_sub, 0, roll_indices.view(-1)).view(n, n, -1)

    # roll_indices.shape:  torch.Size([80, 80])
    # reordered_ptarget_subs.shape:  torch.Size([80, 80, 2])

    distances = torch.norm(p_pred.unsqueeze(
        0) - reordered_ptarget_subs, dim=-1)
    # distances.shape:  torch.Size([80, 80])

    mean_distances = distances.mean(dim=-1)

    i = torch.argmin(mean_distances)

    p_target_sub_reordered = reordered_ptarget_subs[i]
    # p_target_sub_reordered.shape:  torch.Size([80, 2])

    losses = torch.norm(p_pred - p_target_sub_reordered, dim=-1)

    return losses.mean()


def _iou(pred, target, eps=1e-6):
    Iand1 = torch.sum(target * pred, dim=(1, 2, 3))  # Sum over height, width, and channels
    Ior1 = torch.sum(target, dim=(1, 2, 3)) + torch.sum(pred, dim=(1, 2, 3)) - Iand1

    # Compute IoU for each element in the batch
    IoU1 = Iand1 / (Ior1 + eps)

    # IoU loss for the whole batch
    IoU_loss = torch.mean(1 - IoU1)

    return IoU_loss


class IoULoss(torch.nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target):
        return _iou(pred, target)


def curvature_loss(paths):
    loss = 0
    total_points = 0
    for path in paths:
        points = path.points  # Assuming points is a tensor of shape (N, 2)
        if len(points) < 3:
            continue
        # Calculate second differences
        second_differences = points[:-2] - 2 * points[1:-1] + points[2:]
        # Compute squared norms of the second differences
        squared_norms = torch.sum(second_differences ** 2, dim=1)
        loss += torch.sum(squared_norms)
        total_points += len(squared_norms)
    return loss / max(total_points, 1)  # Avoid division by zero
