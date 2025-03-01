import sys
sys.path.append("..")

import torch
import numpy as np
from svglib.svg import SVG
from optim_utils.util import get_dominant_color
import cv2
from PIL import ImageChops
from svglib.svg_primitive import SVGPath
from scipy.ndimage import label
from skimage.measure import regionprops
from svglib.svg import SVG


def get_cubic_segments_from_points(points):
    total_points = points.shape[0]
    seg_num = total_points // 3

    cubics = points.view(seg_num, 3, 2)
    next_points = torch.roll(points, -3, dims=0)[:seg_num*3:3]
    cubics = torch.cat([cubics, next_points.unsqueeze(1)], dim=1)

    return cubics.view(-1, 4, 2)


def sample_bezier(cubics, k=5):
    """
    Sample points on cubic Bezier curves.
    :param cubics: torch.Tensor, shape [num_curves, 4, 2], representing cubic Bezier curves.
    :param k: int, number of sample points per curve.
    :return: torch.Tensor, shape [num_curves * k, 2], representing the sampled points on the Bezier curves.
    """
    # shape [1, k, 1]
    ts = torch.linspace(0, 1, k).view(1, k, 1).to(cubics.device)

    P0, P1, P2, P3 = cubics[:, 0], cubics[:, 1], cubics[:, 2], cubics[:, 3]

    # Calculate cubic Bezier for all curves and all t values at once
    point = (1-ts)**3 * P0.unsqueeze(1) + 3*(1-ts)**2*ts * P1.unsqueeze(1) + \
        3*(1-ts)*ts**2 * P2.unsqueeze(1) + ts**3 * P3.unsqueeze(1)

    # Reshape the tensor to get points in [num_curves * k, 2] format
    point = point.reshape(-1, 2)

    # shape [num_curves * k, 2]
    return point


def is_clockwise(p):
    start, end = p[:-1], p[1:]
    return torch.stack([start, end], dim=-1).det().sum() > 0


def make_clockwise(p):
    if not is_clockwise(p):
        return p.flip(dims=[0])
    return p


def sample_points_by_length_distribution(p_target, n, device="cuda"):
    """
    Compute a subset of target points based on length distribution.

    Args:
        p_target (torch.Tensor): Target points, shape [num_points, 2].
        n (int): Number of points to sample.
        device (str): Device to use for computations.

    Returns:
        tuple: (p_target_sub, matching)
            p_target_sub (torch.Tensor): Subset of target points, shape [n, 2].
            matching (torch.Tensor): Indices of matched points, shape [n].
    """
    assert n > 0, "n must be positive"

    # Assume p_target is already clockwise
    p_target_clockwise = p_target

    # Create evenly spaced distribution for predicted points
    distr_pred = torch.linspace(0., 1., n, device=device)

    # Compute cumulative length distribution of target points
    distr_target = get_length_distribution(p_target_clockwise, normalize=True)

    # Find closest target point for each predicted point
    distances = torch.cdist(distr_pred.unsqueeze(-1), distr_target.unsqueeze(-1))
    matching = distances.argmin(dim=-1)

    # Select subset of target points based on matching
    p_target_sub = p_target_clockwise[matching]

    return p_target_sub, matching


def get_length_distribution(p, normalize=True):
    start, end = p[:-1], p[1:]
    length_distr = torch.norm(end - start, dim=-1).cumsum(dim=0)
    length_distr = torch.cat([length_distr.new_zeros(1),
                              length_distr])

    if normalize:
        length_distr = length_distr / length_distr[-1]

    return length_distr


def align_path_color_to_image(svg: SVG, target_image, logger):
    target_array = np.array(target_image)

    for i, path_group in enumerate(svg.svg_path_groups):
        mask = create_path_mask(svg, path_group, i, target_array.shape[:2])
        
        pixels = target_array[mask.astype(bool)]
        if pixels.size == 0:
            logger.info(f"[Align Color] Path {i} is completely blocked by other paths, skip it.")
            continue
        
        color = get_dominant_color(pixels)
        path_group.color = '#{:02x}{:02x}{:02x}'.format(*color)

    return svg


def create_path_mask(svg, path_group, path_index, target_shape):
    single_path_svg = SVG([path_group], viewbox=svg.viewbox)
    single_path_png = single_path_svg.draw(do_display=False, return_png=True, background_color=None, coordinate_precision=3)
    mask = np.array(single_path_png)
    
    if mask.shape[2] == 3:  # the path occupies entire image (e.g., background)
        mask = np.ones(target_shape, dtype=np.uint8)
        mask = remove_blocked_regions(svg, mask, path_index, after=False)
    else:
        mask = mask[:, :, 3]
        mask = remove_blocked_regions(svg, mask, path_index, after=True)
    
    return mask

def remove_blocked_regions(svg, mask, path_index, after=True):
    for j, other_path_group in enumerate(svg.svg_path_groups):
        if (after and j > path_index) or (not after and j != path_index):
            other_path_png = SVG([other_path_group], viewbox=svg.viewbox).draw(do_display=False, return_png=True, background_color=None)
            other_mask = np.array(other_path_png)
            if other_mask.shape[2] == 4:
                mask[other_mask[:,:,3] > 0] = 0
    return mask


def region_growing(image, seed, threshold, processed_mask):
    height, width = image.shape[:2]
    segmented = np.zeros((height, width), dtype=np.uint8)
    stack = [seed]
    region_pixels = []
    region_color_sum = np.zeros(3, dtype=np.float32)
    
    while stack:
        x, y = stack.pop()
        if segmented[y, x] == 0 and not processed_mask[y, x]:
            segmented[y, x] = 255
            processed_mask[y, x] = True
            
            current_color = image[y, x].astype(np.float32)
            region_pixels.append((x, y))
            region_color_sum += current_color
            mean_color = region_color_sum / len(region_pixels)
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor_color = image[ny, nx].astype(np.float32)
                    # Calculate Euclidean distance between neighbor color and region mean color
                    color_distance = np.linalg.norm(neighbor_color - mean_color)
                    if color_distance < threshold:
                        stack.append((nx, ny))
    return segmented


def resample_contour(contour, n_points=20):
    # Calculate the total length of the contour
    length = cv2.arcLength(contour, True)
    
    # Calculate the length of each segment
    segment_length = length / n_points
    
    resampled_points = []
    accumulated_length = 0
    for i in range(len(contour)):
        if i == 0:
            resampled_points.append(contour[i])
        else:
            accumulated_length += cv2.norm(contour[i] - contour[i-1])
            while accumulated_length >= segment_length:
                # Interpolate to find the point
                t = (accumulated_length - segment_length) / cv2.norm(contour[i] - contour[i-1])
                point = contour[i-1] + t * (contour[i] - contour[i-1])
                resampled_points.append(point)
                accumulated_length -= segment_length
    
    # Ensure we have exactly n_points
    while len(resampled_points) < n_points:
        resampled_points.append(contour[-1])
    
    return np.array(resampled_points, dtype=np.float32)
