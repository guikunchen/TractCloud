"""TractCloud inference pipeline: feature extraction, KNN, and model inference.

Adapted from https://github.com/SlicerDMRI/TractCloud
"""

import json
import logging

import numpy as np
import vtk
import torch
import torch.utils.data as data

from .models import TractDGCNN, PointNetCls

logger = logging.getLogger(__name__)


def extract_ras_features(polydata, num_points=15):
    """Extract RAS coordinate features from VTK polydata.

    Resamples each streamline to equally-spaced points.

    Args:
        polydata: vtkPolyData with lines (streamlines)
        num_points: points per streamline

    Returns:
        (num_fibers, num_points, 3) float64 array
    """
    from vtk.util.numpy_support import vtk_to_numpy

    num_fibers = polydata.GetNumberOfLines()
    all_points = vtk_to_numpy(polydata.GetPoints().GetData())

    cell_array = polydata.GetLines()
    if (hasattr(cell_array, 'GetOffsetsArray')
            and cell_array.GetOffsetsArray()):
        cell_offsets = vtk_to_numpy(cell_array.GetOffsetsArray())
        lengths = np.diff(cell_offsets)
        connectivity = vtk_to_numpy(cell_array.GetConnectivityArray())
    else:
        lengths = np.zeros(num_fibers, dtype=np.int64)
        cell_offsets = np.zeros(num_fibers + 1, dtype=np.int64)
        cell_array.InitTraversal()
        pt_ids = vtk.vtkIdList()
        conn_list = []
        offset = 0
        for i in range(num_fibers):
            cell_array.GetNextCell(pt_ids)
            n = pt_ids.GetNumberOfIds()
            lengths[i] = n
            cell_offsets[i] = offset
            for j in range(n):
                conn_list.append(pt_ids.GetId(j))
            offset += n
        cell_offsets[num_fibers] = offset
        connectivity = np.array(conn_list, dtype=np.int64)

    feat = np.zeros((num_fibers, num_points, 3), dtype=np.float64)
    unique_lengths = np.unique(lengths)

    for n_pts in unique_lengths:
        if n_pts < 2:
            continue
        fiber_indices = np.where(lengths == n_pts)[0]
        batch_size = len(fiber_indices)

        coords = np.zeros((batch_size, n_pts, 3), dtype=np.float64)
        for bi, fi in enumerate(fiber_indices):
            start = cell_offsets[fi]
            pt_indices = connectivity[start:start + n_pts]
            coords[bi] = all_points[pt_indices]

        diffs = np.diff(coords, axis=1)
        seg_lens = np.sqrt(np.sum(diffs ** 2, axis=2))
        cum_len = np.zeros((batch_size, n_pts), dtype=np.float64)
        cum_len[:, 1:] = np.cumsum(seg_lens, axis=1)
        total_len = cum_len[:, -1]

        degenerate = total_len < 1e-12
        if np.any(degenerate):
            feat[fiber_indices[degenerate], :, :] = coords[degenerate, 0:1, :]

        valid = ~degenerate
        if not np.any(valid):
            continue

        valid_coords = coords[valid]
        valid_cum_len = cum_len[valid]
        valid_total_len = total_len[valid]
        valid_fiber_idx = fiber_indices[valid]
        n_valid = len(valid_fiber_idx)

        t = (np.linspace(0, 1, num_points)[None, :]
             * valid_total_len[:, None])

        n_source_pts = valid_coords.shape[1]
        idx = np.empty_like(t, dtype=np.intp)
        for bi in range(n_valid):
            idx[bi] = np.searchsorted(
                valid_cum_len[bi], t[bi], side='right') - 1
        idx = np.clip(idx, 0, n_source_pts - 2)

        row_idx = np.arange(n_valid)[:, None]
        cum_lo = valid_cum_len[row_idx, idx]
        cum_hi = valid_cum_len[row_idx, idx + 1]
        seg_len = cum_hi - cum_lo
        seg_len = np.where(seg_len < 1e-12, 1.0, seg_len)
        frac = ((t - cum_lo) / seg_len)[:, :, None]

        coords_lo = np.take_along_axis(
            valid_coords, idx[:, :, None].repeat(3, axis=2), axis=1)
        coords_hi = np.take_along_axis(
            valid_coords, (idx + 1)[:, :, None].repeat(3, axis=2), axis=1)

        resampled = coords_lo + frac * (coords_hi - coords_lo)
        feat[valid_fiber_idx] = resampled

    return feat


def center_tractography(feat, mass_center):
    """Re-center tractography to match HCP atlas center."""
    subject_center = np.mean(feat, axis=0)
    return feat + (mass_center - subject_center)


def _fiber_distance_efficient(set1, set2, num_points=15):
    """Pairwise distance between streamline sets (quadratic expansion)."""
    s1 = set1.reshape(set1.shape[0], -1)
    s2 = set2.reshape(set2.shape[0], -1)
    s1_sq = (s1 ** 2).sum(1).view(-1, 1)
    s2_t = s2.t()
    s2_sq = (s2 ** 2).sum(1).view(1, -1)
    dist = s1_sq + s2_sq - 2.0 * torch.mm(s1, s2_t)
    dist = torch.sqrt(torch.clamp(dist, 0.0, float("inf")))
    return dist / num_points


def _compute_local_features(feat, k, k_ds_rate=0.1):
    """KNN neighbor features (forward distance only, no flip)."""
    if 0 < k_ds_rate < 1:
        num_ds = int(feat.shape[0] * k_ds_rate)
        ds_indices = np.random.choice(
            feat.shape[0], size=num_ds, replace=False)
        ds_feat = feat[ds_indices, :, :]
    else:
        ds_feat = feat
    dist_mat = _fiber_distance_efficient(feat, ds_feat)
    topk_idx = dist_mat.topk(k=k, largest=False, dim=-1)[1]
    return ds_feat[topk_idx.reshape(-1), ...]


class RealDataDataset(data.Dataset):
    """Dataset for inference on real tractography."""

    def __init__(self, feat, k=20, k_global=80, k_ds_rate=0.1,
                 rough_num_fiber_per_iter=10000, progress_callback=None):
        self.feat = feat.astype(np.float32)
        self.k = k
        self.k_global = k_global
        num_fiber, num_point, num_feat = self.feat.shape

        if self.k_global == 0:
            self.global_feat = np.zeros(
                (1, num_point, num_feat, 1), dtype=np.float32)
        else:
            rand_idx = np.random.randint(0, num_fiber, self.k_global)
            self.global_feat = (
                self.feat[rand_idx]
                .transpose(1, 2, 0)[None, :, :, :]
                .astype(np.float32))

        if self.k == 0:
            self.local_feat = np.zeros(
                (num_fiber, num_point, num_feat, 1), dtype=np.float32)
        else:
            self.local_feat = np.zeros(
                (num_fiber, num_point, num_feat, self.k), dtype=np.float32)
            num_iter = max(num_fiber // rough_num_fiber_per_iter, 1)
            num_per_iter = (num_fiber // num_iter) + 1
            for i_iter in range(num_iter):
                start = i_iter * num_per_iter
                end = min((i_iter + 1) * num_per_iter, num_fiber)
                cur = np.transpose(self.feat[start:end], (0, 2, 1))
                cur_local = _compute_local_features(
                    torch.from_numpy(cur), self.k, k_ds_rate).numpy()
                cur_local = cur_local.reshape(
                    end - start, self.k, num_feat, num_point)
                cur_local = np.transpose(cur_local, (0, 3, 2, 1))
                self.local_feat[start:end] = cur_local
                if progress_callback:
                    progress_callback((i_iter + 1) / num_iter)

    def __getitem__(self, index):
        return (torch.from_numpy(self.feat[index]),
                torch.from_numpy(self.local_feat[index]))

    def __len__(self):
        return self.feat.shape[0]


def load_model(weight_path, args_path, device,
               k_override=None, k_global_override=None):
    """Load a TractCloud model from saved weights."""
    with open(args_path, "r") as f:
        args_dict = json.load(f)
    model_name = args_dict.get("model_name", "dgcnn")
    num_classes = args_dict.get("num_classes", 1600)
    k = k_override if k_override is not None else args_dict.get("k", 20)
    k_global = (k_global_override if k_global_override is not None
                else args_dict.get("k_global", 80))
    k_point_level = args_dict.get("k_point_level", 5)
    emb_dims = args_dict.get("emb_dims", 1024)
    dropout = args_dict.get("dropout", 0.5)

    if model_name == "dgcnn":
        model = TractDGCNN(
            num_classes=num_classes, k=k, k_global=k_global,
            k_point_level=k_point_level, emb_dims=emb_dims,
            dropout=dropout, device=device)
    elif model_name == "pointnet":
        model = PointNetCls(
            k=k, k_global=k_global, num_classes=num_classes,
            feature_transform=False, first_feature_transform=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    weights = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model, args_dict


def run_inference(model, data_loader, global_feat, num_classes, device,
                  progress_callback=None):
    """Run model inference. Returns list of predicted cluster indices."""
    predicted = []
    total_batches = len(data_loader)
    with torch.no_grad():
        for batch_idx, (points, k_local) in enumerate(data_loader):
            num_fiber = points.shape[0]
            points = points.transpose(2, 1)
            k_local = k_local.transpose(2, 1)
            k_global_t = (torch.from_numpy(global_feat)
                          .repeat(num_fiber, 1, 1, 1)
                          .transpose(2, 1))
            info = torch.cat((k_local, k_global_t), dim=3)
            points = points.to(device)
            info = info.to(device)
            pred = model(points, info).view(-1, num_classes)
            predicted.extend(pred.data.max(1)[1].cpu().numpy().tolist())
            if progress_callback:
                progress_callback((batch_idx + 1) / total_batches)
    return predicted
