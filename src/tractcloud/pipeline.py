"""TractCloud pipeline: end-to-end tractography parcellation."""

import time

import numpy as np
import torch

from .model_data import ensure_model_data
from .inference import (
    extract_ras_features, center_tractography,
    RealDataDataset, load_model, run_inference,
)
from .tract_mapping import (
    TRACT_NAMES, TRACT_CATEGORIES, TRACT_FULL_NAMES,
    cluster2tract_label,
)
from .vtk_io import read_polydata, write_polydata, extract_fibers
from .colors import get_tract_color
from .progress import NullReporter


class TractCloudPipeline:
    """End-to-end TractCloud inference pipeline."""

    def __init__(self, reporter=None, device=None, batch_size=2048,
                 num_points=15, include_other=False, data_dir=None,
                 k=20, k_global=80, k_ds_rate=0.1):
        self.reporter = reporter or NullReporter()
        self.batch_size = batch_size
        self.num_points = num_points
        self.include_other = include_other
        self.k = k
        self.k_global = k_global
        self.k_ds_rate = k_ds_rate
        self.num_classes = 1600

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if data_dir:
            import os
            os.environ["TRACTCLOUD_DATA_DIR"] = data_dir

    def run_on_file(self, input_path, output_dir, create_mrb=False):
        """Run parcellation on a VTK/VTP file.

        Writes per-tract VTP files to output_dir, organized by category.

        Args:
            input_path: path to input .vtk/.vtp file
            output_dir: directory for output files
            create_mrb: if True, also create an MRB file

        Returns:
            dict mapping category -> {tract_name: output_file_path}
        """
        import os

        polydata = read_polydata(input_path)
        self.reporter.status(
            f"Loaded {polydata.GetNumberOfLines()} streamlines from "
            f"{os.path.basename(input_path)}")

        result = self.run_on_polydata(polydata)

        os.makedirs(output_dir, exist_ok=True)
        output_files = {}
        color_index = 1

        for category, tracts in result.items():
            cat_dir = os.path.join(output_dir, category)
            os.makedirs(cat_dir, exist_ok=True)
            output_files[category] = {}
            for tract_name, tract_pd in tracts.items():
                full_name = TRACT_FULL_NAMES.get(tract_name, tract_name)
                filename = f"{full_name} ({tract_name}).vtp"
                filepath = os.path.join(cat_dir, filename)
                write_polydata(tract_pd, filepath)
                output_files[category][tract_name] = filepath
                color_index += 1

        total_tracts = sum(len(t) for t in output_files.values())
        self.reporter.result(
            total_tracts, output_dir,
            total_time=None)

        if create_mrb:
            from .mrb_writer import create_mrb
            base_name = os.path.splitext(
                os.path.basename(input_path))[0] + "_TractCloud"
            mrb_path = os.path.join(output_dir, base_name + ".mrb")
            create_mrb(result, mrb_path, base_name)
            self.reporter.status(f"MRB written to {mrb_path}")

        return output_files

    def run_on_polydata(self, polydata):
        """Run parcellation on a vtkPolyData.

        Args:
            polydata: vtkPolyData with lines

        Returns:
            dict: {category_name: {tract_name: vtkPolyData}}
        """
        start_time = time.time()
        total_steps = 4

        # Report device selection
        if self.device.type == "cuda":
            torch.backends.cudnn.enabled = False
            self.reporter.status(
                f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.reporter.status(
                "No GPU detected, using CPU. "
                "Inference will be significantly slower.")

        # Step 1: Extract features
        num_fibers = polydata.GetNumberOfLines()
        self.reporter.status(
            f"Extracting features from {num_fibers} streamlines "
            f"(step 1/{total_steps})...",
            step=1, total_steps=total_steps)
        self.reporter.progress(0.0, step=1)

        weight_path, args_path, mass_center = ensure_model_data(
            progress_callback=lambda f: self.reporter.progress(f * 0.5, step=1))

        model, _ = load_model(
            weight_path, args_path, self.device,
            k_override=self.k, k_global_override=self.k_global)

        feat = extract_ras_features(polydata, num_points=self.num_points)
        self.reporter.progress(1.0, step=1)

        # Step 2: KNN
        self.reporter.status(
            "Re-centering and computing neighbor features "
            f"(step 2/{total_steps}, slowest step)...",
            step=2, total_steps=total_steps)
        centered = center_tractography(feat, mass_center)
        dataset = RealDataDataset(
            centered, k=self.k, k_global=self.k_global,
            k_ds_rate=self.k_ds_rate,
            progress_callback=lambda f: self.reporter.progress(f, step=2))
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)

        # Step 3: Inference (with CPU fallback for incompatible GPUs)
        self.reporter.status(
            f"Running model inference on {self.device} "
            f"(step 3/{total_steps})...",
            step=3, total_steps=total_steps)
        try:
            cluster_preds = run_inference(
                model, data_loader, dataset.global_feat,
                self.num_classes, self.device,
                progress_callback=lambda f: self.reporter.progress(
                    f, step=3))
        except RuntimeError as e:
            if "no kernel image" in str(e) or "CUDA" in str(e):
                logging.warning(
                    f"GPU inference failed ({e}), falling back to CPU")
                self.reporter.status(
                    "GPU incompatible, falling back to CPU "
                    f"(step 3/{total_steps})...",
                    step=3, total_steps=total_steps)
                cpu = torch.device("cpu")
                model = model.to(cpu)
                cluster_preds = run_inference(
                    model, data_loader, dataset.global_feat,
                    self.num_classes, cpu,
                    progress_callback=lambda f: self.reporter.progress(
                        f, step=3))
            else:
                raise

        # Step 4: Output
        self.reporter.status(
            f"Extracting tract bundles (step 4/{total_steps})...",
            step=4, total_steps=total_steps)
        tract_labels = np.array(cluster2tract_label(cluster_preds))

        result = {}
        total_tracts = sum(
            len(tracts) for cat, tracts in TRACT_CATEGORIES.items()
            if cat != "Other" or self.include_other)
        tracts_done = 0

        for category, tract_names_in_cat in TRACT_CATEGORIES.items():
            if category == "Other" and not self.include_other:
                continue
            cat_tracts = {}
            for tract_name in tract_names_in_cat:
                tract_idx = TRACT_NAMES.index(tract_name)
                fiber_idx = np.where(tract_labels == tract_idx)[0]
                if len(fiber_idx) == 0:
                    continue
                cat_tracts[tract_name] = extract_fibers(polydata, fiber_idx)
                tracts_done += 1
                self.reporter.progress(tracts_done / total_tracts, step=4)
            if cat_tracts:
                result[category] = cat_tracts

        elapsed = time.time() - start_time
        total_created = sum(len(t) for t in result.values())
        self.reporter.result(total_created, "", total_time=elapsed)
        return result

    def run_on_features(self, feat_ras):
        """Run on pre-extracted numpy features.

        Args:
            feat_ras: (N, num_points, 3) array

        Returns:
            dict: {tract_name: array of fiber indices}
        """
        weight_path, args_path, mass_center = ensure_model_data()
        model, _ = load_model(
            weight_path, args_path, self.device,
            k_override=self.k, k_global_override=self.k_global)
        centered = center_tractography(feat_ras, mass_center)
        dataset = RealDataDataset(
            centered, k=self.k, k_global=self.k_global,
            k_ds_rate=self.k_ds_rate)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)
        cluster_preds = run_inference(
            model, data_loader, dataset.global_feat,
            self.num_classes, self.device)
        tract_labels = np.array(cluster2tract_label(cluster_preds))

        result = {}
        for tract_name in TRACT_NAMES:
            if tract_name == "Other" and not self.include_other:
                continue
            tract_idx = TRACT_NAMES.index(tract_name)
            indices = np.where(tract_labels == tract_idx)[0]
            if len(indices) > 0:
                result[tract_name] = indices
        return result
