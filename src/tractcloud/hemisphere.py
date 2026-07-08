"""Hemisphere export for TractCloud fine cluster predictions."""

import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import vtk
import whitematteranalysis as wma

from .tract_mapping import TRACT_NAMES, cluster2tract_label
from .vtk_io import extract_fibers, read_polydata, write_polydata


HEMI_OTHER = "other"
HEMI_LEFT = "left"
HEMI_RIGHT = "right"
HEMI_COMMISSURAL = "commissural"

HEMI_CODE = {
    HEMI_OTHER: 0,
    HEMI_LEFT: 1,
    HEMI_RIGHT: 2,
    HEMI_COMMISSURAL: 3,
}

WMA_H_THRESHOLD = 0.5001
WMA_CPC_FINE_ID_FIXES = {145, 159, 557, 677, 770}


def export_hemisphere(input_path, input_pd, cluster_preds, output_dir, atlas_dir):
    """Run WMA registration and write hemisphere-split tract outputs."""
    output_dir = Path(output_dir)
    hemisphere_dir = output_dir / "Hemisphere"
    registered_path = run_wma_registration(input_path, atlas_dir, output_dir)
    registered_pd = read_polydata(str(registered_path))
    cluster_location_file = find_cluster_location_file(atlas_dir)

    return write_hemisphere_outputs(
        input_pd=input_pd,
        registered_pd=registered_pd,
        cluster_preds=cluster_preds,
        output_dir=hemisphere_dir,
        registered_path=registered_path,
        cluster_location_file=cluster_location_file,
    )


def run_wma_registration(input_path, atlas_dir, output_dir):
    atlas_path = find_registration_atlas(atlas_dir)
    registration_dir = Path(output_dir) / "wma_registration"

    cmd = [
        "wm_register_to_atlas_new.py",
        "-mode",
        "rigid_affine_fast",
        str(input_path),
        str(atlas_path),
        str(registration_dir),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=sys.stderr, stderr=sys.stderr)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "wm_register_to_atlas_new.py was not found on PATH. "
            "Install WMA before running --hemisphere-atlas-dir."
        ) from e

    subject_id = Path(input_path).stem
    registered_path = (
        registration_dir / subject_id / "output_tractography" / f"{subject_id}_reg.vtk"
    )
    if not registered_path.exists():
        raise FileNotFoundError(
            f"WMA registration finished but did not write {registered_path}"
        )
    return registered_path


def find_registration_atlas(atlas_dir):
    atlas_dir = Path(atlas_dir).expanduser()
    candidates = [
        atlas_dir / "ORG-Atlases-1.1.1" / "ORG-RegAtlas-100HCP" / "registration_atlas.vtk",
        atlas_dir / "ORG-RegAtlas-100HCP" / "registration_atlas.vtk",
        atlas_dir / "registration_atlas.vtk",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find registration_atlas.vtk in the provided ORG atlas path."
    )


def find_cluster_location_file(atlas_dir):
    atlas_dir = Path(atlas_dir).expanduser()
    candidates = [
        atlas_dir
        / "ORG-Atlases-1.1.1"
        / "ORG-800FC-100HCP"
        / "cluster_hemisphere_location.txt",
        atlas_dir / "ORG-800FC-100HCP" / "cluster_hemisphere_location.txt",
        atlas_dir / "cluster_hemisphere_location.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find ORG-800FC-100HCP/cluster_hemisphere_location.txt "
        "in the provided ORG atlas path."
    )


def write_hemisphere_outputs(
    input_pd,
    registered_pd,
    cluster_preds,
    output_dir,
    registered_path=None,
    cluster_location_file=None,
):
    input_lines = get_polyline_point_ids(input_pd)
    registered_lines = get_polyline_point_ids(registered_pd)
    validate_registered_tractography(input_lines, registered_lines)

    labels = np.asarray(cluster_preds).reshape(-1).astype(np.int64)
    if len(labels) != len(input_lines):
        raise ValueError(
            f"cluster_preds length ({len(labels)}) != streamlines ({len(input_lines)})"
        )

    if cluster_location_file is None:
        raise ValueError("cluster_location_file is required")
    cluster_location = parse_cluster_location_file(cluster_location_file)
    coarse_ids, hemi, hemi_codes = build_assignments(
        registered_pd,
        registered_lines,
        labels,
        cluster_location,
    )

    annotated = annotate_polydata(input_pd, labels, coarse_ids, hemi_codes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for i, hemi_name in enumerate(hemi):
        tract_name = TRACT_NAMES[int(coarse_ids[i])]
        if tract_name == "Other":  # NOTE: discard "Other" tract
            continue
        groups[(hemi_name, tract_name)].append(i)

    written_files = []
    group_counts = defaultdict(dict)
    for (hemi_name, tract_name), idxs in sorted(groups.items()):
        group_counts[hemi_name][tract_name] = len(idxs)
        out_path = output_dir / hemi_name / f"{sanitize_name(tract_name)}_{hemi_name}.vtp"
        write_polydata(extract_fibers(annotated, idxs), str(out_path))
        written_files.append(str(out_path))

    annotated_path = output_dir / "annotated_tractography.vtp"
    write_polydata(annotated, str(annotated_path))
    written_files.append(str(annotated_path))

    summary = {
        "output_dir": str(output_dir),
        "registered_tractography": str(registered_path) if registered_path else None,
        "n_streamlines": int(len(labels)),
        "hemisphere_counts": dict(Counter(hemi)),
        "group_counts": {k: dict(v) for k, v in group_counts.items()},
        "written_files": written_files,
        "hemisphere_code_convention": {
            "0": HEMI_OTHER,
            "1": HEMI_LEFT,
            "2": HEMI_RIGHT,
            "3": HEMI_COMMISSURAL,
        },
    }
    with open(output_dir / "hemisphere_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def parse_cluster_location_file(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.lower().startswith("cluster index"):
                continue

            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue

            match = re.search(r"cluster_(\d+)", parts[0])
            if not match:
                continue

            loc = parts[1].lower()
            if loc not in {"h", "c", "ng"}:
                raise ValueError(f"Unknown cluster location label {loc!r}: {line}")
            mapping[int(match.group(1)) - 1] = loc
    return mapping


def build_assignments(registered_pd, registered_lines, labels, cluster_location):
    coarse_ids = np.asarray(cluster2tract_label(labels), dtype=np.int64)
    hemi = np.asarray([HEMI_OTHER] * len(labels), dtype=object)

    # Use WMA FiberArray for the per-streamline geometric hemisphere test.
    # This matches the core logic in wm_assess_cluster_location_by_hemisphere.py:
    #   points_per_fiber = 40
    #   hemisphere_percent_threshold = 0.5001 for known hemispheric clusters
    #   RAS x/R coordinate: negative = left, positive = right
    fibers = wma.fibers.FiberArray()
    fibers.points_per_fiber = 40
    fibers.hemisphere_percent_threshold = WMA_H_THRESHOLD
    fibers.hemispheres = True
    fibers.convert_from_polydata(registered_pd)

    if fibers.number_of_fibers != len(labels):
        raise ValueError(
            f"WMA FiberArray fibers ({fibers.number_of_fibers}) != labels ({len(labels)})"
        )

    # Convert WMA internal codes to this script's string labels.
    # WMA FiberArray convention:
    #   fiber_hemisphere == -1 -> left
    #   fiber_hemisphere ==  1 -> right
    #   fiber_hemisphere ==  0 -> commissural / ambiguous
    wma_hemi = np.asarray(fibers.fiber_hemisphere, dtype=np.int64)
    raw_hemi = np.asarray([HEMI_COMMISSURAL] * len(labels), dtype=object)
    raw_hemi[wma_hemi == -1] = HEMI_LEFT
    raw_hemi[wma_hemi == 1] = HEMI_RIGHT
    raw_hemi[wma_hemi == 0] = HEMI_COMMISSURAL

    for fine_id in sorted({int(x) for x in labels}):
        idxs = np.where(labels == fine_id)[0]
        tract_name = TRACT_NAMES[int(coarse_ids[idxs[0]])]

        if fine_id < 0 or fine_id >= 800 or tract_name == "Other":
            continue

        loc = "h" if fine_id in WMA_CPC_FINE_ID_FIXES else cluster_location.get(fine_id)
        if loc is None:
            raise ValueError(f"Missing cluster-location entry for fine label {fine_id}")

        if loc in {"c", "ng"}:
            hemi[idxs] = HEMI_COMMISSURAL
            continue

        if loc != "h":
            raise ValueError(f"Unexpected cluster-location label {loc!r} for fine label {fine_id}")

        # For h clusters, use WMA FiberArray's per-streamline result first.
        local = raw_hemi[idxs].copy()

        # Match WMA clusterLocationFile behavior:
        # ambiguous/commissural fibers inside an h cluster are reassigned to
        # the smaller side within that same cluster.
        ambiguous = local == HEMI_COMMISSURAL
        if np.any(ambiguous):
            n_left = int(np.sum(local == HEMI_LEFT))
            n_right = int(np.sum(local == HEMI_RIGHT))
            local[ambiguous] = HEMI_LEFT if n_left <= n_right else HEMI_RIGHT

        hemi[idxs] = local

    hemi_codes = np.asarray([HEMI_CODE[h] for h in hemi], dtype=np.int64)
    return coarse_ids, list(hemi), hemi_codes


def get_polyline_point_ids(pd):
    lines = pd.GetLines()
    lines.InitTraversal()
    ptids = vtk.vtkIdList()
    out = []
    for _ in range(pd.GetNumberOfLines()):
        lines.GetNextCell(ptids)
        out.append([ptids.GetId(i) for i in range(ptids.GetNumberOfIds())])
    return out


def validate_registered_tractography(input_lines, registered_lines):
    if len(input_lines) != len(registered_lines):
        raise ValueError(
            "Input and registered tractography must have the same number of streamlines."
        )

    for i, (input_line, registered_line) in enumerate(zip(input_lines, registered_lines)):
        if len(input_line) != len(registered_line):
            raise ValueError(
                "Input and registered tractography must preserve streamline order "
                f"and point count. Streamline {i} differs."
            )


def annotate_polydata(pd, labels, coarse_ids, hemi_codes):
    annotated = vtk.vtkPolyData()
    annotated.DeepCopy(pd)
    add_int_cell_array(annotated, "FineLabel", labels)
    add_int_cell_array(annotated, "CoarseID", coarse_ids)
    add_int_cell_array(annotated, "HemisphereLocation", hemi_codes)
    return annotated


def add_int_cell_array(pd, name, values):
    pd.GetCellData().RemoveArray(name)
    arr = vtk.vtkIntArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    for value in values:
        arr.InsertNextValue(int(value))
    pd.GetCellData().AddArray(arr)


def sanitize_name(name):
    name = str(name).strip().replace("/", "-").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", name)
