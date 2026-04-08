"""VTK file I/O for tractography data.

Reads and writes VTK/VTP polydata files, preserving all point and cell
data arrays (tensors, scalars, etc.).
"""

import os

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def read_polydata(filepath):
    """Read a VTK or VTP polydata file.

    Args:
        filepath: path to .vtk (legacy) or .vtp (XML) file

    Returns:
        vtkPolyData
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    reader.SetFileName(filepath)
    reader.Update()
    return reader.GetOutput()


def write_polydata(polydata, filepath):
    """Write a vtkPolyData to a VTP (XML) or VTK (legacy) file.

    All point and cell data arrays are preserved automatically.

    Args:
        polydata: vtkPolyData to write
        filepath: output path (.vtp or .vtk)
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".vtp":
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetCompressorTypeToZLib()
    elif ext == ".vtk":
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileTypeToBinary()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    writer.SetFileName(filepath)
    writer.SetInputData(polydata)
    writer.Write()


def extract_fibers(polydata, fiber_indices):
    """Extract a subset of fibers from polydata, preserving data arrays.

    Args:
        polydata: source vtkPolyData with lines
        fiber_indices: array of cell (line) indices to extract

    Returns:
        new vtkPolyData with only the selected fibers and their data
    """
    out_points = vtk.vtkPoints()
    out_lines = vtk.vtkCellArray()
    pt_ids = vtk.vtkIdList()
    in_points = polydata.GetPoints()

    # Map from old point IDs to new point IDs
    old_to_new = {}
    new_cell_indices = []

    for cell_idx in fiber_indices:
        polydata.GetCellPoints(int(cell_idx), pt_ids)
        new_pt_ids = vtk.vtkIdList()
        for j in range(pt_ids.GetNumberOfIds()):
            old_id = pt_ids.GetId(j)
            if old_id not in old_to_new:
                point = in_points.GetPoint(old_id)
                new_id = out_points.InsertNextPoint(point)
                old_to_new[old_id] = new_id
            new_pt_ids.InsertNextId(old_to_new[old_id])
        out_lines.InsertNextCell(new_pt_ids)

    out_pd = vtk.vtkPolyData()
    out_pd.SetPoints(out_points)
    out_pd.SetLines(out_lines)

    # Copy point data arrays for extracted points
    in_point_data = polydata.GetPointData()
    out_point_data = out_pd.GetPointData()
    if in_point_data.GetNumberOfArrays() > 0:
        # Build sorted mapping: new_id -> old_id
        sorted_pairs = sorted(old_to_new.items(), key=lambda x: x[1])
        old_ids = np.array([p[0] for p in sorted_pairs], dtype=np.intp)

        for i in range(in_point_data.GetNumberOfArrays()):
            in_arr = in_point_data.GetArray(i)
            if in_arr is None:
                continue
            np_arr = vtk_to_numpy(in_arr)
            out_np = np_arr[old_ids]
            out_arr = numpy_to_vtk(
                np.ascontiguousarray(out_np), deep=True)
            out_arr.SetName(in_arr.GetName())
            out_point_data.AddArray(out_arr)

    # Copy cell data arrays for extracted cells
    in_cell_data = polydata.GetCellData()
    out_cell_data = out_pd.GetCellData()
    if in_cell_data.GetNumberOfArrays() > 0:
        cell_idx_arr = np.array(fiber_indices, dtype=np.intp)
        for i in range(in_cell_data.GetNumberOfArrays()):
            in_arr = in_cell_data.GetArray(i)
            if in_arr is None:
                continue
            np_arr = vtk_to_numpy(in_arr)
            out_np = np_arr[cell_idx_arr]
            out_arr = numpy_to_vtk(
                np.ascontiguousarray(out_np), deep=True)
            out_arr.SetName(in_arr.GetName())
            out_cell_data.AddArray(out_arr)

    return out_pd
