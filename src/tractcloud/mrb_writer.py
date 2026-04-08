"""Create Slicer-compatible MRB files from parcellation results.

An MRB is a ZIP archive containing MRML scene XML + VTP data files,
loadable in 3D Slicer with SubjectHierarchy, colors, and display settings.
"""

import os
import tempfile
import zipfile
from xml.etree.ElementTree import Element, SubElement, tostring, indent

from .colors import get_tract_color
from .tract_mapping import TRACT_FULL_NAMES
from .vtk_io import write_polydata


def create_mrb(tracts_by_category, output_path, base_name="TractCloud"):
    """Create an MRB file from parcellation results.

    Args:
        tracts_by_category: dict {category: {tract_name: vtkPolyData}}
        output_path: path for the output .mrb file
        base_name: scene name used in MRML
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        scene_dir = os.path.join(tmpdir, base_name)
        data_dir = os.path.join(scene_dir, "Data")
        os.makedirs(data_dir)

        # Build MRML XML
        mrml = Element("MRML", version="Slicer4.4.0", userTags="")

        node_id = 1
        sh_items = []  # (parent_id, item_element) pairs for hierarchy
        sh_root_id = 100
        sh_next_id = sh_root_id + 1
        color_index = 1

        # Category folder IDs for SubjectHierarchy
        cat_sh_ids = {}

        for category, tracts in tracts_by_category.items():
            cat_sh_id = sh_next_id
            sh_next_id += 1
            cat_sh_ids[category] = cat_sh_id

            for tract_name, polydata in tracts.items():
                full_name = TRACT_FULL_NAMES.get(tract_name, tract_name)
                display_name = f"{full_name} ({tract_name})"
                r, g, b = get_tract_color(color_index)
                color_index += 1

                # File
                filename = f"{tract_name}.vtp"
                filepath = os.path.join(data_dir, filename)
                write_polydata(polydata, filepath)

                # Storage node
                storage_id = f"vtkMRMLFiberBundleStorageNode{node_id}"
                storage = SubElement(mrml, "FiberBundleStorage",
                    id=storage_id,
                    name=f"FiberBundleStorage_{node_id}",
                    fileName=f"Data/{filename}",
                    useCompression="1")

                # Line display node
                line_disp_id = f"vtkMRMLFiberBundleLineDisplayNode{node_id}"
                SubElement(mrml, "FiberBundleLineDisplayNode",
                    id=line_disp_id,
                    name=f"FiberBundleLineDisplayNode_{node_id}",
                    color=f"{r:.6f} {g:.6f} {b:.6f}",
                    visibility="true",
                    scalarVisibility="false",
                    colorMode="0")

                # Tube display node
                tube_disp_id = f"vtkMRMLFiberBundleTubeDisplayNode{node_id}"
                SubElement(mrml, "FiberBundleTubeDisplayNode",
                    id=tube_disp_id,
                    name=f"FiberBundleTubeDisplayNode_{node_id}",
                    color=f"{r:.6f} {g:.6f} {b:.6f}",
                    visibility="false",
                    colorMode="0")

                # Glyph display node
                glyph_disp_id = f"vtkMRMLFiberBundleGlyphDisplayNode{node_id}"
                SubElement(mrml, "FiberBundleGlyphDisplayNode",
                    id=glyph_disp_id,
                    name=f"FiberBundleGlyphDisplayNode_{node_id}",
                    visibility="false")

                # FiberBundle node
                fb_id = f"vtkMRMLFiberBundleNode{node_id}"
                SubElement(mrml, "FiberBundle",
                    id=fb_id,
                    name=display_name,
                    displayNodeRef=f"{line_disp_id} {tube_disp_id} {glyph_disp_id}",
                    storageNodeRef=storage_id)

                # Track for SubjectHierarchy
                tract_sh_id = sh_next_id
                sh_next_id += 1
                sh_items.append((cat_sh_id, tract_sh_id, fb_id, display_name))
                node_id += 1

        # SubjectHierarchy node — uses parent= attribute, not XML nesting
        sh_node = SubElement(mrml, "SubjectHierarchy",
            id="vtkMRMLSubjectHierarchyNode1",
            name="SubjectHierarchy",
            attributes="SubjectHierarchyVersion:2")

        root_folder_id = sh_root_id + 50

        # Scene root item
        SubElement(sh_node, "SubjectHierarchyItem",
            id=str(sh_root_id), name="Scene", parent="0",
            type="", expanded="true",
            attributes="Level^Scene|")

        # Root folder
        SubElement(sh_node, "SubjectHierarchyItem",
            id=str(root_folder_id), name=base_name,
            parent=str(sh_root_id), type="",
            expanded="true",
            attributes="Level^Folder|")

        # Category folders
        for category, cat_id in cat_sh_ids.items():
            SubElement(sh_node, "SubjectHierarchyItem",
                id=str(cat_id), name=category,
                parent=str(root_folder_id), type="",
                expanded="true",
                attributes="Level^Folder|")

        # Tract items under their categories (no name — taken from dataNode)
        for cat_id, tract_id, data_node_ref, name in sh_items:
            SubElement(sh_node, "SubjectHierarchyItem",
                id=str(tract_id),
                dataNode=data_node_ref,
                parent=str(cat_id), type="",
                expanded="true")

        # Write MRML
        indent(mrml)
        mrml_content = tostring(mrml, encoding="unicode",
                                xml_declaration=True)
        mrml_path = os.path.join(scene_dir, f"{base_name}.mrml")
        with open(mrml_path, "w") as f:
            f.write(mrml_content)

        # Create ZIP
        os.makedirs(os.path.dirname(os.path.abspath(output_path)),
                     exist_ok=True)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(scene_dir):
                for fname in files:
                    full_path = os.path.join(root, fname)
                    arc_name = os.path.relpath(full_path, tmpdir)
                    zf.write(full_path, arc_name)
