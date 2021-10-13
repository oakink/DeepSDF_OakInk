#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import concurrent.futures
import json
import logging
import os
import pickle
import shutil
import subprocess

import numpy as np
import trimesh

import deep_sdf
import deep_sdf.workspace as ws


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) != 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in scene_or_mesh.geometry.values())
        )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def resize_objs(meshes_targets_and_specific_args, scale, work_path):
    max_norm = 0
    for (mesh_filepath, _, _, _) in meshes_targets_and_specific_args:
        mesh = as_mesh(trimesh.load(mesh_filepath, process=False))
        verts = np.asfarray(mesh.vertices, dtype=np.float32)
        norm = np.linalg.norm(np.max(verts, axis=0) - np.min(verts, axis=0))
        max_norm = max(max_norm, norm)
    pickle.dump({"max_norm": max_norm, "scale": scale}, open(os.path.join(work_path, "rescale.pkl"), "wb"))

    max_norm = max_norm * scale
    for (mesh_filepath, resize_mesh, _, _) in meshes_targets_and_specific_args:
        outmtl = os.path.splitext(resize_mesh)[0] + ".mtl"
        with open(mesh_filepath, "r") as f, open(resize_mesh, "w") as fo:
            for line in f:
                if line.startswith("v "):
                    v = np.fromstring(line[2:], sep=" ")[:, None]  # [3, 1]
                    v = v / max_norm
                    vNormString = "v %f %f %f\n" % (v[0], v[1], v[2])
                    fo.write(vNormString)
                elif line.startswith("mtllib "):
                    fo.write("mtllib " + os.path.basename(outmtl) + "\n")
                else:
                    fo.write(line)
        shutil.copy2(os.path.splitext(mesh_filepath)[0] + ".mtl", outmtl)


def filter_classes_glob(patterns, classes):
    import fnmatch

    passed_classes = set()
    for pattern in patterns:

        passed_classes = passed_classes.union(set(filter(lambda x: fnmatch.fnmatch(x, pattern), classes)))

    return list(passed_classes)


def filter_classes_regex(patterns, classes):
    import re

    passed_classes = set()
    for pattern in patterns:
        regex = re.compile(pattern)
        passed_classes = passed_classes.union(set(filter(regex.match, classes)))

    return list(passed_classes)


def filter_classes(patterns, classes):
    if patterns[0] == "glob":
        return filter_classes_glob(patterns, classes[1:])
    elif patterns[0] == "regex":
        return filter_classes_regex(patterns, classes[1:])
    else:
        return filter_classes_glob(patterns, classes)


def process_mesh(mesh_filepath, target_filepath, executable, additional_args):
    logging.info(mesh_filepath + " --> " + target_filepath)
    command = [executable, "-m", mesh_filepath, "-o", target_filepath] + additional_args

    subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    subproc.wait()


def append_data_source_map(data_dir, name, source):

    data_source_map_filename = ws.get_data_source_map_filename(data_dir)

    print("data sources stored to " + data_source_map_filename)

    data_source_map = {}

    if os.path.isfile(data_source_map_filename):
        with open(data_source_map_filename, "r") as f:
            data_source_map = json.load(f)

    if name in data_source_map:
        if not data_source_map[name] == os.path.abspath(source):
            raise RuntimeError("Cannot add data with the same name and a different source.")

    else:
        data_source_map[name] = os.path.abspath(source)

        with open(data_source_map_filename, "w") as f:
            json.dump(data_source_map, f, indent=2)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source and append the results to " + "a dataset.",
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        required=True,
        help="The directory which holds all preprocessed data.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        default=False,
        action="store_true",
        help="If set, previously-processed shapes will be skipped",
    )
    arg_parser.add_argument(
        "--threads",
        dest="num_threads",
        default=8,
        help="The number of threads to use to process the data.",
    )
    arg_parser.add_argument("--scale", "-s", default=1.0, help="The max size scale of the category.")
    arg_parser.add_argument(
        "--test",
        "-t",
        dest="test_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce SDF samplies for testing",
    )
    arg_parser.add_argument(
        "--surface",
        dest="surface_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce mesh surface samples for evaluation. "
        + "Otherwise, the script will produce SDF samples for training.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    additional_general_args = []

    deepsdf_dir = os.path.dirname(os.path.abspath(__file__))
    if args.surface_sampling:
        executable = os.path.join(deepsdf_dir, "bin/SampleVisibleMeshSurface")
        subdir = ws.surface_samples_subdir
        extension = ".ply"
    else:
        executable = os.path.join(deepsdf_dir, "bin/PreprocessMesh")
        subdir = ws.sdf_samples_subdir
        extension = ".npz"

        if args.test_sampling:
            additional_general_args += ["-t"]

    with open(os.path.join(args.data_dir, "split.json"), "r") as f:
        split = json.load(f)

    dest_dir = os.path.join(args.data_dir, subdir)

    logging.info("Placing the results in " + dest_dir)

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    if args.surface_sampling:
        normalization_param_dir = os.path.join(args.data_dir, ws.normalization_param_subdir)
        if not os.path.isdir(normalization_param_dir):
            os.makedirs(normalization_param_dir)

    real_meta = json.load(open("./data/meta/object_id.json", "r"))
    virtual_meta = json.load(open("./data/meta/virtual_object_id.json", "r"))
    species_ids = dict(split).keys()

    meshes_targets_and_specific_args = []

    for sid in species_ids:
        obj_ids = {}
        if "real" in split[sid]:
            obj_ids.update({oid: True for oid in split[sid]["real"]})
        if "virtual" in split[sid]:
            obj_ids.update({oid: False for oid in split[sid]["virtual"]})

        for oid, is_real in obj_ids.items():

            obj_name = real_meta[oid]["name"] if is_real else virtual_meta[oid]["name"]

            shape_dir = os.path.join("./data", "yodaObjects" if is_real else "yodaVirtualObjects", obj_name, "align")

            processed_filepath = os.path.join(dest_dir, oid + extension)

            if args.skip and os.path.isfile(processed_filepath):
                logging.debug("skipping " + processed_filepath)
                continue

            try:
                mesh_filename = deep_sdf.data.find_mesh_in_directory(shape_dir)

                specific_args = []

                if args.surface_sampling:

                    normalization_param_filename = os.path.join(normalization_param_dir, oid + ".npz")
                    specific_args = ["-n", normalization_param_filename]

                resize_mesh_path, resize_mesh_subpath, resize_mesh_name = (
                    os.path.split(processed_filepath)[0],
                    os.path.split(processed_filepath)[1][:-4],
                    os.path.split(mesh_filename)[1],
                )
                resize_mesh_path = os.path.join(resize_mesh_path + "_resize", resize_mesh_subpath)
                if not os.path.exists(resize_mesh_path):
                    os.makedirs(resize_mesh_path, exist_ok=True)

                meshes_targets_and_specific_args.append(
                    (
                        mesh_filename,
                        os.path.join(resize_mesh_path, resize_mesh_name),
                        processed_filepath,
                        specific_args,
                    )
                )

            except deep_sdf.data.NoMeshFileError:
                logging.warning("No mesh found for instance " + shape_dir)
            except deep_sdf.data.MultipleMeshFileError:
                logging.warning("Multiple meshes found for instance " + shape_dir)

    print(meshes_targets_and_specific_args)
    resize_objs(meshes_targets_and_specific_args, scale=args.scale, work_path=args.data_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.num_threads)) as executor:

        for (
            _,
            resize_mesh,
            target_filepath,
            specific_args,
        ) in meshes_targets_and_specific_args:
            executor.submit(
                process_mesh,
                resize_mesh,
                target_filepath,
                executable,
                specific_args + additional_general_args,
            )

        executor.shutdown()
