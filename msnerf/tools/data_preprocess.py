import json
import math
import os
import subprocess
import sys
from os import makedirs
from os.path import join as pjoin

import cv2
import numpy as np


def extract_frames(video_path, output_dir, num_frames=200):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("无法获取正确的总帧数")
        return

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=np.int32)

    saved = 0
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_name = os.path.join(output_dir, f"{i:04d}.png")
        cv2.imwrite(frame_name, frame)
        saved += 1

    cap.release()
    print(f"提取完成，共保存 {saved} 帧。")


def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2
        ]
    ])


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def closest_point_2_lines(oa, da, ob,
                          db):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def estimate_positions(colmap_exe: str,
                       work_dir,
                       json_name='transforms.json',
                       debug=False):
    images_path = pjoin(work_dir, 'images')
    text_path = pjoin(work_dir, 'text')
    db_path = pjoin(work_dir, 'colmap.db')
    sparse_path = pjoin(work_dir, 'sparse')
    makedirs(text_path, exist_ok=True)
    makedirs(sparse_path, exist_ok=True)

    subprocess.run([colmap_exe, 'feature_extractor',
                    '--ImageReader.camera_model', 'SIMPLE_PINHOLE', '--ImageReader.single_camera', '1',
                    '--database_path', db_path, '--image_path', images_path],
                   check=debug)
    subprocess.run(
        [colmap_exe, 'sequential_matcher', '--SiftMatching.guided_matching=true', '--database_path', db_path],
        check=debug)
    subprocess.run(
        [colmap_exe, 'mapper', '--database_path', db_path, '--image_path', images_path, '--output_path', sparse_path],
        check=debug)
    subprocess.run(
        [colmap_exe, 'bundle_adjuster', '--input_path', pjoin(sparse_path, '0'), '--output_path',
         pjoin(sparse_path, '0'),
         '--BundleAdjustment.refine_principal_point', '1'], check=debug)
    subprocess.run([colmap_exe, 'model_converter', '--input_path', pjoin(sparse_path, '0'), '--output_path', text_path,
                    '--output_type', 'TXT'], check=debug)

    cameras = {}
    with open(pjoin(text_path, "cameras.txt"), "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            els = line.split(" ")
            camera = {}
            camera_id = int(els[0])
            camera["w"] = float(els[2])
            camera["h"] = float(els[3])
            camera["fl_x"] = float(els[4])
            camera["fl_y"] = float(els[4])
            camera["k1"] = 0
            camera["k2"] = 0
            camera["k3"] = 0
            camera["k4"] = 0
            camera["p1"] = 0
            camera["p2"] = 0
            camera["cx"] = float(els[5])
            camera["cy"] = float(els[6])
            camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
            camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
            camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
            camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi
            cameras[camera_id] = camera

    if len(cameras) == 0:
        print("No cameras found!")
        sys.exit(1)

    with open(os.path.join(text_path, "images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        camera = cameras[camera_id]
        out = {
            "camera_angle_x": camera["camera_angle_x"],
            "camera_angle_y": camera["camera_angle_y"],
            "fl_x": camera["fl_x"],
            "fl_y": camera["fl_y"],
            "cx": camera["cx"],
            "cy": camera["cy"],
            "w": camera["w"],
            "h": camera["h"],
            "num_ms": 25,
            "frames": [],
        }

        up = np.zeros(3)
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i % 2 == 1:
                elems = line.split(
                    " ")  # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                # name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
                # why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(images_path)
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)

                c2w[0:3, 2] *= -1  # flip the y and z axis
                c2w[0:3, 1] *= -1
                c2w = c2w[[1, 0, 2, 3], :]
                c2w[2, :] *= -1  # flip whole world upside down
                up += c2w[0:3, 1]

                name = str(f"./images/{'_'.join(elems[9:])}")

                frame = {"file_path": name,
                         "transform_matrix": c2w}
                out["frames"].append(frame)

    up = up / np.linalg.norm(up)
    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1
    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis

    # find a central point they are all looking at
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3, :]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.00001:
                totp += p * w
                totw += w
    if totw > 0.0:
        totp /= totw
    # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= totp

    maxlen = 0.
    for f in out["frames"]:
        maxlen = max(np.max(np.abs(f["transform_matrix"][:3, 3])), maxlen)
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] /= maxlen  # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    with open(pjoin(work_dir, json_name), "w") as outfile:
        json.dump(out, outfile, indent=2)


if __name__ == '__main__':
    estimate_positions(r"D:\files\PHD\myNeRF\COLMAP-3.7-windows-cuda\COLMAP.bat",
                       r"D:\files\PHD\myNeRF\data\91_0", )
