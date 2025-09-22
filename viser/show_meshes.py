"""
Viser script to show meshes from a MultiPly/Hi4D output folder across time.
"""

import argparse
import glob
import os
import time

import cv2
import imageio.v3 as iio
import numpy as np
import trimesh

import viser
import viser.transforms as vtf

COLORS = np.array([
    [0.412, 0.663, 1.0],
    [1.0, 0.749, 0.412],
    [0.412, 1.0, 0.663],
    [0.412, 0.412, 0.663],
    [0.412, 0.0, 0.0],
    [0.0, 0.0, 0.663],
    [0.0, 0.412, 0.0],
    [1.0, 0.0, 0.0],
], dtype=np.float32)


def load_sequence(input_root: str, output_root: str):
    # camera params
    camPs = np.load(os.path.join(input_root, "cameras.npz"))
    norm_camPs = np.load(os.path.join(input_root, "cameras_normalize.npz"))
    scale = norm_camPs["scale_mat_0"].astype(np.float32)[0, 0]

    # images
    image_files = sorted(glob.glob(os.path.join(input_root, "image", "*.png")))
    assert len(image_files), f"No images found in {input_root}/image/*.png"

    # people & frames
    people_dirs = sorted(glob.glob(os.path.join(output_root, "test_mesh", "*")))
    # files include e.g. an extra file/dir; original code did "-1"
    # keep only numeric subdirs
    people_dirs = [d for d in people_dirs if os.path.isdir(d) and os.path.basename(d).isdigit()]
    people_dirs = sorted(people_dirs, key=lambda p: int(os.path.basename(p)))
    number_person = len(people_dirs)
    assert number_person > 0, f"No people folders in {output_root}/test_mesh/*"

    # assume frames based on person 0
    p0 = people_dirs[0]
    # files are like 0000_deformed.ply and 0000_*. second half is something else → //2
    num_frame = len(sorted(glob.glob(os.path.join(p0, "*_deformed.ply"))))
    assert num_frame > 0, f"No *_deformed.ply meshes in {p0}"

    return camPs, scale, image_files, people_dirs, number_person, num_frame


def compute_intrinsics_extrinsics(camPs, image_files, mean_list, scale, start_frame, end_frame):
    """Replicates your OpenCV decomposeProjectionMatrix logic and returns:
       - K (3x3) intrinsics (from first frame)
       - list of T_world_cam (SE3) for frames [start_frame, end_frame)
       - image size (H, W)
    """
    Ks = None
    T_world_cams = []

    # read one image for size
    img0 = cv2.imread(image_files[0], cv2.IMREAD_UNCHANGED)
    H, W = img0.shape[0], img0.shape[1]

    for idx in range(start_frame, end_frame):
        out = cv2.decomposeProjectionMatrix(camPs[f"cam_{idx}"][:3, :])
        if idx == start_frame:
            K = out[0]
            Ks = K / K[2, 2]  # normalize
        R_cam_world = out[1]  # world->cam rotation
        C = out[2]            # homogeneous camera center (world coords)
        C = (C[:3] / C[3])[:, 0]  # (3,)

        # shift by mean, then scale (match original)
        C_shifted = C - (mean_list[idx - start_frame] * scale)

        # We have x_cam = R * x_world + t with t = -R * C
        t = -R_cam_world @ C_shifted

        # Build extrinsic [R|t] (world->cam), then invert to get T_world_cam
        T_world_cam = vtf.SE3.from_matrix(np.block([
            [R_cam_world, t.reshape(3, 1)],
            [np.zeros((1, 3)), np.array([[1.0]])],
        ])).inverse()

        T_world_cams.append(T_world_cam)

    return Ks, T_world_cams, (H, W)


def main(input_root: str, output_root: str, visualize_mesh: bool = True):
    camPs, scale, image_files_all, people_dirs, number_person, num_frame = load_sequence(
        input_root, output_root
    )
    start_frame = 0
    end_frame = num_frame

    # Preload per-person, per-frame meshes (variable topology)
    mean_list = []
    for idx in range(start_frame, end_frame):
        mesh_path = os.path.join(people_dirs[0], f"{idx:04d}_deformed.ply")
        m = trimesh.load(mesh_path, process=False)
        mean_list.append(m.vertices.mean(axis=0))
    mean_list = np.stack(mean_list, axis=0)

    # Preload all frames for all people
    # Store as: per_person_vertices[p][f] / per_person_faces[p][f]
    per_person_vertices = [[] for _ in range(number_person)]
    per_person_faces = [[] for _ in range(number_person)]
    for p, pdir in enumerate(people_dirs):
        for idx in range(start_frame, end_frame):
            mesh_path = os.path.join(pdir, f"{idx:04d}_deformed.ply")
            mesh = trimesh.load(mesh_path, process=False)

            # recentre & scale like original
            mesh.vertices = mesh.vertices - mean_list[idx - start_frame]
            verts = (mesh.vertices * scale).astype(np.float32)
            faces = mesh.faces.astype(np.uint32)

            per_person_vertices[p].append(verts)
            per_person_faces[p].append(faces)

    # restrict images to frame span
    image_files = image_files_all[start_frame:end_frame]

    # Build intrinsics/extrinsics (world poses)
    K, T_world_cams, (H, W) = compute_intrinsics_extrinsics(
        camPs, image_files, mean_list, scale, start_frame, end_frame
    )

    # FOV from fy, H (same as in viser COLMAP demo)
    fy = K[1, 1]
    fov = 2.0 * np.arctan2(H / 2.0, fy)
    aspect = W / H

    # --- VISER ---
    server = viser.ViserServer()  # prints a URL; open it in your browser
    server.scene.add_grid("/grid", width=10.0, height=10.0, position=(0, 0, 0))

    if args.share:
        url = server.request_share_url()
        print(f"[Viser] Share URL: {url}", flush=True)
        if args.share_url_file:
            with open(args.share_url_file, "w") as f:
                f.write(url.strip() + "\n")

    # GUI: frame scrubber
    frame_slider = server.gui.add_slider(
        "frame", min=start_frame, max=end_frame - 1, step=1, initial_value=start_frame
    )

    # after creating frame_slider
    prev_btn = server.gui.add_button("◀ Prev")
    next_btn = server.gui.add_button("Next ▶")

    @prev_btn.on_click
    def _(_):
        frame_slider.value = max(start_frame, frame_slider.value - 1)

    @next_btn.on_click
    def _(_):
        frame_slider.value = min(end_frame - 1, frame_slider.value + 1)

    # Add one mesh handle per person (we’ll replace geometry on updates).
    mesh_handles = []
    mesh_paths   = []  # <— add this
    mesh_colors  = []  # <— optional, if you want to reuse exact color

    if visualize_mesh:
        for p in range(number_person):
            color = (COLORS[p % len(COLORS)] * 255).astype(np.uint8)
            path = f"/mesh_{p}"                    # <— cache path
            h = server.scene.add_mesh_simple(
                path,
                vertices=per_person_vertices[p][frame_slider.value].astype(np.float32, copy=False),
                faces=per_person_faces[p][frame_slider.value].astype(np.uint32, copy=False),
                color=(int(color[0]), int(color[1]), int(color[2])),
            )
            mesh_handles.append(h)
            mesh_paths.append(path)                # <— store
            mesh_colors.append(tuple(int(x) for x in color))  # <— store if desired


    # Add a camera frustum with the first frame’s image; reuse the same handle.
    image0 = iio.imread(image_files[frame_slider.value])
    # Camera pose for current frame:
    T_cam = T_world_cams[frame_slider.value]
    frame_node = server.scene.add_frame(
        "/camera_frame",
        wxyz=T_cam.rotation().wxyz,
        position=T_cam.translation(),
        axes_length=0.05,
        axes_radius=0.002,
    )
    frustum = server.scene.add_camera_frustum(
        "/camera_frame/frustum",
        fov=float(fov),
        aspect=float(aspect),
        scale=0.15,
        image=image0,
    )

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.wxyz = frame_node.wxyz
        client.camera.position = frame_node.position

    # Clicking the frustum jumps the viewer to that pose (handy)
    @frustum.on_click
    def _(_evt: viser.SceneNodePointerEvent) -> None:
        for client in server.get_clients().values():
            client.camera.wxyz = frame_node.wxyz
            client.camera.position = frame_node.position

    def set_frame(f: int):
        f = int(np.clip(f, start_frame, end_frame - 1))
        if visualize_mesh:
            with server.atomic():
                for p, h in enumerate(mesh_handles):
                    v_new = per_person_vertices[p][f]
                    f_new = per_person_faces[p][f]

                    # current dtypes (if handle was removed previously, this block only runs when h is valid)
                    vdt = h.vertices.dtype
                    fdt = h.faces.dtype

                    if h.vertices.shape != v_new.shape or h.faces.shape != f_new.shape:
                        # topology changed → recreate using cached path/color
                        cached_path  = mesh_paths[p]
                        cached_color = mesh_colors[p]  # (r,g,b)

                        # remove the old node and create a fresh one at the same path
                        h.remove()
                        new_h = server.scene.add_mesh_simple(
                            cached_path,
                            vertices=v_new.astype(np.float32, copy=False),
                            faces=f_new.astype(np.uint32, copy=False),
                            color=cached_color,
                        )
                        mesh_handles[p] = new_h
                    else:
                        # topology same → safe in-place update
                        h.vertices = v_new.astype(vdt, copy=False)
                        h.faces    = f_new.astype(fdt, copy=False)

        # Update camera/frustum
        T_cam_f = T_world_cams[f]
        frame_node.wxyz = T_cam_f.rotation().wxyz
        frame_node.position = T_cam_f.translation()
        frustum.image = iio.imread(image_files[f])


    @frame_slider.on_update
    def _(_):
        set_frame(frame_slider.value)


    while True:
        time.sleep(1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        default="../data/taichi01_vitpose_openpose",
        help="Path with cameras.npz, cameras_normalize.npz, and image/*.png",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="../code/outputs/Hi4D/taichi01_sam_delay_depth_loop_2_MLP_vitpose_openpose",
        help="Path with test_mesh/<p>/<frame>_deformed.ply",
    )
    parser.add_argument("--share", action="store_true", default=False, help="Request public share URL via Viser relay")
    parser.add_argument("--share-url-file", type=str, default="", help="Write share URL here")

    parser.add_argument("--no-mesh", action="store_true", help="Disable mesh visualization")
    args = parser.parse_args()

    main(args.input_root, args.output_root, visualize_mesh=not args.no_mesh)
