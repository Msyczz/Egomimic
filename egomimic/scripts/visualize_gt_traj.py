import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm

from egomimic.utils.egomimicUtils import (
    ARIA_INTRINSICS,
    EXTRINSICS,
    ee_pose_to_cam_pixels,
    draw_dot_on_frame,
    AlohaFK
)

from piper_fk import C_PiperForwardKinematics  # ★ 新增：Piper FK

# ====== 需要你改的部分 ======
HDF5_PATH = "/home/symu/data/mytask_2.hdf5"   # 建议先改成 calib_data_left.h5 试一下
EXTRINSICS_KEY = "ariaJul29"
CAM_SIDE = "left"
DEMO_NAME = "demo_0"
OUT_DIR = "./gt_videos"
FPS = 30
# ===========================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # cam->base 外参矩阵（注意：这里的 base 是你标定时用的 base）
    T_cam_base = EXTRINSICS[EXTRINSICS_KEY][CAM_SIDE]
    K = ARIA_INTRINSICS  # 3x4 内参矩阵（现在是改过的 D455 内参）

    print("Using EXTRINSICS key:", EXTRINSICS_KEY, CAM_SIDE)
    print("T_cam_base =\n", T_cam_base)
    print("Intrinsics K =\n", K)

    fk_urdf = AlohaFK()              # URDF FK
    fk_piper = C_PiperForwardKinematics()  # ★ Piper FK

    with h5py.File(HDF5_PATH, "r") as f:
        data_group = f["data"]

        demo_names = [DEMO_NAME] if DEMO_NAME is not None else list(data_group.keys())

        for demo_name in demo_names:
            demo = data_group[demo_name]
            print(f"\nProcessing demo: {demo_name}")

            imgs = demo["obs/front_img_1"][:]           # (T, H, W, 3), uint8
            obs_group = demo["obs"]

            # ---- 1. ee_pos_base: 末端在 base 下的位置 ----
            use_urdf_fk = False  # ★ 先默认用 ee_pose / Piper 数据

            if "ee_pose_robot_frame" in obs_group.keys():
                print("  Found ee_pose_robot_frame, use it directly.")
                ee_pose = obs_group["ee_pose_robot_frame"][:]   # (T, 7)
                ee_pos_base = ee_pose[:, :3]                    # (T, 3)
            elif "joint_positions" in obs_group.keys():
                print("  ee_pose_robot_frame not found, using joint_positions + URDF FK.")
                qpos = obs_group["joint_positions"][:]          # (T, D)
                print("  joint_positions shape:", qpos.shape)

                # Aloha 是 6-DOF，只取前 6 维
                if qpos.shape[1] > 6:
                    print(f"  Detected {qpos.shape[1]}-D joints, using first 6 dims for URDF FK.")
                    qpos_fk = qpos[:, :6]
                elif qpos.shape[1] == 6:
                    qpos_fk = qpos
                else:
                    raise ValueError(
                        f"joint_positions has only {qpos.shape[1]} dims, but Aloha FK expects 6."
                    )

                ee_pos_base = fk_urdf.fk(qpos_fk).numpy()
                use_urdf_fk = True
            else:
                raise KeyError(
                    "Neither 'ee_pose_robot_frame' nor 'joint_positions' in obs. "
                    f"Available obs keys: {list(obs_group.keys())}"
                )

            #---------------------offset------------------------------------------
            tool_offset_base = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            ee_pos_base = ee_pos_base + tool_offset_base[None, :]

            #------------------ ★★ DEBUG 1: Piper FK vs URDF / ee_pose ★★ ----------
            # 这里我们对比同一组关节角下，Piper FK 和 (ee_pos_base) 的差别
            if "joint_positions" in obs_group.keys():
                qpos_all = obs_group["joint_positions"][:]  # (T, D)
                diff_list = []

                T_total = qpos_all.shape[0]
                print("  [DEBUG] Comparing Piper FK vs current ee_pos_base ...")
                for i in range(0, T_total, max(T_total // 20, 1)):  # 采样约 20 个点
                    q = qpos_all[i, :6]  # Piper 只用前 6 维

                    # Piper FK：末端在 Piper base 下的位置 (mm→m)
                    pose_rpy = fk_piper.CalFK(q)[-1]      # link6
                    p_piper = np.array(pose_rpy[:3]) / 1000.0  # (3,)

                    # 当前用的 base 下的末端位置（可能是 URDF FK 或 ee_pose）
                    p_cur = ee_pos_base[i]  # (3,)

                    diff = p_piper - p_cur
                    diff_list.append(diff)

                diffs = np.stack(diff_list)
                print("  [DEBUG] Piper - current_base mean diff [m]:", diffs.mean(axis=0))
                print("  [DEBUG] Piper - current_base std  [m]:", diffs.std(axis=0))
                print("  -----------------------------------------------------")
            #-------------------------------------------------------------------

            #------------------ 打印范围 -------------------
            print("ee_pos_base x range:", ee_pos_base[:, 0].min(), ee_pos_base[:, 0].max())
            print("ee_pos_base y range:", ee_pos_base[:, 1].min(), ee_pos_base[:, 1].max())
            print("ee_pos_base z range:", ee_pos_base[:, 2].min(), ee_pos_base[:, 2].max())

            print("first 5 ee_pos_base (m):")
            print(ee_pos_base[:5])

            #-------------------------------------------------------------
            #-------------------------test: 主点------------------------
            frame0 = imgs[0].copy()   # 第一帧
            H, W, _ = frame0.shape
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            print("fx, fy, cx, cy =", fx, fy, cx, cy)

            # 画一个十字在 (cx, cy)
            u, v = int(cx), int(cy)
            frame_pp = frame0.copy()
            cv2.drawMarker(
                frame_pp,
                (u, v),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )
            pp_path = os.path.join(OUT_DIR, f"{demo_name}_debug_principal_point.png")
            cv2.imwrite(pp_path, cv2.cvtColor(frame_pp, cv2.COLOR_RGB2BGR))
            print("saved:", pp_path)

            #---------------test: base 坐标轴---------------------------
            axis_points = np.array([
                [0.0, 0.0, 0.0],   # origin
                [0.1, 0.0, 0.0],   # +X
                [0.0, 0.1, 0.0],   # +Y
                [0.0, 0.0, 0.1],   # +Z
            ], dtype=np.float32)

            axis_px = ee_pose_to_cam_pixels(
                ee_pose_base=axis_points,
                T_cam_base=T_cam_base,
                intrinsics=K,
            )
            axis_px = np.asarray(axis_px, dtype=np.float32)

            if axis_px.shape[1] == 3:
                axis_px_2d = axis_px[:, :2] / np.clip(axis_px[:, 2:3], 1e-6, None)
            else:
                axis_px_2d = axis_px[:, :2]

            print("base axes projected pixels (u, v):")
            names = ["O", "+X", "+Y", "+Z"]
            H, W, _ = frame0.shape
            for name, p in zip(names, axis_px_2d):
                u, v = float(p[0]), float(p[1])
                in_bounds = (0 <= u < W) and (0 <= v < H)
                print(f"  {name}: ({u:.1f}, {v:.1f}), in_bounds={in_bounds}")

            frame_axes = frame0.copy()
            def to_int(p):
                return (int(p[0]), int(p[1]))

            o, x, y, z = axis_px_2d
            cv2.circle(frame_axes, to_int(o), 6, (0, 255, 0), -1)
            cv2.line(frame_axes, to_int(o), to_int(x), (255, 0, 0), 2)
            cv2.line(frame_axes, to_int(o), to_int(y), (0, 0, 255), 2)
            cv2.line(frame_axes, to_int(o), to_int(z), (0, 255, 255), 2)

            axes_path = os.path.join(OUT_DIR, f"{demo_name}_debug_base_axes.png")
            cv2.imwrite(axes_path, cv2.cvtColor(frame_axes, cv2.COLOR_RGB2BGR))
            print("saved:", axes_path)

            #---------------------------------------------------------------
            T, H, W, _ = imgs.shape
            print(f"  Frames: {T}, resolution: {H}x{W}")

            #-------------------------------test: 正反 T -------------------------
            test_pos = ee_pos_base[:1, :3]

            px1 = ee_pose_to_cam_pixels(
                ee_pose_base=test_pos,
                T_cam_base=T_cam_base,
                intrinsics=K,
            )
            px2 = ee_pose_to_cam_pixels(
                ee_pose_base=test_pos,
                T_cam_base=np.linalg.inv(T_cam_base),
                intrinsics=K,
            )

            print(f"[{demo_name}] first frame: px1 = {px1}, px2 = {px2}")
            # 你现在在这里 exit(0)，所以下面的视频生成逻辑不会跑
            # 先保留 exit(0) 做 debug，如果想看完整视频，把这行删掉
            #exit(0)

            # ---- 3. 投影到像素坐标 (T, 2) ----
            px_vals = ee_pose_to_cam_pixels(
                ee_pose_base=ee_pos_base,
                T_cam_base=T_cam_base,
                intrinsics=K,
            )

            print("  First 5 projected pixels:")
            for i in range(min(5, T)):
                print(f"    t={i}: {px_vals[i]}")

            # ---- 4. 写视频 ----
            out_path = os.path.join(OUT_DIR, f"{demo_name}_gt_traj.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

            for t in tqdm(range(T), desc=f"Writing {demo_name}"):
                frame = imgs[t]
                px = px_vals[t]

                u, v = float(px[0]), float(px[1])
                in_bounds = (0 <= u < W) and (0 <= v < H)
                if not in_bounds and t < 10:
                    print(f"  [WARN] t={t} pixel out of bounds: ({u:.1f}, {v:.1f})")

                frame_with_gt = draw_dot_on_frame(
                    frame,
                    (u, v),
                    show=False,
                    dot_size=5,
                )

                text = f"t={t}  ({u:.1f}, {v:.1f})"
                frame_with_gt = cv2.putText(
                    frame_with_gt,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                frame_bgr = cv2.cvtColor(frame_with_gt.astype(np.uint8), cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            writer.release()
            print(f"Saved GT video to: {out_path}")


if __name__ == "__main__":
    main()
