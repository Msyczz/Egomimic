import h5py
import numpy as np
import os

# ======== 需要改这两项 ========
HDF5_PATH = "/home/symu/data/mytask_2.hdf5"    # 你的训练用 h5
DEMO_NAME = "demo_0"                          # 你想看的 demo 名
# =================================


def main():
    assert os.path.exists(HDF5_PATH), f"{HDF5_PATH} 不存在"

    with h5py.File(HDF5_PATH, "r") as f:
        assert "data" in f, "HDF5 里没有 'data' group"
        data_group = f["data"]

        if DEMO_NAME is not None:
            assert DEMO_NAME in data_group, f"data 里没有 {DEMO_NAME}"
            demo = data_group[DEMO_NAME]
        else:
            # 没指定就随便拿第一个
            first_demo_name = list(data_group.keys())[0]
            demo = data_group[first_demo_name]
            print(f"[INFO] 未指定 DEMO_NAME，使用 {first_demo_name}")
        
        obs = demo["obs"]
        print("obs keys:", list(obs.keys()))
        print("demo keys:", list(demo.keys()))

        # 1. 取关节真值
        assert "joint_positions" in obs, "obs 里没有 'joint_positions'"
        qpos = obs["joint_positions"][:]       # (T, D)
        T, D = qpos.shape
        print(f"joint_positions shape: {qpos.shape}")  # (T, D)

        # 2. 取动作 label（你在训练里用的应该是这个）
        #    下面的键名请按实际 HDF5 修改，
        action_key_candidates = [
            "actions_joints_act",
            "actions",
            "actions_joint",  # whatever你自己实际写的
        ]
        act_key = None
        for k in action_key_candidates:
            if k in demo:
                act_key = k
                break
        assert act_key is not None, f"demo 里找不到这些动作键: {action_key_candidates}"

        actions = demo[act_key][:]   # 形状可能是 (T, H, D) 或 (T, D) 等
        print(f"{act_key} shape:", actions.shape)

        # 3. 理解 actions 的维度含义
        if actions.ndim == 3:
            T_a, H, D_a = actions.shape
            print(f"{act_key} interpreted as: (T={T_a}, horizon={H}, D={D_a})")
        elif actions.ndim == 2:
            T_a, D_a = actions.shape
            H = 1
            print(f"{act_key} interpreted as: (T={T_a}, D={D_a}), no horizon")
        else:
            raise ValueError(f"不支持的 actions 维度: {actions.shape}")

        # 4. 检查时间长度是否对齐
        print("\n=== 时间维度检查 ===")
        print(f"T (joint_positions) = {T},  T_actions = {actions.shape[0]}")
        if actions.shape[0] != T:
            print("⚠️  警告：actions 的时间长度和 joint_positions 不一样，很可能有对齐问题")

        # 5. 抽几个时间步，直接数值对比
        print("\n=== 抽样对比 joint_positions vs actions 第一步 ===")
        indices = np.linspace(0, min(T - 1, actions.shape[0] - 1), num=min(5, T), dtype=int)
        print("sample indices:", indices)

        for t in indices:
            q_t = qpos[t]              # (D,)
            if actions.ndim == 3:
                a_t0 = actions[t, 0]   # 只看 horizon 的第一步
            else:
                a_t0 = actions[t]

            d = a_t0 - q_t[: a_t0.shape[0]]
            print(f"\n[t={t}]")
            print("  qpos[t][:6]   =", np.round(q_t[:6], 4))
            print("  action[t,0][:6] =", np.round(a_t0[:6], 4))
            print("  diff[:6]      =", np.round(d[:6], 4))

        # 6. 如果你怀疑 action 是未来帧，可以再对比 qpos[t+1] / qpos[t+k]
        print("\n=== 额外检查：假设 action 对应 qpos[t+1] ===")
        for t in indices:
            if t + 1 >= T:
                continue
            q_next = qpos[t + 1]
            if actions.ndim == 3:
                a_t0 = actions[t, 0]
            else:
                a_t0 = actions[t]
            d_next = a_t0 - q_next[: a_t0.shape[0]]
            print(f"\n[t={t}] (对比 qpos[t+1])")
            print("  qpos[t+1][:6] =", np.round(q_next[:6], 4))
            print("  action[t,0][:6] =", np.round(a_t0[:6], 4))
            print("  diff_next[:6] =", np.round(d_next[:6], 4))


if __name__ == "__main__":
    main()
