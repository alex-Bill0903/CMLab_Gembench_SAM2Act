import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_and_save(np_path: Path):
    """
    讀取 .npy 檔並對其 obs/rgb 做覆蓋標記，
    將四個視角分別存成四張 png 並放在與 npy 相同資料夾
    """
    data = np.load(np_path, allow_pickle=True).item()
    obs  = data["obs"]
    action_full = data["action"]
    # 只取 XYZ
    action_xyz  = np.array(action_full[:3], dtype=float)

    # stack rgb and point clouds
    rgbs = np.stack(obs["rgb"], axis=0)  # (4, H, W, 3)
    pcs  = np.stack(obs["pc"],  axis=0)  # (4, H, W, 3)

    folder = np_path.parent
    stem   = np_path.stem  # f'{task}+{variation}-{episode}-{step}'

    for i in range(4):
        rgb_img = rgbs[i]
        pc_img  = pcs[i]

        # 計算每像素點到 action_xyz 的距離
        flat_pc = pc_img.reshape(-1, 3)
        dists   = np.linalg.norm(flat_pc - action_xyz, axis=1)
        idx     = np.argmin(dists)
        h, w, _ = pc_img.shape
        row, col = divmod(idx, w)

        # 繪製並存檔
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(rgb_img.astype(np.uint8))
        ax.scatter(col, row, s=80, marker='x')
        ax.axis('off')

        out_name = f"{stem}_view{i}.png"
        out_path = folder / out_name
        fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def batch_process(base_dir: Path):
    """
    遞迴尋找 base_dir 下所有 .npy，並呼叫 visualize_and_save
    """
    for np_path in base_dir.rglob("*.npy"):
        try:
            visualize_and_save(np_path)
            # print(f"Processed: {np_path}")
        except Exception as e:
            print(f"Failed for {np_path}: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Batch visualize .npy obs/action to 4 view PNGs"
    )
    parser.add_argument(
        'base_dir', type=str,
        nargs='?',  # make positional argument optional
        default='/home/bill/Documents/research/CVPR_gembench_baseline/robot-3dlotus/data/experiments/gembench/3dlotus/v1/preds/seed200/record_action',
        help='Record_action base folder, will search recursively for .npy files',
    )
    args = parser.parse_args()
    base = Path(args.base_dir)
    if not base.is_dir():
        raise ValueError(f"Base directory not found: {base}")
    batch_process(base)
