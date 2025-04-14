import cv2
import numpy as np
def process_image(np_img, device, scale=0.5):
    """
    將輸入的 numpy 陣列圖片縮放並轉換為新的 numpy 陣列。
    
    參數:
      np_img: numpy array, shape (H, W, 3)，原始圖片 (H 高度, W 寬度, 3 顏色通道)
      scale: 尺寸縮放比例 (256*scale -> 128, scale=0.5)
    
    返回:
      np_img_resized: numpy array, shape (new_H, new_W, 3)，縮放後的圖片
    """
    # 轉換為 float 並歸一化
    img = np_img.astype(np.float32) / 255.0  # 像素範圍從 [0, 255] -> [0.0, 1.0]
    
    # 縮放尺寸到期望大小, 例如 (256, 256) -> (128, 128)
    new_H = int(img.shape[0] * scale)
    new_W = int(img.shape[1] * scale)
    
    # 使用 cv2 來進行圖像縮放
    img_resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    
    # 恢復到 [0, 255] 範圍並轉換為 uint8
    img_resized = np.clip(img_resized * 255.0, 0, 255).astype(np.uint8)
    
    return np.transpose(img_resized, (2, 0, 1))  # (H, W, C) -> (C, H, W)

from scipy.ndimage import zoom

def resize_point_cloud(pc, scale=0.5):
    """
    將單張點雲圖片（numpy array, shape (256,256,3)）縮小到 (128,128,3)
    使用 scipy.ndimage.zoom 進行線性插值。
    """
    # scale factors: height and width 乘上 scale，通道保持1
    scale_factors = (scale, scale, 1)
    pc_resized = zoom(pc, scale_factors, order=1)  # order=1 為線性插值
    return np.transpose(pc_resized, (2, 0, 1))