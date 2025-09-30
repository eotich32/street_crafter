import cv2
import numpy as np
import json

cam = 'cam01'
idx = 0
img_path = f"/mnt/data/dataset/once/raw/000027/cam01/1616100800400.jpg"
json_path = f"/mnt/data/dataset/once/raw/000027/000027.json"  # 你实际的 json 名字

with open(json_path) as f:
    calib = json.load(f)['calib'][cam]

K = np.array(calib['cam_intrinsic'], dtype=np.float64)
D = np.array(calib['distortion'][:5], dtype=np.float64)  # 取 5 项
img = cv2.imread(img_path)
h, w = img.shape[:2]
new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
undist = cv2.undistort(img, K, D, None, new_K)
print('----------- 内参 -----------')
print('K = \n', K)
print('D = ', D)
print('img.shape = ', (h, w))
print('----------------------------')

cv2.imwrite('test_undist.png', undist)
print('saved: test_undist.png')