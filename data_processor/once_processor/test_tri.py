#show_pkl.py
 
import pickle
import json
import numpy as np

pkl_path = '/mnt/data/dataset/once/processed/000121/track/track_info.pkl'      # 原始 pkl 文件
json_path = '/mnt/data/dataset/once/processed/000121/track/track_info.json'    # 要保存的 json 文件

'''
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# 2. 写入（ensure_ascii=False 保留中文/特殊字符，indent=4 缩进）
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f'JSON 已生成：{json_path}')
'''
# --------- 1. 递归转换函数 ---------
def numpy_to_py(obj):
    """把可能遇到的 numpy 类型全部转成原生 Python 类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: numpy_to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_py(item) for item in obj]
    return obj

# --------- 2. 读取 pkl ---------
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# --------- 3. 清洗 + 写 json ---------
clean_data = numpy_to_py(data)
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(clean_data, f, ensure_ascii=False, indent=4)

print(f'JSON 已生成：{json_path}')