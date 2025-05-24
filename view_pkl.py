import pickle
import numpy as np

# 指定要查看的pkl文件路径
pkl_file_path = './dataset/ts_mon.pkl'  # 根据您的实际路径调整

# 加载pkl文件
with open(pkl_file_path, 'rb') as handle:
    data = pickle.load(handle)

# 如果数据是字典类型
if isinstance(data, dict):
    print(f"数据类型: 字典，包含 {len(data)} 个键")
    
    total_length = 0
    total_samples = 0
    
    # 遍历每个键，计算样本总长度
    max_len = 0
    for site_id, samples in data.items():
        for sample in samples:
            max_len = max(max_len, len(sample))
            total_length += len(sample)
            total_samples += 1
    
    # 计算平均长度
    average_length = total_length / total_samples if total_samples > 0 else 0
    print(f"总样本数: {total_samples}")
    print(f"平均样本长度: {average_length}")
    print(f"最大样本长度: {max_len}")
    # 打印前几个键
#     print("\n前几个键:")
#     for i, key in enumerate(list(data.keys())[:5]):
#         print(f"键 {i}: {key}")
    
#     # 选择第一个键，查看其值的结构
#     first_key = list(data.keys())[0]
#     print(f"\n键 '{first_key}' 的值类型: {type(data[first_key])}")
    
#     # 如果值是列表，显示其长度和前几个元素
#     if isinstance(data[first_key], list):
#         print(f"列表长度: {len(data[first_key])}")
#         print("前几个元素:")
#         max_len = 0
#         for i, item in enumerate(data[first_key]):
#             print(f"元素 {i}: {type(item)}")
#             # 如果元素是数组或列表，显示其形状或长度和前几个值
#             if isinstance(item, (list, np.ndarray)):
#                 max_len = max(max_len, len(item))
#                 print(f"  长度: {len(item)}")
#                 print(f"  前10个值: {item[:10]}")
#         print(f"最大长度: {max_len}")

# # 如果数据是数组类型
# elif isinstance(data, (list, np.ndarray)):
#     print(f"数据类型: {'列表' if isinstance(data, list) else 'NumPy数组'}")
#     print(f"长度: {len(data)}")
#     print("前几个元素:")
#     for i, item in enumerate(data[:5]):
#         print(f"元素 {i}: {type(item)}")
#         # 如果元素是数组或列表，显示其形状或长度
#         if isinstance(item, (list, np.ndarray)):
#             print(f"  长度: {len(item)}")
#             print(f"  前10个值: {item[:10]}")

# else:
#     print(f"数据类型: {type(data)}")
#     print(data)