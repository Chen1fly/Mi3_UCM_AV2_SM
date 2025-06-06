import pickle
import pandas as pd

# 1. 从 pkl 文件里加载对象
with open(r'E:\refav\RefAV\run\output\sm_dataset\val\f3cd0d0d-8b71-3266-9732-d9f0d5778eb6\school bus on intersection_f3cd0d0d_ref_gt.pkl', 'rb') as f:
    obj = pickle.load(f)

def print_dict_contents(d, indent=0, sample_size=1):
    """
    递归打印 dict 的键和值（仅展示前 sample_size 条）：
    - 如果 value 是 DataFrame，则打印它的列名并展示每列前 sample_size 条数据。
    - 如果 value 是 dict，则进一步递归。
    - 如果 value 是列表/元组，则打印其前 sample_size 个元素。
    - 否则直接打印 value。
    """
    prefix = ' ' * indent
    for key, value in d.items():
        print(f"{prefix}Key: {key}")
        # DataFrame
        if isinstance(value, pd.DataFrame):
            df = value
            print(f"{prefix}  (DataFrame with {len(df)} rows and {len(df.columns)} columns)")
            for col in df.columns:
                vals = df[col].tolist()
                # 只取前 sample_size 条
                sample = vals[:sample_size]
                print(f"{prefix}    Column '{col}': {sample}{' ...' if len(vals) > sample_size else ''}")
        # 嵌套 dict
        elif isinstance(value, dict):
            print(f"{prefix}  (Nested dict)")
            print_dict_contents(value, indent + 4, sample_size)
        # list 或 tuple
        elif isinstance(value, (list, tuple)):
            vals = list(value)
            sample = vals[:sample_size]
            print(f"{prefix}  (list/tuple, length={len(vals)}): {sample}{' ...' if len(vals) > sample_size else ''}")
        # 其他类型
        else:
            print(f"{prefix}  Value: {value!r}")

# 根据 obj 类型分发打印
if isinstance(obj, dict):
    print_dict_contents(obj, sample_size=3)
else:
    print(f"Loaded object type: {type(obj)}")
    print(obj)
