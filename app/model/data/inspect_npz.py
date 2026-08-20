import numpy as np

files = [
    "train_tensor_2012_2024.npz",
    "train_tensor_kerala.npz",
    "validation_2025.npz",
    "validation_tensor_kerala.npz",
]

for f in files:
    try:
        npz = np.load(f, allow_pickle=True)
        print(f"\n{f}")
        print("  keys:", npz.files)
        print("  tensor shape:", npz["tensor"].shape)
        print("  date range:", npz["dates"][0], "to", npz["dates"][-1])
    except Exception as e:
        print(f"\n{f}  ERROR: {e}")
