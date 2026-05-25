import torch
m = torch.jit.load("policy.pt", map_location="cpu")

print("=== 子模块列表 ===")
for name, _ in m.named_modules():
    print(" ", name)

print("\n=== buffers (mean/var 一般以 buffer 形式存在) ===")
for name, b in m.named_buffers():
    print(f"  {name}: shape={tuple(b.shape)} head={b.flatten()[:6].tolist()}")