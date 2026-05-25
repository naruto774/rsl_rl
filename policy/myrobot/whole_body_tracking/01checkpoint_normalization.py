import torch
ckpt = torch.load("best_agent.pt", map_location="cpu", weights_only=False)
print(list(ckpt.keys()))
# 一般会看到: ['model_state_dict', 'optimizer_state_dict', 'iter', ...]
# 重点关注是否有 'obs_norm_state_dict' 或 'normalizer' 字段
if "state_preprocessor" in ckpt:
    n = ckpt["state_preprocessor"]
    print("running_mean shape:", n["running_mean"].shape, "head:", n["running_mean"].flatten()[:6])
    print("running_variance  shape:", n["running_variance"].shape,  "head:", n["running_variance"].flatten()[:6])