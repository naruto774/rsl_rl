"""
Export skrl checkpoint directly to self-contained TorchScript policy.pt
and ONNX policy.onnx.

No input policy.pt is required. The script:
  1) reads ckpt["state_preprocessor"] (running_mean / running_variance)
  2) reconstructs actor from ckpt["policy"] state_dict (all Linear layers
     are recovered exactly; the activation between linear layers is
     auto-selected from {ELU, Tanh, ReLU, LeakyReLU, GELU} by L2 to the
     training action recorded in ampobs.csv row 0)
  3) bakes normalization in front of the actor and exports both files:
       - policy.pt   (TorchScript)
       - policy.onnx (ONNX, dynamic batch axis)
       x_hat = clamp((x - mean) / (sqrt(var) + eps), -clip, +clip)
       a     = actor(x_hat)
"""

import os
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional

CKPT_PATH = "best_agent.pt"
JIT_OUT = "policy.pt"
ONNX_OUT = "policy.onnx"
AMPOBS_CSV = "ampobs.csv"
ONNX_OPSET = 17

CANDIDATE_ACTIVATIONS = {
    "ELU": nn.ELU,
    "Tanh": nn.Tanh,
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "GELU": nn.GELU,
}

L2_FAIL_THRESHOLD = 0.5


def get_norm_stats(sp: dict) -> tuple:
    mean_key = "running_mean" if "running_mean" in sp else "mean"
    var_key = (
        "running_variance" if "running_variance" in sp
        else ("running_var" if "running_var" in sp else "var")
    )
    if mean_key not in sp or var_key not in sp:
        raise KeyError(f"state_preprocessor missing mean/var. keys={list(sp.keys())}")
    return sp[mean_key].float().view(-1), sp[var_key].float().view(-1)


def collect_linear_layers(policy_sd: dict, prefix_pattern: str) -> list:
    """Pick out all Linear layers matching <prefix>.{idx}.weight in state_dict, ordered by idx."""
    pattern = re.compile(rf"(?:^|\.){prefix_pattern}\.(\d+)\.weight$")
    linear = {}
    for k, v in policy_sd.items():
        m = pattern.search(k)
        if not m:
            continue
        idx = int(m.group(1))
        bias_key = k[:-len("weight")] + "bias"
        if bias_key not in policy_sd:
            raise KeyError(f"missing bias for {k}")
        linear[idx] = (v.float(), policy_sd[bias_key].float())
    return sorted(linear.items(), key=lambda x: x[0])


def find_extra_head(policy_sd: dict) -> tuple:
    """If the policy has a separate mean head (skrl GaussianMixin pattern),
    pick it up here. Returns (weight, bias) or (None, None) if absent."""
    candidates = [
        "mean_layer", "policy_layer", "action_layer", "mu_head", "mean_net",
        "policy_model.mean_layer", "policy_model.policy_layer",
    ]
    for head_name in candidates:
        wk = f"{head_name}.weight"
        bk = f"{head_name}.bias"
        if wk in policy_sd and bk in policy_sd:
            print(f"[INFO] extra output head detected: {head_name}")
            return policy_sd[wk].float(), policy_sd[bk].float()
    return None, None


def build_actor(policy_sd: dict, activation_cls) -> nn.Module:
    """Reconstruct: net_container(Linear-Activation-Linear-...) + optional mean head."""
    layers = collect_linear_layers(policy_sd, "net_container")
    if not layers:
        layers = collect_linear_layers(policy_sd, "policy_model\\.net_container")
    if not layers:
        raise RuntimeError("No net_container.*.weight found in checkpoint['policy'].")

    head_w, head_b = find_extra_head(policy_sd)

    modules = []
    for i, (_, (w, b)) in enumerate(layers):
        out_f, in_f = w.shape
        fc = nn.Linear(in_f, out_f)
        with torch.no_grad():
            fc.weight.copy_(w)
            fc.bias.copy_(b)
        modules.append(fc)
        is_last_trunk_linear = (i == len(layers) - 1)
        # Insert activation between hidden Linear layers. If a separate head exists,
        # the final trunk linear is also "hidden" and must be followed by activation.
        if (not is_last_trunk_linear) or (head_w is not None):
            modules.append(activation_cls())

    if head_w is not None:
        out_f, in_f = head_w.shape
        fc = nn.Linear(in_f, out_f)
        with torch.no_grad():
            fc.weight.copy_(head_w)
            fc.bias.copy_(head_b)
        modules.append(fc)

    return nn.Sequential(*modules).eval()


class PolicyWithObsNorm(nn.Module):
    def __init__(self, mean: torch.Tensor, var: torch.Tensor, actor: nn.Module,
                 eps: float = 1e-8, clip_threshold: float = 5.0):
        super().__init__()
        self.register_buffer("running_mean", mean.view(1, -1).clone())
        self.register_buffer("running_var", var.view(1, -1).clone())
        self.eps = eps
        self.clip = clip_threshold
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.running_var) + self.eps
        x = (obs - self.running_mean) / std
        x = torch.clamp(x, -self.clip, self.clip)
        return self.actor(x)


def dump_policy_state_dict(policy_sd: dict):
    print("\n=== ckpt['policy'] state_dict structure ===")
    for k, v in policy_sd.items():
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else "?"
        print(f"  {k:55s} {shape}")


def evaluate(wrapped: nn.Module, amp_obs0: torch.Tensor, amp_act0: np.ndarray) -> tuple:
    with torch.no_grad():
        a_pred = wrapped(amp_obs0).squeeze(0).numpy()
    return a_pred, float(np.linalg.norm(a_pred - amp_act0))


def expected_sampling_noise_l2(policy_sd: dict) -> float:
    """
    For skrl GaussianMixin, ampobs may store sampled actions a = mu + sigma * eps.
    The deterministic reconstruction only predicts mu, so the residual L2 lower-bound
    is the expected noise magnitude:
        E[||sigma * eps||_2] ~ sqrt(sum(sigma_i^2))
    where sigma_i = exp(log_std_i). This gives a principled threshold:
        observed_L2 within ~ [0.5, 2.0] * expected_L2 -> reconstruction OK.
    Returns -1.0 if log_std_parameter is absent (e.g. deterministic policy export).
    """
    for k in ("log_std_parameter", "policy_model.log_std_parameter"):
        if k in policy_sd:
            log_std = policy_sd[k].float()
            sigma = torch.exp(log_std)
            return float(torch.sqrt((sigma ** 2).sum()).item())
    return -1.0


def export_policy(
    wrapped: nn.Module,
    obs_dim: int,
    activation_name: str,
    l2: Optional[float] = None,
    verified: bool = True,
) -> None:
    """
    同时导出:
      - policy.pt   : TorchScript（JIT 后端用）
      - policy.onnx : ONNX（ONNX Runtime 后端用）

    关键设计点（与 rl_real_myrobot 中 InferenceRuntime::ONNXModel::forward 对齐）:
      * 部署侧 Ort::Value::CreateTensor 直接使用 ONNX 模型声明的 input shape；
        若 shape 中出现 -1（dynamic axis），ORT 会抛出
        "tried creating tensor with negative value in shape" 并 abort。
      * 实机部署始终是单帧推理 (batch=1)，没有 batch 推理需求，
        所以这里固定 input shape = [1, obs_dim] 导出静态 ONNX。
      * 自检：用 onnxruntime 加载导出的 .onnx，用同一个随机输入与 JIT 比对，
        最大绝对误差 < 1e-4 才视为合格 (ONNX-OK)，否则打印 ONNX-BAD。
    """
    wrapped = wrapped.eval()

    # TorchScript: 与 rl_sdk 的 TorchModel 后端对接
    scripted = torch.jit.script(wrapped)
    torch.jit.save(scripted, JIT_OUT)

    # ONNX: 静态 batch=1，避免 ORT shape inference 异常
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy_obs,
            ONNX_OUT,
            export_params=True,
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
            input_names=["obs"],
            output_names=["action"],
            # 不传 dynamic_axes -> 强制 [1, obs_dim] / [1, act_dim] 全静态
        )

    onnx_ok, onnx_msg = _verify_onnx_against_jit(wrapped, obs_dim)

    l2_msg = f", L2={l2:.6f}" if l2 is not None else ""
    verify_msg = "VERIFIED" if verified else "NOT VERIFIED"
    onnx_status = "ONNX-OK" if onnx_ok else "ONNX-BAD"
    print(
        f"[OK] saved policy -> {JIT_OUT} and {ONNX_OUT} "
        f"(activation={activation_name}{l2_msg}, {verify_msg}, {onnx_status})"
    )
    print(f"     ONNX self-check: {onnx_msg}")


def _verify_onnx_against_jit(wrapped: nn.Module, obs_dim: int) -> tuple:
    """
    用同一份随机 obs 同时跑 PyTorch eager 和 onnxruntime，对比输出。
    返回 (ok, message)。
    - ok=False 说明导出的 .onnx 在 ORT 下不能跑、或与训练侧网络数值不一致，
      不能拿到真机上用。
    - onnxruntime 没装时降级为只检查导出文件本身能被解析。
    """
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        try:
            import onnx  # type: ignore
            onnx.checker.check_model(onnx.load(ONNX_OUT))
            return True, "onnxruntime not installed; static export passed onnx.checker"
        except Exception as e:
            return False, f"onnxruntime missing and onnx.checker failed: {e}"

    try:
        sess = ort.InferenceSession(ONNX_OUT, providers=["CPUExecutionProvider"])
        in_meta = sess.get_inputs()[0]
        in_shape = in_meta.shape

        # 静态 shape 校验：任何 None / 非正整数都说明导出仍含动态轴
        is_static = all(isinstance(d, int) and d > 0 for d in in_shape)
        if not is_static:
            return False, (
                f"input shape not fully static: {in_shape}. "
                "ONNXModel::forward in rl_sar 会直接把这个 shape 丢给 CreateTensor，"
                "出现 -1 / None 时会抛 'tried creating tensor with negative value in shape'."
            )

        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((1, obs_dim)).astype(np.float32)
        y_onnx = sess.run(None, {in_meta.name: x_np})[0]
        with torch.no_grad():
            y_jit = wrapped(torch.from_numpy(x_np)).cpu().numpy()

        max_abs_err = float(np.max(np.abs(y_jit - y_onnx)))
        ok = max_abs_err < 1e-4
        return ok, (
            f"input_shape={in_shape}, output_shape={list(y_onnx.shape)}, "
            f"max|jit-onnx|={max_abs_err:.3e}"
        )
    except Exception as e:
        return False, f"onnxruntime self-check failed: {e}"


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    print("checkpoint keys:", list(ckpt.keys()))
    if "state_preprocessor" not in ckpt or "policy" not in ckpt:
        raise KeyError("checkpoint missing 'policy' or 'state_preprocessor'")

    sp = ckpt["state_preprocessor"]
    print("state_preprocessor keys:", list(sp.keys()))
    running_mean, running_var = get_norm_stats(sp)
    obs_dim = running_mean.numel()
    print(f"obs_dim from norm: {obs_dim}")
    print(f"  mu  head: {running_mean[:6].tolist()}")
    print(f"  var head: {running_var[:6].tolist()}")
    print(f"  count   : {sp.get('current_count', 'NA')}")

    dump_policy_state_dict(ckpt["policy"])

    if not os.path.exists(AMPOBS_CSV):
        print(f"\n[WARN] {AMPOBS_CSV} not found, cannot auto-detect activation."
              " Defaulting to ELU. If sim2sim looks bad, supply ampobs.csv and re-run.")
        actor = build_actor(ckpt["policy"], nn.ELU)
        wrapped = PolicyWithObsNorm(running_mean, running_var, actor).eval()
        export_policy(wrapped, obs_dim, activation_name="ELU", verified=False)
        return

    amp = pd.read_csv(AMPOBS_CSV)
    if amp.shape[1] < obs_dim:
        raise RuntimeError(f"ampobs.csv has {amp.shape[1]} cols, need >= {obs_dim} obs")
    amp_obs0 = torch.tensor(amp.iloc[0, :obs_dim].values, dtype=torch.float32).unsqueeze(0)

    # Probe last Linear out_features to find action_dim used in csv.
    probe_actor = build_actor(ckpt["policy"], nn.ELU)
    act_dim = None
    for m in reversed(list(probe_actor.modules())):
        if isinstance(m, nn.Linear):
            act_dim = m.out_features
            break
    if act_dim is None or amp.shape[1] < obs_dim + act_dim:
        print(f"\n[WARN] ampobs.csv has {amp.shape[1]} cols, expected >= "
              f"{obs_dim}+{act_dim} for action comparison. Skipping L2 check, "
              "defaulting to ELU.")
        wrapped = PolicyWithObsNorm(running_mean, running_var, probe_actor).eval()
        export_policy(wrapped, obs_dim, activation_name="ELU", verified=False)
        return

    amp_act0 = amp.iloc[0, obs_dim:obs_dim + act_dim].values

    print("\n=== auto-selecting activation by L2 to ampobs row 0 ===")
    results = []
    for name, cls in CANDIDATE_ACTIVATIONS.items():
        try:
            actor = build_actor(ckpt["policy"], cls)
            wrapped = PolicyWithObsNorm(running_mean, running_var, actor).eval()
            a_pred, l2 = evaluate(wrapped, amp_obs0, amp_act0)
            print(f"  activation={name:<10s}  L2={l2:.6f}")
            results.append((l2, name, wrapped, a_pred))
        except Exception as e:
            print(f"  activation={name:<10s}  FAILED: {e}")

    if not results:
        raise RuntimeError("All candidate activations failed to build the model.")

    results.sort(key=lambda x: x[0])
    best_l2, best_name, best_wrapped, best_pred = results[0]
    print(f"\n>>> best activation: {best_name}  L2={best_l2:.6f}")
    print("pred :", np.round(best_pred, 4))
    print("train:", np.round(amp_act0, 4))
    diff = np.round(np.abs(best_pred - amp_act0), 4)
    print("|diff|:", diff)

    # Principled threshold from log_std: if observed L2 is within ~2x of the
    # expected sampling noise magnitude, reconstruction is correct and the
    # residual is just from ampobs storing sampled (not deterministic) actions.
    expected_noise_l2 = expected_sampling_noise_l2(ckpt["policy"])
    if expected_noise_l2 > 0:
        ratio = best_l2 / max(expected_noise_l2, 1e-9)
        print(f"\n[NoiseCheck] expected sampling noise L2 = "
              f"sqrt(sum(sigma_i^2)) = {expected_noise_l2:.4f}")
        print(f"[NoiseCheck] observed L2 / expected = {ratio:.3f}")
        if ratio < 2.0:
            print("[NoiseCheck] -> residual is consistent with Gaussian sampling "
                  "noise, reconstruction is OK.")
            verdict_ok = True
        else:
            print("[NoiseCheck] -> observed >> expected, architecture likely wrong.")
            verdict_ok = False
    else:
        # No log_std => fall back to a hard absolute threshold.
        verdict_ok = best_l2 <= L2_FAIL_THRESHOLD

    if not verdict_ok:
        print(
            f"\n[ERROR] L2={best_l2:.3f} too large vs. noise floor."
            " Architecture mismatch is likely (extra heads, different activation,"
            " or different module layout). Refusing to overwrite policy.pt."
        )
        print("Inspect the state_dict dump above and tell which keys/shapes you have.")
        return

    export_policy(
        best_wrapped,
        obs_dim,
        activation_name=best_name,
        l2=best_l2,
        verified=True,
    )


if __name__ == "__main__":
    main()
