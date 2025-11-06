import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


CONFIG_FILENAME = "svd_config.json"
ADAPTER_FILENAME = "adapter_model.bin"


@dataclass
class SVDLinearConfig:
    num_groups: int = 4
    selected_group: int = 1
    adapter_dim: Optional[int] = None
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None

    def to_dict(self) -> Dict[str, Optional[int]]:
        data = {
            "num_groups": self.num_groups,
            "selected_group": self.selected_group,
        }
        if self.adapter_dim is not None:
            data["adapter_dim"] = self.adapter_dim
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Optional[int]]) -> "SVDLinearConfig":
        return cls(
            num_groups=data.get("num_groups", 4),
            selected_group=data.get("selected_group", 1),
            adapter_dim=data.get("adapter_dim"),
        )


class LinearSVDAdapter(nn.Module):
    """Wraps a linear layer with a frozen base weight and an SVD-based adapter."""

    def __init__(self, linear: nn.Linear, config: SVDLinearConfig):
        super().__init__()
        self.base_linear = linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        weight = linear.weight.data
        device = config.device or weight.device
        dtype = config.dtype or weight.dtype
        weight = weight.to(device=device, dtype=dtype)

        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        total_rank = s.shape[0]
        num_groups = max(config.num_groups, 1)
        group_size = max(total_rank // num_groups, 1)
        group_index = max(min(config.selected_group, num_groups - 1), 0)
        start = group_index * group_size
        end = total_rank if group_index == num_groups - 1 else (group_index + 1) * group_size

        u_slice = u[:, start:end].contiguous()
        s_slice = s[start:end].contiguous()
        vh_slice = vh[start:end, :].contiguous()

        # Default adapter width equals the size of the selected singular-value group
        # (i.e. roughly min(in_features, out_features) / num_groups).
        adapter_dim = config.adapter_dim or u_slice.shape[1]
        adapter_dim = min(adapter_dim, u_slice.shape[1])
        if adapter_dim <= 0:
            raise ValueError("Adapter dimension must be positive")

        self.register_buffer("svd_u", u_slice[:, :adapter_dim], persistent=False)
        self.register_buffer("svd_sigma", s_slice[:adapter_dim], persistent=False)
        self.register_buffer("svd_v", vh_slice[:adapter_dim, :].transpose(0, 1), persistent=False)

        self.register_parameter(
            "svd_linear_adapter",
            nn.Parameter(torch.zeros(adapter_dim, adapter_dim, device=device, dtype=dtype)),
        )

    @property
    def adapter_dim(self) -> int:
        return self.svd_linear_adapter.shape[0]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(input)
        if self.svd_linear_adapter.numel() == 0:
            return base_out

        hidden = input @ self.svd_v  # (..., adapter_dim)
        hidden = hidden * self.svd_sigma
        hidden = hidden @ self.svd_linear_adapter.t()
        delta = hidden @ self.svd_u.t()
        return base_out + delta


class HeadAdapter(nn.Module):
    """Adapter applied before computing QKV projections in attention."""

    def __init__(self, num_heads: int, head_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.register_parameter(
            "svd_head_adapter",
            nn.Parameter(torch.zeros(num_heads, head_dim, head_dim, device=device, dtype=dtype)),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.svd_head_adapter.numel() == 0:
            return hidden_states
        batch, seq_len, hidden_dim = hidden_states.shape
        reshaped = hidden_states.view(batch, seq_len, self.num_heads, self.head_dim)
        adapted = torch.einsum("bsnh,nhd->bsnd", reshaped, self.svd_head_adapter)
        adapted = adapted.reshape(batch, seq_len, hidden_dim)
        return hidden_states + adapted


def _replace_module(parent: nn.Module, child_name: str, new_module: nn.Module):
    setattr(parent, child_name, new_module)


def _iterate_named_linears(model: nn.Module):
    for _, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                yield module, child_name, child


def apply_linear_svd_adapters(model: nn.Module, config: SVDLinearConfig) -> List[str]:
    replaced = []
    for parent, child_name, linear in _iterate_named_linears(model):
        weight = getattr(linear, "weight", None)
        if weight is None or weight.ndim < 2 or weight.numel() == 0:
            continue
        adapter = LinearSVDAdapter(linear, config)
        _replace_module(parent, child_name, adapter)
        replaced.append(child_name)
    return replaced


def _resolve_attention_dtype_device(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
    linear_module = getattr(module, "q_proj", None)
    weight = None
    if isinstance(linear_module, LinearSVDAdapter):
        weight = linear_module.base_linear.weight
    elif isinstance(linear_module, nn.Linear):
        weight = linear_module.weight
    if weight is None:
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        device = weight.device
        dtype = weight.dtype
    return device, dtype


def attach_attention_head_adapters(model: nn.Module) -> Dict[str, HeadAdapter]:
    adapters: Dict[str, HeadAdapter] = {}
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
        attention_cls = LlamaAttention
    except Exception:
        attention_cls = None

    def _pre_hook(module: nn.Module, args, kwargs):
        adapter_module: HeadAdapter = getattr(module, "svd_head_adapter_module")
        if "hidden_states" in kwargs and kwargs["hidden_states"] is not None:
            kwargs["hidden_states"] = adapter_module(kwargs["hidden_states"])
            return args, kwargs
        elif len(args) > 0:
            new_hidden = adapter_module(args[0])
            new_args = (new_hidden,) + args[1:]
            return new_args, kwargs
        return args, kwargs

    for module_name, module in model.named_modules():
        if attention_cls is not None and isinstance(module, attention_cls):
            num_heads = module.config.num_attention_heads
            head_dim = module.head_dim
        elif hasattr(module, "head_dim") and hasattr(module, "config") and hasattr(module.config, "num_attention_heads"):
            num_heads = module.config.num_attention_heads
            head_dim = module.head_dim
        else:
            continue
        device, dtype = _resolve_attention_dtype_device(module)
        adapter = HeadAdapter(num_heads, head_dim, device=device, dtype=dtype)
        module.add_module("svd_head_adapter_module", adapter)
        module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        adapters[module_name] = adapter
    return adapters


def apply_svd_tuning(model: nn.Module, config: SVDLinearConfig) -> Dict[str, List[str]]:
    replaced_linears = apply_linear_svd_adapters(model, config)
    head_adapters = attach_attention_head_adapters(model)
    return {"linears": replaced_linears, "attentions": list(head_adapters.keys())}


def collect_svd_state_dict(model: nn.Module, require_grad_only: bool = True) -> Dict[str, torch.Tensor]:
    state = {}
    for name, param in model.named_parameters():
        if "svd_" not in name:
            continue
        if require_grad_only and not param.requires_grad:
            continue
        state[name] = param.detach().cpu()
    return state


def collect_non_svd_state_dict(model: nn.Module, require_grad_only: bool = True) -> Dict[str, torch.Tensor]:
    state = {}
    for name, param in model.named_parameters():
        if "svd_" in name:
            continue
        if require_grad_only and not param.requires_grad:
            continue
        state[name] = param.detach().cpu()
    return state


def save_svd_adapters(model: nn.Module, save_directory: str, config: SVDLinearConfig):
    os.makedirs(save_directory, exist_ok=True)
    state = collect_svd_state_dict(model, require_grad_only=True)
    torch.save(state, os.path.join(save_directory, ADAPTER_FILENAME))
    with open(os.path.join(save_directory, CONFIG_FILENAME), "w") as f:
        json.dump(config.to_dict(), f)


def load_svd_config(load_path: str) -> SVDLinearConfig:
    if os.path.isdir(load_path):
        config_path = os.path.join(load_path, CONFIG_FILENAME)
    else:
        config_path = load_path
    with open(config_path, "r") as f:
        data = json.load(f)
    return SVDLinearConfig.from_dict(data)


def load_svd_adapters(model: nn.Module, load_path: str):
    if os.path.isdir(load_path):
        state_path = os.path.join(load_path, ADAPTER_FILENAME)
    else:
        state_path = load_path
    state = torch.load(state_path, map_location="cpu")
    for name, param in model.named_parameters():
        if name in state:
            param.data.copy_(state[name].to(param.device, dtype=param.dtype))


def filter_svd_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in state_dict.items() if "svd_" in k}
