from typing import Dict, Iterable, List, Optional, Tuple, Unpack

import torch
import torch.nn.functional as F
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.fla.gla import fused_recurrent_gla
from vllm.model_executor.layers.linear import (ColumnParallelLinear, 
                                               RowParallelLinear)

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, row_parallel_weight_loader)
from vllm.model_executor.models.interfaces import (HasInnerState,
                                                   IsAttentionFree)
from vllm.model_executor.models.mamba_cache import (MambaCacheManager,
                                                    MambaCacheParams)
from vllm.model_executor.layers.activation import SiluAndMul, get_act_fn
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, PPMissingLayer, make_empty_intermediate_tensors_factory,
    make_layers, maybe_prefix)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           RowvLLMParameter)
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

class GLAMLP_mohit(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 hidden_ratio: Optional[int] = None,
                 intermediate_size: Optional[int] = None,
                 hidden_act: str = "silu",
                 config = None):
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        
        # TODO: Can add RowParallelLinear here
        self.gate_proj = nn.Linear(self.hidden_size,
                                           self.intermediate_size * 2,
                                           bias = False)
        self.down_proj = nn.Linear(self.intermediate_size,
                                           self.hidden_size,
                                           bias = False)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, 
                x: torch.Tensor,
                **kwargs: Unpack[Dict]) -> torch.Tensor:
        y = self.gate_proj(x)
        # SiluAndMul already applies chunking
        return self.down_proj(self.act_fn(y))
        
class GatedLinearAttention(nn.Module):
    
    def __init__(self, 
                 config,
                 layer_id=None):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.layer_idx = layer_id

        self.key_dim = int(self.hidden_size * config.expand_k)
        self.value_dim = int(self.hidden_size * config.expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.head_dim = self.key_dim // self.num_heads
        self.clamp_min = config.clamp_min

        # Why are we gathering output?
        self.q_proj = ColumnParallelLinear(self.hidden_size,
                                           self.key_dim,
                                           bias=False,
                                           gather_output=True)

        self.k_proj = ColumnParallelLinear(self.hidden_size,
                                           self.key_dim_per_group,
                                           bias=False,
                                           gather_output=True)

        self.q_proj = ColumnParallelLinear(self.hidden_size,
                                           self.value_dim_per_group,
                                           bias=False,
                                           gather_output=True)
        
        # gate projections
        self.g_proj = ColumnParallelLinear(self.hidden_size,
                                           self.value_dim,
                                           bias=False, 
                                           gather_output=True)
        
        if config.use_output_gate:
            self.g_proj = ColumnParallelLinear(self.hidden_size,
                                               self.value_dim,
                                               bias=False,
                                               gather_output=True)
        else:
            self.g_proj = None
        
        gate_low_rank_dim = 16
        self.gk_proj = nn.Sequential(
            ColumnParallelLinear(self.hidden_size, gate_low_rank_dim, bias=False),
            ColumnParallelLinear(gate_low_rank_dim, self.key_dim_per_group, bias=False)
        )

        self.o_proj = RowParallelLinear(self.value_dim,
                                        self.hidden_size,
                                        bias = False)
        
        self.norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.gate_fn = F.silu if config.hidden_act == "swish" else getattr(F, config.hidden_act)
        self.gate_logit_normalizer = 16
        



    def forward(self, 
                hidden_states,
                kv_cache: MambaCacheManager,
                positions: torch.Tensor = None):
        
        # hidden_states: [batch_size, hidden_dim]
        bsz, hidden_dim = hidden_states.size()

        input_state = kv_cache.ssm_state
        input_state = input_state[kv_cache.state_indices_tensor]

        assert bsz % kv_cache.state_indices_tensor.size(0) == 0, (
            f"Batch size {bsz} is not divisible by the number of "
            f"state indices {kv_cache.state_indices_tensor.size(0)}")
        
        qlen = bsz // kv_cache.state_indices_tensor.size(0)
        bsz = kv_cache.state_indices_tensor.size(0)
        x = hidden_states.view(bsz, hidden_dim, -1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.gk_proj_2(self.gk_proj_1(x))

        gk = F.logsigmoid(g) / self.gate_logit_normalizer



class GLABlock(nn.Module):
    
    def __init__(self, ):
        pass
    def forward(self, ):
        pass

class GLAModel(nn.Module):
    
    def __init__(self, ):
        pass
    def forward(self, ):
        pass

class GLAForCausalLM(nn.Module, HasInnerState, IsAttentionFree):
    
    def __init__(self, ):
        pass
    def forward(self, ):
        pass


