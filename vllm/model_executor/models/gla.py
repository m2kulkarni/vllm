from typing import Iterable, List, Optional, Tuple

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
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, PPMissingLayer, make_empty_intermediate_tensors_factory,
    make_layers, maybe_prefix)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           RowvLLMParameter)
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput


class GLAMLP(nn.Module):

    def __init__(self, ):
        pass
    def forward(self, ):
        pass

class GLAAttention(nn.Module):
    
    def __init__(self, ):
        pass
    def forward(self, ):
        pass

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


