# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import enum

class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2

class LayerType(enum.Enum):
    encoder = 1
    decoder = 2
 
class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2

class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2 # Overrides `attention_mask` to be a lower triangular matrix
    prefix = 3
    custom = 4 # Forces one to pass an `attention_mask` that's 1 if we need to mask. Tensor that can be broadcast to [micro_batch_size, n_head, seq_length, seq_length]

class PositionEmbeddingType(enum.Enum):
    rotary = 1 # NOTE: this one is not used so far, however for future compatibility the enum left as is
    absolute = 2
    alibi = 3

class UL2ModelType(enum.Enum):
    encoder_decoder = 'ED'
    non_causal_decoder = 'ND'
    causal_decoder = 'CD'
