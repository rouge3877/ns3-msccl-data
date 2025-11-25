########## Generated Model Code ###########
from typing import *
from pathlib import Path
import torch
import torch.utils.checkpoint as ckpt
import nnscaler
import nnscaler.flags
import _operator
from numpy import inf
import builtins

runtime_version = '0.7'


import nnscaler.graph.function.wrapnn

import apex.normalization.fused_layer_norm

import modeling.modeling_deepseek_modifier

class GenModel(nnscaler.runtime.module.ParallelModule):
    use_scheduler = False
    nmicros_per_scheduler_step = 1
    rank = 6
    world_size = 8
    
    def __init__(self, init_params=True, build_buckets=True, *args, async_op=False, max_bucket_size_bytes=None, zero_use_reduce_scatter=False, **kwargs):
        super().__init__()
        # communication groups
        self.init_group(ranks=[0, 1, 2, 3, 4, 5, 6, 7])
         
        self.register_parameter('model_model_embed_tokens_weight_12833', torch.nn.Parameter(torch.empty((102400, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_embed_tokens_weight_12833', 18, True, 'model.model.embed_tokens.weight', (102400, 2048), (slice(0, 102400, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_0_input_layernorm_weight_8008', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_0_input_layernorm_weight_8008', 22, True, 'model.model.layers.0.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_0_self_attn_q_proj_weight_8010', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_0_self_attn_q_proj_weight_8010', 36, True, 'model.model.layers.0.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_0_self_attn_kv_a_proj_with_mqa_weight_13065', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_0_self_attn_kv_a_proj_with_mqa_weight_13065', 52, True, 'model.model.layers.0.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_0_self_attn_kv_a_layernorm_weight_8022', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_0_self_attn_kv_a_layernorm_weight_8022', 68, True, 'model.model.layers.0.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_0_self_attn_kv_b_proj_weight_8024', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_0_self_attn_kv_b_proj_weight_8024', 76, True, 'model.model.layers.0.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_0_self_attn_rotary_emb_cos_cached_8030', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_0_self_attn_rotary_emb_cos_cached_8030', 98, False, 'model.model.layers.0.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_0_self_attn_rotary_emb_sin_cached_8033', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_0_self_attn_rotary_emb_sin_cached_8033', 106, False, 'model.model.layers.0.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_0_self_attn_o_proj_weight_8071', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_0_self_attn_o_proj_weight_8071', 247, True, 'model.model.layers.0.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_0_post_attention_layernorm_weight_8074', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_0_post_attention_layernorm_weight_8074', 253, True, 'model.model.layers.0.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_0_mlp_gate_proj_weight_8076', torch.nn.Parameter(torch.empty((10944, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_0_mlp_gate_proj_weight_8076', 261, True, 'model.model.layers.0.mlp.gate_proj.weight', (10944, 2048), (slice(0, 10944, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_0_mlp_up_proj_weight_8079', torch.nn.Parameter(torch.empty((10944, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_0_mlp_up_proj_weight_8079', 267, True, 'model.model.layers.0.mlp.up_proj.weight', (10944, 2048), (slice(0, 10944, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_0_mlp_down_proj_weight_8082', torch.nn.Parameter(torch.empty((2048, 10944), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_0_mlp_down_proj_weight_8082', 273, True, 'model.model.layers.0.mlp.down_proj.weight', (2048, 10944), (slice(0, 2048, None), slice(0, 10944, None)), 1)
        
        self.register_parameter('model_model_layers_1_input_layernorm_weight_8085', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_input_layernorm_weight_8085', 279, True, 'model.model.layers.1.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_1_self_attn_q_proj_weight_8087', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_self_attn_q_proj_weight_8087', 293, True, 'model.model.layers.1.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_1_self_attn_kv_a_proj_with_mqa_weight_14817', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_self_attn_kv_a_proj_with_mqa_weight_14817', 309, True, 'model.model.layers.1.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_1_self_attn_kv_a_layernorm_weight_8099', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_self_attn_kv_a_layernorm_weight_8099', 325, True, 'model.model.layers.1.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_1_self_attn_kv_b_proj_weight_8101', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_self_attn_kv_b_proj_weight_8101', 333, True, 'model.model.layers.1.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_1_self_attn_rotary_emb_cos_cached_8107', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_1_self_attn_rotary_emb_cos_cached_8107', 355, False, 'model.model.layers.1.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_1_self_attn_rotary_emb_sin_cached_8110', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_1_self_attn_rotary_emb_sin_cached_8110', 363, False, 'model.model.layers.1.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_1_self_attn_o_proj_weight_8148', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_self_attn_o_proj_weight_8148', 504, True, 'model.model.layers.1.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_1_post_attention_layernorm_weight_8151', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_post_attention_layernorm_weight_8151', 510, True, 'model.model.layers.1.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_1_mlp_gate_weight_8153', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_mlp_gate_weight_8153', 518, True, 'model.model.layers.1.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_1_mlp_gate_projs_16145', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_mlp_gate_projs_16145', 531, True, 'model.model.layers.1.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_1_mlp_up_projs_16153', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_mlp_up_projs_16153', 533, True, 'model.model.layers.1.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_1_mlp_down_projs_16161', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_mlp_down_projs_16161', 535, True, 'model.model.layers.1.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_1_mlp_shared_experts_gate_proj_weight_8161', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_mlp_shared_experts_gate_proj_weight_8161', 539, True, 'model.model.layers.1.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_1_mlp_shared_experts_up_proj_weight_8164', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_mlp_shared_experts_up_proj_weight_8164', 545, True, 'model.model.layers.1.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_1_mlp_shared_experts_down_proj_weight_8167', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_1_mlp_shared_experts_down_proj_weight_8167', 551, True, 'model.model.layers.1.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_2_input_layernorm_weight_8171', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_input_layernorm_weight_8171', 559, True, 'model.model.layers.2.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_2_self_attn_q_proj_weight_8173', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_self_attn_q_proj_weight_8173', 573, True, 'model.model.layers.2.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_2_self_attn_kv_a_proj_with_mqa_weight_16689', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_self_attn_kv_a_proj_with_mqa_weight_16689', 589, True, 'model.model.layers.2.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_2_self_attn_kv_a_layernorm_weight_8185', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_self_attn_kv_a_layernorm_weight_8185', 605, True, 'model.model.layers.2.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_2_self_attn_kv_b_proj_weight_8187', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_self_attn_kv_b_proj_weight_8187', 613, True, 'model.model.layers.2.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_2_self_attn_rotary_emb_cos_cached_8193', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_2_self_attn_rotary_emb_cos_cached_8193', 635, False, 'model.model.layers.2.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_2_self_attn_rotary_emb_sin_cached_8196', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_2_self_attn_rotary_emb_sin_cached_8196', 643, False, 'model.model.layers.2.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_2_self_attn_o_proj_weight_8234', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_self_attn_o_proj_weight_8234', 784, True, 'model.model.layers.2.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_2_post_attention_layernorm_weight_8237', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_post_attention_layernorm_weight_8237', 790, True, 'model.model.layers.2.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_2_mlp_gate_weight_8239', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_mlp_gate_weight_8239', 798, True, 'model.model.layers.2.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_2_mlp_gate_projs_18017', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_mlp_gate_projs_18017', 811, True, 'model.model.layers.2.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_2_mlp_up_projs_18025', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_mlp_up_projs_18025', 813, True, 'model.model.layers.2.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_2_mlp_down_projs_18033', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_mlp_down_projs_18033', 815, True, 'model.model.layers.2.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_2_mlp_shared_experts_gate_proj_weight_8247', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_mlp_shared_experts_gate_proj_weight_8247', 819, True, 'model.model.layers.2.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_2_mlp_shared_experts_up_proj_weight_8250', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_mlp_shared_experts_up_proj_weight_8250', 825, True, 'model.model.layers.2.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_2_mlp_shared_experts_down_proj_weight_8253', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_2_mlp_shared_experts_down_proj_weight_8253', 831, True, 'model.model.layers.2.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_3_input_layernorm_weight_8257', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_input_layernorm_weight_8257', 839, True, 'model.model.layers.3.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_3_self_attn_q_proj_weight_8259', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_self_attn_q_proj_weight_8259', 853, True, 'model.model.layers.3.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_3_self_attn_kv_a_proj_with_mqa_weight_18561', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_self_attn_kv_a_proj_with_mqa_weight_18561', 869, True, 'model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_3_self_attn_kv_a_layernorm_weight_8271', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_self_attn_kv_a_layernorm_weight_8271', 885, True, 'model.model.layers.3.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_3_self_attn_kv_b_proj_weight_8273', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_self_attn_kv_b_proj_weight_8273', 893, True, 'model.model.layers.3.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_3_self_attn_rotary_emb_cos_cached_8279', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_3_self_attn_rotary_emb_cos_cached_8279', 915, False, 'model.model.layers.3.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_3_self_attn_rotary_emb_sin_cached_8282', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_3_self_attn_rotary_emb_sin_cached_8282', 923, False, 'model.model.layers.3.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_3_self_attn_o_proj_weight_8320', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_self_attn_o_proj_weight_8320', 1064, True, 'model.model.layers.3.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_3_post_attention_layernorm_weight_8323', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_post_attention_layernorm_weight_8323', 1070, True, 'model.model.layers.3.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_3_mlp_gate_weight_8325', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_mlp_gate_weight_8325', 1078, True, 'model.model.layers.3.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_3_mlp_gate_projs_19889', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_mlp_gate_projs_19889', 1091, True, 'model.model.layers.3.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_3_mlp_up_projs_19897', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_mlp_up_projs_19897', 1093, True, 'model.model.layers.3.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_3_mlp_down_projs_19905', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_mlp_down_projs_19905', 1095, True, 'model.model.layers.3.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_3_mlp_shared_experts_gate_proj_weight_8333', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_mlp_shared_experts_gate_proj_weight_8333', 1099, True, 'model.model.layers.3.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_3_mlp_shared_experts_up_proj_weight_8336', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_mlp_shared_experts_up_proj_weight_8336', 1105, True, 'model.model.layers.3.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_3_mlp_shared_experts_down_proj_weight_8339', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_3_mlp_shared_experts_down_proj_weight_8339', 1111, True, 'model.model.layers.3.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_4_input_layernorm_weight_8343', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_input_layernorm_weight_8343', 1119, True, 'model.model.layers.4.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_4_self_attn_q_proj_weight_8345', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_self_attn_q_proj_weight_8345', 1133, True, 'model.model.layers.4.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_4_self_attn_kv_a_proj_with_mqa_weight_20433', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_self_attn_kv_a_proj_with_mqa_weight_20433', 1149, True, 'model.model.layers.4.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_4_self_attn_kv_a_layernorm_weight_8357', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_self_attn_kv_a_layernorm_weight_8357', 1165, True, 'model.model.layers.4.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_4_self_attn_kv_b_proj_weight_8359', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_self_attn_kv_b_proj_weight_8359', 1173, True, 'model.model.layers.4.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_4_self_attn_rotary_emb_cos_cached_8365', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_4_self_attn_rotary_emb_cos_cached_8365', 1195, False, 'model.model.layers.4.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_4_self_attn_rotary_emb_sin_cached_8368', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_4_self_attn_rotary_emb_sin_cached_8368', 1203, False, 'model.model.layers.4.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_4_self_attn_o_proj_weight_8406', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_self_attn_o_proj_weight_8406', 1344, True, 'model.model.layers.4.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_4_post_attention_layernorm_weight_8409', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_post_attention_layernorm_weight_8409', 1350, True, 'model.model.layers.4.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_4_mlp_gate_weight_8411', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_mlp_gate_weight_8411', 1358, True, 'model.model.layers.4.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_4_mlp_gate_projs_21761', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_mlp_gate_projs_21761', 1371, True, 'model.model.layers.4.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_4_mlp_up_projs_21769', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_mlp_up_projs_21769', 1373, True, 'model.model.layers.4.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_4_mlp_down_projs_21777', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_mlp_down_projs_21777', 1375, True, 'model.model.layers.4.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_4_mlp_shared_experts_gate_proj_weight_8419', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_mlp_shared_experts_gate_proj_weight_8419', 1379, True, 'model.model.layers.4.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_4_mlp_shared_experts_up_proj_weight_8422', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_mlp_shared_experts_up_proj_weight_8422', 1385, True, 'model.model.layers.4.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_4_mlp_shared_experts_down_proj_weight_8425', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_4_mlp_shared_experts_down_proj_weight_8425', 1391, True, 'model.model.layers.4.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_5_input_layernorm_weight_8429', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_input_layernorm_weight_8429', 1399, True, 'model.model.layers.5.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_5_self_attn_q_proj_weight_8431', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_self_attn_q_proj_weight_8431', 1413, True, 'model.model.layers.5.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_5_self_attn_kv_a_proj_with_mqa_weight_22305', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_self_attn_kv_a_proj_with_mqa_weight_22305', 1429, True, 'model.model.layers.5.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_5_self_attn_kv_a_layernorm_weight_8443', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_self_attn_kv_a_layernorm_weight_8443', 1445, True, 'model.model.layers.5.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_5_self_attn_kv_b_proj_weight_8445', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_self_attn_kv_b_proj_weight_8445', 1453, True, 'model.model.layers.5.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_5_self_attn_rotary_emb_cos_cached_8451', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_5_self_attn_rotary_emb_cos_cached_8451', 1475, False, 'model.model.layers.5.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_5_self_attn_rotary_emb_sin_cached_8454', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_5_self_attn_rotary_emb_sin_cached_8454', 1483, False, 'model.model.layers.5.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_5_self_attn_o_proj_weight_8492', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_self_attn_o_proj_weight_8492', 1624, True, 'model.model.layers.5.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_5_post_attention_layernorm_weight_8495', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_post_attention_layernorm_weight_8495', 1630, True, 'model.model.layers.5.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_5_mlp_gate_weight_8497', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_mlp_gate_weight_8497', 1638, True, 'model.model.layers.5.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_5_mlp_gate_projs_23633', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_mlp_gate_projs_23633', 1651, True, 'model.model.layers.5.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_5_mlp_up_projs_23641', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_mlp_up_projs_23641', 1653, True, 'model.model.layers.5.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_5_mlp_down_projs_23649', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_mlp_down_projs_23649', 1655, True, 'model.model.layers.5.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_5_mlp_shared_experts_gate_proj_weight_8505', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_mlp_shared_experts_gate_proj_weight_8505', 1659, True, 'model.model.layers.5.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_5_mlp_shared_experts_up_proj_weight_8508', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_mlp_shared_experts_up_proj_weight_8508', 1665, True, 'model.model.layers.5.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_5_mlp_shared_experts_down_proj_weight_8511', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_5_mlp_shared_experts_down_proj_weight_8511', 1671, True, 'model.model.layers.5.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_6_input_layernorm_weight_8515', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_input_layernorm_weight_8515', 1679, True, 'model.model.layers.6.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_6_self_attn_q_proj_weight_8517', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_self_attn_q_proj_weight_8517', 1693, True, 'model.model.layers.6.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_6_self_attn_kv_a_proj_with_mqa_weight_24177', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_self_attn_kv_a_proj_with_mqa_weight_24177', 1709, True, 'model.model.layers.6.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_6_self_attn_kv_a_layernorm_weight_8529', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_self_attn_kv_a_layernorm_weight_8529', 1725, True, 'model.model.layers.6.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_6_self_attn_kv_b_proj_weight_8531', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_self_attn_kv_b_proj_weight_8531', 1733, True, 'model.model.layers.6.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_6_self_attn_rotary_emb_cos_cached_8537', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_6_self_attn_rotary_emb_cos_cached_8537', 1755, False, 'model.model.layers.6.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_6_self_attn_rotary_emb_sin_cached_8540', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_6_self_attn_rotary_emb_sin_cached_8540', 1763, False, 'model.model.layers.6.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_6_self_attn_o_proj_weight_8578', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_self_attn_o_proj_weight_8578', 1904, True, 'model.model.layers.6.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_6_post_attention_layernorm_weight_8581', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_post_attention_layernorm_weight_8581', 1910, True, 'model.model.layers.6.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_6_mlp_gate_weight_8583', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_mlp_gate_weight_8583', 1918, True, 'model.model.layers.6.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_6_mlp_gate_projs_25505', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_mlp_gate_projs_25505', 1931, True, 'model.model.layers.6.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_6_mlp_up_projs_25513', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_mlp_up_projs_25513', 1933, True, 'model.model.layers.6.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_6_mlp_down_projs_25521', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_mlp_down_projs_25521', 1935, True, 'model.model.layers.6.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_6_mlp_shared_experts_gate_proj_weight_8591', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_mlp_shared_experts_gate_proj_weight_8591', 1939, True, 'model.model.layers.6.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_6_mlp_shared_experts_up_proj_weight_8594', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_mlp_shared_experts_up_proj_weight_8594', 1945, True, 'model.model.layers.6.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_6_mlp_shared_experts_down_proj_weight_8597', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_6_mlp_shared_experts_down_proj_weight_8597', 1951, True, 'model.model.layers.6.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_7_input_layernorm_weight_8601', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_input_layernorm_weight_8601', 1959, True, 'model.model.layers.7.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_7_self_attn_q_proj_weight_8603', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_self_attn_q_proj_weight_8603', 1973, True, 'model.model.layers.7.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_7_self_attn_kv_a_proj_with_mqa_weight_26049', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_self_attn_kv_a_proj_with_mqa_weight_26049', 1989, True, 'model.model.layers.7.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_7_self_attn_kv_a_layernorm_weight_8615', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_self_attn_kv_a_layernorm_weight_8615', 2005, True, 'model.model.layers.7.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_7_self_attn_kv_b_proj_weight_8617', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_self_attn_kv_b_proj_weight_8617', 2013, True, 'model.model.layers.7.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_7_self_attn_rotary_emb_cos_cached_8623', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_7_self_attn_rotary_emb_cos_cached_8623', 2035, False, 'model.model.layers.7.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_7_self_attn_rotary_emb_sin_cached_8626', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_7_self_attn_rotary_emb_sin_cached_8626', 2043, False, 'model.model.layers.7.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_7_self_attn_o_proj_weight_8664', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_self_attn_o_proj_weight_8664', 2184, True, 'model.model.layers.7.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_7_post_attention_layernorm_weight_8667', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_post_attention_layernorm_weight_8667', 2190, True, 'model.model.layers.7.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_7_mlp_gate_weight_8669', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_mlp_gate_weight_8669', 2198, True, 'model.model.layers.7.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_7_mlp_gate_projs_27377', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_mlp_gate_projs_27377', 2211, True, 'model.model.layers.7.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_7_mlp_up_projs_27385', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_mlp_up_projs_27385', 2213, True, 'model.model.layers.7.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_7_mlp_down_projs_27393', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_mlp_down_projs_27393', 2215, True, 'model.model.layers.7.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_7_mlp_shared_experts_gate_proj_weight_8677', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_mlp_shared_experts_gate_proj_weight_8677', 2219, True, 'model.model.layers.7.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_7_mlp_shared_experts_up_proj_weight_8680', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_mlp_shared_experts_up_proj_weight_8680', 2225, True, 'model.model.layers.7.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_7_mlp_shared_experts_down_proj_weight_8683', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_7_mlp_shared_experts_down_proj_weight_8683', 2231, True, 'model.model.layers.7.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_8_input_layernorm_weight_8687', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_input_layernorm_weight_8687', 2239, True, 'model.model.layers.8.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_8_self_attn_q_proj_weight_8689', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_self_attn_q_proj_weight_8689', 2253, True, 'model.model.layers.8.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_8_self_attn_kv_a_proj_with_mqa_weight_27921', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_self_attn_kv_a_proj_with_mqa_weight_27921', 2269, True, 'model.model.layers.8.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_8_self_attn_kv_a_layernorm_weight_8701', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_self_attn_kv_a_layernorm_weight_8701', 2285, True, 'model.model.layers.8.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_8_self_attn_kv_b_proj_weight_8703', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_self_attn_kv_b_proj_weight_8703', 2293, True, 'model.model.layers.8.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_8_self_attn_rotary_emb_cos_cached_8709', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_8_self_attn_rotary_emb_cos_cached_8709', 2315, False, 'model.model.layers.8.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_8_self_attn_rotary_emb_sin_cached_8712', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_8_self_attn_rotary_emb_sin_cached_8712', 2323, False, 'model.model.layers.8.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_8_self_attn_o_proj_weight_8750', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_self_attn_o_proj_weight_8750', 2464, True, 'model.model.layers.8.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_8_post_attention_layernorm_weight_8753', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_post_attention_layernorm_weight_8753', 2470, True, 'model.model.layers.8.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_8_mlp_gate_weight_8755', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_mlp_gate_weight_8755', 2478, True, 'model.model.layers.8.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_8_mlp_gate_projs_29249', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_mlp_gate_projs_29249', 2491, True, 'model.model.layers.8.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_8_mlp_up_projs_29257', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_mlp_up_projs_29257', 2493, True, 'model.model.layers.8.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_8_mlp_down_projs_29265', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_mlp_down_projs_29265', 2495, True, 'model.model.layers.8.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_8_mlp_shared_experts_gate_proj_weight_8763', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_mlp_shared_experts_gate_proj_weight_8763', 2499, True, 'model.model.layers.8.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_8_mlp_shared_experts_up_proj_weight_8766', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_mlp_shared_experts_up_proj_weight_8766', 2505, True, 'model.model.layers.8.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_8_mlp_shared_experts_down_proj_weight_8769', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_8_mlp_shared_experts_down_proj_weight_8769', 2511, True, 'model.model.layers.8.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_9_input_layernorm_weight_8773', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_input_layernorm_weight_8773', 2519, True, 'model.model.layers.9.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_9_self_attn_q_proj_weight_8775', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_self_attn_q_proj_weight_8775', 2533, True, 'model.model.layers.9.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_9_self_attn_kv_a_proj_with_mqa_weight_29793', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_self_attn_kv_a_proj_with_mqa_weight_29793', 2549, True, 'model.model.layers.9.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_9_self_attn_kv_a_layernorm_weight_8787', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_self_attn_kv_a_layernorm_weight_8787', 2565, True, 'model.model.layers.9.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_9_self_attn_kv_b_proj_weight_8789', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_self_attn_kv_b_proj_weight_8789', 2573, True, 'model.model.layers.9.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_9_self_attn_rotary_emb_cos_cached_8795', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_9_self_attn_rotary_emb_cos_cached_8795', 2595, False, 'model.model.layers.9.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_9_self_attn_rotary_emb_sin_cached_8798', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_9_self_attn_rotary_emb_sin_cached_8798', 2603, False, 'model.model.layers.9.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_9_self_attn_o_proj_weight_8836', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_self_attn_o_proj_weight_8836', 2744, True, 'model.model.layers.9.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_9_post_attention_layernorm_weight_8839', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_post_attention_layernorm_weight_8839', 2750, True, 'model.model.layers.9.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_9_mlp_gate_weight_8841', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_mlp_gate_weight_8841', 2758, True, 'model.model.layers.9.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_9_mlp_gate_projs_31121', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_mlp_gate_projs_31121', 2771, True, 'model.model.layers.9.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_9_mlp_up_projs_31129', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_mlp_up_projs_31129', 2773, True, 'model.model.layers.9.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_9_mlp_down_projs_31137', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_mlp_down_projs_31137', 2775, True, 'model.model.layers.9.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_9_mlp_shared_experts_gate_proj_weight_8849', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_mlp_shared_experts_gate_proj_weight_8849', 2779, True, 'model.model.layers.9.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_9_mlp_shared_experts_up_proj_weight_8852', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_mlp_shared_experts_up_proj_weight_8852', 2785, True, 'model.model.layers.9.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_9_mlp_shared_experts_down_proj_weight_8855', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_9_mlp_shared_experts_down_proj_weight_8855', 2791, True, 'model.model.layers.9.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_10_input_layernorm_weight_8859', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_input_layernorm_weight_8859', 2799, True, 'model.model.layers.10.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_10_self_attn_q_proj_weight_8861', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_self_attn_q_proj_weight_8861', 2813, True, 'model.model.layers.10.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_10_self_attn_kv_a_proj_with_mqa_weight_31665', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_self_attn_kv_a_proj_with_mqa_weight_31665', 2829, True, 'model.model.layers.10.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_10_self_attn_kv_a_layernorm_weight_8873', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_self_attn_kv_a_layernorm_weight_8873', 2845, True, 'model.model.layers.10.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_10_self_attn_kv_b_proj_weight_8875', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_self_attn_kv_b_proj_weight_8875', 2853, True, 'model.model.layers.10.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_10_self_attn_rotary_emb_cos_cached_8881', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_10_self_attn_rotary_emb_cos_cached_8881', 2875, False, 'model.model.layers.10.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_10_self_attn_rotary_emb_sin_cached_8884', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_10_self_attn_rotary_emb_sin_cached_8884', 2883, False, 'model.model.layers.10.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_10_self_attn_o_proj_weight_8922', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_self_attn_o_proj_weight_8922', 3024, True, 'model.model.layers.10.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_10_post_attention_layernorm_weight_8925', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_post_attention_layernorm_weight_8925', 3030, True, 'model.model.layers.10.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_10_mlp_gate_weight_8927', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_mlp_gate_weight_8927', 3038, True, 'model.model.layers.10.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_10_mlp_gate_projs_32993', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_mlp_gate_projs_32993', 3051, True, 'model.model.layers.10.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_10_mlp_up_projs_33001', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_mlp_up_projs_33001', 3053, True, 'model.model.layers.10.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_10_mlp_down_projs_33009', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_mlp_down_projs_33009', 3055, True, 'model.model.layers.10.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_10_mlp_shared_experts_gate_proj_weight_8935', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_mlp_shared_experts_gate_proj_weight_8935', 3059, True, 'model.model.layers.10.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_10_mlp_shared_experts_up_proj_weight_8938', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_mlp_shared_experts_up_proj_weight_8938', 3065, True, 'model.model.layers.10.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_10_mlp_shared_experts_down_proj_weight_8941', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_10_mlp_shared_experts_down_proj_weight_8941', 3071, True, 'model.model.layers.10.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_11_input_layernorm_weight_8945', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_input_layernorm_weight_8945', 3079, True, 'model.model.layers.11.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_11_self_attn_q_proj_weight_8947', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_self_attn_q_proj_weight_8947', 3093, True, 'model.model.layers.11.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_11_self_attn_kv_a_proj_with_mqa_weight_33537', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_self_attn_kv_a_proj_with_mqa_weight_33537', 3109, True, 'model.model.layers.11.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_11_self_attn_kv_a_layernorm_weight_8959', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_self_attn_kv_a_layernorm_weight_8959', 3125, True, 'model.model.layers.11.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_11_self_attn_kv_b_proj_weight_8961', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_self_attn_kv_b_proj_weight_8961', 3133, True, 'model.model.layers.11.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_11_self_attn_rotary_emb_cos_cached_8967', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_11_self_attn_rotary_emb_cos_cached_8967', 3155, False, 'model.model.layers.11.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_11_self_attn_rotary_emb_sin_cached_8970', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_11_self_attn_rotary_emb_sin_cached_8970', 3163, False, 'model.model.layers.11.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_11_self_attn_o_proj_weight_9008', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_self_attn_o_proj_weight_9008', 3304, True, 'model.model.layers.11.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_11_post_attention_layernorm_weight_9011', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_post_attention_layernorm_weight_9011', 3310, True, 'model.model.layers.11.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_11_mlp_gate_weight_9013', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_mlp_gate_weight_9013', 3318, True, 'model.model.layers.11.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_11_mlp_gate_projs_34865', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_mlp_gate_projs_34865', 3331, True, 'model.model.layers.11.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_11_mlp_up_projs_34873', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_mlp_up_projs_34873', 3333, True, 'model.model.layers.11.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_11_mlp_down_projs_34881', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_mlp_down_projs_34881', 3335, True, 'model.model.layers.11.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_11_mlp_shared_experts_gate_proj_weight_9021', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_mlp_shared_experts_gate_proj_weight_9021', 3339, True, 'model.model.layers.11.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_11_mlp_shared_experts_up_proj_weight_9024', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_mlp_shared_experts_up_proj_weight_9024', 3345, True, 'model.model.layers.11.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_11_mlp_shared_experts_down_proj_weight_9027', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_11_mlp_shared_experts_down_proj_weight_9027', 3351, True, 'model.model.layers.11.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_12_input_layernorm_weight_9031', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_input_layernorm_weight_9031', 3359, True, 'model.model.layers.12.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_12_self_attn_q_proj_weight_9033', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_self_attn_q_proj_weight_9033', 3373, True, 'model.model.layers.12.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_12_self_attn_kv_a_proj_with_mqa_weight_35409', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_self_attn_kv_a_proj_with_mqa_weight_35409', 3389, True, 'model.model.layers.12.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_12_self_attn_kv_a_layernorm_weight_9045', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_self_attn_kv_a_layernorm_weight_9045', 3405, True, 'model.model.layers.12.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_12_self_attn_kv_b_proj_weight_9047', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_self_attn_kv_b_proj_weight_9047', 3413, True, 'model.model.layers.12.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_12_self_attn_rotary_emb_cos_cached_9053', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_12_self_attn_rotary_emb_cos_cached_9053', 3435, False, 'model.model.layers.12.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_12_self_attn_rotary_emb_sin_cached_9056', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_12_self_attn_rotary_emb_sin_cached_9056', 3443, False, 'model.model.layers.12.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_12_self_attn_o_proj_weight_9094', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_self_attn_o_proj_weight_9094', 3584, True, 'model.model.layers.12.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_12_post_attention_layernorm_weight_9097', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_post_attention_layernorm_weight_9097', 3590, True, 'model.model.layers.12.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_12_mlp_gate_weight_9099', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_mlp_gate_weight_9099', 3598, True, 'model.model.layers.12.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_12_mlp_gate_projs_36737', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_mlp_gate_projs_36737', 3611, True, 'model.model.layers.12.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_12_mlp_up_projs_36745', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_mlp_up_projs_36745', 3613, True, 'model.model.layers.12.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_12_mlp_down_projs_36753', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_mlp_down_projs_36753', 3615, True, 'model.model.layers.12.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_12_mlp_shared_experts_gate_proj_weight_9107', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_mlp_shared_experts_gate_proj_weight_9107', 3619, True, 'model.model.layers.12.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_12_mlp_shared_experts_up_proj_weight_9110', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_mlp_shared_experts_up_proj_weight_9110', 3625, True, 'model.model.layers.12.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_12_mlp_shared_experts_down_proj_weight_9113', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_12_mlp_shared_experts_down_proj_weight_9113', 3631, True, 'model.model.layers.12.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_13_input_layernorm_weight_9117', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_input_layernorm_weight_9117', 3639, True, 'model.model.layers.13.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_13_self_attn_q_proj_weight_9119', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_self_attn_q_proj_weight_9119', 3653, True, 'model.model.layers.13.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_13_self_attn_kv_a_proj_with_mqa_weight_37281', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_self_attn_kv_a_proj_with_mqa_weight_37281', 3669, True, 'model.model.layers.13.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_13_self_attn_kv_a_layernorm_weight_9131', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_self_attn_kv_a_layernorm_weight_9131', 3685, True, 'model.model.layers.13.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_13_self_attn_kv_b_proj_weight_9133', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_self_attn_kv_b_proj_weight_9133', 3693, True, 'model.model.layers.13.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_13_self_attn_rotary_emb_cos_cached_9139', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_13_self_attn_rotary_emb_cos_cached_9139', 3715, False, 'model.model.layers.13.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_13_self_attn_rotary_emb_sin_cached_9142', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_13_self_attn_rotary_emb_sin_cached_9142', 3723, False, 'model.model.layers.13.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_13_self_attn_o_proj_weight_9180', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_self_attn_o_proj_weight_9180', 3864, True, 'model.model.layers.13.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_13_post_attention_layernorm_weight_9183', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_post_attention_layernorm_weight_9183', 3870, True, 'model.model.layers.13.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_13_mlp_gate_weight_9185', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_mlp_gate_weight_9185', 3878, True, 'model.model.layers.13.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_13_mlp_gate_projs_38609', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_mlp_gate_projs_38609', 3891, True, 'model.model.layers.13.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_13_mlp_up_projs_38617', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_mlp_up_projs_38617', 3893, True, 'model.model.layers.13.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_13_mlp_down_projs_38625', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_mlp_down_projs_38625', 3895, True, 'model.model.layers.13.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_13_mlp_shared_experts_gate_proj_weight_9193', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_mlp_shared_experts_gate_proj_weight_9193', 3899, True, 'model.model.layers.13.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_13_mlp_shared_experts_up_proj_weight_9196', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_mlp_shared_experts_up_proj_weight_9196', 3905, True, 'model.model.layers.13.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_13_mlp_shared_experts_down_proj_weight_9199', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_13_mlp_shared_experts_down_proj_weight_9199', 3911, True, 'model.model.layers.13.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_14_input_layernorm_weight_9203', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_input_layernorm_weight_9203', 3919, True, 'model.model.layers.14.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_14_self_attn_q_proj_weight_9205', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_self_attn_q_proj_weight_9205', 3933, True, 'model.model.layers.14.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_14_self_attn_kv_a_proj_with_mqa_weight_39153', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_self_attn_kv_a_proj_with_mqa_weight_39153', 3949, True, 'model.model.layers.14.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_14_self_attn_kv_a_layernorm_weight_9217', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_self_attn_kv_a_layernorm_weight_9217', 3965, True, 'model.model.layers.14.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_14_self_attn_kv_b_proj_weight_9219', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_self_attn_kv_b_proj_weight_9219', 3973, True, 'model.model.layers.14.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_14_self_attn_rotary_emb_cos_cached_9225', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_14_self_attn_rotary_emb_cos_cached_9225', 3995, False, 'model.model.layers.14.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_14_self_attn_rotary_emb_sin_cached_9228', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_14_self_attn_rotary_emb_sin_cached_9228', 4003, False, 'model.model.layers.14.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_14_self_attn_o_proj_weight_9266', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_self_attn_o_proj_weight_9266', 4144, True, 'model.model.layers.14.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_14_post_attention_layernorm_weight_9269', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_post_attention_layernorm_weight_9269', 4150, True, 'model.model.layers.14.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_14_mlp_gate_weight_9271', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_mlp_gate_weight_9271', 4158, True, 'model.model.layers.14.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_14_mlp_gate_projs_40481', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_mlp_gate_projs_40481', 4171, True, 'model.model.layers.14.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_14_mlp_up_projs_40489', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_mlp_up_projs_40489', 4173, True, 'model.model.layers.14.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_14_mlp_down_projs_40497', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_mlp_down_projs_40497', 4175, True, 'model.model.layers.14.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_14_mlp_shared_experts_gate_proj_weight_9279', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_mlp_shared_experts_gate_proj_weight_9279', 4179, True, 'model.model.layers.14.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_14_mlp_shared_experts_up_proj_weight_9282', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_mlp_shared_experts_up_proj_weight_9282', 4185, True, 'model.model.layers.14.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_14_mlp_shared_experts_down_proj_weight_9285', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_14_mlp_shared_experts_down_proj_weight_9285', 4191, True, 'model.model.layers.14.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_15_input_layernorm_weight_9289', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_input_layernorm_weight_9289', 4199, True, 'model.model.layers.15.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_15_self_attn_q_proj_weight_9291', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_self_attn_q_proj_weight_9291', 4213, True, 'model.model.layers.15.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_15_self_attn_kv_a_proj_with_mqa_weight_41025', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_self_attn_kv_a_proj_with_mqa_weight_41025', 4229, True, 'model.model.layers.15.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_15_self_attn_kv_a_layernorm_weight_9303', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_self_attn_kv_a_layernorm_weight_9303', 4245, True, 'model.model.layers.15.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_15_self_attn_kv_b_proj_weight_9305', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_self_attn_kv_b_proj_weight_9305', 4253, True, 'model.model.layers.15.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_15_self_attn_rotary_emb_cos_cached_9311', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_15_self_attn_rotary_emb_cos_cached_9311', 4275, False, 'model.model.layers.15.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_15_self_attn_rotary_emb_sin_cached_9314', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_15_self_attn_rotary_emb_sin_cached_9314', 4283, False, 'model.model.layers.15.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_15_self_attn_o_proj_weight_9352', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_self_attn_o_proj_weight_9352', 4424, True, 'model.model.layers.15.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_15_post_attention_layernorm_weight_9355', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_post_attention_layernorm_weight_9355', 4430, True, 'model.model.layers.15.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_15_mlp_gate_weight_9357', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_mlp_gate_weight_9357', 4438, True, 'model.model.layers.15.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_15_mlp_gate_projs_42353', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_mlp_gate_projs_42353', 4451, True, 'model.model.layers.15.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_15_mlp_up_projs_42361', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_mlp_up_projs_42361', 4453, True, 'model.model.layers.15.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_15_mlp_down_projs_42369', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_mlp_down_projs_42369', 4455, True, 'model.model.layers.15.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_15_mlp_shared_experts_gate_proj_weight_9365', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_mlp_shared_experts_gate_proj_weight_9365', 4459, True, 'model.model.layers.15.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_15_mlp_shared_experts_up_proj_weight_9368', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_mlp_shared_experts_up_proj_weight_9368', 4465, True, 'model.model.layers.15.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_15_mlp_shared_experts_down_proj_weight_9371', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_15_mlp_shared_experts_down_proj_weight_9371', 4471, True, 'model.model.layers.15.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_16_input_layernorm_weight_9375', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_input_layernorm_weight_9375', 4479, True, 'model.model.layers.16.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_16_self_attn_q_proj_weight_9377', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_self_attn_q_proj_weight_9377', 4493, True, 'model.model.layers.16.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_16_self_attn_kv_a_proj_with_mqa_weight_42897', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_self_attn_kv_a_proj_with_mqa_weight_42897', 4509, True, 'model.model.layers.16.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_16_self_attn_kv_a_layernorm_weight_9389', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_self_attn_kv_a_layernorm_weight_9389', 4525, True, 'model.model.layers.16.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_16_self_attn_kv_b_proj_weight_9391', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_self_attn_kv_b_proj_weight_9391', 4533, True, 'model.model.layers.16.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_16_self_attn_rotary_emb_cos_cached_9397', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_16_self_attn_rotary_emb_cos_cached_9397', 4555, False, 'model.model.layers.16.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_16_self_attn_rotary_emb_sin_cached_9400', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_16_self_attn_rotary_emb_sin_cached_9400', 4563, False, 'model.model.layers.16.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_16_self_attn_o_proj_weight_9438', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_self_attn_o_proj_weight_9438', 4704, True, 'model.model.layers.16.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_16_post_attention_layernorm_weight_9441', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_post_attention_layernorm_weight_9441', 4710, True, 'model.model.layers.16.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_16_mlp_gate_weight_9443', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_mlp_gate_weight_9443', 4718, True, 'model.model.layers.16.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_16_mlp_gate_projs_44225', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_mlp_gate_projs_44225', 4731, True, 'model.model.layers.16.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_16_mlp_up_projs_44233', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_mlp_up_projs_44233', 4733, True, 'model.model.layers.16.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_16_mlp_down_projs_44241', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_mlp_down_projs_44241', 4735, True, 'model.model.layers.16.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_16_mlp_shared_experts_gate_proj_weight_9451', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_mlp_shared_experts_gate_proj_weight_9451', 4739, True, 'model.model.layers.16.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_16_mlp_shared_experts_up_proj_weight_9454', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_mlp_shared_experts_up_proj_weight_9454', 4745, True, 'model.model.layers.16.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_16_mlp_shared_experts_down_proj_weight_9457', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_16_mlp_shared_experts_down_proj_weight_9457', 4751, True, 'model.model.layers.16.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_17_input_layernorm_weight_9461', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_input_layernorm_weight_9461', 4759, True, 'model.model.layers.17.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_17_self_attn_q_proj_weight_9463', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_self_attn_q_proj_weight_9463', 4773, True, 'model.model.layers.17.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_17_self_attn_kv_a_proj_with_mqa_weight_44769', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_self_attn_kv_a_proj_with_mqa_weight_44769', 4789, True, 'model.model.layers.17.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_17_self_attn_kv_a_layernorm_weight_9475', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_self_attn_kv_a_layernorm_weight_9475', 4805, True, 'model.model.layers.17.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_17_self_attn_kv_b_proj_weight_9477', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_self_attn_kv_b_proj_weight_9477', 4813, True, 'model.model.layers.17.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_17_self_attn_rotary_emb_cos_cached_9483', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_17_self_attn_rotary_emb_cos_cached_9483', 4835, False, 'model.model.layers.17.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_17_self_attn_rotary_emb_sin_cached_9486', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_17_self_attn_rotary_emb_sin_cached_9486', 4843, False, 'model.model.layers.17.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_17_self_attn_o_proj_weight_9524', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_self_attn_o_proj_weight_9524', 4984, True, 'model.model.layers.17.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_17_post_attention_layernorm_weight_9527', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_post_attention_layernorm_weight_9527', 4990, True, 'model.model.layers.17.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_17_mlp_gate_weight_9529', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_mlp_gate_weight_9529', 4998, True, 'model.model.layers.17.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_17_mlp_gate_projs_46097', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_mlp_gate_projs_46097', 5011, True, 'model.model.layers.17.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_17_mlp_up_projs_46105', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_mlp_up_projs_46105', 5013, True, 'model.model.layers.17.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_17_mlp_down_projs_46113', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_mlp_down_projs_46113', 5015, True, 'model.model.layers.17.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_17_mlp_shared_experts_gate_proj_weight_9537', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_mlp_shared_experts_gate_proj_weight_9537', 5019, True, 'model.model.layers.17.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_17_mlp_shared_experts_up_proj_weight_9540', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_mlp_shared_experts_up_proj_weight_9540', 5025, True, 'model.model.layers.17.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_17_mlp_shared_experts_down_proj_weight_9543', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_17_mlp_shared_experts_down_proj_weight_9543', 5031, True, 'model.model.layers.17.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_18_input_layernorm_weight_9547', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_input_layernorm_weight_9547', 5039, True, 'model.model.layers.18.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_18_self_attn_q_proj_weight_9549', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_self_attn_q_proj_weight_9549', 5053, True, 'model.model.layers.18.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_18_self_attn_kv_a_proj_with_mqa_weight_46641', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_self_attn_kv_a_proj_with_mqa_weight_46641', 5069, True, 'model.model.layers.18.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_18_self_attn_kv_a_layernorm_weight_9561', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_self_attn_kv_a_layernorm_weight_9561', 5085, True, 'model.model.layers.18.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_18_self_attn_kv_b_proj_weight_9563', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_self_attn_kv_b_proj_weight_9563', 5093, True, 'model.model.layers.18.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_18_self_attn_rotary_emb_cos_cached_9569', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_18_self_attn_rotary_emb_cos_cached_9569', 5115, False, 'model.model.layers.18.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_18_self_attn_rotary_emb_sin_cached_9572', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_18_self_attn_rotary_emb_sin_cached_9572', 5123, False, 'model.model.layers.18.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_18_self_attn_o_proj_weight_9610', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_self_attn_o_proj_weight_9610', 5264, True, 'model.model.layers.18.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_18_post_attention_layernorm_weight_9613', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_post_attention_layernorm_weight_9613', 5270, True, 'model.model.layers.18.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_18_mlp_gate_weight_9615', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_mlp_gate_weight_9615', 5278, True, 'model.model.layers.18.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_18_mlp_gate_projs_47969', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_mlp_gate_projs_47969', 5291, True, 'model.model.layers.18.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_18_mlp_up_projs_47977', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_mlp_up_projs_47977', 5293, True, 'model.model.layers.18.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_18_mlp_down_projs_47985', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_mlp_down_projs_47985', 5295, True, 'model.model.layers.18.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_18_mlp_shared_experts_gate_proj_weight_9623', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_mlp_shared_experts_gate_proj_weight_9623', 5299, True, 'model.model.layers.18.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_18_mlp_shared_experts_up_proj_weight_9626', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_mlp_shared_experts_up_proj_weight_9626', 5305, True, 'model.model.layers.18.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_18_mlp_shared_experts_down_proj_weight_9629', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_18_mlp_shared_experts_down_proj_weight_9629', 5311, True, 'model.model.layers.18.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_19_input_layernorm_weight_9633', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_input_layernorm_weight_9633', 5319, True, 'model.model.layers.19.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_19_self_attn_q_proj_weight_9635', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_self_attn_q_proj_weight_9635', 5333, True, 'model.model.layers.19.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_19_self_attn_kv_a_proj_with_mqa_weight_48513', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_self_attn_kv_a_proj_with_mqa_weight_48513', 5349, True, 'model.model.layers.19.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_19_self_attn_kv_a_layernorm_weight_9647', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_self_attn_kv_a_layernorm_weight_9647', 5365, True, 'model.model.layers.19.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_19_self_attn_kv_b_proj_weight_9649', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_self_attn_kv_b_proj_weight_9649', 5373, True, 'model.model.layers.19.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_19_self_attn_rotary_emb_cos_cached_9655', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_19_self_attn_rotary_emb_cos_cached_9655', 5395, False, 'model.model.layers.19.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_19_self_attn_rotary_emb_sin_cached_9658', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_19_self_attn_rotary_emb_sin_cached_9658', 5403, False, 'model.model.layers.19.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_19_self_attn_o_proj_weight_9696', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_self_attn_o_proj_weight_9696', 5544, True, 'model.model.layers.19.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_19_post_attention_layernorm_weight_9699', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_post_attention_layernorm_weight_9699', 5550, True, 'model.model.layers.19.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_19_mlp_gate_weight_9701', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_mlp_gate_weight_9701', 5558, True, 'model.model.layers.19.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_19_mlp_gate_projs_49841', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_mlp_gate_projs_49841', 5571, True, 'model.model.layers.19.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_19_mlp_up_projs_49849', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_mlp_up_projs_49849', 5573, True, 'model.model.layers.19.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_19_mlp_down_projs_49857', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_mlp_down_projs_49857', 5575, True, 'model.model.layers.19.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_19_mlp_shared_experts_gate_proj_weight_9709', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_mlp_shared_experts_gate_proj_weight_9709', 5579, True, 'model.model.layers.19.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_19_mlp_shared_experts_up_proj_weight_9712', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_mlp_shared_experts_up_proj_weight_9712', 5585, True, 'model.model.layers.19.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_19_mlp_shared_experts_down_proj_weight_9715', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_19_mlp_shared_experts_down_proj_weight_9715', 5591, True, 'model.model.layers.19.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_20_input_layernorm_weight_9719', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_input_layernorm_weight_9719', 5599, True, 'model.model.layers.20.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_20_self_attn_q_proj_weight_9721', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_self_attn_q_proj_weight_9721', 5613, True, 'model.model.layers.20.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_20_self_attn_kv_a_proj_with_mqa_weight_50385', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_self_attn_kv_a_proj_with_mqa_weight_50385', 5629, True, 'model.model.layers.20.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_20_self_attn_kv_a_layernorm_weight_9733', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_self_attn_kv_a_layernorm_weight_9733', 5645, True, 'model.model.layers.20.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_20_self_attn_kv_b_proj_weight_9735', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_self_attn_kv_b_proj_weight_9735', 5653, True, 'model.model.layers.20.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_20_self_attn_rotary_emb_cos_cached_9741', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_20_self_attn_rotary_emb_cos_cached_9741', 5675, False, 'model.model.layers.20.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_20_self_attn_rotary_emb_sin_cached_9744', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_20_self_attn_rotary_emb_sin_cached_9744', 5683, False, 'model.model.layers.20.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_20_self_attn_o_proj_weight_9782', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_self_attn_o_proj_weight_9782', 5824, True, 'model.model.layers.20.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_20_post_attention_layernorm_weight_9785', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_post_attention_layernorm_weight_9785', 5830, True, 'model.model.layers.20.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_20_mlp_gate_weight_9787', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_mlp_gate_weight_9787', 5838, True, 'model.model.layers.20.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_20_mlp_gate_projs_51713', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_mlp_gate_projs_51713', 5851, True, 'model.model.layers.20.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_20_mlp_up_projs_51721', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_mlp_up_projs_51721', 5853, True, 'model.model.layers.20.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_20_mlp_down_projs_51729', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_mlp_down_projs_51729', 5855, True, 'model.model.layers.20.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_20_mlp_shared_experts_gate_proj_weight_9795', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_mlp_shared_experts_gate_proj_weight_9795', 5859, True, 'model.model.layers.20.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_20_mlp_shared_experts_up_proj_weight_9798', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_mlp_shared_experts_up_proj_weight_9798', 5865, True, 'model.model.layers.20.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_20_mlp_shared_experts_down_proj_weight_9801', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_20_mlp_shared_experts_down_proj_weight_9801', 5871, True, 'model.model.layers.20.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_21_input_layernorm_weight_9805', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_input_layernorm_weight_9805', 5879, True, 'model.model.layers.21.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_21_self_attn_q_proj_weight_9807', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_self_attn_q_proj_weight_9807', 5893, True, 'model.model.layers.21.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_21_self_attn_kv_a_proj_with_mqa_weight_52257', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_self_attn_kv_a_proj_with_mqa_weight_52257', 5909, True, 'model.model.layers.21.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_21_self_attn_kv_a_layernorm_weight_9819', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_self_attn_kv_a_layernorm_weight_9819', 5925, True, 'model.model.layers.21.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_21_self_attn_kv_b_proj_weight_9821', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_self_attn_kv_b_proj_weight_9821', 5933, True, 'model.model.layers.21.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_21_self_attn_rotary_emb_cos_cached_9827', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_21_self_attn_rotary_emb_cos_cached_9827', 5955, False, 'model.model.layers.21.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_21_self_attn_rotary_emb_sin_cached_9830', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_21_self_attn_rotary_emb_sin_cached_9830', 5963, False, 'model.model.layers.21.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_21_self_attn_o_proj_weight_9868', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_self_attn_o_proj_weight_9868', 6104, True, 'model.model.layers.21.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_21_post_attention_layernorm_weight_9871', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_post_attention_layernorm_weight_9871', 6110, True, 'model.model.layers.21.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_21_mlp_gate_weight_9873', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_mlp_gate_weight_9873', 6118, True, 'model.model.layers.21.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_21_mlp_gate_projs_53585', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_mlp_gate_projs_53585', 6131, True, 'model.model.layers.21.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_21_mlp_up_projs_53593', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_mlp_up_projs_53593', 6133, True, 'model.model.layers.21.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_21_mlp_down_projs_53601', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_mlp_down_projs_53601', 6135, True, 'model.model.layers.21.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_21_mlp_shared_experts_gate_proj_weight_9881', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_mlp_shared_experts_gate_proj_weight_9881', 6139, True, 'model.model.layers.21.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_21_mlp_shared_experts_up_proj_weight_9884', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_mlp_shared_experts_up_proj_weight_9884', 6145, True, 'model.model.layers.21.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_21_mlp_shared_experts_down_proj_weight_9887', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_21_mlp_shared_experts_down_proj_weight_9887', 6151, True, 'model.model.layers.21.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_22_input_layernorm_weight_9891', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_input_layernorm_weight_9891', 6159, True, 'model.model.layers.22.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_22_self_attn_q_proj_weight_9893', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_self_attn_q_proj_weight_9893', 6173, True, 'model.model.layers.22.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_22_self_attn_kv_a_proj_with_mqa_weight_54129', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_self_attn_kv_a_proj_with_mqa_weight_54129', 6189, True, 'model.model.layers.22.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_22_self_attn_kv_a_layernorm_weight_9905', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_self_attn_kv_a_layernorm_weight_9905', 6205, True, 'model.model.layers.22.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_22_self_attn_kv_b_proj_weight_9907', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_self_attn_kv_b_proj_weight_9907', 6213, True, 'model.model.layers.22.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_22_self_attn_rotary_emb_cos_cached_9913', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_22_self_attn_rotary_emb_cos_cached_9913', 6235, False, 'model.model.layers.22.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_22_self_attn_rotary_emb_sin_cached_9916', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_22_self_attn_rotary_emb_sin_cached_9916', 6243, False, 'model.model.layers.22.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_22_self_attn_o_proj_weight_9954', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_self_attn_o_proj_weight_9954', 6384, True, 'model.model.layers.22.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_22_post_attention_layernorm_weight_9957', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_post_attention_layernorm_weight_9957', 6390, True, 'model.model.layers.22.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_22_mlp_gate_weight_9959', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_mlp_gate_weight_9959', 6398, True, 'model.model.layers.22.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_22_mlp_gate_projs_55457', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_mlp_gate_projs_55457', 6411, True, 'model.model.layers.22.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_22_mlp_up_projs_55465', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_mlp_up_projs_55465', 6413, True, 'model.model.layers.22.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_22_mlp_down_projs_55473', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_mlp_down_projs_55473', 6415, True, 'model.model.layers.22.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_22_mlp_shared_experts_gate_proj_weight_9967', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_mlp_shared_experts_gate_proj_weight_9967', 6419, True, 'model.model.layers.22.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_22_mlp_shared_experts_up_proj_weight_9970', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_mlp_shared_experts_up_proj_weight_9970', 6425, True, 'model.model.layers.22.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_22_mlp_shared_experts_down_proj_weight_9973', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_22_mlp_shared_experts_down_proj_weight_9973', 6431, True, 'model.model.layers.22.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_23_input_layernorm_weight_9977', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_input_layernorm_weight_9977', 6439, True, 'model.model.layers.23.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_23_self_attn_q_proj_weight_9979', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_self_attn_q_proj_weight_9979', 6453, True, 'model.model.layers.23.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_23_self_attn_kv_a_proj_with_mqa_weight_56001', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_self_attn_kv_a_proj_with_mqa_weight_56001', 6469, True, 'model.model.layers.23.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_23_self_attn_kv_a_layernorm_weight_9991', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_self_attn_kv_a_layernorm_weight_9991', 6485, True, 'model.model.layers.23.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_23_self_attn_kv_b_proj_weight_9993', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_self_attn_kv_b_proj_weight_9993', 6493, True, 'model.model.layers.23.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_23_self_attn_rotary_emb_cos_cached_9999', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_23_self_attn_rotary_emb_cos_cached_9999', 6515, False, 'model.model.layers.23.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_23_self_attn_rotary_emb_sin_cached_10002', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_23_self_attn_rotary_emb_sin_cached_10002', 6523, False, 'model.model.layers.23.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_23_self_attn_o_proj_weight_10040', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_self_attn_o_proj_weight_10040', 6664, True, 'model.model.layers.23.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_23_post_attention_layernorm_weight_10043', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_post_attention_layernorm_weight_10043', 6670, True, 'model.model.layers.23.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_23_mlp_gate_weight_10045', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_mlp_gate_weight_10045', 6678, True, 'model.model.layers.23.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_23_mlp_gate_projs_57329', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_mlp_gate_projs_57329', 6691, True, 'model.model.layers.23.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_23_mlp_up_projs_57337', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_mlp_up_projs_57337', 6693, True, 'model.model.layers.23.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_23_mlp_down_projs_57345', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_mlp_down_projs_57345', 6695, True, 'model.model.layers.23.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_23_mlp_shared_experts_gate_proj_weight_10053', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_mlp_shared_experts_gate_proj_weight_10053', 6699, True, 'model.model.layers.23.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_23_mlp_shared_experts_up_proj_weight_10056', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_mlp_shared_experts_up_proj_weight_10056', 6705, True, 'model.model.layers.23.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_23_mlp_shared_experts_down_proj_weight_10059', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_23_mlp_shared_experts_down_proj_weight_10059', 6711, True, 'model.model.layers.23.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_24_input_layernorm_weight_10063', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_input_layernorm_weight_10063', 6719, True, 'model.model.layers.24.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_24_self_attn_q_proj_weight_10065', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_self_attn_q_proj_weight_10065', 6733, True, 'model.model.layers.24.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_24_self_attn_kv_a_proj_with_mqa_weight_57873', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_self_attn_kv_a_proj_with_mqa_weight_57873', 6749, True, 'model.model.layers.24.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_24_self_attn_kv_a_layernorm_weight_10077', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_self_attn_kv_a_layernorm_weight_10077', 6765, True, 'model.model.layers.24.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_24_self_attn_kv_b_proj_weight_10079', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_self_attn_kv_b_proj_weight_10079', 6773, True, 'model.model.layers.24.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_24_self_attn_rotary_emb_cos_cached_10085', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_24_self_attn_rotary_emb_cos_cached_10085', 6795, False, 'model.model.layers.24.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_24_self_attn_rotary_emb_sin_cached_10088', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_24_self_attn_rotary_emb_sin_cached_10088', 6803, False, 'model.model.layers.24.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_24_self_attn_o_proj_weight_10126', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_self_attn_o_proj_weight_10126', 6944, True, 'model.model.layers.24.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_24_post_attention_layernorm_weight_10129', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_post_attention_layernorm_weight_10129', 6950, True, 'model.model.layers.24.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_24_mlp_gate_weight_10131', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_mlp_gate_weight_10131', 6958, True, 'model.model.layers.24.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_24_mlp_gate_projs_59201', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_mlp_gate_projs_59201', 6971, True, 'model.model.layers.24.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_24_mlp_up_projs_59209', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_mlp_up_projs_59209', 6973, True, 'model.model.layers.24.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_24_mlp_down_projs_59217', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_mlp_down_projs_59217', 6975, True, 'model.model.layers.24.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_24_mlp_shared_experts_gate_proj_weight_10139', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_mlp_shared_experts_gate_proj_weight_10139', 6979, True, 'model.model.layers.24.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_24_mlp_shared_experts_up_proj_weight_10142', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_mlp_shared_experts_up_proj_weight_10142', 6985, True, 'model.model.layers.24.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_24_mlp_shared_experts_down_proj_weight_10145', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_24_mlp_shared_experts_down_proj_weight_10145', 6991, True, 'model.model.layers.24.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_25_input_layernorm_weight_10149', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_input_layernorm_weight_10149', 6999, True, 'model.model.layers.25.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_25_self_attn_q_proj_weight_10151', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_self_attn_q_proj_weight_10151', 7013, True, 'model.model.layers.25.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_25_self_attn_kv_a_proj_with_mqa_weight_59745', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_self_attn_kv_a_proj_with_mqa_weight_59745', 7029, True, 'model.model.layers.25.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_25_self_attn_kv_a_layernorm_weight_10163', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_self_attn_kv_a_layernorm_weight_10163', 7045, True, 'model.model.layers.25.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_25_self_attn_kv_b_proj_weight_10165', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_self_attn_kv_b_proj_weight_10165', 7053, True, 'model.model.layers.25.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_25_self_attn_rotary_emb_cos_cached_10171', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_25_self_attn_rotary_emb_cos_cached_10171', 7075, False, 'model.model.layers.25.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_25_self_attn_rotary_emb_sin_cached_10174', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_25_self_attn_rotary_emb_sin_cached_10174', 7083, False, 'model.model.layers.25.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_25_self_attn_o_proj_weight_10212', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_self_attn_o_proj_weight_10212', 7224, True, 'model.model.layers.25.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_25_post_attention_layernorm_weight_10215', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_post_attention_layernorm_weight_10215', 7230, True, 'model.model.layers.25.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_25_mlp_gate_weight_10217', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_mlp_gate_weight_10217', 7238, True, 'model.model.layers.25.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_25_mlp_gate_projs_61073', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_mlp_gate_projs_61073', 7251, True, 'model.model.layers.25.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_25_mlp_up_projs_61081', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_mlp_up_projs_61081', 7253, True, 'model.model.layers.25.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_25_mlp_down_projs_61089', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_mlp_down_projs_61089', 7255, True, 'model.model.layers.25.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_25_mlp_shared_experts_gate_proj_weight_10225', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_mlp_shared_experts_gate_proj_weight_10225', 7259, True, 'model.model.layers.25.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_25_mlp_shared_experts_up_proj_weight_10228', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_mlp_shared_experts_up_proj_weight_10228', 7265, True, 'model.model.layers.25.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_25_mlp_shared_experts_down_proj_weight_10231', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_25_mlp_shared_experts_down_proj_weight_10231', 7271, True, 'model.model.layers.25.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_layers_26_input_layernorm_weight_10235', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_input_layernorm_weight_10235', 7279, True, 'model.model.layers.26.input_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_26_self_attn_q_proj_weight_10237', torch.nn.Parameter(torch.empty((3072, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_self_attn_q_proj_weight_10237', 7293, True, 'model.model.layers.26.self_attn.q_proj.weight', (3072, 2048), (slice(0, 3072, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_26_self_attn_kv_a_proj_with_mqa_weight_61617', torch.nn.Parameter(torch.empty((576, 256), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_self_attn_kv_a_proj_with_mqa_weight_61617', 7309, True, 'model.model.layers.26.self_attn.kv_a_proj_with_mqa.weight', (576, 2048), (slice(0, 576, None), slice(1536, 1792, None)), 1)
        
        self.register_parameter('model_model_layers_26_self_attn_kv_a_layernorm_weight_10249', torch.nn.Parameter(torch.empty((512,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_self_attn_kv_a_layernorm_weight_10249', 7325, True, 'model.model.layers.26.self_attn.kv_a_layernorm.weight', (512,), (slice(0, 512, None),), 1)
        
        self.register_parameter('model_model_layers_26_self_attn_kv_b_proj_weight_10251', torch.nn.Parameter(torch.empty((4096, 512), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_self_attn_kv_b_proj_weight_10251', 7333, True, 'model.model.layers.26.self_attn.kv_b_proj.weight', (4096, 512), (slice(0, 4096, None), slice(0, 512, None)), 1)
        
        self.register_buffer('model_model_layers_26_self_attn_rotary_emb_cos_cached_10257', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_26_self_attn_rotary_emb_cos_cached_10257', 7355, False, 'model.model.layers.26.self_attn.rotary_emb.cos_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_buffer('model_model_layers_26_self_attn_rotary_emb_sin_cached_10260', torch.empty((163840, 64), dtype=torch.bfloat16), persistent=False)
        self.add_full_map('model_model_layers_26_self_attn_rotary_emb_sin_cached_10260', 7363, False, 'model.model.layers.26.self_attn.rotary_emb.sin_cached', (163840, 64), (slice(0, 163840, None), slice(0, 64, None)), 1)
        
        self.register_parameter('model_model_layers_26_self_attn_o_proj_weight_10298', torch.nn.Parameter(torch.empty((2048, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_self_attn_o_proj_weight_10298', 7504, True, 'model.model.layers.26.self_attn.o_proj.weight', (2048, 2048), (slice(0, 2048, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_26_post_attention_layernorm_weight_10301', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_post_attention_layernorm_weight_10301', 7510, True, 'model.model.layers.26.post_attention_layernorm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_model_layers_26_mlp_gate_weight_10303', torch.nn.Parameter(torch.empty((64, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_mlp_gate_weight_10303', 7518, True, 'model.model.layers.26.mlp.gate.weight', (64, 2048), (slice(0, 64, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_26_mlp_gate_projs_62945', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_mlp_gate_projs_62945', 7531, True, 'model.model.layers.26.mlp.gate_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_26_mlp_up_projs_62953', torch.nn.Parameter(torch.empty((8, 1408, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_mlp_up_projs_62953', 7533, True, 'model.model.layers.26.mlp.up_projs', (64, 1408, 2048), (slice(48, 56, None), slice(0, 1408, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_26_mlp_down_projs_62961', torch.nn.Parameter(torch.empty((8, 2048, 1408), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_mlp_down_projs_62961', 7535, True, 'model.model.layers.26.mlp.down_projs', (64, 2048, 1408), (slice(48, 56, None), slice(0, 2048, None), slice(0, 1408, None)), 1)
        
        self.register_parameter('model_model_layers_26_mlp_shared_experts_gate_proj_weight_10311', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_mlp_shared_experts_gate_proj_weight_10311', 7539, True, 'model.model.layers.26.mlp.shared_experts.gate_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_26_mlp_shared_experts_up_proj_weight_10314', torch.nn.Parameter(torch.empty((2816, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_mlp_shared_experts_up_proj_weight_10314', 7545, True, 'model.model.layers.26.mlp.shared_experts.up_proj.weight', (2816, 2048), (slice(0, 2816, None), slice(0, 2048, None)), 1)
        
        self.register_parameter('model_model_layers_26_mlp_shared_experts_down_proj_weight_10317', torch.nn.Parameter(torch.empty((2048, 2816), dtype=torch.bfloat16)))
        self.add_full_map('model_model_layers_26_mlp_shared_experts_down_proj_weight_10317', 7551, True, 'model.model.layers.26.mlp.shared_experts.down_proj.weight', (2048, 2816), (slice(0, 2048, None), slice(0, 2816, None)), 1)
        
        self.register_parameter('model_model_norm_weight_10321', torch.nn.Parameter(torch.empty((2048,), dtype=torch.bfloat16)))
        self.add_full_map('model_model_norm_weight_10321', 7559, True, 'model.model.norm.weight', (2048,), (slice(0, 2048, None),), 1)
        
        self.register_parameter('model_lm_head_weight_63345', torch.nn.Parameter(torch.empty((12800, 2048), dtype=torch.bfloat16)))
        self.add_full_map('model_lm_head_weight_63345', 7567, True, 'model.lm_head.weight', (102400, 2048), (slice(76800, 89600, None), slice(0, 2048, None)), 1)
        
        
        self.wreducer150596 = nnscaler.runtime.adapter.Reducer(ranks=[0, 1, 2, 3, 4, 5, 6, 7], reduce_op='sum', async_op=async_op, zero=True, max_bucket_size_bytes=max_bucket_size_bytes, zero_use_reduce_scatter=zero_use_reduce_scatter, zero_ngroups=1)
        self.wreducer150596.add_param(self.model_model_layers_0_self_attn_q_proj_weight_8010)
        self.wreducer150596.add_param(self.model_model_layers_0_self_attn_kv_a_layernorm_weight_8022)
        self.wreducer150596.add_param(self.model_model_layers_0_self_attn_kv_b_proj_weight_8024)
        self.wreducer150596.add_param(self.model_model_layers_0_mlp_gate_proj_weight_8076)
        self.wreducer150596.add_param(self.model_model_layers_0_mlp_up_proj_weight_8079)
        self.wreducer150596.add_param(self.model_model_layers_0_mlp_down_proj_weight_8082)
        self.wreducer150596.add_param(self.model_model_layers_1_self_attn_q_proj_weight_8087)
        self.wreducer150596.add_param(self.model_model_layers_1_self_attn_kv_a_layernorm_weight_8099)
        self.wreducer150596.add_param(self.model_model_layers_1_self_attn_kv_b_proj_weight_8101)
        self.wreducer150596.add_param(self.model_model_layers_1_mlp_shared_experts_gate_proj_weight_8161)
        self.wreducer150596.add_param(self.model_model_layers_1_mlp_shared_experts_up_proj_weight_8164)
        self.wreducer150596.add_param(self.model_model_layers_1_mlp_shared_experts_down_proj_weight_8167)
        self.wreducer150596.add_param(self.model_model_layers_2_self_attn_q_proj_weight_8173)
        self.wreducer150596.add_param(self.model_model_layers_2_self_attn_kv_a_layernorm_weight_8185)
        self.wreducer150596.add_param(self.model_model_layers_2_self_attn_kv_b_proj_weight_8187)
        self.wreducer150596.add_param(self.model_model_layers_2_mlp_shared_experts_gate_proj_weight_8247)
        self.wreducer150596.add_param(self.model_model_layers_2_mlp_shared_experts_up_proj_weight_8250)
        self.wreducer150596.add_param(self.model_model_layers_2_mlp_shared_experts_down_proj_weight_8253)
        self.wreducer150596.add_param(self.model_model_layers_3_self_attn_q_proj_weight_8259)
        self.wreducer150596.add_param(self.model_model_layers_3_self_attn_kv_a_layernorm_weight_8271)
        self.wreducer150596.add_param(self.model_model_layers_3_self_attn_kv_b_proj_weight_8273)
        self.wreducer150596.add_param(self.model_model_layers_3_mlp_shared_experts_gate_proj_weight_8333)
        self.wreducer150596.add_param(self.model_model_layers_3_mlp_shared_experts_up_proj_weight_8336)
        self.wreducer150596.add_param(self.model_model_layers_3_mlp_shared_experts_down_proj_weight_8339)
        self.wreducer150596.add_param(self.model_model_layers_4_self_attn_q_proj_weight_8345)
        self.wreducer150596.add_param(self.model_model_layers_4_self_attn_kv_a_layernorm_weight_8357)
        self.wreducer150596.add_param(self.model_model_layers_4_self_attn_kv_b_proj_weight_8359)
        self.wreducer150596.add_param(self.model_model_layers_4_mlp_shared_experts_gate_proj_weight_8419)
        self.wreducer150596.add_param(self.model_model_layers_4_mlp_shared_experts_up_proj_weight_8422)
        self.wreducer150596.add_param(self.model_model_layers_4_mlp_shared_experts_down_proj_weight_8425)
        self.wreducer150596.add_param(self.model_model_layers_5_self_attn_q_proj_weight_8431)
        self.wreducer150596.add_param(self.model_model_layers_5_self_attn_kv_a_layernorm_weight_8443)
        self.wreducer150596.add_param(self.model_model_layers_5_self_attn_kv_b_proj_weight_8445)
        self.wreducer150596.add_param(self.model_model_layers_5_mlp_shared_experts_gate_proj_weight_8505)
        self.wreducer150596.add_param(self.model_model_layers_5_mlp_shared_experts_up_proj_weight_8508)
        self.wreducer150596.add_param(self.model_model_layers_5_mlp_shared_experts_down_proj_weight_8511)
        self.wreducer150596.add_param(self.model_model_layers_6_self_attn_q_proj_weight_8517)
        self.wreducer150596.add_param(self.model_model_layers_6_self_attn_kv_a_layernorm_weight_8529)
        self.wreducer150596.add_param(self.model_model_layers_6_self_attn_kv_b_proj_weight_8531)
        self.wreducer150596.add_param(self.model_model_layers_6_mlp_shared_experts_gate_proj_weight_8591)
        self.wreducer150596.add_param(self.model_model_layers_6_mlp_shared_experts_up_proj_weight_8594)
        self.wreducer150596.add_param(self.model_model_layers_6_mlp_shared_experts_down_proj_weight_8597)
        self.wreducer150596.add_param(self.model_model_layers_7_self_attn_q_proj_weight_8603)
        self.wreducer150596.add_param(self.model_model_layers_7_self_attn_kv_a_layernorm_weight_8615)
        self.wreducer150596.add_param(self.model_model_layers_7_self_attn_kv_b_proj_weight_8617)
        self.wreducer150596.add_param(self.model_model_layers_7_mlp_shared_experts_gate_proj_weight_8677)
        self.wreducer150596.add_param(self.model_model_layers_7_mlp_shared_experts_up_proj_weight_8680)
        self.wreducer150596.add_param(self.model_model_layers_7_mlp_shared_experts_down_proj_weight_8683)
        self.wreducer150596.add_param(self.model_model_layers_8_self_attn_q_proj_weight_8689)
        self.wreducer150596.add_param(self.model_model_layers_8_self_attn_kv_a_layernorm_weight_8701)
        self.wreducer150596.add_param(self.model_model_layers_8_self_attn_kv_b_proj_weight_8703)
        self.wreducer150596.add_param(self.model_model_layers_8_mlp_shared_experts_gate_proj_weight_8763)
        self.wreducer150596.add_param(self.model_model_layers_8_mlp_shared_experts_up_proj_weight_8766)
        self.wreducer150596.add_param(self.model_model_layers_8_mlp_shared_experts_down_proj_weight_8769)
        self.wreducer150596.add_param(self.model_model_layers_9_self_attn_q_proj_weight_8775)
        self.wreducer150596.add_param(self.model_model_layers_9_self_attn_kv_a_layernorm_weight_8787)
        self.wreducer150596.add_param(self.model_model_layers_9_self_attn_kv_b_proj_weight_8789)
        self.wreducer150596.add_param(self.model_model_layers_9_mlp_shared_experts_gate_proj_weight_8849)
        self.wreducer150596.add_param(self.model_model_layers_9_mlp_shared_experts_up_proj_weight_8852)
        self.wreducer150596.add_param(self.model_model_layers_9_mlp_shared_experts_down_proj_weight_8855)
        self.wreducer150596.add_param(self.model_model_layers_10_self_attn_q_proj_weight_8861)
        self.wreducer150596.add_param(self.model_model_layers_10_self_attn_kv_a_layernorm_weight_8873)
        self.wreducer150596.add_param(self.model_model_layers_10_self_attn_kv_b_proj_weight_8875)
        self.wreducer150596.add_param(self.model_model_layers_10_mlp_shared_experts_gate_proj_weight_8935)
        self.wreducer150596.add_param(self.model_model_layers_10_mlp_shared_experts_up_proj_weight_8938)
        self.wreducer150596.add_param(self.model_model_layers_10_mlp_shared_experts_down_proj_weight_8941)
        self.wreducer150596.add_param(self.model_model_layers_11_self_attn_q_proj_weight_8947)
        self.wreducer150596.add_param(self.model_model_layers_11_self_attn_kv_a_layernorm_weight_8959)
        self.wreducer150596.add_param(self.model_model_layers_11_self_attn_kv_b_proj_weight_8961)
        self.wreducer150596.add_param(self.model_model_layers_11_mlp_shared_experts_gate_proj_weight_9021)
        self.wreducer150596.add_param(self.model_model_layers_11_mlp_shared_experts_up_proj_weight_9024)
        self.wreducer150596.add_param(self.model_model_layers_11_mlp_shared_experts_down_proj_weight_9027)
        self.wreducer150596.add_param(self.model_model_layers_12_self_attn_q_proj_weight_9033)
        self.wreducer150596.add_param(self.model_model_layers_12_self_attn_kv_a_layernorm_weight_9045)
        self.wreducer150596.add_param(self.model_model_layers_12_self_attn_kv_b_proj_weight_9047)
        self.wreducer150596.add_param(self.model_model_layers_12_mlp_shared_experts_gate_proj_weight_9107)
        self.wreducer150596.add_param(self.model_model_layers_12_mlp_shared_experts_up_proj_weight_9110)
        self.wreducer150596.add_param(self.model_model_layers_12_mlp_shared_experts_down_proj_weight_9113)
        self.wreducer150596.add_param(self.model_model_layers_13_self_attn_q_proj_weight_9119)
        self.wreducer150596.add_param(self.model_model_layers_13_self_attn_kv_a_layernorm_weight_9131)
        self.wreducer150596.add_param(self.model_model_layers_13_self_attn_kv_b_proj_weight_9133)
        self.wreducer150596.add_param(self.model_model_layers_13_mlp_shared_experts_gate_proj_weight_9193)
        self.wreducer150596.add_param(self.model_model_layers_13_mlp_shared_experts_up_proj_weight_9196)
        self.wreducer150596.add_param(self.model_model_layers_13_mlp_shared_experts_down_proj_weight_9199)
        self.wreducer150596.add_param(self.model_model_layers_14_self_attn_q_proj_weight_9205)
        self.wreducer150596.add_param(self.model_model_layers_14_self_attn_kv_a_layernorm_weight_9217)
        self.wreducer150596.add_param(self.model_model_layers_14_self_attn_kv_b_proj_weight_9219)
        self.wreducer150596.add_param(self.model_model_layers_14_mlp_shared_experts_gate_proj_weight_9279)
        self.wreducer150596.add_param(self.model_model_layers_14_mlp_shared_experts_up_proj_weight_9282)
        self.wreducer150596.add_param(self.model_model_layers_14_mlp_shared_experts_down_proj_weight_9285)
        self.wreducer150596.add_param(self.model_model_layers_15_self_attn_q_proj_weight_9291)
        self.wreducer150596.add_param(self.model_model_layers_15_self_attn_kv_a_layernorm_weight_9303)
        self.wreducer150596.add_param(self.model_model_layers_15_self_attn_kv_b_proj_weight_9305)
        self.wreducer150596.add_param(self.model_model_layers_15_mlp_shared_experts_gate_proj_weight_9365)
        self.wreducer150596.add_param(self.model_model_layers_15_mlp_shared_experts_up_proj_weight_9368)
        self.wreducer150596.add_param(self.model_model_layers_15_mlp_shared_experts_down_proj_weight_9371)
        self.wreducer150596.add_param(self.model_model_layers_16_self_attn_q_proj_weight_9377)
        self.wreducer150596.add_param(self.model_model_layers_16_self_attn_kv_a_layernorm_weight_9389)
        self.wreducer150596.add_param(self.model_model_layers_16_self_attn_kv_b_proj_weight_9391)
        self.wreducer150596.add_param(self.model_model_layers_16_mlp_shared_experts_gate_proj_weight_9451)
        self.wreducer150596.add_param(self.model_model_layers_16_mlp_shared_experts_up_proj_weight_9454)
        self.wreducer150596.add_param(self.model_model_layers_16_mlp_shared_experts_down_proj_weight_9457)
        self.wreducer150596.add_param(self.model_model_layers_17_self_attn_q_proj_weight_9463)
        self.wreducer150596.add_param(self.model_model_layers_17_self_attn_kv_a_layernorm_weight_9475)
        self.wreducer150596.add_param(self.model_model_layers_17_self_attn_kv_b_proj_weight_9477)
        self.wreducer150596.add_param(self.model_model_layers_17_mlp_shared_experts_gate_proj_weight_9537)
        self.wreducer150596.add_param(self.model_model_layers_17_mlp_shared_experts_up_proj_weight_9540)
        self.wreducer150596.add_param(self.model_model_layers_17_mlp_shared_experts_down_proj_weight_9543)
        self.wreducer150596.add_param(self.model_model_layers_18_self_attn_q_proj_weight_9549)
        self.wreducer150596.add_param(self.model_model_layers_18_self_attn_kv_a_layernorm_weight_9561)
        self.wreducer150596.add_param(self.model_model_layers_18_self_attn_kv_b_proj_weight_9563)
        self.wreducer150596.add_param(self.model_model_layers_18_mlp_shared_experts_gate_proj_weight_9623)
        self.wreducer150596.add_param(self.model_model_layers_18_mlp_shared_experts_up_proj_weight_9626)
        self.wreducer150596.add_param(self.model_model_layers_18_mlp_shared_experts_down_proj_weight_9629)
        self.wreducer150596.add_param(self.model_model_layers_19_self_attn_q_proj_weight_9635)
        self.wreducer150596.add_param(self.model_model_layers_19_self_attn_kv_a_layernorm_weight_9647)
        self.wreducer150596.add_param(self.model_model_layers_19_self_attn_kv_b_proj_weight_9649)
        self.wreducer150596.add_param(self.model_model_layers_19_mlp_shared_experts_gate_proj_weight_9709)
        self.wreducer150596.add_param(self.model_model_layers_19_mlp_shared_experts_up_proj_weight_9712)
        self.wreducer150596.add_param(self.model_model_layers_19_mlp_shared_experts_down_proj_weight_9715)
        self.wreducer150596.add_param(self.model_model_layers_20_self_attn_q_proj_weight_9721)
        self.wreducer150596.add_param(self.model_model_layers_20_self_attn_kv_a_layernorm_weight_9733)
        self.wreducer150596.add_param(self.model_model_layers_20_self_attn_kv_b_proj_weight_9735)
        self.wreducer150596.add_param(self.model_model_layers_20_mlp_shared_experts_gate_proj_weight_9795)
        self.wreducer150596.add_param(self.model_model_layers_20_mlp_shared_experts_up_proj_weight_9798)
        self.wreducer150596.add_param(self.model_model_layers_20_mlp_shared_experts_down_proj_weight_9801)
        self.wreducer150596.add_param(self.model_model_layers_21_self_attn_q_proj_weight_9807)
        self.wreducer150596.add_param(self.model_model_layers_21_self_attn_kv_a_layernorm_weight_9819)
        self.wreducer150596.add_param(self.model_model_layers_21_self_attn_kv_b_proj_weight_9821)
        self.wreducer150596.add_param(self.model_model_layers_21_mlp_shared_experts_gate_proj_weight_9881)
        self.wreducer150596.add_param(self.model_model_layers_21_mlp_shared_experts_up_proj_weight_9884)
        self.wreducer150596.add_param(self.model_model_layers_21_mlp_shared_experts_down_proj_weight_9887)
        self.wreducer150596.add_param(self.model_model_layers_22_self_attn_q_proj_weight_9893)
        self.wreducer150596.add_param(self.model_model_layers_22_self_attn_kv_a_layernorm_weight_9905)
        self.wreducer150596.add_param(self.model_model_layers_22_self_attn_kv_b_proj_weight_9907)
        self.wreducer150596.add_param(self.model_model_layers_22_mlp_shared_experts_gate_proj_weight_9967)
        self.wreducer150596.add_param(self.model_model_layers_22_mlp_shared_experts_up_proj_weight_9970)
        self.wreducer150596.add_param(self.model_model_layers_22_mlp_shared_experts_down_proj_weight_9973)
        self.wreducer150596.add_param(self.model_model_layers_23_self_attn_q_proj_weight_9979)
        self.wreducer150596.add_param(self.model_model_layers_23_self_attn_kv_a_layernorm_weight_9991)
        self.wreducer150596.add_param(self.model_model_layers_23_self_attn_kv_b_proj_weight_9993)
        self.wreducer150596.add_param(self.model_model_layers_23_mlp_shared_experts_gate_proj_weight_10053)
        self.wreducer150596.add_param(self.model_model_layers_23_mlp_shared_experts_up_proj_weight_10056)
        self.wreducer150596.add_param(self.model_model_layers_23_mlp_shared_experts_down_proj_weight_10059)
        self.wreducer150596.add_param(self.model_model_layers_24_self_attn_q_proj_weight_10065)
        self.wreducer150596.add_param(self.model_model_layers_24_self_attn_kv_a_layernorm_weight_10077)
        self.wreducer150596.add_param(self.model_model_layers_24_self_attn_kv_b_proj_weight_10079)
        self.wreducer150596.add_param(self.model_model_layers_24_mlp_shared_experts_gate_proj_weight_10139)
        self.wreducer150596.add_param(self.model_model_layers_24_mlp_shared_experts_up_proj_weight_10142)
        self.wreducer150596.add_param(self.model_model_layers_24_mlp_shared_experts_down_proj_weight_10145)
        self.wreducer150596.add_param(self.model_model_layers_25_self_attn_q_proj_weight_10151)
        self.wreducer150596.add_param(self.model_model_layers_25_self_attn_kv_a_layernorm_weight_10163)
        self.wreducer150596.add_param(self.model_model_layers_25_self_attn_kv_b_proj_weight_10165)
        self.wreducer150596.add_param(self.model_model_layers_25_mlp_shared_experts_gate_proj_weight_10225)
        self.wreducer150596.add_param(self.model_model_layers_25_mlp_shared_experts_up_proj_weight_10228)
        self.wreducer150596.add_param(self.model_model_layers_25_mlp_shared_experts_down_proj_weight_10231)
        self.wreducer150596.add_param(self.model_model_layers_26_self_attn_q_proj_weight_10237)
        self.wreducer150596.add_param(self.model_model_layers_26_self_attn_kv_a_layernorm_weight_10249)
        self.wreducer150596.add_param(self.model_model_layers_26_self_attn_kv_b_proj_weight_10251)
        self.wreducer150596.add_param(self.model_model_layers_26_mlp_shared_experts_gate_proj_weight_10311)
        self.wreducer150596.add_param(self.model_model_layers_26_mlp_shared_experts_up_proj_weight_10314)
        self.wreducer150596.add_param(self.model_model_layers_26_mlp_shared_experts_down_proj_weight_10317)
        self.add_reducer(self.wreducer150596)
        
        self._post_init(init_params, build_buckets)
    
    def segment160545(self, samples_10336):
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/train.py", line 126, in forward,  input_ids=samples['net_input']['src_tokens'],
        getitem_7591 = _operator.getitem(samples_10336, 'net_input')
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/train.py", line 126, in forward,  input_ids=samples['net_input']['src_tokens'],
        getitem_1_8003 = _operator.getitem(getitem_7591, 'src_tokens')
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1495, in forward,  position_ids = torch.arange(
        arange_8004 = nnscaler.runtime.function.arange(start=0, end=2048, step=1, dtype=torch.int64, requires_grad=False)
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1501, in forward,  position_ids = position_ids.unsqueeze(0)
        unsqueeze_8005 = torch.unsqueeze(arange_8004, dim=0)
        del arange_8004
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1504, in forward,  inputs_embeds = self.embed_tokens(input_ids)
        embedding_12841 = nnscaler.runtime.function.embedding(getitem_1_8003, self.model_model_embed_tokens_weight_12833, padding_idx=None, start=0, stop=102400)
        del getitem_1_8003
        embedding_8007 = nnscaler.runtime.adapter.nn.allgather_split(embedding_12841, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del embedding_12841
        # created at IRAdapterGener:local_consumer_multiref
        embedding_80212, embedding_80216 = nnscaler.runtime.function.multiref(embedding_8007, times=2)
        del embedding_8007
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_8009 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(embedding_80212, self.model_model_layers_0_input_layernorm_weight_8008, (2048,), 1e-06)
        del embedding_80212
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_63423, fused_rms_norm_affine_63424 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_8009, times=2)
        del fused_rms_norm_affine_8009
        fused_rms_norm_affine_64616 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_63423, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_63423
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_12889 = torch.nn.functional.linear(fused_rms_norm_affine_64616, self.model_model_layers_0_self_attn_q_proj_weight_8010, bias=None)
        del fused_rms_norm_affine_64616
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_12945 = torch.Tensor.view(linear_12889, size=(8, 256, 16, 192))
        del linear_12889
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_12969 = torch.transpose(view_12945, dim0=1, dim1=2)
        del view_12945
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_13033, split_13041 = torch.functional.split(transpose_12969, split_size_or_sections=[128, 64], dim=-1)
        del transpose_12969
        fused_rms_norm_affine_64680 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_63424, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_63424
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_1_13073 = torch.nn.functional.linear(fused_rms_norm_affine_64680, self.model_model_layers_0_self_attn_kv_a_proj_with_mqa_weight_13065, bias=None)
        del fused_rms_norm_affine_64680
        linear_1_8017 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_1_13073, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_1_13073
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_1_8018, split_1_8019 = torch.functional.split(linear_1_8017, split_size_or_sections=[512, 64], dim=-1)
        del linear_1_8017
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_1_8020 = torch.Tensor.view(split_1_8019, size=(8, 2048, 1, 64))
        del split_1_8019
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_1_8021 = torch.transpose(view_1_8020, dim0=1, dim1=2)
        del view_1_8020
        split_1_13113 = nnscaler.runtime.adapter.nn.split_allgather(split_1_8018, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_1_8018
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_1_13193 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_1_13113, self.model_model_layers_0_self_attn_kv_a_layernorm_weight_8022, (512,), 1e-06)
        del split_1_13113
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_2_13209 = torch.nn.functional.linear(fused_rms_norm_affine_1_13193, self.model_model_layers_0_self_attn_kv_b_proj_weight_8024, bias=None)
        del fused_rms_norm_affine_1_13193
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_2_13265 = torch.Tensor.view(linear_2_13209, size=(8, 256, 16, 256))
        del linear_2_13209
        view_2_13257 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_2_13265, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_2_13265
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_2_13281 = torch.transpose(view_2_13257, dim0=1, dim1=2)
        del view_2_13257
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_2_13321, split_2_13329 = torch.functional.split(transpose_2_13281, split_size_or_sections=[128, 128], dim=-1)
        del transpose_2_13281
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_16_8031 = nnscaler.runtime.function.fullslice(self.model_model_layers_0_self_attn_rotary_emb_cos_cached_8030, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_8032 = torch.Tensor.to(getitem_16_8031, dtype=torch.bfloat16)
        del getitem_16_8031
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_17_8034 = nnscaler.runtime.function.fullslice(self.model_model_layers_0_self_attn_rotary_emb_sin_cached_8033, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_1_8035 = torch.Tensor.to(getitem_17_8034, dtype=torch.bfloat16)
        del getitem_17_8034
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_18_8036 = nnscaler.runtime.function.fullslice(to_8032, unsqueeze_8005)
        del to_8032
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_1_8037 = torch.unsqueeze(getitem_18_8036, dim=1)
        del getitem_18_8036
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_1_63429, unsqueeze_1_63430 = nnscaler.runtime.function.multiref(unsqueeze_1_8037, times=2)
        del unsqueeze_1_8037
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_19_8038 = nnscaler.runtime.function.fullslice(to_1_8035, unsqueeze_8005)
        del to_1_8035
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_2_8039 = torch.unsqueeze(getitem_19_8038, dim=1)
        del getitem_19_8038
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_2_63433, unsqueeze_2_63434 = nnscaler.runtime.function.multiref(unsqueeze_2_8039, times=2)
        del unsqueeze_2_8039
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_3_13417 = torch.Tensor.view(split_13041, size=(8, 16, 256, 32, 2))
        del split_13041
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_3_13457 = torch.transpose(view_3_13417, dim0=4, dim1=3)
        del view_3_13417
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_13489 = torch.Tensor.reshape(transpose_3_13457, shape=(8, 16, 256, 64))
        del transpose_3_13457
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_4_8043 = torch.Tensor.view(transpose_1_8021, size=(8, 1, 2048, 32, 2))
        del transpose_1_8021
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_4_8044 = torch.transpose(view_4_8043, dim0=4, dim1=3)
        del view_4_8043
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_1_8045 = torch.Tensor.reshape(transpose_4_8044, shape=(8, 1, 2048, 64))
        del transpose_4_8044
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_1_63441, reshape_1_63442, reshape_1_63443 = nnscaler.runtime.function.multiref(reshape_1_8045, times=3)
        del reshape_1_8045
        unsqueeze_1_64808 = nnscaler.runtime.adapter.chunk(unsqueeze_1_63429, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_1_63429
        # created at IRAdapterGener:local_consumer_multiref
        reshape_80307, reshape_80311, reshape_80315 = nnscaler.runtime.function.multiref(reshape_13489, times=3)
        del reshape_13489
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_13585 = torch.mul(reshape_80307, unsqueeze_1_64808)
        del unsqueeze_1_64808, reshape_80307
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_29_13633 = nnscaler.runtime.function.fullslice(reshape_80311, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_80311
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_31_13657 = nnscaler.runtime.function.fullslice(reshape_80315, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_80315
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_13681 = _operator.neg(getitem_31_13657)
        del getitem_31_13657
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_13721 = nnscaler.runtime.function.cat(neg_13681, getitem_29_13633, dim=-1)
        del getitem_29_13633, neg_13681
        unsqueeze_2_64880 = nnscaler.runtime.adapter.chunk(unsqueeze_2_63433, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_2_63433
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_1_13753 = torch.mul(cat_13721, unsqueeze_2_64880)
        del cat_13721, unsqueeze_2_64880
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_1_13801 = torch.add(mul_13585, mul_1_13753, alpha=1)
        del mul_13585, mul_1_13753
        unsqueeze_1_64912 = nnscaler.runtime.adapter.chunk(unsqueeze_1_63430, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_1_63430
        reshape_1_64904 = nnscaler.runtime.adapter.nn.split_allgather(reshape_1_63443, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_1_63443
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_2_13841 = torch.mul(reshape_1_64904, unsqueeze_1_64912)
        del unsqueeze_1_64912, reshape_1_64904
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_33_8054 = nnscaler.runtime.function.fullslice(reshape_1_63441, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_1_63441
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_35_8055 = nnscaler.runtime.function.fullslice(reshape_1_63442, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_1_63442
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_1_8056 = _operator.neg(getitem_35_8055)
        del getitem_35_8055
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_1_8057 = nnscaler.runtime.function.cat(neg_1_8056, getitem_33_8054, dim=-1)
        del getitem_33_8054, neg_1_8056
        cat_1_13945 = nnscaler.runtime.adapter.nn.split_allgather(cat_1_8057, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_1_8057
        unsqueeze_2_64936 = nnscaler.runtime.adapter.chunk(unsqueeze_2_63434, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_2_63434
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_3_13953 = torch.mul(cat_1_13945, unsqueeze_2_64936)
        del cat_1_13945, unsqueeze_2_64936
        mul_2_8053 = nnscaler.runtime.adapter.nn.allgather_split(mul_2_13841, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_2_13841
        mul_3_8058 = nnscaler.runtime.adapter.nn.allgather_split(mul_3_13953, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_3_13953
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_2_8059 = torch.add(mul_2_8053, mul_3_8058, alpha=1)
        del mul_2_8053, mul_3_8058
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_2_14001 = nnscaler.runtime.function.cat(split_13033, add_1_13801, dim=-1)
        del split_13033, add_1_13801
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_8061 = torch.Tensor.expand(add_2_8059, size=[-1, 16, -1, -1])
        del add_2_8059
        split_2_13337 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_2_13321, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_2_13321
        expand_14041 = nnscaler.runtime.adapter.nn.split_allgather(expand_8061, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_8061
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_3_14049 = nnscaler.runtime.function.cat(split_2_13337, expand_14041, dim=-1)
        del split_2_13337, expand_14041
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_14065 = torch.nn.functional.pad(split_2_13329, pad=[0, 64], mode='constant', value=0.0)
        del split_2_13329
        cat_2_13993 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_2_14001, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_2_14001
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_5_14097 = torch.transpose(cat_2_13993, dim0=1, dim1=2)
        del cat_2_13993
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_6_14137 = torch.transpose(cat_3_14049, dim0=1, dim1=2)
        del cat_3_14049
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_7_14169 = torch.transpose(pad_14065, dim0=1, dim1=2)
        del pad_14065
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_0_self_attn_training_7604 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_7605 = 0.0 if model_model_layers_0_self_attn_training_7604 else 0.0
        transpose_7_14177 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_7_14169, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_7_14169
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_14217 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_5_14097, transpose_6_14137, transpose_7_14177, dropout=ifexpr_7605, causal=True, attention_mask=None, query_length=2048)
        del transpose_5_14097, transpose_6_14137, transpose_7_14177
        nnscaler_flash_attention_forward_8067 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_14217, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_14217
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_36_8068 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_8067, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_8067
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_2_8069 = torch.Tensor.reshape(getitem_36_8068, shape=(8, 2048, 2048))
        del getitem_36_8068
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_8070 = torch.Tensor.contiguous(reshape_2_8069)
        del reshape_2_8069
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_3_8072 = torch.nn.functional.linear(contiguous_8070, self.model_model_layers_0_self_attn_o_proj_weight_8071, bias=None)
        del contiguous_8070
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_3_8073 = torch.add(embedding_80216, linear_3_8072, alpha=1)
        del embedding_80216, linear_3_8072
        # created at IRAdapterGener:local_consumer_multiref
        add_3_80380, add_3_80384 = nnscaler.runtime.function.multiref(add_3_8073, times=2)
        del add_3_8073
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_2_8075 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_3_80380, self.model_model_layers_0_post_attention_layernorm_weight_8074, (2048,), 1e-06)
        del add_3_80380
        fused_rms_norm_affine_2_14385 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_2_8075, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_2_8075
        # created at IRAdapterGener:local_consumer_multiref
        fused_rms_norm_affine_2_80451, fused_rms_norm_affine_2_80455 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_2_14385, times=2)
        del fused_rms_norm_affine_2_14385
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_4_14401 = torch.nn.functional.linear(fused_rms_norm_affine_2_80451, self.model_model_layers_0_mlp_gate_proj_weight_8076, bias=None)
        del fused_rms_norm_affine_2_80451
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_14457 = torch.nn.functional.silu(linear_4_14401, inplace=False)
        del linear_4_14401
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_5_14481 = torch.nn.functional.linear(fused_rms_norm_affine_2_80455, self.model_model_layers_0_mlp_up_proj_weight_8079, bias=None)
        del fused_rms_norm_affine_2_80455
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_4_14529 = torch.mul(silu_14457, linear_5_14481)
        del silu_14457, linear_5_14481
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_6_14553 = torch.nn.functional.linear(mul_4_14529, self.model_model_layers_0_mlp_down_proj_weight_8082, bias=None)
        del mul_4_14529
        linear_6_8083 = nnscaler.runtime.adapter.nn.allgather_split(linear_6_14553, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_6_14553
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_4_8084 = torch.add(add_3_80384, linear_6_8083, alpha=1)
        del add_3_80384, linear_6_8083
        # created at IRAdapterGener:local_consumer_multiref
        add_4_80516, add_4_80520 = nnscaler.runtime.function.multiref(add_4_8084, times=2)
        del add_4_8084
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_3_8086 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_4_80516, self.model_model_layers_1_input_layernorm_weight_8085, (2048,), 1e-06)
        del add_4_80516
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_3_63451, fused_rms_norm_affine_3_63452 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_3_8086, times=2)
        del fused_rms_norm_affine_3_8086
        fused_rms_norm_affine_3_65128 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_3_63451, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_3_63451
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_7_14641 = torch.nn.functional.linear(fused_rms_norm_affine_3_65128, self.model_model_layers_1_self_attn_q_proj_weight_8087, bias=None)
        del fused_rms_norm_affine_3_65128
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_5_14697 = torch.Tensor.view(linear_7_14641, size=(8, 256, 16, 192))
        del linear_7_14641
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_8_14721 = torch.transpose(view_5_14697, dim0=1, dim1=2)
        del view_5_14697
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_3_14785, split_3_14793 = torch.functional.split(transpose_8_14721, split_size_or_sections=[128, 64], dim=-1)
        del transpose_8_14721
        fused_rms_norm_affine_3_65192 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_3_63452, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_3_63452
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_8_14825 = torch.nn.functional.linear(fused_rms_norm_affine_3_65192, self.model_model_layers_1_self_attn_kv_a_proj_with_mqa_weight_14817, bias=None)
        del fused_rms_norm_affine_3_65192
        linear_8_8094 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_8_14825, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_8_14825
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_4_8095, split_4_8096 = torch.functional.split(linear_8_8094, split_size_or_sections=[512, 64], dim=-1)
        del linear_8_8094
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_6_8097 = torch.Tensor.view(split_4_8096, size=(8, 2048, 1, 64))
        del split_4_8096
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_9_8098 = torch.transpose(view_6_8097, dim0=1, dim1=2)
        del view_6_8097
        split_4_14865 = nnscaler.runtime.adapter.nn.split_allgather(split_4_8095, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_4_8095
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_4_14945 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_4_14865, self.model_model_layers_1_self_attn_kv_a_layernorm_weight_8099, (512,), 1e-06)
        del split_4_14865
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_9_14961 = torch.nn.functional.linear(fused_rms_norm_affine_4_14945, self.model_model_layers_1_self_attn_kv_b_proj_weight_8101, bias=None)
        del fused_rms_norm_affine_4_14945
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_7_15017 = torch.Tensor.view(linear_9_14961, size=(8, 256, 16, 256))
        del linear_9_14961
        view_7_15009 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_7_15017, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_7_15017
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_10_15033 = torch.transpose(view_7_15009, dim0=1, dim1=2)
        del view_7_15009
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_5_15073, split_5_15081 = torch.functional.split(transpose_10_15033, split_size_or_sections=[128, 128], dim=-1)
        del transpose_10_15033
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_48_8108 = nnscaler.runtime.function.fullslice(self.model_model_layers_1_self_attn_rotary_emb_cos_cached_8107, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_2_8109 = torch.Tensor.to(getitem_48_8108, dtype=torch.bfloat16)
        del getitem_48_8108
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_49_8111 = nnscaler.runtime.function.fullslice(self.model_model_layers_1_self_attn_rotary_emb_sin_cached_8110, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_3_8112 = torch.Tensor.to(getitem_49_8111, dtype=torch.bfloat16)
        del getitem_49_8111
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_50_8113 = nnscaler.runtime.function.fullslice(to_2_8109, unsqueeze_8005)
        del to_2_8109
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_3_8114 = torch.unsqueeze(getitem_50_8113, dim=1)
        del getitem_50_8113
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_3_63457, unsqueeze_3_63458 = nnscaler.runtime.function.multiref(unsqueeze_3_8114, times=2)
        del unsqueeze_3_8114
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_51_8115 = nnscaler.runtime.function.fullslice(to_3_8112, unsqueeze_8005)
        del to_3_8112
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_4_8116 = torch.unsqueeze(getitem_51_8115, dim=1)
        del getitem_51_8115
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_4_63461, unsqueeze_4_63462 = nnscaler.runtime.function.multiref(unsqueeze_4_8116, times=2)
        del unsqueeze_4_8116
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_8_15169 = torch.Tensor.view(split_3_14793, size=(8, 16, 256, 32, 2))
        del split_3_14793
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_11_15209 = torch.transpose(view_8_15169, dim0=4, dim1=3)
        del view_8_15169
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_3_15241 = torch.Tensor.reshape(transpose_11_15209, shape=(8, 16, 256, 64))
        del transpose_11_15209
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_9_8120 = torch.Tensor.view(transpose_9_8098, size=(8, 1, 2048, 32, 2))
        del transpose_9_8098
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_12_8121 = torch.transpose(view_9_8120, dim0=4, dim1=3)
        del view_9_8120
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_4_8122 = torch.Tensor.reshape(transpose_12_8121, shape=(8, 1, 2048, 64))
        del transpose_12_8121
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_4_63469, reshape_4_63470, reshape_4_63471 = nnscaler.runtime.function.multiref(reshape_4_8122, times=3)
        del reshape_4_8122
        unsqueeze_3_65320 = nnscaler.runtime.adapter.chunk(unsqueeze_3_63457, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_3_63457
        # created at IRAdapterGener:local_consumer_multiref
        reshape_3_80611, reshape_3_80615, reshape_3_80619 = nnscaler.runtime.function.multiref(reshape_3_15241, times=3)
        del reshape_3_15241
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_5_15337 = torch.mul(reshape_3_80611, unsqueeze_3_65320)
        del unsqueeze_3_65320, reshape_3_80611
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_61_15385 = nnscaler.runtime.function.fullslice(reshape_3_80615, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_3_80615
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_63_15409 = nnscaler.runtime.function.fullslice(reshape_3_80619, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_3_80619
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_2_15433 = _operator.neg(getitem_63_15409)
        del getitem_63_15409
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_4_15473 = nnscaler.runtime.function.cat(neg_2_15433, getitem_61_15385, dim=-1)
        del getitem_61_15385, neg_2_15433
        unsqueeze_4_65392 = nnscaler.runtime.adapter.chunk(unsqueeze_4_63461, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_4_63461
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_6_15505 = torch.mul(cat_4_15473, unsqueeze_4_65392)
        del cat_4_15473, unsqueeze_4_65392
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_5_15553 = torch.add(mul_5_15337, mul_6_15505, alpha=1)
        del mul_5_15337, mul_6_15505
        unsqueeze_3_65424 = nnscaler.runtime.adapter.chunk(unsqueeze_3_63458, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_3_63458
        reshape_4_65416 = nnscaler.runtime.adapter.nn.split_allgather(reshape_4_63471, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_4_63471
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_7_15593 = torch.mul(reshape_4_65416, unsqueeze_3_65424)
        del unsqueeze_3_65424, reshape_4_65416
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_65_8131 = nnscaler.runtime.function.fullslice(reshape_4_63469, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_4_63469
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_67_8132 = nnscaler.runtime.function.fullslice(reshape_4_63470, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_4_63470
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_3_8133 = _operator.neg(getitem_67_8132)
        del getitem_67_8132
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_5_8134 = nnscaler.runtime.function.cat(neg_3_8133, getitem_65_8131, dim=-1)
        del getitem_65_8131, neg_3_8133
        cat_5_15697 = nnscaler.runtime.adapter.nn.split_allgather(cat_5_8134, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_5_8134
        unsqueeze_4_65448 = nnscaler.runtime.adapter.chunk(unsqueeze_4_63462, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_4_63462
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_8_15705 = torch.mul(cat_5_15697, unsqueeze_4_65448)
        del cat_5_15697, unsqueeze_4_65448
        mul_7_8130 = nnscaler.runtime.adapter.nn.allgather_split(mul_7_15593, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_7_15593
        mul_8_8135 = nnscaler.runtime.adapter.nn.allgather_split(mul_8_15705, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_8_15705
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_6_8136 = torch.add(mul_7_8130, mul_8_8135, alpha=1)
        del mul_7_8130, mul_8_8135
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_6_15753 = nnscaler.runtime.function.cat(split_3_14785, add_5_15553, dim=-1)
        del split_3_14785, add_5_15553
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_1_8138 = torch.Tensor.expand(add_6_8136, size=[-1, 16, -1, -1])
        del add_6_8136
        split_5_15089 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_5_15073, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_5_15073
        expand_1_15793 = nnscaler.runtime.adapter.nn.split_allgather(expand_1_8138, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_1_8138
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_7_15801 = nnscaler.runtime.function.cat(split_5_15089, expand_1_15793, dim=-1)
        del split_5_15089, expand_1_15793
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_1_15817 = torch.nn.functional.pad(split_5_15081, pad=[0, 64], mode='constant', value=0.0)
        del split_5_15081
        cat_6_15745 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_6_15753, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_6_15753
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_13_15849 = torch.transpose(cat_6_15745, dim0=1, dim1=2)
        del cat_6_15745
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_14_15889 = torch.transpose(cat_7_15801, dim0=1, dim1=2)
        del cat_7_15801
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_15_15921 = torch.transpose(pad_1_15817, dim0=1, dim1=2)
        del pad_1_15817
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_1_self_attn_training_7618 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_1_7619 = 0.0 if model_model_layers_1_self_attn_training_7618 else 0.0
        transpose_15_15929 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_15_15921, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_15_15921
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_1_15969 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_13_15849, transpose_14_15889, transpose_15_15929, dropout=ifexpr_1_7619, causal=True, attention_mask=None, query_length=2048)
        del transpose_13_15849, transpose_14_15889, transpose_15_15929
        nnscaler_flash_attention_forward_1_8144 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_1_15969, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_1_15969
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_68_8145 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_1_8144, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_1_8144
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_5_8146 = torch.Tensor.reshape(getitem_68_8145, shape=(8, 2048, 2048))
        del getitem_68_8145
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_1_8147 = torch.Tensor.contiguous(reshape_5_8146)
        del reshape_5_8146
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_10_8149 = torch.nn.functional.linear(contiguous_1_8147, self.model_model_layers_1_self_attn_o_proj_weight_8148, bias=None)
        del contiguous_1_8147
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_7_8150 = torch.add(add_4_80520, linear_10_8149, alpha=1)
        del add_4_80520, linear_10_8149
        # created at IRAdapterGener:local_consumer_multiref
        add_7_80684, add_7_80688 = nnscaler.runtime.function.multiref(add_7_8150, times=2)
        del add_7_8150
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_5_8152 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_7_80684, self.model_model_layers_1_post_attention_layernorm_weight_8151, (2048,), 1e-06)
        del add_7_80684
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_5_63483, fused_rms_norm_affine_5_63484, fused_rms_norm_affine_5_63485, fused_rms_norm_affine_5_63486 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_5_8152, times=4)
        del fused_rms_norm_affine_5_8152
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_1_mlp_gate_training_7621 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_8154, moe_route_8155, moe_route_8156 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_5_63483, self.model_model_layers_1_mlp_gate_weight_8153, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_1_mlp_gate_training_7621, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_5_63483
        moe_route_8155 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_8155, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_8156 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_8156, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_5_63484 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_5_63484, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_16169 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_5_63484, moe_route_8154, moe_route_8155, moe_route_8156, self.model_model_layers_1_mlp_gate_projs_16145, self.model_model_layers_1_mlp_up_projs_16153, self.model_model_layers_1_mlp_down_projs_16161, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_5_63484, moe_route_8154, moe_route_8155, moe_route_8156
        fused_rms_norm_affine_5_65608 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_5_63485, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_5_63485
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_11_16233 = torch.nn.functional.linear(fused_rms_norm_affine_5_65608, self.model_model_layers_1_mlp_shared_experts_gate_proj_weight_8161, bias=None)
        del fused_rms_norm_affine_5_65608
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_1_16289 = torch.nn.functional.silu(linear_11_16233, inplace=False)
        del linear_11_16233
        fused_rms_norm_affine_5_65648 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_5_63486, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_5_63486
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_12_16313 = torch.nn.functional.linear(fused_rms_norm_affine_5_65648, self.model_model_layers_1_mlp_shared_experts_up_proj_weight_8164, bias=None)
        del fused_rms_norm_affine_5_65648
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_9_16361 = torch.mul(silu_1_16289, linear_12_16313)
        del silu_1_16289, linear_12_16313
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_13_16385 = torch.nn.functional.linear(mul_9_16361, self.model_model_layers_1_mlp_shared_experts_down_proj_weight_8167, bias=None)
        del mul_9_16361
        nnscaler_moe_gmm_8160 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_16169, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_16169
        linear_13_8168 = nnscaler.runtime.adapter.nn.allgather_split(linear_13_16385, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_13_16385
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_8_8169 = torch.add(nnscaler_moe_gmm_8160, linear_13_8168, alpha=1)
        del nnscaler_moe_gmm_8160, linear_13_8168
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_9_8170 = torch.add(add_7_80688, add_8_8169, alpha=1)
        del add_7_80688, add_8_8169
        # created at IRAdapterGener:local_consumer_multiref
        add_9_80748, add_9_80752 = nnscaler.runtime.function.multiref(add_9_8170, times=2)
        del add_9_8170
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_6_8172 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_9_80748, self.model_model_layers_2_input_layernorm_weight_8171, (2048,), 1e-06)
        del add_9_80748
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_6_63495, fused_rms_norm_affine_6_63496 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_6_8172, times=2)
        del fused_rms_norm_affine_6_8172
        fused_rms_norm_affine_6_65704 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_6_63495, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_6_63495
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_14_16513 = torch.nn.functional.linear(fused_rms_norm_affine_6_65704, self.model_model_layers_2_self_attn_q_proj_weight_8173, bias=None)
        del fused_rms_norm_affine_6_65704
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_10_16569 = torch.Tensor.view(linear_14_16513, size=(8, 256, 16, 192))
        del linear_14_16513
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_16_16593 = torch.transpose(view_10_16569, dim0=1, dim1=2)
        del view_10_16569
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_6_16657, split_6_16665 = torch.functional.split(transpose_16_16593, split_size_or_sections=[128, 64], dim=-1)
        del transpose_16_16593
        fused_rms_norm_affine_6_65768 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_6_63496, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_6_63496
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_15_16697 = torch.nn.functional.linear(fused_rms_norm_affine_6_65768, self.model_model_layers_2_self_attn_kv_a_proj_with_mqa_weight_16689, bias=None)
        del fused_rms_norm_affine_6_65768
        linear_15_8180 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_15_16697, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_15_16697
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_7_8181, split_7_8182 = torch.functional.split(linear_15_8180, split_size_or_sections=[512, 64], dim=-1)
        del linear_15_8180
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_11_8183 = torch.Tensor.view(split_7_8182, size=(8, 2048, 1, 64))
        del split_7_8182
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_17_8184 = torch.transpose(view_11_8183, dim0=1, dim1=2)
        del view_11_8183
        split_7_16737 = nnscaler.runtime.adapter.nn.split_allgather(split_7_8181, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_7_8181
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_7_16817 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_7_16737, self.model_model_layers_2_self_attn_kv_a_layernorm_weight_8185, (512,), 1e-06)
        del split_7_16737
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_16_16833 = torch.nn.functional.linear(fused_rms_norm_affine_7_16817, self.model_model_layers_2_self_attn_kv_b_proj_weight_8187, bias=None)
        del fused_rms_norm_affine_7_16817
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_12_16889 = torch.Tensor.view(linear_16_16833, size=(8, 256, 16, 256))
        del linear_16_16833
        view_12_16881 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_12_16889, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_12_16889
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_18_16905 = torch.transpose(view_12_16881, dim0=1, dim1=2)
        del view_12_16881
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_8_16945, split_8_16953 = torch.functional.split(transpose_18_16905, split_size_or_sections=[128, 128], dim=-1)
        del transpose_18_16905
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_83_8194 = nnscaler.runtime.function.fullslice(self.model_model_layers_2_self_attn_rotary_emb_cos_cached_8193, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_4_8195 = torch.Tensor.to(getitem_83_8194, dtype=torch.bfloat16)
        del getitem_83_8194
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_84_8197 = nnscaler.runtime.function.fullslice(self.model_model_layers_2_self_attn_rotary_emb_sin_cached_8196, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_5_8198 = torch.Tensor.to(getitem_84_8197, dtype=torch.bfloat16)
        del getitem_84_8197
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_85_8199 = nnscaler.runtime.function.fullslice(to_4_8195, unsqueeze_8005)
        del to_4_8195
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_5_8200 = torch.unsqueeze(getitem_85_8199, dim=1)
        del getitem_85_8199
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_5_63501, unsqueeze_5_63502 = nnscaler.runtime.function.multiref(unsqueeze_5_8200, times=2)
        del unsqueeze_5_8200
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_86_8201 = nnscaler.runtime.function.fullslice(to_5_8198, unsqueeze_8005)
        del to_5_8198
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_6_8202 = torch.unsqueeze(getitem_86_8201, dim=1)
        del getitem_86_8201
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_6_63505, unsqueeze_6_63506 = nnscaler.runtime.function.multiref(unsqueeze_6_8202, times=2)
        del unsqueeze_6_8202
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_13_17041 = torch.Tensor.view(split_6_16665, size=(8, 16, 256, 32, 2))
        del split_6_16665
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_19_17081 = torch.transpose(view_13_17041, dim0=4, dim1=3)
        del view_13_17041
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_6_17113 = torch.Tensor.reshape(transpose_19_17081, shape=(8, 16, 256, 64))
        del transpose_19_17081
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_14_8206 = torch.Tensor.view(transpose_17_8184, size=(8, 1, 2048, 32, 2))
        del transpose_17_8184
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_20_8207 = torch.transpose(view_14_8206, dim0=4, dim1=3)
        del view_14_8206
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_7_8208 = torch.Tensor.reshape(transpose_20_8207, shape=(8, 1, 2048, 64))
        del transpose_20_8207
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_7_63513, reshape_7_63514, reshape_7_63515 = nnscaler.runtime.function.multiref(reshape_7_8208, times=3)
        del reshape_7_8208
        unsqueeze_5_65896 = nnscaler.runtime.adapter.chunk(unsqueeze_5_63501, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_5_63501
        # created at IRAdapterGener:local_consumer_multiref
        reshape_6_80843, reshape_6_80847, reshape_6_80851 = nnscaler.runtime.function.multiref(reshape_6_17113, times=3)
        del reshape_6_17113
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_10_17209 = torch.mul(reshape_6_80843, unsqueeze_5_65896)
        del unsqueeze_5_65896, reshape_6_80843
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_96_17257 = nnscaler.runtime.function.fullslice(reshape_6_80847, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_6_80847
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_98_17281 = nnscaler.runtime.function.fullslice(reshape_6_80851, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_6_80851
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_4_17305 = _operator.neg(getitem_98_17281)
        del getitem_98_17281
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_8_17345 = nnscaler.runtime.function.cat(neg_4_17305, getitem_96_17257, dim=-1)
        del getitem_96_17257, neg_4_17305
        unsqueeze_6_65968 = nnscaler.runtime.adapter.chunk(unsqueeze_6_63505, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_6_63505
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_11_17377 = torch.mul(cat_8_17345, unsqueeze_6_65968)
        del cat_8_17345, unsqueeze_6_65968
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_10_17425 = torch.add(mul_10_17209, mul_11_17377, alpha=1)
        del mul_10_17209, mul_11_17377
        unsqueeze_5_66000 = nnscaler.runtime.adapter.chunk(unsqueeze_5_63502, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_5_63502
        reshape_7_65992 = nnscaler.runtime.adapter.nn.split_allgather(reshape_7_63515, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_7_63515
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_12_17465 = torch.mul(reshape_7_65992, unsqueeze_5_66000)
        del unsqueeze_5_66000, reshape_7_65992
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_100_8217 = nnscaler.runtime.function.fullslice(reshape_7_63513, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_7_63513
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_102_8218 = nnscaler.runtime.function.fullslice(reshape_7_63514, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_7_63514
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_5_8219 = _operator.neg(getitem_102_8218)
        del getitem_102_8218
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_9_8220 = nnscaler.runtime.function.cat(neg_5_8219, getitem_100_8217, dim=-1)
        del getitem_100_8217, neg_5_8219
        cat_9_17569 = nnscaler.runtime.adapter.nn.split_allgather(cat_9_8220, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_9_8220
        unsqueeze_6_66024 = nnscaler.runtime.adapter.chunk(unsqueeze_6_63506, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_6_63506
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_13_17577 = torch.mul(cat_9_17569, unsqueeze_6_66024)
        del cat_9_17569, unsqueeze_6_66024
        mul_12_8216 = nnscaler.runtime.adapter.nn.allgather_split(mul_12_17465, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_12_17465
        mul_13_8221 = nnscaler.runtime.adapter.nn.allgather_split(mul_13_17577, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_13_17577
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_11_8222 = torch.add(mul_12_8216, mul_13_8221, alpha=1)
        del mul_12_8216, mul_13_8221
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_10_17625 = nnscaler.runtime.function.cat(split_6_16657, add_10_17425, dim=-1)
        del split_6_16657, add_10_17425
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_2_8224 = torch.Tensor.expand(add_11_8222, size=[-1, 16, -1, -1])
        del add_11_8222
        split_8_16961 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_8_16945, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_8_16945
        expand_2_17665 = nnscaler.runtime.adapter.nn.split_allgather(expand_2_8224, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_2_8224
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_11_17673 = nnscaler.runtime.function.cat(split_8_16961, expand_2_17665, dim=-1)
        del split_8_16961, expand_2_17665
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_2_17689 = torch.nn.functional.pad(split_8_16953, pad=[0, 64], mode='constant', value=0.0)
        del split_8_16953
        cat_10_17617 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_10_17625, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_10_17625
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_21_17721 = torch.transpose(cat_10_17617, dim0=1, dim1=2)
        del cat_10_17617
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_22_17761 = torch.transpose(cat_11_17673, dim0=1, dim1=2)
        del cat_11_17673
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_23_17793 = torch.transpose(pad_2_17689, dim0=1, dim1=2)
        del pad_2_17689
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_2_self_attn_training_7633 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_2_7634 = 0.0 if model_model_layers_2_self_attn_training_7633 else 0.0
        transpose_23_17801 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_23_17793, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_23_17793
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_2_17841 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_21_17721, transpose_22_17761, transpose_23_17801, dropout=ifexpr_2_7634, causal=True, attention_mask=None, query_length=2048)
        del transpose_21_17721, transpose_22_17761, transpose_23_17801
        nnscaler_flash_attention_forward_2_8230 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_2_17841, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_2_17841
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_103_8231 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_2_8230, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_2_8230
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_8_8232 = torch.Tensor.reshape(getitem_103_8231, shape=(8, 2048, 2048))
        del getitem_103_8231
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_2_8233 = torch.Tensor.contiguous(reshape_8_8232)
        del reshape_8_8232
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_17_8235 = torch.nn.functional.linear(contiguous_2_8233, self.model_model_layers_2_self_attn_o_proj_weight_8234, bias=None)
        del contiguous_2_8233
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_12_8236 = torch.add(add_9_80752, linear_17_8235, alpha=1)
        del add_9_80752, linear_17_8235
        # created at IRAdapterGener:local_consumer_multiref
        add_12_80916, add_12_80920 = nnscaler.runtime.function.multiref(add_12_8236, times=2)
        del add_12_8236
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_8_8238 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_12_80916, self.model_model_layers_2_post_attention_layernorm_weight_8237, (2048,), 1e-06)
        del add_12_80916
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_8_63527, fused_rms_norm_affine_8_63528, fused_rms_norm_affine_8_63529, fused_rms_norm_affine_8_63530 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_8_8238, times=4)
        del fused_rms_norm_affine_8_8238
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_2_mlp_gate_training_7636 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_1_8240, moe_route_1_8241, moe_route_1_8242 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_8_63527, self.model_model_layers_2_mlp_gate_weight_8239, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_2_mlp_gate_training_7636, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_8_63527
        moe_route_1_8241 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_1_8241, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_1_8242 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_1_8242, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_8_63528 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_8_63528, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_1_18041 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_8_63528, moe_route_1_8240, moe_route_1_8241, moe_route_1_8242, self.model_model_layers_2_mlp_gate_projs_18017, self.model_model_layers_2_mlp_up_projs_18025, self.model_model_layers_2_mlp_down_projs_18033, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_8_63528, moe_route_1_8240, moe_route_1_8241, moe_route_1_8242
        fused_rms_norm_affine_8_66184 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_8_63529, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_8_63529
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_18_18105 = torch.nn.functional.linear(fused_rms_norm_affine_8_66184, self.model_model_layers_2_mlp_shared_experts_gate_proj_weight_8247, bias=None)
        del fused_rms_norm_affine_8_66184
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_2_18161 = torch.nn.functional.silu(linear_18_18105, inplace=False)
        del linear_18_18105
        fused_rms_norm_affine_8_66224 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_8_63530, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_8_63530
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_19_18185 = torch.nn.functional.linear(fused_rms_norm_affine_8_66224, self.model_model_layers_2_mlp_shared_experts_up_proj_weight_8250, bias=None)
        del fused_rms_norm_affine_8_66224
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_14_18233 = torch.mul(silu_2_18161, linear_19_18185)
        del silu_2_18161, linear_19_18185
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_20_18257 = torch.nn.functional.linear(mul_14_18233, self.model_model_layers_2_mlp_shared_experts_down_proj_weight_8253, bias=None)
        del mul_14_18233
        nnscaler_moe_gmm_1_8246 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_1_18041, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_1_18041
        linear_20_8254 = nnscaler.runtime.adapter.nn.allgather_split(linear_20_18257, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_20_18257
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_13_8255 = torch.add(nnscaler_moe_gmm_1_8246, linear_20_8254, alpha=1)
        del nnscaler_moe_gmm_1_8246, linear_20_8254
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_14_8256 = torch.add(add_12_80920, add_13_8255, alpha=1)
        del add_12_80920, add_13_8255
        # created at IRAdapterGener:local_consumer_multiref
        add_14_80980, add_14_80984 = nnscaler.runtime.function.multiref(add_14_8256, times=2)
        del add_14_8256
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_9_8258 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_14_80980, self.model_model_layers_3_input_layernorm_weight_8257, (2048,), 1e-06)
        del add_14_80980
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_9_63539, fused_rms_norm_affine_9_63540 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_9_8258, times=2)
        del fused_rms_norm_affine_9_8258
        fused_rms_norm_affine_9_66280 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_9_63539, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_9_63539
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_21_18385 = torch.nn.functional.linear(fused_rms_norm_affine_9_66280, self.model_model_layers_3_self_attn_q_proj_weight_8259, bias=None)
        del fused_rms_norm_affine_9_66280
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_15_18441 = torch.Tensor.view(linear_21_18385, size=(8, 256, 16, 192))
        del linear_21_18385
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_24_18465 = torch.transpose(view_15_18441, dim0=1, dim1=2)
        del view_15_18441
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_9_18529, split_9_18537 = torch.functional.split(transpose_24_18465, split_size_or_sections=[128, 64], dim=-1)
        del transpose_24_18465
        fused_rms_norm_affine_9_66344 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_9_63540, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_9_63540
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_22_18569 = torch.nn.functional.linear(fused_rms_norm_affine_9_66344, self.model_model_layers_3_self_attn_kv_a_proj_with_mqa_weight_18561, bias=None)
        del fused_rms_norm_affine_9_66344
        linear_22_8266 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_22_18569, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_22_18569
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_10_8267, split_10_8268 = torch.functional.split(linear_22_8266, split_size_or_sections=[512, 64], dim=-1)
        del linear_22_8266
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_16_8269 = torch.Tensor.view(split_10_8268, size=(8, 2048, 1, 64))
        del split_10_8268
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_25_8270 = torch.transpose(view_16_8269, dim0=1, dim1=2)
        del view_16_8269
        split_10_18609 = nnscaler.runtime.adapter.nn.split_allgather(split_10_8267, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_10_8267
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_10_18689 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_10_18609, self.model_model_layers_3_self_attn_kv_a_layernorm_weight_8271, (512,), 1e-06)
        del split_10_18609
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_23_18705 = torch.nn.functional.linear(fused_rms_norm_affine_10_18689, self.model_model_layers_3_self_attn_kv_b_proj_weight_8273, bias=None)
        del fused_rms_norm_affine_10_18689
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_17_18761 = torch.Tensor.view(linear_23_18705, size=(8, 256, 16, 256))
        del linear_23_18705
        view_17_18753 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_17_18761, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_17_18761
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_26_18777 = torch.transpose(view_17_18753, dim0=1, dim1=2)
        del view_17_18753
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_11_18817, split_11_18825 = torch.functional.split(transpose_26_18777, split_size_or_sections=[128, 128], dim=-1)
        del transpose_26_18777
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_118_8280 = nnscaler.runtime.function.fullslice(self.model_model_layers_3_self_attn_rotary_emb_cos_cached_8279, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_6_8281 = torch.Tensor.to(getitem_118_8280, dtype=torch.bfloat16)
        del getitem_118_8280
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_119_8283 = nnscaler.runtime.function.fullslice(self.model_model_layers_3_self_attn_rotary_emb_sin_cached_8282, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_7_8284 = torch.Tensor.to(getitem_119_8283, dtype=torch.bfloat16)
        del getitem_119_8283
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_120_8285 = nnscaler.runtime.function.fullslice(to_6_8281, unsqueeze_8005)
        del to_6_8281
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_7_8286 = torch.unsqueeze(getitem_120_8285, dim=1)
        del getitem_120_8285
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_7_63545, unsqueeze_7_63546 = nnscaler.runtime.function.multiref(unsqueeze_7_8286, times=2)
        del unsqueeze_7_8286
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_121_8287 = nnscaler.runtime.function.fullslice(to_7_8284, unsqueeze_8005)
        del to_7_8284
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_8_8288 = torch.unsqueeze(getitem_121_8287, dim=1)
        del getitem_121_8287
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_8_63549, unsqueeze_8_63550 = nnscaler.runtime.function.multiref(unsqueeze_8_8288, times=2)
        del unsqueeze_8_8288
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_18_18913 = torch.Tensor.view(split_9_18537, size=(8, 16, 256, 32, 2))
        del split_9_18537
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_27_18953 = torch.transpose(view_18_18913, dim0=4, dim1=3)
        del view_18_18913
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_9_18985 = torch.Tensor.reshape(transpose_27_18953, shape=(8, 16, 256, 64))
        del transpose_27_18953
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_19_8292 = torch.Tensor.view(transpose_25_8270, size=(8, 1, 2048, 32, 2))
        del transpose_25_8270
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_28_8293 = torch.transpose(view_19_8292, dim0=4, dim1=3)
        del view_19_8292
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_10_8294 = torch.Tensor.reshape(transpose_28_8293, shape=(8, 1, 2048, 64))
        del transpose_28_8293
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_10_63557, reshape_10_63558, reshape_10_63559 = nnscaler.runtime.function.multiref(reshape_10_8294, times=3)
        del reshape_10_8294
        unsqueeze_7_66472 = nnscaler.runtime.adapter.chunk(unsqueeze_7_63545, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_7_63545
        # created at IRAdapterGener:local_consumer_multiref
        reshape_9_81075, reshape_9_81079, reshape_9_81083 = nnscaler.runtime.function.multiref(reshape_9_18985, times=3)
        del reshape_9_18985
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_15_19081 = torch.mul(reshape_9_81075, unsqueeze_7_66472)
        del unsqueeze_7_66472, reshape_9_81075
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_131_19129 = nnscaler.runtime.function.fullslice(reshape_9_81079, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_9_81079
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_133_19153 = nnscaler.runtime.function.fullslice(reshape_9_81083, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_9_81083
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_6_19177 = _operator.neg(getitem_133_19153)
        del getitem_133_19153
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_12_19217 = nnscaler.runtime.function.cat(neg_6_19177, getitem_131_19129, dim=-1)
        del getitem_131_19129, neg_6_19177
        unsqueeze_8_66544 = nnscaler.runtime.adapter.chunk(unsqueeze_8_63549, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_8_63549
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_16_19249 = torch.mul(cat_12_19217, unsqueeze_8_66544)
        del cat_12_19217, unsqueeze_8_66544
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_15_19297 = torch.add(mul_15_19081, mul_16_19249, alpha=1)
        del mul_15_19081, mul_16_19249
        unsqueeze_7_66576 = nnscaler.runtime.adapter.chunk(unsqueeze_7_63546, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_7_63546
        reshape_10_66568 = nnscaler.runtime.adapter.nn.split_allgather(reshape_10_63559, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_10_63559
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_17_19337 = torch.mul(reshape_10_66568, unsqueeze_7_66576)
        del unsqueeze_7_66576, reshape_10_66568
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_135_8303 = nnscaler.runtime.function.fullslice(reshape_10_63557, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_10_63557
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_137_8304 = nnscaler.runtime.function.fullslice(reshape_10_63558, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_10_63558
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_7_8305 = _operator.neg(getitem_137_8304)
        del getitem_137_8304
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_13_8306 = nnscaler.runtime.function.cat(neg_7_8305, getitem_135_8303, dim=-1)
        del getitem_135_8303, neg_7_8305
        cat_13_19441 = nnscaler.runtime.adapter.nn.split_allgather(cat_13_8306, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_13_8306
        unsqueeze_8_66600 = nnscaler.runtime.adapter.chunk(unsqueeze_8_63550, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_8_63550
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_18_19449 = torch.mul(cat_13_19441, unsqueeze_8_66600)
        del cat_13_19441, unsqueeze_8_66600
        mul_17_8302 = nnscaler.runtime.adapter.nn.allgather_split(mul_17_19337, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_17_19337
        mul_18_8307 = nnscaler.runtime.adapter.nn.allgather_split(mul_18_19449, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_18_19449
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_16_8308 = torch.add(mul_17_8302, mul_18_8307, alpha=1)
        del mul_17_8302, mul_18_8307
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_14_19497 = nnscaler.runtime.function.cat(split_9_18529, add_15_19297, dim=-1)
        del split_9_18529, add_15_19297
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_3_8310 = torch.Tensor.expand(add_16_8308, size=[-1, 16, -1, -1])
        del add_16_8308
        split_11_18833 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_11_18817, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_11_18817
        expand_3_19537 = nnscaler.runtime.adapter.nn.split_allgather(expand_3_8310, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_3_8310
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_15_19545 = nnscaler.runtime.function.cat(split_11_18833, expand_3_19537, dim=-1)
        del split_11_18833, expand_3_19537
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_3_19561 = torch.nn.functional.pad(split_11_18825, pad=[0, 64], mode='constant', value=0.0)
        del split_11_18825
        cat_14_19489 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_14_19497, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_14_19497
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_29_19593 = torch.transpose(cat_14_19489, dim0=1, dim1=2)
        del cat_14_19489
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_30_19633 = torch.transpose(cat_15_19545, dim0=1, dim1=2)
        del cat_15_19545
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_31_19665 = torch.transpose(pad_3_19561, dim0=1, dim1=2)
        del pad_3_19561
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_3_self_attn_training_7648 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_3_7649 = 0.0 if model_model_layers_3_self_attn_training_7648 else 0.0
        transpose_31_19673 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_31_19665, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_31_19665
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_3_19713 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_29_19593, transpose_30_19633, transpose_31_19673, dropout=ifexpr_3_7649, causal=True, attention_mask=None, query_length=2048)
        del transpose_29_19593, transpose_30_19633, transpose_31_19673
        nnscaler_flash_attention_forward_3_8316 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_3_19713, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_3_19713
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_138_8317 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_3_8316, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_3_8316
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_11_8318 = torch.Tensor.reshape(getitem_138_8317, shape=(8, 2048, 2048))
        del getitem_138_8317
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_3_8319 = torch.Tensor.contiguous(reshape_11_8318)
        del reshape_11_8318
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_24_8321 = torch.nn.functional.linear(contiguous_3_8319, self.model_model_layers_3_self_attn_o_proj_weight_8320, bias=None)
        del contiguous_3_8319
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_17_8322 = torch.add(add_14_80984, linear_24_8321, alpha=1)
        del add_14_80984, linear_24_8321
        # created at IRAdapterGener:local_consumer_multiref
        add_17_81148, add_17_81152 = nnscaler.runtime.function.multiref(add_17_8322, times=2)
        del add_17_8322
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_11_8324 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_17_81148, self.model_model_layers_3_post_attention_layernorm_weight_8323, (2048,), 1e-06)
        del add_17_81148
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_11_63571, fused_rms_norm_affine_11_63572, fused_rms_norm_affine_11_63573, fused_rms_norm_affine_11_63574 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_11_8324, times=4)
        del fused_rms_norm_affine_11_8324
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_3_mlp_gate_training_7651 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_2_8326, moe_route_2_8327, moe_route_2_8328 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_11_63571, self.model_model_layers_3_mlp_gate_weight_8325, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_3_mlp_gate_training_7651, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_11_63571
        moe_route_2_8327 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_2_8327, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_2_8328 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_2_8328, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_11_63572 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_11_63572, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_2_19913 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_11_63572, moe_route_2_8326, moe_route_2_8327, moe_route_2_8328, self.model_model_layers_3_mlp_gate_projs_19889, self.model_model_layers_3_mlp_up_projs_19897, self.model_model_layers_3_mlp_down_projs_19905, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_11_63572, moe_route_2_8326, moe_route_2_8327, moe_route_2_8328
        fused_rms_norm_affine_11_66760 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_11_63573, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_11_63573
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_25_19977 = torch.nn.functional.linear(fused_rms_norm_affine_11_66760, self.model_model_layers_3_mlp_shared_experts_gate_proj_weight_8333, bias=None)
        del fused_rms_norm_affine_11_66760
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_3_20033 = torch.nn.functional.silu(linear_25_19977, inplace=False)
        del linear_25_19977
        fused_rms_norm_affine_11_66800 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_11_63574, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_11_63574
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_26_20057 = torch.nn.functional.linear(fused_rms_norm_affine_11_66800, self.model_model_layers_3_mlp_shared_experts_up_proj_weight_8336, bias=None)
        del fused_rms_norm_affine_11_66800
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_19_20105 = torch.mul(silu_3_20033, linear_26_20057)
        del silu_3_20033, linear_26_20057
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_27_20129 = torch.nn.functional.linear(mul_19_20105, self.model_model_layers_3_mlp_shared_experts_down_proj_weight_8339, bias=None)
        del mul_19_20105
        nnscaler_moe_gmm_2_8332 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_2_19913, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_2_19913
        linear_27_8340 = nnscaler.runtime.adapter.nn.allgather_split(linear_27_20129, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_27_20129
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_18_8341 = torch.add(nnscaler_moe_gmm_2_8332, linear_27_8340, alpha=1)
        del nnscaler_moe_gmm_2_8332, linear_27_8340
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_19_8342 = torch.add(add_17_81152, add_18_8341, alpha=1)
        del add_17_81152, add_18_8341
        # created at IRAdapterGener:local_consumer_multiref
        add_19_81212, add_19_81216 = nnscaler.runtime.function.multiref(add_19_8342, times=2)
        del add_19_8342
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_12_8344 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_19_81212, self.model_model_layers_4_input_layernorm_weight_8343, (2048,), 1e-06)
        del add_19_81212
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_12_63583, fused_rms_norm_affine_12_63584 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_12_8344, times=2)
        del fused_rms_norm_affine_12_8344
        fused_rms_norm_affine_12_66856 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_12_63583, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_12_63583
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_28_20257 = torch.nn.functional.linear(fused_rms_norm_affine_12_66856, self.model_model_layers_4_self_attn_q_proj_weight_8345, bias=None)
        del fused_rms_norm_affine_12_66856
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_20_20313 = torch.Tensor.view(linear_28_20257, size=(8, 256, 16, 192))
        del linear_28_20257
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_32_20337 = torch.transpose(view_20_20313, dim0=1, dim1=2)
        del view_20_20313
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_12_20401, split_12_20409 = torch.functional.split(transpose_32_20337, split_size_or_sections=[128, 64], dim=-1)
        del transpose_32_20337
        fused_rms_norm_affine_12_66920 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_12_63584, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_12_63584
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_29_20441 = torch.nn.functional.linear(fused_rms_norm_affine_12_66920, self.model_model_layers_4_self_attn_kv_a_proj_with_mqa_weight_20433, bias=None)
        del fused_rms_norm_affine_12_66920
        linear_29_8352 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_29_20441, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_29_20441
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_13_8353, split_13_8354 = torch.functional.split(linear_29_8352, split_size_or_sections=[512, 64], dim=-1)
        del linear_29_8352
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_21_8355 = torch.Tensor.view(split_13_8354, size=(8, 2048, 1, 64))
        del split_13_8354
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_33_8356 = torch.transpose(view_21_8355, dim0=1, dim1=2)
        del view_21_8355
        split_13_20481 = nnscaler.runtime.adapter.nn.split_allgather(split_13_8353, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_13_8353
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_13_20561 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_13_20481, self.model_model_layers_4_self_attn_kv_a_layernorm_weight_8357, (512,), 1e-06)
        del split_13_20481
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_30_20577 = torch.nn.functional.linear(fused_rms_norm_affine_13_20561, self.model_model_layers_4_self_attn_kv_b_proj_weight_8359, bias=None)
        del fused_rms_norm_affine_13_20561
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_22_20633 = torch.Tensor.view(linear_30_20577, size=(8, 256, 16, 256))
        del linear_30_20577
        view_22_20625 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_22_20633, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_22_20633
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_34_20649 = torch.transpose(view_22_20625, dim0=1, dim1=2)
        del view_22_20625
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_14_20689, split_14_20697 = torch.functional.split(transpose_34_20649, split_size_or_sections=[128, 128], dim=-1)
        del transpose_34_20649
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_153_8366 = nnscaler.runtime.function.fullslice(self.model_model_layers_4_self_attn_rotary_emb_cos_cached_8365, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_8_8367 = torch.Tensor.to(getitem_153_8366, dtype=torch.bfloat16)
        del getitem_153_8366
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_154_8369 = nnscaler.runtime.function.fullslice(self.model_model_layers_4_self_attn_rotary_emb_sin_cached_8368, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_9_8370 = torch.Tensor.to(getitem_154_8369, dtype=torch.bfloat16)
        del getitem_154_8369
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_155_8371 = nnscaler.runtime.function.fullslice(to_8_8367, unsqueeze_8005)
        del to_8_8367
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_9_8372 = torch.unsqueeze(getitem_155_8371, dim=1)
        del getitem_155_8371
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_9_63589, unsqueeze_9_63590 = nnscaler.runtime.function.multiref(unsqueeze_9_8372, times=2)
        del unsqueeze_9_8372
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_156_8373 = nnscaler.runtime.function.fullslice(to_9_8370, unsqueeze_8005)
        del to_9_8370
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_10_8374 = torch.unsqueeze(getitem_156_8373, dim=1)
        del getitem_156_8373
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_10_63593, unsqueeze_10_63594 = nnscaler.runtime.function.multiref(unsqueeze_10_8374, times=2)
        del unsqueeze_10_8374
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_23_20785 = torch.Tensor.view(split_12_20409, size=(8, 16, 256, 32, 2))
        del split_12_20409
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_35_20825 = torch.transpose(view_23_20785, dim0=4, dim1=3)
        del view_23_20785
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_12_20857 = torch.Tensor.reshape(transpose_35_20825, shape=(8, 16, 256, 64))
        del transpose_35_20825
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_24_8378 = torch.Tensor.view(transpose_33_8356, size=(8, 1, 2048, 32, 2))
        del transpose_33_8356
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_36_8379 = torch.transpose(view_24_8378, dim0=4, dim1=3)
        del view_24_8378
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_13_8380 = torch.Tensor.reshape(transpose_36_8379, shape=(8, 1, 2048, 64))
        del transpose_36_8379
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_13_63601, reshape_13_63602, reshape_13_63603 = nnscaler.runtime.function.multiref(reshape_13_8380, times=3)
        del reshape_13_8380
        unsqueeze_9_67048 = nnscaler.runtime.adapter.chunk(unsqueeze_9_63589, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_9_63589
        # created at IRAdapterGener:local_consumer_multiref
        reshape_12_81307, reshape_12_81311, reshape_12_81315 = nnscaler.runtime.function.multiref(reshape_12_20857, times=3)
        del reshape_12_20857
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_20_20953 = torch.mul(reshape_12_81307, unsqueeze_9_67048)
        del unsqueeze_9_67048, reshape_12_81307
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_166_21001 = nnscaler.runtime.function.fullslice(reshape_12_81311, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_12_81311
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_168_21025 = nnscaler.runtime.function.fullslice(reshape_12_81315, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_12_81315
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_8_21049 = _operator.neg(getitem_168_21025)
        del getitem_168_21025
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_16_21089 = nnscaler.runtime.function.cat(neg_8_21049, getitem_166_21001, dim=-1)
        del getitem_166_21001, neg_8_21049
        unsqueeze_10_67120 = nnscaler.runtime.adapter.chunk(unsqueeze_10_63593, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_10_63593
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_21_21121 = torch.mul(cat_16_21089, unsqueeze_10_67120)
        del cat_16_21089, unsqueeze_10_67120
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_20_21169 = torch.add(mul_20_20953, mul_21_21121, alpha=1)
        del mul_20_20953, mul_21_21121
        unsqueeze_9_67152 = nnscaler.runtime.adapter.chunk(unsqueeze_9_63590, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_9_63590
        reshape_13_67144 = nnscaler.runtime.adapter.nn.split_allgather(reshape_13_63603, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_13_63603
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_22_21209 = torch.mul(reshape_13_67144, unsqueeze_9_67152)
        del unsqueeze_9_67152, reshape_13_67144
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_170_8389 = nnscaler.runtime.function.fullslice(reshape_13_63601, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_13_63601
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_172_8390 = nnscaler.runtime.function.fullslice(reshape_13_63602, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_13_63602
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_9_8391 = _operator.neg(getitem_172_8390)
        del getitem_172_8390
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_17_8392 = nnscaler.runtime.function.cat(neg_9_8391, getitem_170_8389, dim=-1)
        del getitem_170_8389, neg_9_8391
        cat_17_21313 = nnscaler.runtime.adapter.nn.split_allgather(cat_17_8392, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_17_8392
        unsqueeze_10_67176 = nnscaler.runtime.adapter.chunk(unsqueeze_10_63594, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_10_63594
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_23_21321 = torch.mul(cat_17_21313, unsqueeze_10_67176)
        del cat_17_21313, unsqueeze_10_67176
        mul_22_8388 = nnscaler.runtime.adapter.nn.allgather_split(mul_22_21209, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_22_21209
        mul_23_8393 = nnscaler.runtime.adapter.nn.allgather_split(mul_23_21321, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_23_21321
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_21_8394 = torch.add(mul_22_8388, mul_23_8393, alpha=1)
        del mul_22_8388, mul_23_8393
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_18_21369 = nnscaler.runtime.function.cat(split_12_20401, add_20_21169, dim=-1)
        del split_12_20401, add_20_21169
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_4_8396 = torch.Tensor.expand(add_21_8394, size=[-1, 16, -1, -1])
        del add_21_8394
        split_14_20705 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_14_20689, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_14_20689
        expand_4_21409 = nnscaler.runtime.adapter.nn.split_allgather(expand_4_8396, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_4_8396
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_19_21417 = nnscaler.runtime.function.cat(split_14_20705, expand_4_21409, dim=-1)
        del split_14_20705, expand_4_21409
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_4_21433 = torch.nn.functional.pad(split_14_20697, pad=[0, 64], mode='constant', value=0.0)
        del split_14_20697
        cat_18_21361 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_18_21369, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_18_21369
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_37_21465 = torch.transpose(cat_18_21361, dim0=1, dim1=2)
        del cat_18_21361
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_38_21505 = torch.transpose(cat_19_21417, dim0=1, dim1=2)
        del cat_19_21417
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_39_21537 = torch.transpose(pad_4_21433, dim0=1, dim1=2)
        del pad_4_21433
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_4_self_attn_training_7663 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_4_7664 = 0.0 if model_model_layers_4_self_attn_training_7663 else 0.0
        transpose_39_21545 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_39_21537, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_39_21537
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_4_21585 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_37_21465, transpose_38_21505, transpose_39_21545, dropout=ifexpr_4_7664, causal=True, attention_mask=None, query_length=2048)
        del transpose_37_21465, transpose_38_21505, transpose_39_21545
        nnscaler_flash_attention_forward_4_8402 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_4_21585, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_4_21585
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_173_8403 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_4_8402, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_4_8402
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_14_8404 = torch.Tensor.reshape(getitem_173_8403, shape=(8, 2048, 2048))
        del getitem_173_8403
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_4_8405 = torch.Tensor.contiguous(reshape_14_8404)
        del reshape_14_8404
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_31_8407 = torch.nn.functional.linear(contiguous_4_8405, self.model_model_layers_4_self_attn_o_proj_weight_8406, bias=None)
        del contiguous_4_8405
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_22_8408 = torch.add(add_19_81216, linear_31_8407, alpha=1)
        del add_19_81216, linear_31_8407
        # created at IRAdapterGener:local_consumer_multiref
        add_22_81380, add_22_81384 = nnscaler.runtime.function.multiref(add_22_8408, times=2)
        del add_22_8408
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_14_8410 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_22_81380, self.model_model_layers_4_post_attention_layernorm_weight_8409, (2048,), 1e-06)
        del add_22_81380
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_14_63615, fused_rms_norm_affine_14_63616, fused_rms_norm_affine_14_63617, fused_rms_norm_affine_14_63618 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_14_8410, times=4)
        del fused_rms_norm_affine_14_8410
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_4_mlp_gate_training_7666 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_3_8412, moe_route_3_8413, moe_route_3_8414 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_14_63615, self.model_model_layers_4_mlp_gate_weight_8411, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_4_mlp_gate_training_7666, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_14_63615
        moe_route_3_8413 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_3_8413, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_3_8414 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_3_8414, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_14_63616 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_14_63616, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_3_21785 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_14_63616, moe_route_3_8412, moe_route_3_8413, moe_route_3_8414, self.model_model_layers_4_mlp_gate_projs_21761, self.model_model_layers_4_mlp_up_projs_21769, self.model_model_layers_4_mlp_down_projs_21777, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_14_63616, moe_route_3_8412, moe_route_3_8413, moe_route_3_8414
        fused_rms_norm_affine_14_67336 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_14_63617, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_14_63617
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_32_21849 = torch.nn.functional.linear(fused_rms_norm_affine_14_67336, self.model_model_layers_4_mlp_shared_experts_gate_proj_weight_8419, bias=None)
        del fused_rms_norm_affine_14_67336
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_4_21905 = torch.nn.functional.silu(linear_32_21849, inplace=False)
        del linear_32_21849
        fused_rms_norm_affine_14_67376 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_14_63618, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_14_63618
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_33_21929 = torch.nn.functional.linear(fused_rms_norm_affine_14_67376, self.model_model_layers_4_mlp_shared_experts_up_proj_weight_8422, bias=None)
        del fused_rms_norm_affine_14_67376
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_24_21977 = torch.mul(silu_4_21905, linear_33_21929)
        del silu_4_21905, linear_33_21929
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_34_22001 = torch.nn.functional.linear(mul_24_21977, self.model_model_layers_4_mlp_shared_experts_down_proj_weight_8425, bias=None)
        del mul_24_21977
        nnscaler_moe_gmm_3_8418 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_3_21785, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_3_21785
        linear_34_8426 = nnscaler.runtime.adapter.nn.allgather_split(linear_34_22001, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_34_22001
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_23_8427 = torch.add(nnscaler_moe_gmm_3_8418, linear_34_8426, alpha=1)
        del nnscaler_moe_gmm_3_8418, linear_34_8426
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_24_8428 = torch.add(add_22_81384, add_23_8427, alpha=1)
        del add_22_81384, add_23_8427
        # created at IRAdapterGener:local_consumer_multiref
        add_24_81444, add_24_81448 = nnscaler.runtime.function.multiref(add_24_8428, times=2)
        del add_24_8428
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_15_8430 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_24_81444, self.model_model_layers_5_input_layernorm_weight_8429, (2048,), 1e-06)
        del add_24_81444
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_15_63627, fused_rms_norm_affine_15_63628 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_15_8430, times=2)
        del fused_rms_norm_affine_15_8430
        fused_rms_norm_affine_15_67432 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_15_63627, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_15_63627
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_35_22129 = torch.nn.functional.linear(fused_rms_norm_affine_15_67432, self.model_model_layers_5_self_attn_q_proj_weight_8431, bias=None)
        del fused_rms_norm_affine_15_67432
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_25_22185 = torch.Tensor.view(linear_35_22129, size=(8, 256, 16, 192))
        del linear_35_22129
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_40_22209 = torch.transpose(view_25_22185, dim0=1, dim1=2)
        del view_25_22185
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_15_22273, split_15_22281 = torch.functional.split(transpose_40_22209, split_size_or_sections=[128, 64], dim=-1)
        del transpose_40_22209
        fused_rms_norm_affine_15_67496 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_15_63628, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_15_63628
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_36_22313 = torch.nn.functional.linear(fused_rms_norm_affine_15_67496, self.model_model_layers_5_self_attn_kv_a_proj_with_mqa_weight_22305, bias=None)
        del fused_rms_norm_affine_15_67496
        linear_36_8438 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_36_22313, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_36_22313
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_16_8439, split_16_8440 = torch.functional.split(linear_36_8438, split_size_or_sections=[512, 64], dim=-1)
        del linear_36_8438
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_26_8441 = torch.Tensor.view(split_16_8440, size=(8, 2048, 1, 64))
        del split_16_8440
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_41_8442 = torch.transpose(view_26_8441, dim0=1, dim1=2)
        del view_26_8441
        split_16_22353 = nnscaler.runtime.adapter.nn.split_allgather(split_16_8439, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_16_8439
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_16_22433 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_16_22353, self.model_model_layers_5_self_attn_kv_a_layernorm_weight_8443, (512,), 1e-06)
        del split_16_22353
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_37_22449 = torch.nn.functional.linear(fused_rms_norm_affine_16_22433, self.model_model_layers_5_self_attn_kv_b_proj_weight_8445, bias=None)
        del fused_rms_norm_affine_16_22433
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_27_22505 = torch.Tensor.view(linear_37_22449, size=(8, 256, 16, 256))
        del linear_37_22449
        view_27_22497 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_27_22505, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_27_22505
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_42_22521 = torch.transpose(view_27_22497, dim0=1, dim1=2)
        del view_27_22497
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_17_22561, split_17_22569 = torch.functional.split(transpose_42_22521, split_size_or_sections=[128, 128], dim=-1)
        del transpose_42_22521
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_188_8452 = nnscaler.runtime.function.fullslice(self.model_model_layers_5_self_attn_rotary_emb_cos_cached_8451, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_10_8453 = torch.Tensor.to(getitem_188_8452, dtype=torch.bfloat16)
        del getitem_188_8452
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_189_8455 = nnscaler.runtime.function.fullslice(self.model_model_layers_5_self_attn_rotary_emb_sin_cached_8454, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_11_8456 = torch.Tensor.to(getitem_189_8455, dtype=torch.bfloat16)
        del getitem_189_8455
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_190_8457 = nnscaler.runtime.function.fullslice(to_10_8453, unsqueeze_8005)
        del to_10_8453
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_11_8458 = torch.unsqueeze(getitem_190_8457, dim=1)
        del getitem_190_8457
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_11_63633, unsqueeze_11_63634 = nnscaler.runtime.function.multiref(unsqueeze_11_8458, times=2)
        del unsqueeze_11_8458
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_191_8459 = nnscaler.runtime.function.fullslice(to_11_8456, unsqueeze_8005)
        del to_11_8456
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_12_8460 = torch.unsqueeze(getitem_191_8459, dim=1)
        del getitem_191_8459
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_12_63637, unsqueeze_12_63638 = nnscaler.runtime.function.multiref(unsqueeze_12_8460, times=2)
        del unsqueeze_12_8460
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_28_22657 = torch.Tensor.view(split_15_22281, size=(8, 16, 256, 32, 2))
        del split_15_22281
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_43_22697 = torch.transpose(view_28_22657, dim0=4, dim1=3)
        del view_28_22657
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_15_22729 = torch.Tensor.reshape(transpose_43_22697, shape=(8, 16, 256, 64))
        del transpose_43_22697
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_29_8464 = torch.Tensor.view(transpose_41_8442, size=(8, 1, 2048, 32, 2))
        del transpose_41_8442
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_44_8465 = torch.transpose(view_29_8464, dim0=4, dim1=3)
        del view_29_8464
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_16_8466 = torch.Tensor.reshape(transpose_44_8465, shape=(8, 1, 2048, 64))
        del transpose_44_8465
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_16_63645, reshape_16_63646, reshape_16_63647 = nnscaler.runtime.function.multiref(reshape_16_8466, times=3)
        del reshape_16_8466
        unsqueeze_11_67624 = nnscaler.runtime.adapter.chunk(unsqueeze_11_63633, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_11_63633
        # created at IRAdapterGener:local_consumer_multiref
        reshape_15_81539, reshape_15_81543, reshape_15_81547 = nnscaler.runtime.function.multiref(reshape_15_22729, times=3)
        del reshape_15_22729
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_25_22825 = torch.mul(reshape_15_81539, unsqueeze_11_67624)
        del unsqueeze_11_67624, reshape_15_81539
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_201_22873 = nnscaler.runtime.function.fullslice(reshape_15_81543, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_15_81543
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_203_22897 = nnscaler.runtime.function.fullslice(reshape_15_81547, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_15_81547
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_10_22921 = _operator.neg(getitem_203_22897)
        del getitem_203_22897
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_20_22961 = nnscaler.runtime.function.cat(neg_10_22921, getitem_201_22873, dim=-1)
        del getitem_201_22873, neg_10_22921
        unsqueeze_12_67696 = nnscaler.runtime.adapter.chunk(unsqueeze_12_63637, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_12_63637
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_26_22993 = torch.mul(cat_20_22961, unsqueeze_12_67696)
        del cat_20_22961, unsqueeze_12_67696
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_25_23041 = torch.add(mul_25_22825, mul_26_22993, alpha=1)
        del mul_25_22825, mul_26_22993
        unsqueeze_11_67728 = nnscaler.runtime.adapter.chunk(unsqueeze_11_63634, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_11_63634
        reshape_16_67720 = nnscaler.runtime.adapter.nn.split_allgather(reshape_16_63647, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_16_63647
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_27_23081 = torch.mul(reshape_16_67720, unsqueeze_11_67728)
        del unsqueeze_11_67728, reshape_16_67720
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_205_8475 = nnscaler.runtime.function.fullslice(reshape_16_63645, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_16_63645
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_207_8476 = nnscaler.runtime.function.fullslice(reshape_16_63646, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_16_63646
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_11_8477 = _operator.neg(getitem_207_8476)
        del getitem_207_8476
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_21_8478 = nnscaler.runtime.function.cat(neg_11_8477, getitem_205_8475, dim=-1)
        del getitem_205_8475, neg_11_8477
        cat_21_23185 = nnscaler.runtime.adapter.nn.split_allgather(cat_21_8478, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_21_8478
        unsqueeze_12_67752 = nnscaler.runtime.adapter.chunk(unsqueeze_12_63638, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_12_63638
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_28_23193 = torch.mul(cat_21_23185, unsqueeze_12_67752)
        del cat_21_23185, unsqueeze_12_67752
        mul_27_8474 = nnscaler.runtime.adapter.nn.allgather_split(mul_27_23081, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_27_23081
        mul_28_8479 = nnscaler.runtime.adapter.nn.allgather_split(mul_28_23193, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_28_23193
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_26_8480 = torch.add(mul_27_8474, mul_28_8479, alpha=1)
        del mul_27_8474, mul_28_8479
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_22_23241 = nnscaler.runtime.function.cat(split_15_22273, add_25_23041, dim=-1)
        del split_15_22273, add_25_23041
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_5_8482 = torch.Tensor.expand(add_26_8480, size=[-1, 16, -1, -1])
        del add_26_8480
        split_17_22577 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_17_22561, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_17_22561
        expand_5_23281 = nnscaler.runtime.adapter.nn.split_allgather(expand_5_8482, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_5_8482
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_23_23289 = nnscaler.runtime.function.cat(split_17_22577, expand_5_23281, dim=-1)
        del split_17_22577, expand_5_23281
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_5_23305 = torch.nn.functional.pad(split_17_22569, pad=[0, 64], mode='constant', value=0.0)
        del split_17_22569
        cat_22_23233 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_22_23241, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_22_23241
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_45_23337 = torch.transpose(cat_22_23233, dim0=1, dim1=2)
        del cat_22_23233
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_46_23377 = torch.transpose(cat_23_23289, dim0=1, dim1=2)
        del cat_23_23289
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_47_23409 = torch.transpose(pad_5_23305, dim0=1, dim1=2)
        del pad_5_23305
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_5_self_attn_training_7678 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_5_7679 = 0.0 if model_model_layers_5_self_attn_training_7678 else 0.0
        transpose_47_23417 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_47_23409, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_47_23409
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_5_23457 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_45_23337, transpose_46_23377, transpose_47_23417, dropout=ifexpr_5_7679, causal=True, attention_mask=None, query_length=2048)
        del transpose_45_23337, transpose_46_23377, transpose_47_23417
        nnscaler_flash_attention_forward_5_8488 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_5_23457, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_5_23457
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_208_8489 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_5_8488, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_5_8488
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_17_8490 = torch.Tensor.reshape(getitem_208_8489, shape=(8, 2048, 2048))
        del getitem_208_8489
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_5_8491 = torch.Tensor.contiguous(reshape_17_8490)
        del reshape_17_8490
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_38_8493 = torch.nn.functional.linear(contiguous_5_8491, self.model_model_layers_5_self_attn_o_proj_weight_8492, bias=None)
        del contiguous_5_8491
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_27_8494 = torch.add(add_24_81448, linear_38_8493, alpha=1)
        del add_24_81448, linear_38_8493
        # created at IRAdapterGener:local_consumer_multiref
        add_27_81612, add_27_81616 = nnscaler.runtime.function.multiref(add_27_8494, times=2)
        del add_27_8494
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_17_8496 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_27_81612, self.model_model_layers_5_post_attention_layernorm_weight_8495, (2048,), 1e-06)
        del add_27_81612
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_17_63659, fused_rms_norm_affine_17_63660, fused_rms_norm_affine_17_63661, fused_rms_norm_affine_17_63662 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_17_8496, times=4)
        del fused_rms_norm_affine_17_8496
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_5_mlp_gate_training_7681 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_4_8498, moe_route_4_8499, moe_route_4_8500 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_17_63659, self.model_model_layers_5_mlp_gate_weight_8497, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_5_mlp_gate_training_7681, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_17_63659
        moe_route_4_8499 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_4_8499, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_4_8500 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_4_8500, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_17_63660 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_17_63660, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_4_23657 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_17_63660, moe_route_4_8498, moe_route_4_8499, moe_route_4_8500, self.model_model_layers_5_mlp_gate_projs_23633, self.model_model_layers_5_mlp_up_projs_23641, self.model_model_layers_5_mlp_down_projs_23649, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_17_63660, moe_route_4_8498, moe_route_4_8499, moe_route_4_8500
        fused_rms_norm_affine_17_67912 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_17_63661, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_17_63661
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_39_23721 = torch.nn.functional.linear(fused_rms_norm_affine_17_67912, self.model_model_layers_5_mlp_shared_experts_gate_proj_weight_8505, bias=None)
        del fused_rms_norm_affine_17_67912
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_5_23777 = torch.nn.functional.silu(linear_39_23721, inplace=False)
        del linear_39_23721
        fused_rms_norm_affine_17_67952 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_17_63662, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_17_63662
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_40_23801 = torch.nn.functional.linear(fused_rms_norm_affine_17_67952, self.model_model_layers_5_mlp_shared_experts_up_proj_weight_8508, bias=None)
        del fused_rms_norm_affine_17_67952
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_29_23849 = torch.mul(silu_5_23777, linear_40_23801)
        del silu_5_23777, linear_40_23801
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_41_23873 = torch.nn.functional.linear(mul_29_23849, self.model_model_layers_5_mlp_shared_experts_down_proj_weight_8511, bias=None)
        del mul_29_23849
        nnscaler_moe_gmm_4_8504 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_4_23657, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_4_23657
        linear_41_8512 = nnscaler.runtime.adapter.nn.allgather_split(linear_41_23873, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_41_23873
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_28_8513 = torch.add(nnscaler_moe_gmm_4_8504, linear_41_8512, alpha=1)
        del nnscaler_moe_gmm_4_8504, linear_41_8512
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_29_8514 = torch.add(add_27_81616, add_28_8513, alpha=1)
        del add_27_81616, add_28_8513
        # created at IRAdapterGener:local_consumer_multiref
        add_29_81676, add_29_81680 = nnscaler.runtime.function.multiref(add_29_8514, times=2)
        del add_29_8514
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_18_8516 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_29_81676, self.model_model_layers_6_input_layernorm_weight_8515, (2048,), 1e-06)
        del add_29_81676
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_18_63671, fused_rms_norm_affine_18_63672 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_18_8516, times=2)
        del fused_rms_norm_affine_18_8516
        fused_rms_norm_affine_18_68008 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_18_63671, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_18_63671
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_42_24001 = torch.nn.functional.linear(fused_rms_norm_affine_18_68008, self.model_model_layers_6_self_attn_q_proj_weight_8517, bias=None)
        del fused_rms_norm_affine_18_68008
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_30_24057 = torch.Tensor.view(linear_42_24001, size=(8, 256, 16, 192))
        del linear_42_24001
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_48_24081 = torch.transpose(view_30_24057, dim0=1, dim1=2)
        del view_30_24057
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_18_24145, split_18_24153 = torch.functional.split(transpose_48_24081, split_size_or_sections=[128, 64], dim=-1)
        del transpose_48_24081
        fused_rms_norm_affine_18_68072 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_18_63672, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_18_63672
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_43_24185 = torch.nn.functional.linear(fused_rms_norm_affine_18_68072, self.model_model_layers_6_self_attn_kv_a_proj_with_mqa_weight_24177, bias=None)
        del fused_rms_norm_affine_18_68072
        linear_43_8524 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_43_24185, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_43_24185
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_19_8525, split_19_8526 = torch.functional.split(linear_43_8524, split_size_or_sections=[512, 64], dim=-1)
        del linear_43_8524
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_31_8527 = torch.Tensor.view(split_19_8526, size=(8, 2048, 1, 64))
        del split_19_8526
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_49_8528 = torch.transpose(view_31_8527, dim0=1, dim1=2)
        del view_31_8527
        split_19_24225 = nnscaler.runtime.adapter.nn.split_allgather(split_19_8525, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_19_8525
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_19_24305 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_19_24225, self.model_model_layers_6_self_attn_kv_a_layernorm_weight_8529, (512,), 1e-06)
        del split_19_24225
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_44_24321 = torch.nn.functional.linear(fused_rms_norm_affine_19_24305, self.model_model_layers_6_self_attn_kv_b_proj_weight_8531, bias=None)
        del fused_rms_norm_affine_19_24305
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_32_24377 = torch.Tensor.view(linear_44_24321, size=(8, 256, 16, 256))
        del linear_44_24321
        view_32_24369 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_32_24377, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_32_24377
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_50_24393 = torch.transpose(view_32_24369, dim0=1, dim1=2)
        del view_32_24369
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_20_24433, split_20_24441 = torch.functional.split(transpose_50_24393, split_size_or_sections=[128, 128], dim=-1)
        del transpose_50_24393
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_223_8538 = nnscaler.runtime.function.fullslice(self.model_model_layers_6_self_attn_rotary_emb_cos_cached_8537, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_12_8539 = torch.Tensor.to(getitem_223_8538, dtype=torch.bfloat16)
        del getitem_223_8538
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_224_8541 = nnscaler.runtime.function.fullslice(self.model_model_layers_6_self_attn_rotary_emb_sin_cached_8540, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_13_8542 = torch.Tensor.to(getitem_224_8541, dtype=torch.bfloat16)
        del getitem_224_8541
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_225_8543 = nnscaler.runtime.function.fullslice(to_12_8539, unsqueeze_8005)
        del to_12_8539
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_13_8544 = torch.unsqueeze(getitem_225_8543, dim=1)
        del getitem_225_8543
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_13_63677, unsqueeze_13_63678 = nnscaler.runtime.function.multiref(unsqueeze_13_8544, times=2)
        del unsqueeze_13_8544
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_226_8545 = nnscaler.runtime.function.fullslice(to_13_8542, unsqueeze_8005)
        del to_13_8542
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_14_8546 = torch.unsqueeze(getitem_226_8545, dim=1)
        del getitem_226_8545
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_14_63681, unsqueeze_14_63682 = nnscaler.runtime.function.multiref(unsqueeze_14_8546, times=2)
        del unsqueeze_14_8546
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_33_24529 = torch.Tensor.view(split_18_24153, size=(8, 16, 256, 32, 2))
        del split_18_24153
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_51_24569 = torch.transpose(view_33_24529, dim0=4, dim1=3)
        del view_33_24529
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_18_24601 = torch.Tensor.reshape(transpose_51_24569, shape=(8, 16, 256, 64))
        del transpose_51_24569
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_34_8550 = torch.Tensor.view(transpose_49_8528, size=(8, 1, 2048, 32, 2))
        del transpose_49_8528
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_52_8551 = torch.transpose(view_34_8550, dim0=4, dim1=3)
        del view_34_8550
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_19_8552 = torch.Tensor.reshape(transpose_52_8551, shape=(8, 1, 2048, 64))
        del transpose_52_8551
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_19_63689, reshape_19_63690, reshape_19_63691 = nnscaler.runtime.function.multiref(reshape_19_8552, times=3)
        del reshape_19_8552
        unsqueeze_13_68200 = nnscaler.runtime.adapter.chunk(unsqueeze_13_63677, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_13_63677
        # created at IRAdapterGener:local_consumer_multiref
        reshape_18_81771, reshape_18_81775, reshape_18_81779 = nnscaler.runtime.function.multiref(reshape_18_24601, times=3)
        del reshape_18_24601
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_30_24697 = torch.mul(reshape_18_81771, unsqueeze_13_68200)
        del unsqueeze_13_68200, reshape_18_81771
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_236_24745 = nnscaler.runtime.function.fullslice(reshape_18_81775, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_18_81775
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_238_24769 = nnscaler.runtime.function.fullslice(reshape_18_81779, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_18_81779
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_12_24793 = _operator.neg(getitem_238_24769)
        del getitem_238_24769
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_24_24833 = nnscaler.runtime.function.cat(neg_12_24793, getitem_236_24745, dim=-1)
        del getitem_236_24745, neg_12_24793
        unsqueeze_14_68272 = nnscaler.runtime.adapter.chunk(unsqueeze_14_63681, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_14_63681
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_31_24865 = torch.mul(cat_24_24833, unsqueeze_14_68272)
        del cat_24_24833, unsqueeze_14_68272
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_30_24913 = torch.add(mul_30_24697, mul_31_24865, alpha=1)
        del mul_30_24697, mul_31_24865
        unsqueeze_13_68304 = nnscaler.runtime.adapter.chunk(unsqueeze_13_63678, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_13_63678
        reshape_19_68296 = nnscaler.runtime.adapter.nn.split_allgather(reshape_19_63691, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_19_63691
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_32_24953 = torch.mul(reshape_19_68296, unsqueeze_13_68304)
        del unsqueeze_13_68304, reshape_19_68296
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_240_8561 = nnscaler.runtime.function.fullslice(reshape_19_63689, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_19_63689
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_242_8562 = nnscaler.runtime.function.fullslice(reshape_19_63690, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_19_63690
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_13_8563 = _operator.neg(getitem_242_8562)
        del getitem_242_8562
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_25_8564 = nnscaler.runtime.function.cat(neg_13_8563, getitem_240_8561, dim=-1)
        del getitem_240_8561, neg_13_8563
        cat_25_25057 = nnscaler.runtime.adapter.nn.split_allgather(cat_25_8564, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_25_8564
        unsqueeze_14_68328 = nnscaler.runtime.adapter.chunk(unsqueeze_14_63682, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_14_63682
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_33_25065 = torch.mul(cat_25_25057, unsqueeze_14_68328)
        del cat_25_25057, unsqueeze_14_68328
        mul_32_8560 = nnscaler.runtime.adapter.nn.allgather_split(mul_32_24953, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_32_24953
        mul_33_8565 = nnscaler.runtime.adapter.nn.allgather_split(mul_33_25065, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_33_25065
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_31_8566 = torch.add(mul_32_8560, mul_33_8565, alpha=1)
        del mul_32_8560, mul_33_8565
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_26_25113 = nnscaler.runtime.function.cat(split_18_24145, add_30_24913, dim=-1)
        del split_18_24145, add_30_24913
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_6_8568 = torch.Tensor.expand(add_31_8566, size=[-1, 16, -1, -1])
        del add_31_8566
        split_20_24449 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_20_24433, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_20_24433
        expand_6_25153 = nnscaler.runtime.adapter.nn.split_allgather(expand_6_8568, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_6_8568
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_27_25161 = nnscaler.runtime.function.cat(split_20_24449, expand_6_25153, dim=-1)
        del split_20_24449, expand_6_25153
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_6_25177 = torch.nn.functional.pad(split_20_24441, pad=[0, 64], mode='constant', value=0.0)
        del split_20_24441
        cat_26_25105 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_26_25113, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_26_25113
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_53_25209 = torch.transpose(cat_26_25105, dim0=1, dim1=2)
        del cat_26_25105
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_54_25249 = torch.transpose(cat_27_25161, dim0=1, dim1=2)
        del cat_27_25161
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_55_25281 = torch.transpose(pad_6_25177, dim0=1, dim1=2)
        del pad_6_25177
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_6_self_attn_training_7693 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_6_7694 = 0.0 if model_model_layers_6_self_attn_training_7693 else 0.0
        transpose_55_25289 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_55_25281, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_55_25281
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_6_25329 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_53_25209, transpose_54_25249, transpose_55_25289, dropout=ifexpr_6_7694, causal=True, attention_mask=None, query_length=2048)
        del transpose_53_25209, transpose_54_25249, transpose_55_25289
        nnscaler_flash_attention_forward_6_8574 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_6_25329, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_6_25329
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_243_8575 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_6_8574, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_6_8574
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_20_8576 = torch.Tensor.reshape(getitem_243_8575, shape=(8, 2048, 2048))
        del getitem_243_8575
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_6_8577 = torch.Tensor.contiguous(reshape_20_8576)
        del reshape_20_8576
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_45_8579 = torch.nn.functional.linear(contiguous_6_8577, self.model_model_layers_6_self_attn_o_proj_weight_8578, bias=None)
        del contiguous_6_8577
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_32_8580 = torch.add(add_29_81680, linear_45_8579, alpha=1)
        del add_29_81680, linear_45_8579
        # created at IRAdapterGener:local_consumer_multiref
        add_32_81844, add_32_81848 = nnscaler.runtime.function.multiref(add_32_8580, times=2)
        del add_32_8580
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_20_8582 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_32_81844, self.model_model_layers_6_post_attention_layernorm_weight_8581, (2048,), 1e-06)
        del add_32_81844
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_20_63703, fused_rms_norm_affine_20_63704, fused_rms_norm_affine_20_63705, fused_rms_norm_affine_20_63706 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_20_8582, times=4)
        del fused_rms_norm_affine_20_8582
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_6_mlp_gate_training_7696 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_5_8584, moe_route_5_8585, moe_route_5_8586 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_20_63703, self.model_model_layers_6_mlp_gate_weight_8583, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_6_mlp_gate_training_7696, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_20_63703
        moe_route_5_8585 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_5_8585, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_5_8586 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_5_8586, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_20_63704 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_20_63704, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_5_25529 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_20_63704, moe_route_5_8584, moe_route_5_8585, moe_route_5_8586, self.model_model_layers_6_mlp_gate_projs_25505, self.model_model_layers_6_mlp_up_projs_25513, self.model_model_layers_6_mlp_down_projs_25521, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_20_63704, moe_route_5_8584, moe_route_5_8585, moe_route_5_8586
        fused_rms_norm_affine_20_68488 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_20_63705, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_20_63705
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_46_25593 = torch.nn.functional.linear(fused_rms_norm_affine_20_68488, self.model_model_layers_6_mlp_shared_experts_gate_proj_weight_8591, bias=None)
        del fused_rms_norm_affine_20_68488
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_6_25649 = torch.nn.functional.silu(linear_46_25593, inplace=False)
        del linear_46_25593
        fused_rms_norm_affine_20_68528 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_20_63706, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_20_63706
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_47_25673 = torch.nn.functional.linear(fused_rms_norm_affine_20_68528, self.model_model_layers_6_mlp_shared_experts_up_proj_weight_8594, bias=None)
        del fused_rms_norm_affine_20_68528
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_34_25721 = torch.mul(silu_6_25649, linear_47_25673)
        del silu_6_25649, linear_47_25673
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_48_25745 = torch.nn.functional.linear(mul_34_25721, self.model_model_layers_6_mlp_shared_experts_down_proj_weight_8597, bias=None)
        del mul_34_25721
        nnscaler_moe_gmm_5_8590 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_5_25529, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_5_25529
        linear_48_8598 = nnscaler.runtime.adapter.nn.allgather_split(linear_48_25745, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_48_25745
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_33_8599 = torch.add(nnscaler_moe_gmm_5_8590, linear_48_8598, alpha=1)
        del nnscaler_moe_gmm_5_8590, linear_48_8598
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_34_8600 = torch.add(add_32_81848, add_33_8599, alpha=1)
        del add_32_81848, add_33_8599
        # created at IRAdapterGener:local_consumer_multiref
        add_34_81908, add_34_81912 = nnscaler.runtime.function.multiref(add_34_8600, times=2)
        del add_34_8600
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_21_8602 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_34_81908, self.model_model_layers_7_input_layernorm_weight_8601, (2048,), 1e-06)
        del add_34_81908
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_21_63715, fused_rms_norm_affine_21_63716 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_21_8602, times=2)
        del fused_rms_norm_affine_21_8602
        fused_rms_norm_affine_21_68584 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_21_63715, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_21_63715
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_49_25873 = torch.nn.functional.linear(fused_rms_norm_affine_21_68584, self.model_model_layers_7_self_attn_q_proj_weight_8603, bias=None)
        del fused_rms_norm_affine_21_68584
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_35_25929 = torch.Tensor.view(linear_49_25873, size=(8, 256, 16, 192))
        del linear_49_25873
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_56_25953 = torch.transpose(view_35_25929, dim0=1, dim1=2)
        del view_35_25929
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_21_26017, split_21_26025 = torch.functional.split(transpose_56_25953, split_size_or_sections=[128, 64], dim=-1)
        del transpose_56_25953
        fused_rms_norm_affine_21_68648 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_21_63716, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_21_63716
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_50_26057 = torch.nn.functional.linear(fused_rms_norm_affine_21_68648, self.model_model_layers_7_self_attn_kv_a_proj_with_mqa_weight_26049, bias=None)
        del fused_rms_norm_affine_21_68648
        linear_50_8610 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_50_26057, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_50_26057
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_22_8611, split_22_8612 = torch.functional.split(linear_50_8610, split_size_or_sections=[512, 64], dim=-1)
        del linear_50_8610
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_36_8613 = torch.Tensor.view(split_22_8612, size=(8, 2048, 1, 64))
        del split_22_8612
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_57_8614 = torch.transpose(view_36_8613, dim0=1, dim1=2)
        del view_36_8613
        split_22_26097 = nnscaler.runtime.adapter.nn.split_allgather(split_22_8611, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_22_8611
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_22_26177 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_22_26097, self.model_model_layers_7_self_attn_kv_a_layernorm_weight_8615, (512,), 1e-06)
        del split_22_26097
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_51_26193 = torch.nn.functional.linear(fused_rms_norm_affine_22_26177, self.model_model_layers_7_self_attn_kv_b_proj_weight_8617, bias=None)
        del fused_rms_norm_affine_22_26177
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_37_26249 = torch.Tensor.view(linear_51_26193, size=(8, 256, 16, 256))
        del linear_51_26193
        view_37_26241 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_37_26249, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_37_26249
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_58_26265 = torch.transpose(view_37_26241, dim0=1, dim1=2)
        del view_37_26241
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_23_26305, split_23_26313 = torch.functional.split(transpose_58_26265, split_size_or_sections=[128, 128], dim=-1)
        del transpose_58_26265
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_258_8624 = nnscaler.runtime.function.fullslice(self.model_model_layers_7_self_attn_rotary_emb_cos_cached_8623, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_14_8625 = torch.Tensor.to(getitem_258_8624, dtype=torch.bfloat16)
        del getitem_258_8624
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_259_8627 = nnscaler.runtime.function.fullslice(self.model_model_layers_7_self_attn_rotary_emb_sin_cached_8626, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_15_8628 = torch.Tensor.to(getitem_259_8627, dtype=torch.bfloat16)
        del getitem_259_8627
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_260_8629 = nnscaler.runtime.function.fullslice(to_14_8625, unsqueeze_8005)
        del to_14_8625
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_15_8630 = torch.unsqueeze(getitem_260_8629, dim=1)
        del getitem_260_8629
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_15_63721, unsqueeze_15_63722 = nnscaler.runtime.function.multiref(unsqueeze_15_8630, times=2)
        del unsqueeze_15_8630
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_261_8631 = nnscaler.runtime.function.fullslice(to_15_8628, unsqueeze_8005)
        del to_15_8628
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_16_8632 = torch.unsqueeze(getitem_261_8631, dim=1)
        del getitem_261_8631
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_16_63725, unsqueeze_16_63726 = nnscaler.runtime.function.multiref(unsqueeze_16_8632, times=2)
        del unsqueeze_16_8632
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_38_26401 = torch.Tensor.view(split_21_26025, size=(8, 16, 256, 32, 2))
        del split_21_26025
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_59_26441 = torch.transpose(view_38_26401, dim0=4, dim1=3)
        del view_38_26401
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_21_26473 = torch.Tensor.reshape(transpose_59_26441, shape=(8, 16, 256, 64))
        del transpose_59_26441
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_39_8636 = torch.Tensor.view(transpose_57_8614, size=(8, 1, 2048, 32, 2))
        del transpose_57_8614
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_60_8637 = torch.transpose(view_39_8636, dim0=4, dim1=3)
        del view_39_8636
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_22_8638 = torch.Tensor.reshape(transpose_60_8637, shape=(8, 1, 2048, 64))
        del transpose_60_8637
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_22_63733, reshape_22_63734, reshape_22_63735 = nnscaler.runtime.function.multiref(reshape_22_8638, times=3)
        del reshape_22_8638
        unsqueeze_15_68776 = nnscaler.runtime.adapter.chunk(unsqueeze_15_63721, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_15_63721
        # created at IRAdapterGener:local_consumer_multiref
        reshape_21_82003, reshape_21_82007, reshape_21_82011 = nnscaler.runtime.function.multiref(reshape_21_26473, times=3)
        del reshape_21_26473
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_35_26569 = torch.mul(reshape_21_82003, unsqueeze_15_68776)
        del unsqueeze_15_68776, reshape_21_82003
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_271_26617 = nnscaler.runtime.function.fullslice(reshape_21_82007, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_21_82007
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_273_26641 = nnscaler.runtime.function.fullslice(reshape_21_82011, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_21_82011
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_14_26665 = _operator.neg(getitem_273_26641)
        del getitem_273_26641
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_28_26705 = nnscaler.runtime.function.cat(neg_14_26665, getitem_271_26617, dim=-1)
        del getitem_271_26617, neg_14_26665
        unsqueeze_16_68848 = nnscaler.runtime.adapter.chunk(unsqueeze_16_63725, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_16_63725
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_36_26737 = torch.mul(cat_28_26705, unsqueeze_16_68848)
        del cat_28_26705, unsqueeze_16_68848
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_35_26785 = torch.add(mul_35_26569, mul_36_26737, alpha=1)
        del mul_35_26569, mul_36_26737
        unsqueeze_15_68880 = nnscaler.runtime.adapter.chunk(unsqueeze_15_63722, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_15_63722
        reshape_22_68872 = nnscaler.runtime.adapter.nn.split_allgather(reshape_22_63735, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_22_63735
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_37_26825 = torch.mul(reshape_22_68872, unsqueeze_15_68880)
        del unsqueeze_15_68880, reshape_22_68872
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_275_8647 = nnscaler.runtime.function.fullslice(reshape_22_63733, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_22_63733
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_277_8648 = nnscaler.runtime.function.fullslice(reshape_22_63734, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_22_63734
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_15_8649 = _operator.neg(getitem_277_8648)
        del getitem_277_8648
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_29_8650 = nnscaler.runtime.function.cat(neg_15_8649, getitem_275_8647, dim=-1)
        del getitem_275_8647, neg_15_8649
        cat_29_26929 = nnscaler.runtime.adapter.nn.split_allgather(cat_29_8650, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_29_8650
        unsqueeze_16_68904 = nnscaler.runtime.adapter.chunk(unsqueeze_16_63726, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_16_63726
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_38_26937 = torch.mul(cat_29_26929, unsqueeze_16_68904)
        del cat_29_26929, unsqueeze_16_68904
        mul_37_8646 = nnscaler.runtime.adapter.nn.allgather_split(mul_37_26825, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_37_26825
        mul_38_8651 = nnscaler.runtime.adapter.nn.allgather_split(mul_38_26937, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_38_26937
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_36_8652 = torch.add(mul_37_8646, mul_38_8651, alpha=1)
        del mul_37_8646, mul_38_8651
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_30_26985 = nnscaler.runtime.function.cat(split_21_26017, add_35_26785, dim=-1)
        del split_21_26017, add_35_26785
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_7_8654 = torch.Tensor.expand(add_36_8652, size=[-1, 16, -1, -1])
        del add_36_8652
        split_23_26321 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_23_26305, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_23_26305
        expand_7_27025 = nnscaler.runtime.adapter.nn.split_allgather(expand_7_8654, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_7_8654
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_31_27033 = nnscaler.runtime.function.cat(split_23_26321, expand_7_27025, dim=-1)
        del split_23_26321, expand_7_27025
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_7_27049 = torch.nn.functional.pad(split_23_26313, pad=[0, 64], mode='constant', value=0.0)
        del split_23_26313
        cat_30_26977 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_30_26985, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_30_26985
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_61_27081 = torch.transpose(cat_30_26977, dim0=1, dim1=2)
        del cat_30_26977
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_62_27121 = torch.transpose(cat_31_27033, dim0=1, dim1=2)
        del cat_31_27033
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_63_27153 = torch.transpose(pad_7_27049, dim0=1, dim1=2)
        del pad_7_27049
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_7_self_attn_training_7708 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_7_7709 = 0.0 if model_model_layers_7_self_attn_training_7708 else 0.0
        transpose_63_27161 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_63_27153, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_63_27153
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_7_27201 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_61_27081, transpose_62_27121, transpose_63_27161, dropout=ifexpr_7_7709, causal=True, attention_mask=None, query_length=2048)
        del transpose_61_27081, transpose_62_27121, transpose_63_27161
        nnscaler_flash_attention_forward_7_8660 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_7_27201, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_7_27201
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_278_8661 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_7_8660, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_7_8660
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_23_8662 = torch.Tensor.reshape(getitem_278_8661, shape=(8, 2048, 2048))
        del getitem_278_8661
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_7_8663 = torch.Tensor.contiguous(reshape_23_8662)
        del reshape_23_8662
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_52_8665 = torch.nn.functional.linear(contiguous_7_8663, self.model_model_layers_7_self_attn_o_proj_weight_8664, bias=None)
        del contiguous_7_8663
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_37_8666 = torch.add(add_34_81912, linear_52_8665, alpha=1)
        del add_34_81912, linear_52_8665
        # created at IRAdapterGener:local_consumer_multiref
        add_37_82076, add_37_82080 = nnscaler.runtime.function.multiref(add_37_8666, times=2)
        del add_37_8666
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_23_8668 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_37_82076, self.model_model_layers_7_post_attention_layernorm_weight_8667, (2048,), 1e-06)
        del add_37_82076
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_23_63747, fused_rms_norm_affine_23_63748, fused_rms_norm_affine_23_63749, fused_rms_norm_affine_23_63750 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_23_8668, times=4)
        del fused_rms_norm_affine_23_8668
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_7_mlp_gate_training_7711 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_6_8670, moe_route_6_8671, moe_route_6_8672 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_23_63747, self.model_model_layers_7_mlp_gate_weight_8669, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_7_mlp_gate_training_7711, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_23_63747
        moe_route_6_8671 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_6_8671, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_6_8672 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_6_8672, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_23_63748 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_23_63748, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_6_27401 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_23_63748, moe_route_6_8670, moe_route_6_8671, moe_route_6_8672, self.model_model_layers_7_mlp_gate_projs_27377, self.model_model_layers_7_mlp_up_projs_27385, self.model_model_layers_7_mlp_down_projs_27393, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_23_63748, moe_route_6_8670, moe_route_6_8671, moe_route_6_8672
        fused_rms_norm_affine_23_69064 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_23_63749, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_23_63749
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_53_27465 = torch.nn.functional.linear(fused_rms_norm_affine_23_69064, self.model_model_layers_7_mlp_shared_experts_gate_proj_weight_8677, bias=None)
        del fused_rms_norm_affine_23_69064
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_7_27521 = torch.nn.functional.silu(linear_53_27465, inplace=False)
        del linear_53_27465
        fused_rms_norm_affine_23_69104 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_23_63750, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_23_63750
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_54_27545 = torch.nn.functional.linear(fused_rms_norm_affine_23_69104, self.model_model_layers_7_mlp_shared_experts_up_proj_weight_8680, bias=None)
        del fused_rms_norm_affine_23_69104
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_39_27593 = torch.mul(silu_7_27521, linear_54_27545)
        del silu_7_27521, linear_54_27545
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_55_27617 = torch.nn.functional.linear(mul_39_27593, self.model_model_layers_7_mlp_shared_experts_down_proj_weight_8683, bias=None)
        del mul_39_27593
        nnscaler_moe_gmm_6_8676 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_6_27401, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_6_27401
        linear_55_8684 = nnscaler.runtime.adapter.nn.allgather_split(linear_55_27617, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_55_27617
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_38_8685 = torch.add(nnscaler_moe_gmm_6_8676, linear_55_8684, alpha=1)
        del nnscaler_moe_gmm_6_8676, linear_55_8684
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_39_8686 = torch.add(add_37_82080, add_38_8685, alpha=1)
        del add_37_82080, add_38_8685
        # created at IRAdapterGener:local_consumer_multiref
        add_39_82140, add_39_82144 = nnscaler.runtime.function.multiref(add_39_8686, times=2)
        del add_39_8686
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_24_8688 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_39_82140, self.model_model_layers_8_input_layernorm_weight_8687, (2048,), 1e-06)
        del add_39_82140
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_24_63759, fused_rms_norm_affine_24_63760 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_24_8688, times=2)
        del fused_rms_norm_affine_24_8688
        fused_rms_norm_affine_24_69160 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_24_63759, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_24_63759
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_56_27745 = torch.nn.functional.linear(fused_rms_norm_affine_24_69160, self.model_model_layers_8_self_attn_q_proj_weight_8689, bias=None)
        del fused_rms_norm_affine_24_69160
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_40_27801 = torch.Tensor.view(linear_56_27745, size=(8, 256, 16, 192))
        del linear_56_27745
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_64_27825 = torch.transpose(view_40_27801, dim0=1, dim1=2)
        del view_40_27801
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_24_27889, split_24_27897 = torch.functional.split(transpose_64_27825, split_size_or_sections=[128, 64], dim=-1)
        del transpose_64_27825
        fused_rms_norm_affine_24_69224 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_24_63760, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_24_63760
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_57_27929 = torch.nn.functional.linear(fused_rms_norm_affine_24_69224, self.model_model_layers_8_self_attn_kv_a_proj_with_mqa_weight_27921, bias=None)
        del fused_rms_norm_affine_24_69224
        linear_57_8696 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_57_27929, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_57_27929
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_25_8697, split_25_8698 = torch.functional.split(linear_57_8696, split_size_or_sections=[512, 64], dim=-1)
        del linear_57_8696
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_41_8699 = torch.Tensor.view(split_25_8698, size=(8, 2048, 1, 64))
        del split_25_8698
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_65_8700 = torch.transpose(view_41_8699, dim0=1, dim1=2)
        del view_41_8699
        split_25_27969 = nnscaler.runtime.adapter.nn.split_allgather(split_25_8697, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_25_8697
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_25_28049 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_25_27969, self.model_model_layers_8_self_attn_kv_a_layernorm_weight_8701, (512,), 1e-06)
        del split_25_27969
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_58_28065 = torch.nn.functional.linear(fused_rms_norm_affine_25_28049, self.model_model_layers_8_self_attn_kv_b_proj_weight_8703, bias=None)
        del fused_rms_norm_affine_25_28049
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_42_28121 = torch.Tensor.view(linear_58_28065, size=(8, 256, 16, 256))
        del linear_58_28065
        view_42_28113 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_42_28121, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_42_28121
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_66_28137 = torch.transpose(view_42_28113, dim0=1, dim1=2)
        del view_42_28113
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_26_28177, split_26_28185 = torch.functional.split(transpose_66_28137, split_size_or_sections=[128, 128], dim=-1)
        del transpose_66_28137
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_293_8710 = nnscaler.runtime.function.fullslice(self.model_model_layers_8_self_attn_rotary_emb_cos_cached_8709, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_16_8711 = torch.Tensor.to(getitem_293_8710, dtype=torch.bfloat16)
        del getitem_293_8710
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_294_8713 = nnscaler.runtime.function.fullslice(self.model_model_layers_8_self_attn_rotary_emb_sin_cached_8712, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_17_8714 = torch.Tensor.to(getitem_294_8713, dtype=torch.bfloat16)
        del getitem_294_8713
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_295_8715 = nnscaler.runtime.function.fullslice(to_16_8711, unsqueeze_8005)
        del to_16_8711
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_17_8716 = torch.unsqueeze(getitem_295_8715, dim=1)
        del getitem_295_8715
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_17_63765, unsqueeze_17_63766 = nnscaler.runtime.function.multiref(unsqueeze_17_8716, times=2)
        del unsqueeze_17_8716
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_296_8717 = nnscaler.runtime.function.fullslice(to_17_8714, unsqueeze_8005)
        del to_17_8714
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_18_8718 = torch.unsqueeze(getitem_296_8717, dim=1)
        del getitem_296_8717
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_18_63769, unsqueeze_18_63770 = nnscaler.runtime.function.multiref(unsqueeze_18_8718, times=2)
        del unsqueeze_18_8718
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_43_28273 = torch.Tensor.view(split_24_27897, size=(8, 16, 256, 32, 2))
        del split_24_27897
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_67_28313 = torch.transpose(view_43_28273, dim0=4, dim1=3)
        del view_43_28273
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_24_28345 = torch.Tensor.reshape(transpose_67_28313, shape=(8, 16, 256, 64))
        del transpose_67_28313
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_44_8722 = torch.Tensor.view(transpose_65_8700, size=(8, 1, 2048, 32, 2))
        del transpose_65_8700
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_68_8723 = torch.transpose(view_44_8722, dim0=4, dim1=3)
        del view_44_8722
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_25_8724 = torch.Tensor.reshape(transpose_68_8723, shape=(8, 1, 2048, 64))
        del transpose_68_8723
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_25_63777, reshape_25_63778, reshape_25_63779 = nnscaler.runtime.function.multiref(reshape_25_8724, times=3)
        del reshape_25_8724
        unsqueeze_17_69352 = nnscaler.runtime.adapter.chunk(unsqueeze_17_63765, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_17_63765
        # created at IRAdapterGener:local_consumer_multiref
        reshape_24_82235, reshape_24_82239, reshape_24_82243 = nnscaler.runtime.function.multiref(reshape_24_28345, times=3)
        del reshape_24_28345
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_40_28441 = torch.mul(reshape_24_82235, unsqueeze_17_69352)
        del unsqueeze_17_69352, reshape_24_82235
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_306_28489 = nnscaler.runtime.function.fullslice(reshape_24_82239, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_24_82239
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_308_28513 = nnscaler.runtime.function.fullslice(reshape_24_82243, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_24_82243
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_16_28537 = _operator.neg(getitem_308_28513)
        del getitem_308_28513
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_32_28577 = nnscaler.runtime.function.cat(neg_16_28537, getitem_306_28489, dim=-1)
        del getitem_306_28489, neg_16_28537
        unsqueeze_18_69424 = nnscaler.runtime.adapter.chunk(unsqueeze_18_63769, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_18_63769
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_41_28609 = torch.mul(cat_32_28577, unsqueeze_18_69424)
        del cat_32_28577, unsqueeze_18_69424
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_40_28657 = torch.add(mul_40_28441, mul_41_28609, alpha=1)
        del mul_40_28441, mul_41_28609
        unsqueeze_17_69456 = nnscaler.runtime.adapter.chunk(unsqueeze_17_63766, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_17_63766
        reshape_25_69448 = nnscaler.runtime.adapter.nn.split_allgather(reshape_25_63779, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_25_63779
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_42_28697 = torch.mul(reshape_25_69448, unsqueeze_17_69456)
        del unsqueeze_17_69456, reshape_25_69448
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_310_8733 = nnscaler.runtime.function.fullslice(reshape_25_63777, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_25_63777
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_312_8734 = nnscaler.runtime.function.fullslice(reshape_25_63778, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_25_63778
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_17_8735 = _operator.neg(getitem_312_8734)
        del getitem_312_8734
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_33_8736 = nnscaler.runtime.function.cat(neg_17_8735, getitem_310_8733, dim=-1)
        del getitem_310_8733, neg_17_8735
        cat_33_28801 = nnscaler.runtime.adapter.nn.split_allgather(cat_33_8736, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_33_8736
        unsqueeze_18_69480 = nnscaler.runtime.adapter.chunk(unsqueeze_18_63770, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_18_63770
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_43_28809 = torch.mul(cat_33_28801, unsqueeze_18_69480)
        del cat_33_28801, unsqueeze_18_69480
        mul_42_8732 = nnscaler.runtime.adapter.nn.allgather_split(mul_42_28697, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_42_28697
        mul_43_8737 = nnscaler.runtime.adapter.nn.allgather_split(mul_43_28809, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_43_28809
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_41_8738 = torch.add(mul_42_8732, mul_43_8737, alpha=1)
        del mul_42_8732, mul_43_8737
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_34_28857 = nnscaler.runtime.function.cat(split_24_27889, add_40_28657, dim=-1)
        del split_24_27889, add_40_28657
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_8_8740 = torch.Tensor.expand(add_41_8738, size=[-1, 16, -1, -1])
        del add_41_8738
        split_26_28193 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_26_28177, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_26_28177
        expand_8_28897 = nnscaler.runtime.adapter.nn.split_allgather(expand_8_8740, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_8_8740
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_35_28905 = nnscaler.runtime.function.cat(split_26_28193, expand_8_28897, dim=-1)
        del split_26_28193, expand_8_28897
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_8_28921 = torch.nn.functional.pad(split_26_28185, pad=[0, 64], mode='constant', value=0.0)
        del split_26_28185
        cat_34_28849 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_34_28857, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_34_28857
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_69_28953 = torch.transpose(cat_34_28849, dim0=1, dim1=2)
        del cat_34_28849
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_70_28993 = torch.transpose(cat_35_28905, dim0=1, dim1=2)
        del cat_35_28905
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_71_29025 = torch.transpose(pad_8_28921, dim0=1, dim1=2)
        del pad_8_28921
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_8_self_attn_training_7723 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_8_7724 = 0.0 if model_model_layers_8_self_attn_training_7723 else 0.0
        transpose_71_29033 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_71_29025, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_71_29025
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_8_29073 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_69_28953, transpose_70_28993, transpose_71_29033, dropout=ifexpr_8_7724, causal=True, attention_mask=None, query_length=2048)
        del transpose_69_28953, transpose_70_28993, transpose_71_29033
        nnscaler_flash_attention_forward_8_8746 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_8_29073, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_8_29073
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_313_8747 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_8_8746, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_8_8746
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_26_8748 = torch.Tensor.reshape(getitem_313_8747, shape=(8, 2048, 2048))
        del getitem_313_8747
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_8_8749 = torch.Tensor.contiguous(reshape_26_8748)
        del reshape_26_8748
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_59_8751 = torch.nn.functional.linear(contiguous_8_8749, self.model_model_layers_8_self_attn_o_proj_weight_8750, bias=None)
        del contiguous_8_8749
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_42_8752 = torch.add(add_39_82144, linear_59_8751, alpha=1)
        del add_39_82144, linear_59_8751
        # created at IRAdapterGener:local_consumer_multiref
        add_42_82308, add_42_82312 = nnscaler.runtime.function.multiref(add_42_8752, times=2)
        del add_42_8752
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_26_8754 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_42_82308, self.model_model_layers_8_post_attention_layernorm_weight_8753, (2048,), 1e-06)
        del add_42_82308
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_26_63791, fused_rms_norm_affine_26_63792, fused_rms_norm_affine_26_63793, fused_rms_norm_affine_26_63794 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_26_8754, times=4)
        del fused_rms_norm_affine_26_8754
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_8_mlp_gate_training_7726 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_7_8756, moe_route_7_8757, moe_route_7_8758 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_26_63791, self.model_model_layers_8_mlp_gate_weight_8755, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_8_mlp_gate_training_7726, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_26_63791
        moe_route_7_8757 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_7_8757, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_7_8758 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_7_8758, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_26_63792 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_26_63792, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_7_29273 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_26_63792, moe_route_7_8756, moe_route_7_8757, moe_route_7_8758, self.model_model_layers_8_mlp_gate_projs_29249, self.model_model_layers_8_mlp_up_projs_29257, self.model_model_layers_8_mlp_down_projs_29265, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_26_63792, moe_route_7_8756, moe_route_7_8757, moe_route_7_8758
        fused_rms_norm_affine_26_69640 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_26_63793, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_26_63793
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_60_29337 = torch.nn.functional.linear(fused_rms_norm_affine_26_69640, self.model_model_layers_8_mlp_shared_experts_gate_proj_weight_8763, bias=None)
        del fused_rms_norm_affine_26_69640
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_8_29393 = torch.nn.functional.silu(linear_60_29337, inplace=False)
        del linear_60_29337
        fused_rms_norm_affine_26_69680 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_26_63794, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_26_63794
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_61_29417 = torch.nn.functional.linear(fused_rms_norm_affine_26_69680, self.model_model_layers_8_mlp_shared_experts_up_proj_weight_8766, bias=None)
        del fused_rms_norm_affine_26_69680
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_44_29465 = torch.mul(silu_8_29393, linear_61_29417)
        del silu_8_29393, linear_61_29417
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_62_29489 = torch.nn.functional.linear(mul_44_29465, self.model_model_layers_8_mlp_shared_experts_down_proj_weight_8769, bias=None)
        del mul_44_29465
        nnscaler_moe_gmm_7_8762 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_7_29273, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_7_29273
        linear_62_8770 = nnscaler.runtime.adapter.nn.allgather_split(linear_62_29489, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_62_29489
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_43_8771 = torch.add(nnscaler_moe_gmm_7_8762, linear_62_8770, alpha=1)
        del nnscaler_moe_gmm_7_8762, linear_62_8770
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_44_8772 = torch.add(add_42_82312, add_43_8771, alpha=1)
        del add_42_82312, add_43_8771
        # created at IRAdapterGener:local_consumer_multiref
        add_44_82372, add_44_82376 = nnscaler.runtime.function.multiref(add_44_8772, times=2)
        del add_44_8772
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_27_8774 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_44_82372, self.model_model_layers_9_input_layernorm_weight_8773, (2048,), 1e-06)
        del add_44_82372
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_27_63803, fused_rms_norm_affine_27_63804 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_27_8774, times=2)
        del fused_rms_norm_affine_27_8774
        fused_rms_norm_affine_27_69736 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_27_63803, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_27_63803
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_63_29617 = torch.nn.functional.linear(fused_rms_norm_affine_27_69736, self.model_model_layers_9_self_attn_q_proj_weight_8775, bias=None)
        del fused_rms_norm_affine_27_69736
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_45_29673 = torch.Tensor.view(linear_63_29617, size=(8, 256, 16, 192))
        del linear_63_29617
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_72_29697 = torch.transpose(view_45_29673, dim0=1, dim1=2)
        del view_45_29673
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_27_29761, split_27_29769 = torch.functional.split(transpose_72_29697, split_size_or_sections=[128, 64], dim=-1)
        del transpose_72_29697
        fused_rms_norm_affine_27_69800 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_27_63804, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_27_63804
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_64_29801 = torch.nn.functional.linear(fused_rms_norm_affine_27_69800, self.model_model_layers_9_self_attn_kv_a_proj_with_mqa_weight_29793, bias=None)
        del fused_rms_norm_affine_27_69800
        linear_64_8782 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_64_29801, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_64_29801
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_28_8783, split_28_8784 = torch.functional.split(linear_64_8782, split_size_or_sections=[512, 64], dim=-1)
        del linear_64_8782
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_46_8785 = torch.Tensor.view(split_28_8784, size=(8, 2048, 1, 64))
        del split_28_8784
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_73_8786 = torch.transpose(view_46_8785, dim0=1, dim1=2)
        del view_46_8785
        split_28_29841 = nnscaler.runtime.adapter.nn.split_allgather(split_28_8783, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_28_8783
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_28_29921 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_28_29841, self.model_model_layers_9_self_attn_kv_a_layernorm_weight_8787, (512,), 1e-06)
        del split_28_29841
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_65_29937 = torch.nn.functional.linear(fused_rms_norm_affine_28_29921, self.model_model_layers_9_self_attn_kv_b_proj_weight_8789, bias=None)
        del fused_rms_norm_affine_28_29921
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_47_29993 = torch.Tensor.view(linear_65_29937, size=(8, 256, 16, 256))
        del linear_65_29937
        view_47_29985 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_47_29993, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_47_29993
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_74_30009 = torch.transpose(view_47_29985, dim0=1, dim1=2)
        del view_47_29985
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_29_30049, split_29_30057 = torch.functional.split(transpose_74_30009, split_size_or_sections=[128, 128], dim=-1)
        del transpose_74_30009
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_328_8796 = nnscaler.runtime.function.fullslice(self.model_model_layers_9_self_attn_rotary_emb_cos_cached_8795, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_18_8797 = torch.Tensor.to(getitem_328_8796, dtype=torch.bfloat16)
        del getitem_328_8796
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_329_8799 = nnscaler.runtime.function.fullslice(self.model_model_layers_9_self_attn_rotary_emb_sin_cached_8798, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_19_8800 = torch.Tensor.to(getitem_329_8799, dtype=torch.bfloat16)
        del getitem_329_8799
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_330_8801 = nnscaler.runtime.function.fullslice(to_18_8797, unsqueeze_8005)
        del to_18_8797
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_19_8802 = torch.unsqueeze(getitem_330_8801, dim=1)
        del getitem_330_8801
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_19_63809, unsqueeze_19_63810 = nnscaler.runtime.function.multiref(unsqueeze_19_8802, times=2)
        del unsqueeze_19_8802
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_331_8803 = nnscaler.runtime.function.fullslice(to_19_8800, unsqueeze_8005)
        del to_19_8800
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_20_8804 = torch.unsqueeze(getitem_331_8803, dim=1)
        del getitem_331_8803
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_20_63813, unsqueeze_20_63814 = nnscaler.runtime.function.multiref(unsqueeze_20_8804, times=2)
        del unsqueeze_20_8804
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_48_30145 = torch.Tensor.view(split_27_29769, size=(8, 16, 256, 32, 2))
        del split_27_29769
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_75_30185 = torch.transpose(view_48_30145, dim0=4, dim1=3)
        del view_48_30145
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_27_30217 = torch.Tensor.reshape(transpose_75_30185, shape=(8, 16, 256, 64))
        del transpose_75_30185
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_49_8808 = torch.Tensor.view(transpose_73_8786, size=(8, 1, 2048, 32, 2))
        del transpose_73_8786
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_76_8809 = torch.transpose(view_49_8808, dim0=4, dim1=3)
        del view_49_8808
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_28_8810 = torch.Tensor.reshape(transpose_76_8809, shape=(8, 1, 2048, 64))
        del transpose_76_8809
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_28_63821, reshape_28_63822, reshape_28_63823 = nnscaler.runtime.function.multiref(reshape_28_8810, times=3)
        del reshape_28_8810
        unsqueeze_19_69928 = nnscaler.runtime.adapter.chunk(unsqueeze_19_63809, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_19_63809
        # created at IRAdapterGener:local_consumer_multiref
        reshape_27_82467, reshape_27_82471, reshape_27_82475 = nnscaler.runtime.function.multiref(reshape_27_30217, times=3)
        del reshape_27_30217
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_45_30313 = torch.mul(reshape_27_82467, unsqueeze_19_69928)
        del unsqueeze_19_69928, reshape_27_82467
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_341_30361 = nnscaler.runtime.function.fullslice(reshape_27_82471, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_27_82471
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_343_30385 = nnscaler.runtime.function.fullslice(reshape_27_82475, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_27_82475
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_18_30409 = _operator.neg(getitem_343_30385)
        del getitem_343_30385
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_36_30449 = nnscaler.runtime.function.cat(neg_18_30409, getitem_341_30361, dim=-1)
        del getitem_341_30361, neg_18_30409
        unsqueeze_20_70000 = nnscaler.runtime.adapter.chunk(unsqueeze_20_63813, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_20_63813
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_46_30481 = torch.mul(cat_36_30449, unsqueeze_20_70000)
        del cat_36_30449, unsqueeze_20_70000
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_45_30529 = torch.add(mul_45_30313, mul_46_30481, alpha=1)
        del mul_45_30313, mul_46_30481
        unsqueeze_19_70032 = nnscaler.runtime.adapter.chunk(unsqueeze_19_63810, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_19_63810
        reshape_28_70024 = nnscaler.runtime.adapter.nn.split_allgather(reshape_28_63823, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_28_63823
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_47_30569 = torch.mul(reshape_28_70024, unsqueeze_19_70032)
        del unsqueeze_19_70032, reshape_28_70024
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_345_8819 = nnscaler.runtime.function.fullslice(reshape_28_63821, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_28_63821
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_347_8820 = nnscaler.runtime.function.fullslice(reshape_28_63822, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_28_63822
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_19_8821 = _operator.neg(getitem_347_8820)
        del getitem_347_8820
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_37_8822 = nnscaler.runtime.function.cat(neg_19_8821, getitem_345_8819, dim=-1)
        del getitem_345_8819, neg_19_8821
        cat_37_30673 = nnscaler.runtime.adapter.nn.split_allgather(cat_37_8822, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_37_8822
        unsqueeze_20_70056 = nnscaler.runtime.adapter.chunk(unsqueeze_20_63814, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_20_63814
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_48_30681 = torch.mul(cat_37_30673, unsqueeze_20_70056)
        del cat_37_30673, unsqueeze_20_70056
        mul_47_8818 = nnscaler.runtime.adapter.nn.allgather_split(mul_47_30569, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_47_30569
        mul_48_8823 = nnscaler.runtime.adapter.nn.allgather_split(mul_48_30681, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_48_30681
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_46_8824 = torch.add(mul_47_8818, mul_48_8823, alpha=1)
        del mul_47_8818, mul_48_8823
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_38_30729 = nnscaler.runtime.function.cat(split_27_29761, add_45_30529, dim=-1)
        del split_27_29761, add_45_30529
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_9_8826 = torch.Tensor.expand(add_46_8824, size=[-1, 16, -1, -1])
        del add_46_8824
        split_29_30065 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_29_30049, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_29_30049
        expand_9_30769 = nnscaler.runtime.adapter.nn.split_allgather(expand_9_8826, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_9_8826
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_39_30777 = nnscaler.runtime.function.cat(split_29_30065, expand_9_30769, dim=-1)
        del split_29_30065, expand_9_30769
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_9_30793 = torch.nn.functional.pad(split_29_30057, pad=[0, 64], mode='constant', value=0.0)
        del split_29_30057
        cat_38_30721 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_38_30729, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_38_30729
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_77_30825 = torch.transpose(cat_38_30721, dim0=1, dim1=2)
        del cat_38_30721
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_78_30865 = torch.transpose(cat_39_30777, dim0=1, dim1=2)
        del cat_39_30777
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_79_30897 = torch.transpose(pad_9_30793, dim0=1, dim1=2)
        del pad_9_30793
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_9_self_attn_training_7738 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_9_7739 = 0.0 if model_model_layers_9_self_attn_training_7738 else 0.0
        transpose_79_30905 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_79_30897, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_79_30897
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_9_30945 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_77_30825, transpose_78_30865, transpose_79_30905, dropout=ifexpr_9_7739, causal=True, attention_mask=None, query_length=2048)
        del transpose_77_30825, transpose_78_30865, transpose_79_30905
        nnscaler_flash_attention_forward_9_8832 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_9_30945, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_9_30945
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_348_8833 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_9_8832, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_9_8832
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_29_8834 = torch.Tensor.reshape(getitem_348_8833, shape=(8, 2048, 2048))
        del getitem_348_8833
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_9_8835 = torch.Tensor.contiguous(reshape_29_8834)
        del reshape_29_8834
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_66_8837 = torch.nn.functional.linear(contiguous_9_8835, self.model_model_layers_9_self_attn_o_proj_weight_8836, bias=None)
        del contiguous_9_8835
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_47_8838 = torch.add(add_44_82376, linear_66_8837, alpha=1)
        del add_44_82376, linear_66_8837
        # created at IRAdapterGener:local_consumer_multiref
        add_47_82540, add_47_82544 = nnscaler.runtime.function.multiref(add_47_8838, times=2)
        del add_47_8838
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_29_8840 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_47_82540, self.model_model_layers_9_post_attention_layernorm_weight_8839, (2048,), 1e-06)
        del add_47_82540
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_29_63835, fused_rms_norm_affine_29_63836, fused_rms_norm_affine_29_63837, fused_rms_norm_affine_29_63838 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_29_8840, times=4)
        del fused_rms_norm_affine_29_8840
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_9_mlp_gate_training_7741 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_8_8842, moe_route_8_8843, moe_route_8_8844 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_29_63835, self.model_model_layers_9_mlp_gate_weight_8841, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_9_mlp_gate_training_7741, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_29_63835
        moe_route_8_8843 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_8_8843, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_8_8844 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_8_8844, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_29_63836 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_29_63836, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_8_31145 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_29_63836, moe_route_8_8842, moe_route_8_8843, moe_route_8_8844, self.model_model_layers_9_mlp_gate_projs_31121, self.model_model_layers_9_mlp_up_projs_31129, self.model_model_layers_9_mlp_down_projs_31137, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_29_63836, moe_route_8_8842, moe_route_8_8843, moe_route_8_8844
        fused_rms_norm_affine_29_70216 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_29_63837, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_29_63837
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_67_31209 = torch.nn.functional.linear(fused_rms_norm_affine_29_70216, self.model_model_layers_9_mlp_shared_experts_gate_proj_weight_8849, bias=None)
        del fused_rms_norm_affine_29_70216
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_9_31265 = torch.nn.functional.silu(linear_67_31209, inplace=False)
        del linear_67_31209
        fused_rms_norm_affine_29_70256 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_29_63838, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_29_63838
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_68_31289 = torch.nn.functional.linear(fused_rms_norm_affine_29_70256, self.model_model_layers_9_mlp_shared_experts_up_proj_weight_8852, bias=None)
        del fused_rms_norm_affine_29_70256
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_49_31337 = torch.mul(silu_9_31265, linear_68_31289)
        del silu_9_31265, linear_68_31289
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_69_31361 = torch.nn.functional.linear(mul_49_31337, self.model_model_layers_9_mlp_shared_experts_down_proj_weight_8855, bias=None)
        del mul_49_31337
        nnscaler_moe_gmm_8_8848 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_8_31145, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_8_31145
        linear_69_8856 = nnscaler.runtime.adapter.nn.allgather_split(linear_69_31361, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_69_31361
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_48_8857 = torch.add(nnscaler_moe_gmm_8_8848, linear_69_8856, alpha=1)
        del nnscaler_moe_gmm_8_8848, linear_69_8856
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_49_8858 = torch.add(add_47_82544, add_48_8857, alpha=1)
        del add_47_82544, add_48_8857
        # created at IRAdapterGener:local_consumer_multiref
        add_49_82604, add_49_82608 = nnscaler.runtime.function.multiref(add_49_8858, times=2)
        del add_49_8858
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_30_8860 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_49_82604, self.model_model_layers_10_input_layernorm_weight_8859, (2048,), 1e-06)
        del add_49_82604
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_30_63847, fused_rms_norm_affine_30_63848 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_30_8860, times=2)
        del fused_rms_norm_affine_30_8860
        fused_rms_norm_affine_30_70312 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_30_63847, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_30_63847
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_70_31489 = torch.nn.functional.linear(fused_rms_norm_affine_30_70312, self.model_model_layers_10_self_attn_q_proj_weight_8861, bias=None)
        del fused_rms_norm_affine_30_70312
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_50_31545 = torch.Tensor.view(linear_70_31489, size=(8, 256, 16, 192))
        del linear_70_31489
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_80_31569 = torch.transpose(view_50_31545, dim0=1, dim1=2)
        del view_50_31545
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_30_31633, split_30_31641 = torch.functional.split(transpose_80_31569, split_size_or_sections=[128, 64], dim=-1)
        del transpose_80_31569
        fused_rms_norm_affine_30_70376 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_30_63848, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_30_63848
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_71_31673 = torch.nn.functional.linear(fused_rms_norm_affine_30_70376, self.model_model_layers_10_self_attn_kv_a_proj_with_mqa_weight_31665, bias=None)
        del fused_rms_norm_affine_30_70376
        linear_71_8868 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_71_31673, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_71_31673
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_31_8869, split_31_8870 = torch.functional.split(linear_71_8868, split_size_or_sections=[512, 64], dim=-1)
        del linear_71_8868
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_51_8871 = torch.Tensor.view(split_31_8870, size=(8, 2048, 1, 64))
        del split_31_8870
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_81_8872 = torch.transpose(view_51_8871, dim0=1, dim1=2)
        del view_51_8871
        split_31_31713 = nnscaler.runtime.adapter.nn.split_allgather(split_31_8869, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_31_8869
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_31_31793 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_31_31713, self.model_model_layers_10_self_attn_kv_a_layernorm_weight_8873, (512,), 1e-06)
        del split_31_31713
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_72_31809 = torch.nn.functional.linear(fused_rms_norm_affine_31_31793, self.model_model_layers_10_self_attn_kv_b_proj_weight_8875, bias=None)
        del fused_rms_norm_affine_31_31793
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_52_31865 = torch.Tensor.view(linear_72_31809, size=(8, 256, 16, 256))
        del linear_72_31809
        view_52_31857 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_52_31865, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_52_31865
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_82_31881 = torch.transpose(view_52_31857, dim0=1, dim1=2)
        del view_52_31857
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_32_31921, split_32_31929 = torch.functional.split(transpose_82_31881, split_size_or_sections=[128, 128], dim=-1)
        del transpose_82_31881
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_363_8882 = nnscaler.runtime.function.fullslice(self.model_model_layers_10_self_attn_rotary_emb_cos_cached_8881, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_20_8883 = torch.Tensor.to(getitem_363_8882, dtype=torch.bfloat16)
        del getitem_363_8882
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_364_8885 = nnscaler.runtime.function.fullslice(self.model_model_layers_10_self_attn_rotary_emb_sin_cached_8884, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_21_8886 = torch.Tensor.to(getitem_364_8885, dtype=torch.bfloat16)
        del getitem_364_8885
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_365_8887 = nnscaler.runtime.function.fullslice(to_20_8883, unsqueeze_8005)
        del to_20_8883
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_21_8888 = torch.unsqueeze(getitem_365_8887, dim=1)
        del getitem_365_8887
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_21_63853, unsqueeze_21_63854 = nnscaler.runtime.function.multiref(unsqueeze_21_8888, times=2)
        del unsqueeze_21_8888
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_366_8889 = nnscaler.runtime.function.fullslice(to_21_8886, unsqueeze_8005)
        del to_21_8886
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_22_8890 = torch.unsqueeze(getitem_366_8889, dim=1)
        del getitem_366_8889
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_22_63857, unsqueeze_22_63858 = nnscaler.runtime.function.multiref(unsqueeze_22_8890, times=2)
        del unsqueeze_22_8890
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_53_32017 = torch.Tensor.view(split_30_31641, size=(8, 16, 256, 32, 2))
        del split_30_31641
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_83_32057 = torch.transpose(view_53_32017, dim0=4, dim1=3)
        del view_53_32017
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_30_32089 = torch.Tensor.reshape(transpose_83_32057, shape=(8, 16, 256, 64))
        del transpose_83_32057
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_54_8894 = torch.Tensor.view(transpose_81_8872, size=(8, 1, 2048, 32, 2))
        del transpose_81_8872
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_84_8895 = torch.transpose(view_54_8894, dim0=4, dim1=3)
        del view_54_8894
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_31_8896 = torch.Tensor.reshape(transpose_84_8895, shape=(8, 1, 2048, 64))
        del transpose_84_8895
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_31_63865, reshape_31_63866, reshape_31_63867 = nnscaler.runtime.function.multiref(reshape_31_8896, times=3)
        del reshape_31_8896
        unsqueeze_21_70504 = nnscaler.runtime.adapter.chunk(unsqueeze_21_63853, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_21_63853
        # created at IRAdapterGener:local_consumer_multiref
        reshape_30_82699, reshape_30_82703, reshape_30_82707 = nnscaler.runtime.function.multiref(reshape_30_32089, times=3)
        del reshape_30_32089
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_50_32185 = torch.mul(reshape_30_82699, unsqueeze_21_70504)
        del unsqueeze_21_70504, reshape_30_82699
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_376_32233 = nnscaler.runtime.function.fullslice(reshape_30_82703, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_30_82703
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_378_32257 = nnscaler.runtime.function.fullslice(reshape_30_82707, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_30_82707
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_20_32281 = _operator.neg(getitem_378_32257)
        del getitem_378_32257
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_40_32321 = nnscaler.runtime.function.cat(neg_20_32281, getitem_376_32233, dim=-1)
        del getitem_376_32233, neg_20_32281
        unsqueeze_22_70576 = nnscaler.runtime.adapter.chunk(unsqueeze_22_63857, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_22_63857
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_51_32353 = torch.mul(cat_40_32321, unsqueeze_22_70576)
        del cat_40_32321, unsqueeze_22_70576
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_50_32401 = torch.add(mul_50_32185, mul_51_32353, alpha=1)
        del mul_50_32185, mul_51_32353
        unsqueeze_21_70608 = nnscaler.runtime.adapter.chunk(unsqueeze_21_63854, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_21_63854
        reshape_31_70600 = nnscaler.runtime.adapter.nn.split_allgather(reshape_31_63867, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_31_63867
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_52_32441 = torch.mul(reshape_31_70600, unsqueeze_21_70608)
        del unsqueeze_21_70608, reshape_31_70600
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_380_8905 = nnscaler.runtime.function.fullslice(reshape_31_63865, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_31_63865
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_382_8906 = nnscaler.runtime.function.fullslice(reshape_31_63866, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_31_63866
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_21_8907 = _operator.neg(getitem_382_8906)
        del getitem_382_8906
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_41_8908 = nnscaler.runtime.function.cat(neg_21_8907, getitem_380_8905, dim=-1)
        del getitem_380_8905, neg_21_8907
        cat_41_32545 = nnscaler.runtime.adapter.nn.split_allgather(cat_41_8908, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_41_8908
        unsqueeze_22_70632 = nnscaler.runtime.adapter.chunk(unsqueeze_22_63858, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_22_63858
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_53_32553 = torch.mul(cat_41_32545, unsqueeze_22_70632)
        del cat_41_32545, unsqueeze_22_70632
        mul_52_8904 = nnscaler.runtime.adapter.nn.allgather_split(mul_52_32441, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_52_32441
        mul_53_8909 = nnscaler.runtime.adapter.nn.allgather_split(mul_53_32553, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_53_32553
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_51_8910 = torch.add(mul_52_8904, mul_53_8909, alpha=1)
        del mul_52_8904, mul_53_8909
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_42_32601 = nnscaler.runtime.function.cat(split_30_31633, add_50_32401, dim=-1)
        del split_30_31633, add_50_32401
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_10_8912 = torch.Tensor.expand(add_51_8910, size=[-1, 16, -1, -1])
        del add_51_8910
        split_32_31937 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_32_31921, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_32_31921
        expand_10_32641 = nnscaler.runtime.adapter.nn.split_allgather(expand_10_8912, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_10_8912
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_43_32649 = nnscaler.runtime.function.cat(split_32_31937, expand_10_32641, dim=-1)
        del split_32_31937, expand_10_32641
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_10_32665 = torch.nn.functional.pad(split_32_31929, pad=[0, 64], mode='constant', value=0.0)
        del split_32_31929
        cat_42_32593 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_42_32601, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_42_32601
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_85_32697 = torch.transpose(cat_42_32593, dim0=1, dim1=2)
        del cat_42_32593
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_86_32737 = torch.transpose(cat_43_32649, dim0=1, dim1=2)
        del cat_43_32649
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_87_32769 = torch.transpose(pad_10_32665, dim0=1, dim1=2)
        del pad_10_32665
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_10_self_attn_training_7753 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_10_7754 = 0.0 if model_model_layers_10_self_attn_training_7753 else 0.0
        transpose_87_32777 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_87_32769, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_87_32769
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_10_32817 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_85_32697, transpose_86_32737, transpose_87_32777, dropout=ifexpr_10_7754, causal=True, attention_mask=None, query_length=2048)
        del transpose_85_32697, transpose_86_32737, transpose_87_32777
        nnscaler_flash_attention_forward_10_8918 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_10_32817, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_10_32817
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_383_8919 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_10_8918, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_10_8918
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_32_8920 = torch.Tensor.reshape(getitem_383_8919, shape=(8, 2048, 2048))
        del getitem_383_8919
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_10_8921 = torch.Tensor.contiguous(reshape_32_8920)
        del reshape_32_8920
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_73_8923 = torch.nn.functional.linear(contiguous_10_8921, self.model_model_layers_10_self_attn_o_proj_weight_8922, bias=None)
        del contiguous_10_8921
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_52_8924 = torch.add(add_49_82608, linear_73_8923, alpha=1)
        del add_49_82608, linear_73_8923
        # created at IRAdapterGener:local_consumer_multiref
        add_52_82772, add_52_82776 = nnscaler.runtime.function.multiref(add_52_8924, times=2)
        del add_52_8924
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_32_8926 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_52_82772, self.model_model_layers_10_post_attention_layernorm_weight_8925, (2048,), 1e-06)
        del add_52_82772
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_32_63879, fused_rms_norm_affine_32_63880, fused_rms_norm_affine_32_63881, fused_rms_norm_affine_32_63882 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_32_8926, times=4)
        del fused_rms_norm_affine_32_8926
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_10_mlp_gate_training_7756 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_9_8928, moe_route_9_8929, moe_route_9_8930 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_32_63879, self.model_model_layers_10_mlp_gate_weight_8927, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_10_mlp_gate_training_7756, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_32_63879
        moe_route_9_8929 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_9_8929, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_9_8930 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_9_8930, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_32_63880 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_32_63880, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_9_33017 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_32_63880, moe_route_9_8928, moe_route_9_8929, moe_route_9_8930, self.model_model_layers_10_mlp_gate_projs_32993, self.model_model_layers_10_mlp_up_projs_33001, self.model_model_layers_10_mlp_down_projs_33009, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_32_63880, moe_route_9_8928, moe_route_9_8929, moe_route_9_8930
        fused_rms_norm_affine_32_70792 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_32_63881, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_32_63881
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_74_33081 = torch.nn.functional.linear(fused_rms_norm_affine_32_70792, self.model_model_layers_10_mlp_shared_experts_gate_proj_weight_8935, bias=None)
        del fused_rms_norm_affine_32_70792
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_10_33137 = torch.nn.functional.silu(linear_74_33081, inplace=False)
        del linear_74_33081
        fused_rms_norm_affine_32_70832 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_32_63882, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_32_63882
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_75_33161 = torch.nn.functional.linear(fused_rms_norm_affine_32_70832, self.model_model_layers_10_mlp_shared_experts_up_proj_weight_8938, bias=None)
        del fused_rms_norm_affine_32_70832
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_54_33209 = torch.mul(silu_10_33137, linear_75_33161)
        del silu_10_33137, linear_75_33161
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_76_33233 = torch.nn.functional.linear(mul_54_33209, self.model_model_layers_10_mlp_shared_experts_down_proj_weight_8941, bias=None)
        del mul_54_33209
        nnscaler_moe_gmm_9_8934 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_9_33017, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_9_33017
        linear_76_8942 = nnscaler.runtime.adapter.nn.allgather_split(linear_76_33233, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_76_33233
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_53_8943 = torch.add(nnscaler_moe_gmm_9_8934, linear_76_8942, alpha=1)
        del nnscaler_moe_gmm_9_8934, linear_76_8942
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_54_8944 = torch.add(add_52_82776, add_53_8943, alpha=1)
        del add_52_82776, add_53_8943
        # created at IRAdapterGener:local_consumer_multiref
        add_54_82836, add_54_82840 = nnscaler.runtime.function.multiref(add_54_8944, times=2)
        del add_54_8944
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_33_8946 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_54_82836, self.model_model_layers_11_input_layernorm_weight_8945, (2048,), 1e-06)
        del add_54_82836
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_33_63891, fused_rms_norm_affine_33_63892 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_33_8946, times=2)
        del fused_rms_norm_affine_33_8946
        fused_rms_norm_affine_33_70888 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_33_63891, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_33_63891
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_77_33361 = torch.nn.functional.linear(fused_rms_norm_affine_33_70888, self.model_model_layers_11_self_attn_q_proj_weight_8947, bias=None)
        del fused_rms_norm_affine_33_70888
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_55_33417 = torch.Tensor.view(linear_77_33361, size=(8, 256, 16, 192))
        del linear_77_33361
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_88_33441 = torch.transpose(view_55_33417, dim0=1, dim1=2)
        del view_55_33417
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_33_33505, split_33_33513 = torch.functional.split(transpose_88_33441, split_size_or_sections=[128, 64], dim=-1)
        del transpose_88_33441
        fused_rms_norm_affine_33_70952 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_33_63892, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_33_63892
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_78_33545 = torch.nn.functional.linear(fused_rms_norm_affine_33_70952, self.model_model_layers_11_self_attn_kv_a_proj_with_mqa_weight_33537, bias=None)
        del fused_rms_norm_affine_33_70952
        linear_78_8954 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_78_33545, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_78_33545
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_34_8955, split_34_8956 = torch.functional.split(linear_78_8954, split_size_or_sections=[512, 64], dim=-1)
        del linear_78_8954
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_56_8957 = torch.Tensor.view(split_34_8956, size=(8, 2048, 1, 64))
        del split_34_8956
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_89_8958 = torch.transpose(view_56_8957, dim0=1, dim1=2)
        del view_56_8957
        split_34_33585 = nnscaler.runtime.adapter.nn.split_allgather(split_34_8955, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_34_8955
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_34_33665 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_34_33585, self.model_model_layers_11_self_attn_kv_a_layernorm_weight_8959, (512,), 1e-06)
        del split_34_33585
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_79_33681 = torch.nn.functional.linear(fused_rms_norm_affine_34_33665, self.model_model_layers_11_self_attn_kv_b_proj_weight_8961, bias=None)
        del fused_rms_norm_affine_34_33665
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_57_33737 = torch.Tensor.view(linear_79_33681, size=(8, 256, 16, 256))
        del linear_79_33681
        view_57_33729 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_57_33737, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_57_33737
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_90_33753 = torch.transpose(view_57_33729, dim0=1, dim1=2)
        del view_57_33729
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_35_33793, split_35_33801 = torch.functional.split(transpose_90_33753, split_size_or_sections=[128, 128], dim=-1)
        del transpose_90_33753
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_398_8968 = nnscaler.runtime.function.fullslice(self.model_model_layers_11_self_attn_rotary_emb_cos_cached_8967, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_22_8969 = torch.Tensor.to(getitem_398_8968, dtype=torch.bfloat16)
        del getitem_398_8968
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_399_8971 = nnscaler.runtime.function.fullslice(self.model_model_layers_11_self_attn_rotary_emb_sin_cached_8970, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_23_8972 = torch.Tensor.to(getitem_399_8971, dtype=torch.bfloat16)
        del getitem_399_8971
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_400_8973 = nnscaler.runtime.function.fullslice(to_22_8969, unsqueeze_8005)
        del to_22_8969
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_23_8974 = torch.unsqueeze(getitem_400_8973, dim=1)
        del getitem_400_8973
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_23_63897, unsqueeze_23_63898 = nnscaler.runtime.function.multiref(unsqueeze_23_8974, times=2)
        del unsqueeze_23_8974
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_401_8975 = nnscaler.runtime.function.fullslice(to_23_8972, unsqueeze_8005)
        del to_23_8972
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_24_8976 = torch.unsqueeze(getitem_401_8975, dim=1)
        del getitem_401_8975
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_24_63901, unsqueeze_24_63902 = nnscaler.runtime.function.multiref(unsqueeze_24_8976, times=2)
        del unsqueeze_24_8976
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_58_33889 = torch.Tensor.view(split_33_33513, size=(8, 16, 256, 32, 2))
        del split_33_33513
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_91_33929 = torch.transpose(view_58_33889, dim0=4, dim1=3)
        del view_58_33889
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_33_33961 = torch.Tensor.reshape(transpose_91_33929, shape=(8, 16, 256, 64))
        del transpose_91_33929
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_59_8980 = torch.Tensor.view(transpose_89_8958, size=(8, 1, 2048, 32, 2))
        del transpose_89_8958
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_92_8981 = torch.transpose(view_59_8980, dim0=4, dim1=3)
        del view_59_8980
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_34_8982 = torch.Tensor.reshape(transpose_92_8981, shape=(8, 1, 2048, 64))
        del transpose_92_8981
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_34_63909, reshape_34_63910, reshape_34_63911 = nnscaler.runtime.function.multiref(reshape_34_8982, times=3)
        del reshape_34_8982
        unsqueeze_23_71080 = nnscaler.runtime.adapter.chunk(unsqueeze_23_63897, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_23_63897
        # created at IRAdapterGener:local_consumer_multiref
        reshape_33_82931, reshape_33_82935, reshape_33_82939 = nnscaler.runtime.function.multiref(reshape_33_33961, times=3)
        del reshape_33_33961
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_55_34057 = torch.mul(reshape_33_82931, unsqueeze_23_71080)
        del unsqueeze_23_71080, reshape_33_82931
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_411_34105 = nnscaler.runtime.function.fullslice(reshape_33_82935, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_33_82935
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_413_34129 = nnscaler.runtime.function.fullslice(reshape_33_82939, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_33_82939
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_22_34153 = _operator.neg(getitem_413_34129)
        del getitem_413_34129
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_44_34193 = nnscaler.runtime.function.cat(neg_22_34153, getitem_411_34105, dim=-1)
        del getitem_411_34105, neg_22_34153
        unsqueeze_24_71152 = nnscaler.runtime.adapter.chunk(unsqueeze_24_63901, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_24_63901
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_56_34225 = torch.mul(cat_44_34193, unsqueeze_24_71152)
        del cat_44_34193, unsqueeze_24_71152
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_55_34273 = torch.add(mul_55_34057, mul_56_34225, alpha=1)
        del mul_55_34057, mul_56_34225
        unsqueeze_23_71184 = nnscaler.runtime.adapter.chunk(unsqueeze_23_63898, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_23_63898
        reshape_34_71176 = nnscaler.runtime.adapter.nn.split_allgather(reshape_34_63911, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_34_63911
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_57_34313 = torch.mul(reshape_34_71176, unsqueeze_23_71184)
        del unsqueeze_23_71184, reshape_34_71176
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_415_8991 = nnscaler.runtime.function.fullslice(reshape_34_63909, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_34_63909
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_417_8992 = nnscaler.runtime.function.fullslice(reshape_34_63910, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_34_63910
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_23_8993 = _operator.neg(getitem_417_8992)
        del getitem_417_8992
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_45_8994 = nnscaler.runtime.function.cat(neg_23_8993, getitem_415_8991, dim=-1)
        del getitem_415_8991, neg_23_8993
        cat_45_34417 = nnscaler.runtime.adapter.nn.split_allgather(cat_45_8994, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_45_8994
        unsqueeze_24_71208 = nnscaler.runtime.adapter.chunk(unsqueeze_24_63902, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_24_63902
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_58_34425 = torch.mul(cat_45_34417, unsqueeze_24_71208)
        del cat_45_34417, unsqueeze_24_71208
        mul_57_8990 = nnscaler.runtime.adapter.nn.allgather_split(mul_57_34313, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_57_34313
        mul_58_8995 = nnscaler.runtime.adapter.nn.allgather_split(mul_58_34425, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_58_34425
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_56_8996 = torch.add(mul_57_8990, mul_58_8995, alpha=1)
        del mul_57_8990, mul_58_8995
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_46_34473 = nnscaler.runtime.function.cat(split_33_33505, add_55_34273, dim=-1)
        del split_33_33505, add_55_34273
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_11_8998 = torch.Tensor.expand(add_56_8996, size=[-1, 16, -1, -1])
        del add_56_8996
        split_35_33809 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_35_33793, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_35_33793
        expand_11_34513 = nnscaler.runtime.adapter.nn.split_allgather(expand_11_8998, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_11_8998
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_47_34521 = nnscaler.runtime.function.cat(split_35_33809, expand_11_34513, dim=-1)
        del split_35_33809, expand_11_34513
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_11_34537 = torch.nn.functional.pad(split_35_33801, pad=[0, 64], mode='constant', value=0.0)
        del split_35_33801
        cat_46_34465 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_46_34473, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_46_34473
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_93_34569 = torch.transpose(cat_46_34465, dim0=1, dim1=2)
        del cat_46_34465
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_94_34609 = torch.transpose(cat_47_34521, dim0=1, dim1=2)
        del cat_47_34521
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_95_34641 = torch.transpose(pad_11_34537, dim0=1, dim1=2)
        del pad_11_34537
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_11_self_attn_training_7768 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_11_7769 = 0.0 if model_model_layers_11_self_attn_training_7768 else 0.0
        transpose_95_34649 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_95_34641, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_95_34641
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_11_34689 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_93_34569, transpose_94_34609, transpose_95_34649, dropout=ifexpr_11_7769, causal=True, attention_mask=None, query_length=2048)
        del transpose_93_34569, transpose_94_34609, transpose_95_34649
        nnscaler_flash_attention_forward_11_9004 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_11_34689, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_11_34689
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_418_9005 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_11_9004, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_11_9004
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_35_9006 = torch.Tensor.reshape(getitem_418_9005, shape=(8, 2048, 2048))
        del getitem_418_9005
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_11_9007 = torch.Tensor.contiguous(reshape_35_9006)
        del reshape_35_9006
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_80_9009 = torch.nn.functional.linear(contiguous_11_9007, self.model_model_layers_11_self_attn_o_proj_weight_9008, bias=None)
        del contiguous_11_9007
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_57_9010 = torch.add(add_54_82840, linear_80_9009, alpha=1)
        del add_54_82840, linear_80_9009
        # created at IRAdapterGener:local_consumer_multiref
        add_57_83004, add_57_83008 = nnscaler.runtime.function.multiref(add_57_9010, times=2)
        del add_57_9010
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_35_9012 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_57_83004, self.model_model_layers_11_post_attention_layernorm_weight_9011, (2048,), 1e-06)
        del add_57_83004
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_35_63923, fused_rms_norm_affine_35_63924, fused_rms_norm_affine_35_63925, fused_rms_norm_affine_35_63926 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_35_9012, times=4)
        del fused_rms_norm_affine_35_9012
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_11_mlp_gate_training_7771 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_10_9014, moe_route_10_9015, moe_route_10_9016 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_35_63923, self.model_model_layers_11_mlp_gate_weight_9013, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_11_mlp_gate_training_7771, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_35_63923
        moe_route_10_9015 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_10_9015, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_10_9016 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_10_9016, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_35_63924 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_35_63924, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_10_34889 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_35_63924, moe_route_10_9014, moe_route_10_9015, moe_route_10_9016, self.model_model_layers_11_mlp_gate_projs_34865, self.model_model_layers_11_mlp_up_projs_34873, self.model_model_layers_11_mlp_down_projs_34881, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_35_63924, moe_route_10_9014, moe_route_10_9015, moe_route_10_9016
        fused_rms_norm_affine_35_71368 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_35_63925, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_35_63925
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_81_34953 = torch.nn.functional.linear(fused_rms_norm_affine_35_71368, self.model_model_layers_11_mlp_shared_experts_gate_proj_weight_9021, bias=None)
        del fused_rms_norm_affine_35_71368
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_11_35009 = torch.nn.functional.silu(linear_81_34953, inplace=False)
        del linear_81_34953
        fused_rms_norm_affine_35_71408 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_35_63926, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_35_63926
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_82_35033 = torch.nn.functional.linear(fused_rms_norm_affine_35_71408, self.model_model_layers_11_mlp_shared_experts_up_proj_weight_9024, bias=None)
        del fused_rms_norm_affine_35_71408
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_59_35081 = torch.mul(silu_11_35009, linear_82_35033)
        del silu_11_35009, linear_82_35033
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_83_35105 = torch.nn.functional.linear(mul_59_35081, self.model_model_layers_11_mlp_shared_experts_down_proj_weight_9027, bias=None)
        del mul_59_35081
        nnscaler_moe_gmm_10_9020 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_10_34889, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_10_34889
        linear_83_9028 = nnscaler.runtime.adapter.nn.allgather_split(linear_83_35105, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_83_35105
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_58_9029 = torch.add(nnscaler_moe_gmm_10_9020, linear_83_9028, alpha=1)
        del nnscaler_moe_gmm_10_9020, linear_83_9028
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_59_9030 = torch.add(add_57_83008, add_58_9029, alpha=1)
        del add_57_83008, add_58_9029
        # created at IRAdapterGener:local_consumer_multiref
        add_59_83068, add_59_83072 = nnscaler.runtime.function.multiref(add_59_9030, times=2)
        del add_59_9030
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_36_9032 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_59_83068, self.model_model_layers_12_input_layernorm_weight_9031, (2048,), 1e-06)
        del add_59_83068
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_36_63935, fused_rms_norm_affine_36_63936 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_36_9032, times=2)
        del fused_rms_norm_affine_36_9032
        fused_rms_norm_affine_36_71464 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_36_63935, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_36_63935
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_84_35233 = torch.nn.functional.linear(fused_rms_norm_affine_36_71464, self.model_model_layers_12_self_attn_q_proj_weight_9033, bias=None)
        del fused_rms_norm_affine_36_71464
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_60_35289 = torch.Tensor.view(linear_84_35233, size=(8, 256, 16, 192))
        del linear_84_35233
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_96_35313 = torch.transpose(view_60_35289, dim0=1, dim1=2)
        del view_60_35289
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_36_35377, split_36_35385 = torch.functional.split(transpose_96_35313, split_size_or_sections=[128, 64], dim=-1)
        del transpose_96_35313
        fused_rms_norm_affine_36_71528 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_36_63936, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_36_63936
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_85_35417 = torch.nn.functional.linear(fused_rms_norm_affine_36_71528, self.model_model_layers_12_self_attn_kv_a_proj_with_mqa_weight_35409, bias=None)
        del fused_rms_norm_affine_36_71528
        linear_85_9040 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_85_35417, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_85_35417
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_37_9041, split_37_9042 = torch.functional.split(linear_85_9040, split_size_or_sections=[512, 64], dim=-1)
        del linear_85_9040
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_61_9043 = torch.Tensor.view(split_37_9042, size=(8, 2048, 1, 64))
        del split_37_9042
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_97_9044 = torch.transpose(view_61_9043, dim0=1, dim1=2)
        del view_61_9043
        split_37_35457 = nnscaler.runtime.adapter.nn.split_allgather(split_37_9041, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_37_9041
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_37_35537 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_37_35457, self.model_model_layers_12_self_attn_kv_a_layernorm_weight_9045, (512,), 1e-06)
        del split_37_35457
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_86_35553 = torch.nn.functional.linear(fused_rms_norm_affine_37_35537, self.model_model_layers_12_self_attn_kv_b_proj_weight_9047, bias=None)
        del fused_rms_norm_affine_37_35537
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_62_35609 = torch.Tensor.view(linear_86_35553, size=(8, 256, 16, 256))
        del linear_86_35553
        view_62_35601 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_62_35609, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_62_35609
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_98_35625 = torch.transpose(view_62_35601, dim0=1, dim1=2)
        del view_62_35601
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_38_35665, split_38_35673 = torch.functional.split(transpose_98_35625, split_size_or_sections=[128, 128], dim=-1)
        del transpose_98_35625
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_433_9054 = nnscaler.runtime.function.fullslice(self.model_model_layers_12_self_attn_rotary_emb_cos_cached_9053, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_24_9055 = torch.Tensor.to(getitem_433_9054, dtype=torch.bfloat16)
        del getitem_433_9054
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_434_9057 = nnscaler.runtime.function.fullslice(self.model_model_layers_12_self_attn_rotary_emb_sin_cached_9056, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_25_9058 = torch.Tensor.to(getitem_434_9057, dtype=torch.bfloat16)
        del getitem_434_9057
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_435_9059 = nnscaler.runtime.function.fullslice(to_24_9055, unsqueeze_8005)
        del to_24_9055
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_25_9060 = torch.unsqueeze(getitem_435_9059, dim=1)
        del getitem_435_9059
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_25_63941, unsqueeze_25_63942 = nnscaler.runtime.function.multiref(unsqueeze_25_9060, times=2)
        del unsqueeze_25_9060
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_436_9061 = nnscaler.runtime.function.fullslice(to_25_9058, unsqueeze_8005)
        del to_25_9058
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_26_9062 = torch.unsqueeze(getitem_436_9061, dim=1)
        del getitem_436_9061
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_26_63945, unsqueeze_26_63946 = nnscaler.runtime.function.multiref(unsqueeze_26_9062, times=2)
        del unsqueeze_26_9062
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_63_35761 = torch.Tensor.view(split_36_35385, size=(8, 16, 256, 32, 2))
        del split_36_35385
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_99_35801 = torch.transpose(view_63_35761, dim0=4, dim1=3)
        del view_63_35761
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_36_35833 = torch.Tensor.reshape(transpose_99_35801, shape=(8, 16, 256, 64))
        del transpose_99_35801
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_64_9066 = torch.Tensor.view(transpose_97_9044, size=(8, 1, 2048, 32, 2))
        del transpose_97_9044
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_100_9067 = torch.transpose(view_64_9066, dim0=4, dim1=3)
        del view_64_9066
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_37_9068 = torch.Tensor.reshape(transpose_100_9067, shape=(8, 1, 2048, 64))
        del transpose_100_9067
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_37_63953, reshape_37_63954, reshape_37_63955 = nnscaler.runtime.function.multiref(reshape_37_9068, times=3)
        del reshape_37_9068
        unsqueeze_25_71656 = nnscaler.runtime.adapter.chunk(unsqueeze_25_63941, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_25_63941
        # created at IRAdapterGener:local_consumer_multiref
        reshape_36_83163, reshape_36_83167, reshape_36_83171 = nnscaler.runtime.function.multiref(reshape_36_35833, times=3)
        del reshape_36_35833
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_60_35929 = torch.mul(reshape_36_83163, unsqueeze_25_71656)
        del unsqueeze_25_71656, reshape_36_83163
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_446_35977 = nnscaler.runtime.function.fullslice(reshape_36_83167, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_36_83167
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_448_36001 = nnscaler.runtime.function.fullslice(reshape_36_83171, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_36_83171
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_24_36025 = _operator.neg(getitem_448_36001)
        del getitem_448_36001
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_48_36065 = nnscaler.runtime.function.cat(neg_24_36025, getitem_446_35977, dim=-1)
        del getitem_446_35977, neg_24_36025
        unsqueeze_26_71728 = nnscaler.runtime.adapter.chunk(unsqueeze_26_63945, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_26_63945
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_61_36097 = torch.mul(cat_48_36065, unsqueeze_26_71728)
        del cat_48_36065, unsqueeze_26_71728
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_60_36145 = torch.add(mul_60_35929, mul_61_36097, alpha=1)
        del mul_60_35929, mul_61_36097
        unsqueeze_25_71760 = nnscaler.runtime.adapter.chunk(unsqueeze_25_63942, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_25_63942
        reshape_37_71752 = nnscaler.runtime.adapter.nn.split_allgather(reshape_37_63955, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_37_63955
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_62_36185 = torch.mul(reshape_37_71752, unsqueeze_25_71760)
        del unsqueeze_25_71760, reshape_37_71752
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_450_9077 = nnscaler.runtime.function.fullslice(reshape_37_63953, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_37_63953
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_452_9078 = nnscaler.runtime.function.fullslice(reshape_37_63954, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_37_63954
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_25_9079 = _operator.neg(getitem_452_9078)
        del getitem_452_9078
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_49_9080 = nnscaler.runtime.function.cat(neg_25_9079, getitem_450_9077, dim=-1)
        del getitem_450_9077, neg_25_9079
        cat_49_36289 = nnscaler.runtime.adapter.nn.split_allgather(cat_49_9080, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_49_9080
        unsqueeze_26_71784 = nnscaler.runtime.adapter.chunk(unsqueeze_26_63946, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_26_63946
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_63_36297 = torch.mul(cat_49_36289, unsqueeze_26_71784)
        del cat_49_36289, unsqueeze_26_71784
        mul_62_9076 = nnscaler.runtime.adapter.nn.allgather_split(mul_62_36185, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_62_36185
        mul_63_9081 = nnscaler.runtime.adapter.nn.allgather_split(mul_63_36297, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_63_36297
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_61_9082 = torch.add(mul_62_9076, mul_63_9081, alpha=1)
        del mul_62_9076, mul_63_9081
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_50_36345 = nnscaler.runtime.function.cat(split_36_35377, add_60_36145, dim=-1)
        del split_36_35377, add_60_36145
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_12_9084 = torch.Tensor.expand(add_61_9082, size=[-1, 16, -1, -1])
        del add_61_9082
        split_38_35681 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_38_35665, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_38_35665
        expand_12_36385 = nnscaler.runtime.adapter.nn.split_allgather(expand_12_9084, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_12_9084
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_51_36393 = nnscaler.runtime.function.cat(split_38_35681, expand_12_36385, dim=-1)
        del split_38_35681, expand_12_36385
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_12_36409 = torch.nn.functional.pad(split_38_35673, pad=[0, 64], mode='constant', value=0.0)
        del split_38_35673
        cat_50_36337 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_50_36345, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_50_36345
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_101_36441 = torch.transpose(cat_50_36337, dim0=1, dim1=2)
        del cat_50_36337
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_102_36481 = torch.transpose(cat_51_36393, dim0=1, dim1=2)
        del cat_51_36393
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_103_36513 = torch.transpose(pad_12_36409, dim0=1, dim1=2)
        del pad_12_36409
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_12_self_attn_training_7783 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_12_7784 = 0.0 if model_model_layers_12_self_attn_training_7783 else 0.0
        transpose_103_36521 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_103_36513, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_103_36513
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_12_36561 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_101_36441, transpose_102_36481, transpose_103_36521, dropout=ifexpr_12_7784, causal=True, attention_mask=None, query_length=2048)
        del transpose_101_36441, transpose_102_36481, transpose_103_36521
        nnscaler_flash_attention_forward_12_9090 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_12_36561, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_12_36561
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_453_9091 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_12_9090, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_12_9090
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_38_9092 = torch.Tensor.reshape(getitem_453_9091, shape=(8, 2048, 2048))
        del getitem_453_9091
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_12_9093 = torch.Tensor.contiguous(reshape_38_9092)
        del reshape_38_9092
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_87_9095 = torch.nn.functional.linear(contiguous_12_9093, self.model_model_layers_12_self_attn_o_proj_weight_9094, bias=None)
        del contiguous_12_9093
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_62_9096 = torch.add(add_59_83072, linear_87_9095, alpha=1)
        del add_59_83072, linear_87_9095
        # created at IRAdapterGener:local_consumer_multiref
        add_62_83236, add_62_83240 = nnscaler.runtime.function.multiref(add_62_9096, times=2)
        del add_62_9096
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_38_9098 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_62_83236, self.model_model_layers_12_post_attention_layernorm_weight_9097, (2048,), 1e-06)
        del add_62_83236
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_38_63967, fused_rms_norm_affine_38_63968, fused_rms_norm_affine_38_63969, fused_rms_norm_affine_38_63970 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_38_9098, times=4)
        del fused_rms_norm_affine_38_9098
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_12_mlp_gate_training_7786 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_11_9100, moe_route_11_9101, moe_route_11_9102 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_38_63967, self.model_model_layers_12_mlp_gate_weight_9099, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_12_mlp_gate_training_7786, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_38_63967
        moe_route_11_9101 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_11_9101, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_11_9102 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_11_9102, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_38_63968 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_38_63968, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_11_36761 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_38_63968, moe_route_11_9100, moe_route_11_9101, moe_route_11_9102, self.model_model_layers_12_mlp_gate_projs_36737, self.model_model_layers_12_mlp_up_projs_36745, self.model_model_layers_12_mlp_down_projs_36753, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_38_63968, moe_route_11_9100, moe_route_11_9101, moe_route_11_9102
        fused_rms_norm_affine_38_71944 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_38_63969, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_38_63969
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_88_36825 = torch.nn.functional.linear(fused_rms_norm_affine_38_71944, self.model_model_layers_12_mlp_shared_experts_gate_proj_weight_9107, bias=None)
        del fused_rms_norm_affine_38_71944
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_12_36881 = torch.nn.functional.silu(linear_88_36825, inplace=False)
        del linear_88_36825
        fused_rms_norm_affine_38_71984 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_38_63970, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_38_63970
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_89_36905 = torch.nn.functional.linear(fused_rms_norm_affine_38_71984, self.model_model_layers_12_mlp_shared_experts_up_proj_weight_9110, bias=None)
        del fused_rms_norm_affine_38_71984
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_64_36953 = torch.mul(silu_12_36881, linear_89_36905)
        del silu_12_36881, linear_89_36905
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_90_36977 = torch.nn.functional.linear(mul_64_36953, self.model_model_layers_12_mlp_shared_experts_down_proj_weight_9113, bias=None)
        del mul_64_36953
        nnscaler_moe_gmm_11_9106 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_11_36761, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_11_36761
        linear_90_9114 = nnscaler.runtime.adapter.nn.allgather_split(linear_90_36977, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_90_36977
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_63_9115 = torch.add(nnscaler_moe_gmm_11_9106, linear_90_9114, alpha=1)
        del nnscaler_moe_gmm_11_9106, linear_90_9114
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_64_9116 = torch.add(add_62_83240, add_63_9115, alpha=1)
        del add_62_83240, add_63_9115
        # created at IRAdapterGener:local_consumer_multiref
        add_64_83300, add_64_83304 = nnscaler.runtime.function.multiref(add_64_9116, times=2)
        del add_64_9116
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_39_9118 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_64_83300, self.model_model_layers_13_input_layernorm_weight_9117, (2048,), 1e-06)
        del add_64_83300
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_39_63979, fused_rms_norm_affine_39_63980 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_39_9118, times=2)
        del fused_rms_norm_affine_39_9118
        fused_rms_norm_affine_39_72040 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_39_63979, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_39_63979
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_91_37105 = torch.nn.functional.linear(fused_rms_norm_affine_39_72040, self.model_model_layers_13_self_attn_q_proj_weight_9119, bias=None)
        del fused_rms_norm_affine_39_72040
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_65_37161 = torch.Tensor.view(linear_91_37105, size=(8, 256, 16, 192))
        del linear_91_37105
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_104_37185 = torch.transpose(view_65_37161, dim0=1, dim1=2)
        del view_65_37161
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_39_37249, split_39_37257 = torch.functional.split(transpose_104_37185, split_size_or_sections=[128, 64], dim=-1)
        del transpose_104_37185
        fused_rms_norm_affine_39_72104 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_39_63980, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_39_63980
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_92_37289 = torch.nn.functional.linear(fused_rms_norm_affine_39_72104, self.model_model_layers_13_self_attn_kv_a_proj_with_mqa_weight_37281, bias=None)
        del fused_rms_norm_affine_39_72104
        linear_92_9126 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_92_37289, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_92_37289
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_40_9127, split_40_9128 = torch.functional.split(linear_92_9126, split_size_or_sections=[512, 64], dim=-1)
        del linear_92_9126
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_66_9129 = torch.Tensor.view(split_40_9128, size=(8, 2048, 1, 64))
        del split_40_9128
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_105_9130 = torch.transpose(view_66_9129, dim0=1, dim1=2)
        del view_66_9129
        split_40_37329 = nnscaler.runtime.adapter.nn.split_allgather(split_40_9127, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_40_9127
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_40_37409 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_40_37329, self.model_model_layers_13_self_attn_kv_a_layernorm_weight_9131, (512,), 1e-06)
        del split_40_37329
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_93_37425 = torch.nn.functional.linear(fused_rms_norm_affine_40_37409, self.model_model_layers_13_self_attn_kv_b_proj_weight_9133, bias=None)
        del fused_rms_norm_affine_40_37409
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_67_37481 = torch.Tensor.view(linear_93_37425, size=(8, 256, 16, 256))
        del linear_93_37425
        view_67_37473 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_67_37481, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_67_37481
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_106_37497 = torch.transpose(view_67_37473, dim0=1, dim1=2)
        del view_67_37473
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_41_37537, split_41_37545 = torch.functional.split(transpose_106_37497, split_size_or_sections=[128, 128], dim=-1)
        del transpose_106_37497
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_468_9140 = nnscaler.runtime.function.fullslice(self.model_model_layers_13_self_attn_rotary_emb_cos_cached_9139, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_26_9141 = torch.Tensor.to(getitem_468_9140, dtype=torch.bfloat16)
        del getitem_468_9140
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_469_9143 = nnscaler.runtime.function.fullslice(self.model_model_layers_13_self_attn_rotary_emb_sin_cached_9142, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_27_9144 = torch.Tensor.to(getitem_469_9143, dtype=torch.bfloat16)
        del getitem_469_9143
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_470_9145 = nnscaler.runtime.function.fullslice(to_26_9141, unsqueeze_8005)
        del to_26_9141
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_27_9146 = torch.unsqueeze(getitem_470_9145, dim=1)
        del getitem_470_9145
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_27_63985, unsqueeze_27_63986 = nnscaler.runtime.function.multiref(unsqueeze_27_9146, times=2)
        del unsqueeze_27_9146
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_471_9147 = nnscaler.runtime.function.fullslice(to_27_9144, unsqueeze_8005)
        del to_27_9144
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_28_9148 = torch.unsqueeze(getitem_471_9147, dim=1)
        del getitem_471_9147
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_28_63989, unsqueeze_28_63990 = nnscaler.runtime.function.multiref(unsqueeze_28_9148, times=2)
        del unsqueeze_28_9148
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_68_37633 = torch.Tensor.view(split_39_37257, size=(8, 16, 256, 32, 2))
        del split_39_37257
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_107_37673 = torch.transpose(view_68_37633, dim0=4, dim1=3)
        del view_68_37633
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_39_37705 = torch.Tensor.reshape(transpose_107_37673, shape=(8, 16, 256, 64))
        del transpose_107_37673
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_69_9152 = torch.Tensor.view(transpose_105_9130, size=(8, 1, 2048, 32, 2))
        del transpose_105_9130
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_108_9153 = torch.transpose(view_69_9152, dim0=4, dim1=3)
        del view_69_9152
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_40_9154 = torch.Tensor.reshape(transpose_108_9153, shape=(8, 1, 2048, 64))
        del transpose_108_9153
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_40_63997, reshape_40_63998, reshape_40_63999 = nnscaler.runtime.function.multiref(reshape_40_9154, times=3)
        del reshape_40_9154
        unsqueeze_27_72232 = nnscaler.runtime.adapter.chunk(unsqueeze_27_63985, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_27_63985
        # created at IRAdapterGener:local_consumer_multiref
        reshape_39_83395, reshape_39_83399, reshape_39_83403 = nnscaler.runtime.function.multiref(reshape_39_37705, times=3)
        del reshape_39_37705
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_65_37801 = torch.mul(reshape_39_83395, unsqueeze_27_72232)
        del unsqueeze_27_72232, reshape_39_83395
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_481_37849 = nnscaler.runtime.function.fullslice(reshape_39_83399, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_39_83399
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_483_37873 = nnscaler.runtime.function.fullslice(reshape_39_83403, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_39_83403
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_26_37897 = _operator.neg(getitem_483_37873)
        del getitem_483_37873
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_52_37937 = nnscaler.runtime.function.cat(neg_26_37897, getitem_481_37849, dim=-1)
        del getitem_481_37849, neg_26_37897
        unsqueeze_28_72304 = nnscaler.runtime.adapter.chunk(unsqueeze_28_63989, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_28_63989
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_66_37969 = torch.mul(cat_52_37937, unsqueeze_28_72304)
        del cat_52_37937, unsqueeze_28_72304
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_65_38017 = torch.add(mul_65_37801, mul_66_37969, alpha=1)
        del mul_65_37801, mul_66_37969
        unsqueeze_27_72336 = nnscaler.runtime.adapter.chunk(unsqueeze_27_63986, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_27_63986
        reshape_40_72328 = nnscaler.runtime.adapter.nn.split_allgather(reshape_40_63999, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_40_63999
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_67_38057 = torch.mul(reshape_40_72328, unsqueeze_27_72336)
        del unsqueeze_27_72336, reshape_40_72328
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_485_9163 = nnscaler.runtime.function.fullslice(reshape_40_63997, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_40_63997
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_487_9164 = nnscaler.runtime.function.fullslice(reshape_40_63998, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_40_63998
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_27_9165 = _operator.neg(getitem_487_9164)
        del getitem_487_9164
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_53_9166 = nnscaler.runtime.function.cat(neg_27_9165, getitem_485_9163, dim=-1)
        del getitem_485_9163, neg_27_9165
        cat_53_38161 = nnscaler.runtime.adapter.nn.split_allgather(cat_53_9166, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_53_9166
        unsqueeze_28_72360 = nnscaler.runtime.adapter.chunk(unsqueeze_28_63990, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_28_63990
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_68_38169 = torch.mul(cat_53_38161, unsqueeze_28_72360)
        del cat_53_38161, unsqueeze_28_72360
        mul_67_9162 = nnscaler.runtime.adapter.nn.allgather_split(mul_67_38057, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_67_38057
        mul_68_9167 = nnscaler.runtime.adapter.nn.allgather_split(mul_68_38169, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_68_38169
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_66_9168 = torch.add(mul_67_9162, mul_68_9167, alpha=1)
        del mul_67_9162, mul_68_9167
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_54_38217 = nnscaler.runtime.function.cat(split_39_37249, add_65_38017, dim=-1)
        del split_39_37249, add_65_38017
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_13_9170 = torch.Tensor.expand(add_66_9168, size=[-1, 16, -1, -1])
        del add_66_9168
        split_41_37553 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_41_37537, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_41_37537
        expand_13_38257 = nnscaler.runtime.adapter.nn.split_allgather(expand_13_9170, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_13_9170
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_55_38265 = nnscaler.runtime.function.cat(split_41_37553, expand_13_38257, dim=-1)
        del split_41_37553, expand_13_38257
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_13_38281 = torch.nn.functional.pad(split_41_37545, pad=[0, 64], mode='constant', value=0.0)
        del split_41_37545
        cat_54_38209 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_54_38217, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_54_38217
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_109_38313 = torch.transpose(cat_54_38209, dim0=1, dim1=2)
        del cat_54_38209
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_110_38353 = torch.transpose(cat_55_38265, dim0=1, dim1=2)
        del cat_55_38265
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_111_38385 = torch.transpose(pad_13_38281, dim0=1, dim1=2)
        del pad_13_38281
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_13_self_attn_training_7798 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_13_7799 = 0.0 if model_model_layers_13_self_attn_training_7798 else 0.0
        transpose_111_38393 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_111_38385, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_111_38385
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_13_38433 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_109_38313, transpose_110_38353, transpose_111_38393, dropout=ifexpr_13_7799, causal=True, attention_mask=None, query_length=2048)
        del transpose_109_38313, transpose_110_38353, transpose_111_38393
        nnscaler_flash_attention_forward_13_9176 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_13_38433, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_13_38433
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_488_9177 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_13_9176, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_13_9176
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_41_9178 = torch.Tensor.reshape(getitem_488_9177, shape=(8, 2048, 2048))
        del getitem_488_9177
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_13_9179 = torch.Tensor.contiguous(reshape_41_9178)
        del reshape_41_9178
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_94_9181 = torch.nn.functional.linear(contiguous_13_9179, self.model_model_layers_13_self_attn_o_proj_weight_9180, bias=None)
        del contiguous_13_9179
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_67_9182 = torch.add(add_64_83304, linear_94_9181, alpha=1)
        del add_64_83304, linear_94_9181
        # created at IRAdapterGener:local_consumer_multiref
        add_67_83468, add_67_83472 = nnscaler.runtime.function.multiref(add_67_9182, times=2)
        del add_67_9182
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_41_9184 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_67_83468, self.model_model_layers_13_post_attention_layernorm_weight_9183, (2048,), 1e-06)
        del add_67_83468
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_41_64011, fused_rms_norm_affine_41_64012, fused_rms_norm_affine_41_64013, fused_rms_norm_affine_41_64014 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_41_9184, times=4)
        del fused_rms_norm_affine_41_9184
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_13_mlp_gate_training_7801 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_12_9186, moe_route_12_9187, moe_route_12_9188 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_41_64011, self.model_model_layers_13_mlp_gate_weight_9185, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_13_mlp_gate_training_7801, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_41_64011
        moe_route_12_9187 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_12_9187, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_12_9188 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_12_9188, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_41_64012 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_41_64012, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_12_38633 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_41_64012, moe_route_12_9186, moe_route_12_9187, moe_route_12_9188, self.model_model_layers_13_mlp_gate_projs_38609, self.model_model_layers_13_mlp_up_projs_38617, self.model_model_layers_13_mlp_down_projs_38625, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_41_64012, moe_route_12_9186, moe_route_12_9187, moe_route_12_9188
        fused_rms_norm_affine_41_72520 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_41_64013, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_41_64013
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_95_38697 = torch.nn.functional.linear(fused_rms_norm_affine_41_72520, self.model_model_layers_13_mlp_shared_experts_gate_proj_weight_9193, bias=None)
        del fused_rms_norm_affine_41_72520
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_13_38753 = torch.nn.functional.silu(linear_95_38697, inplace=False)
        del linear_95_38697
        fused_rms_norm_affine_41_72560 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_41_64014, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_41_64014
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_96_38777 = torch.nn.functional.linear(fused_rms_norm_affine_41_72560, self.model_model_layers_13_mlp_shared_experts_up_proj_weight_9196, bias=None)
        del fused_rms_norm_affine_41_72560
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_69_38825 = torch.mul(silu_13_38753, linear_96_38777)
        del silu_13_38753, linear_96_38777
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_97_38849 = torch.nn.functional.linear(mul_69_38825, self.model_model_layers_13_mlp_shared_experts_down_proj_weight_9199, bias=None)
        del mul_69_38825
        nnscaler_moe_gmm_12_9192 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_12_38633, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_12_38633
        linear_97_9200 = nnscaler.runtime.adapter.nn.allgather_split(linear_97_38849, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_97_38849
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_68_9201 = torch.add(nnscaler_moe_gmm_12_9192, linear_97_9200, alpha=1)
        del nnscaler_moe_gmm_12_9192, linear_97_9200
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_69_9202 = torch.add(add_67_83472, add_68_9201, alpha=1)
        del add_67_83472, add_68_9201
        # created at IRAdapterGener:local_consumer_multiref
        add_69_83532, add_69_83536 = nnscaler.runtime.function.multiref(add_69_9202, times=2)
        del add_69_9202
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_42_9204 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_69_83532, self.model_model_layers_14_input_layernorm_weight_9203, (2048,), 1e-06)
        del add_69_83532
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_42_64023, fused_rms_norm_affine_42_64024 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_42_9204, times=2)
        del fused_rms_norm_affine_42_9204
        fused_rms_norm_affine_42_72616 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_42_64023, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_42_64023
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_98_38977 = torch.nn.functional.linear(fused_rms_norm_affine_42_72616, self.model_model_layers_14_self_attn_q_proj_weight_9205, bias=None)
        del fused_rms_norm_affine_42_72616
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_70_39033 = torch.Tensor.view(linear_98_38977, size=(8, 256, 16, 192))
        del linear_98_38977
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_112_39057 = torch.transpose(view_70_39033, dim0=1, dim1=2)
        del view_70_39033
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_42_39121, split_42_39129 = torch.functional.split(transpose_112_39057, split_size_or_sections=[128, 64], dim=-1)
        del transpose_112_39057
        fused_rms_norm_affine_42_72680 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_42_64024, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_42_64024
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_99_39161 = torch.nn.functional.linear(fused_rms_norm_affine_42_72680, self.model_model_layers_14_self_attn_kv_a_proj_with_mqa_weight_39153, bias=None)
        del fused_rms_norm_affine_42_72680
        linear_99_9212 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_99_39161, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_99_39161
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_43_9213, split_43_9214 = torch.functional.split(linear_99_9212, split_size_or_sections=[512, 64], dim=-1)
        del linear_99_9212
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_71_9215 = torch.Tensor.view(split_43_9214, size=(8, 2048, 1, 64))
        del split_43_9214
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_113_9216 = torch.transpose(view_71_9215, dim0=1, dim1=2)
        del view_71_9215
        split_43_39201 = nnscaler.runtime.adapter.nn.split_allgather(split_43_9213, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_43_9213
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_43_39281 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_43_39201, self.model_model_layers_14_self_attn_kv_a_layernorm_weight_9217, (512,), 1e-06)
        del split_43_39201
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_100_39297 = torch.nn.functional.linear(fused_rms_norm_affine_43_39281, self.model_model_layers_14_self_attn_kv_b_proj_weight_9219, bias=None)
        del fused_rms_norm_affine_43_39281
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_72_39353 = torch.Tensor.view(linear_100_39297, size=(8, 256, 16, 256))
        del linear_100_39297
        view_72_39345 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_72_39353, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_72_39353
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_114_39369 = torch.transpose(view_72_39345, dim0=1, dim1=2)
        del view_72_39345
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_44_39409, split_44_39417 = torch.functional.split(transpose_114_39369, split_size_or_sections=[128, 128], dim=-1)
        del transpose_114_39369
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_503_9226 = nnscaler.runtime.function.fullslice(self.model_model_layers_14_self_attn_rotary_emb_cos_cached_9225, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_28_9227 = torch.Tensor.to(getitem_503_9226, dtype=torch.bfloat16)
        del getitem_503_9226
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_504_9229 = nnscaler.runtime.function.fullslice(self.model_model_layers_14_self_attn_rotary_emb_sin_cached_9228, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_29_9230 = torch.Tensor.to(getitem_504_9229, dtype=torch.bfloat16)
        del getitem_504_9229
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_505_9231 = nnscaler.runtime.function.fullslice(to_28_9227, unsqueeze_8005)
        del to_28_9227
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_29_9232 = torch.unsqueeze(getitem_505_9231, dim=1)
        del getitem_505_9231
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_29_64029, unsqueeze_29_64030 = nnscaler.runtime.function.multiref(unsqueeze_29_9232, times=2)
        del unsqueeze_29_9232
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_506_9233 = nnscaler.runtime.function.fullslice(to_29_9230, unsqueeze_8005)
        del to_29_9230
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_30_9234 = torch.unsqueeze(getitem_506_9233, dim=1)
        del getitem_506_9233
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_30_64033, unsqueeze_30_64034 = nnscaler.runtime.function.multiref(unsqueeze_30_9234, times=2)
        del unsqueeze_30_9234
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_73_39505 = torch.Tensor.view(split_42_39129, size=(8, 16, 256, 32, 2))
        del split_42_39129
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_115_39545 = torch.transpose(view_73_39505, dim0=4, dim1=3)
        del view_73_39505
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_42_39577 = torch.Tensor.reshape(transpose_115_39545, shape=(8, 16, 256, 64))
        del transpose_115_39545
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_74_9238 = torch.Tensor.view(transpose_113_9216, size=(8, 1, 2048, 32, 2))
        del transpose_113_9216
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_116_9239 = torch.transpose(view_74_9238, dim0=4, dim1=3)
        del view_74_9238
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_43_9240 = torch.Tensor.reshape(transpose_116_9239, shape=(8, 1, 2048, 64))
        del transpose_116_9239
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_43_64041, reshape_43_64042, reshape_43_64043 = nnscaler.runtime.function.multiref(reshape_43_9240, times=3)
        del reshape_43_9240
        unsqueeze_29_72808 = nnscaler.runtime.adapter.chunk(unsqueeze_29_64029, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_29_64029
        # created at IRAdapterGener:local_consumer_multiref
        reshape_42_83627, reshape_42_83631, reshape_42_83635 = nnscaler.runtime.function.multiref(reshape_42_39577, times=3)
        del reshape_42_39577
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_70_39673 = torch.mul(reshape_42_83627, unsqueeze_29_72808)
        del unsqueeze_29_72808, reshape_42_83627
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_516_39721 = nnscaler.runtime.function.fullslice(reshape_42_83631, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_42_83631
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_518_39745 = nnscaler.runtime.function.fullslice(reshape_42_83635, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_42_83635
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_28_39769 = _operator.neg(getitem_518_39745)
        del getitem_518_39745
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_56_39809 = nnscaler.runtime.function.cat(neg_28_39769, getitem_516_39721, dim=-1)
        del getitem_516_39721, neg_28_39769
        unsqueeze_30_72880 = nnscaler.runtime.adapter.chunk(unsqueeze_30_64033, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_30_64033
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_71_39841 = torch.mul(cat_56_39809, unsqueeze_30_72880)
        del cat_56_39809, unsqueeze_30_72880
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_70_39889 = torch.add(mul_70_39673, mul_71_39841, alpha=1)
        del mul_70_39673, mul_71_39841
        unsqueeze_29_72912 = nnscaler.runtime.adapter.chunk(unsqueeze_29_64030, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_29_64030
        reshape_43_72904 = nnscaler.runtime.adapter.nn.split_allgather(reshape_43_64043, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_43_64043
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_72_39929 = torch.mul(reshape_43_72904, unsqueeze_29_72912)
        del unsqueeze_29_72912, reshape_43_72904
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_520_9249 = nnscaler.runtime.function.fullslice(reshape_43_64041, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_43_64041
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_522_9250 = nnscaler.runtime.function.fullslice(reshape_43_64042, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_43_64042
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_29_9251 = _operator.neg(getitem_522_9250)
        del getitem_522_9250
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_57_9252 = nnscaler.runtime.function.cat(neg_29_9251, getitem_520_9249, dim=-1)
        del getitem_520_9249, neg_29_9251
        cat_57_40033 = nnscaler.runtime.adapter.nn.split_allgather(cat_57_9252, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_57_9252
        unsqueeze_30_72936 = nnscaler.runtime.adapter.chunk(unsqueeze_30_64034, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_30_64034
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_73_40041 = torch.mul(cat_57_40033, unsqueeze_30_72936)
        del cat_57_40033, unsqueeze_30_72936
        mul_72_9248 = nnscaler.runtime.adapter.nn.allgather_split(mul_72_39929, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_72_39929
        mul_73_9253 = nnscaler.runtime.adapter.nn.allgather_split(mul_73_40041, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_73_40041
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_71_9254 = torch.add(mul_72_9248, mul_73_9253, alpha=1)
        del mul_72_9248, mul_73_9253
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_58_40089 = nnscaler.runtime.function.cat(split_42_39121, add_70_39889, dim=-1)
        del split_42_39121, add_70_39889
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_14_9256 = torch.Tensor.expand(add_71_9254, size=[-1, 16, -1, -1])
        del add_71_9254
        split_44_39425 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_44_39409, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_44_39409
        expand_14_40129 = nnscaler.runtime.adapter.nn.split_allgather(expand_14_9256, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_14_9256
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_59_40137 = nnscaler.runtime.function.cat(split_44_39425, expand_14_40129, dim=-1)
        del split_44_39425, expand_14_40129
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_14_40153 = torch.nn.functional.pad(split_44_39417, pad=[0, 64], mode='constant', value=0.0)
        del split_44_39417
        cat_58_40081 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_58_40089, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_58_40089
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_117_40185 = torch.transpose(cat_58_40081, dim0=1, dim1=2)
        del cat_58_40081
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_118_40225 = torch.transpose(cat_59_40137, dim0=1, dim1=2)
        del cat_59_40137
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_119_40257 = torch.transpose(pad_14_40153, dim0=1, dim1=2)
        del pad_14_40153
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_14_self_attn_training_7813 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_14_7814 = 0.0 if model_model_layers_14_self_attn_training_7813 else 0.0
        transpose_119_40265 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_119_40257, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_119_40257
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_14_40305 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_117_40185, transpose_118_40225, transpose_119_40265, dropout=ifexpr_14_7814, causal=True, attention_mask=None, query_length=2048)
        del transpose_117_40185, transpose_118_40225, transpose_119_40265
        nnscaler_flash_attention_forward_14_9262 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_14_40305, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_14_40305
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_523_9263 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_14_9262, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_14_9262
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_44_9264 = torch.Tensor.reshape(getitem_523_9263, shape=(8, 2048, 2048))
        del getitem_523_9263
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_14_9265 = torch.Tensor.contiguous(reshape_44_9264)
        del reshape_44_9264
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_101_9267 = torch.nn.functional.linear(contiguous_14_9265, self.model_model_layers_14_self_attn_o_proj_weight_9266, bias=None)
        del contiguous_14_9265
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_72_9268 = torch.add(add_69_83536, linear_101_9267, alpha=1)
        del add_69_83536, linear_101_9267
        # created at IRAdapterGener:local_consumer_multiref
        add_72_83700, add_72_83704 = nnscaler.runtime.function.multiref(add_72_9268, times=2)
        del add_72_9268
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_44_9270 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_72_83700, self.model_model_layers_14_post_attention_layernorm_weight_9269, (2048,), 1e-06)
        del add_72_83700
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_44_64055, fused_rms_norm_affine_44_64056, fused_rms_norm_affine_44_64057, fused_rms_norm_affine_44_64058 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_44_9270, times=4)
        del fused_rms_norm_affine_44_9270
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_14_mlp_gate_training_7816 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_13_9272, moe_route_13_9273, moe_route_13_9274 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_44_64055, self.model_model_layers_14_mlp_gate_weight_9271, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_14_mlp_gate_training_7816, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_44_64055
        moe_route_13_9273 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_13_9273, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_13_9274 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_13_9274, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_44_64056 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_44_64056, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_13_40505 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_44_64056, moe_route_13_9272, moe_route_13_9273, moe_route_13_9274, self.model_model_layers_14_mlp_gate_projs_40481, self.model_model_layers_14_mlp_up_projs_40489, self.model_model_layers_14_mlp_down_projs_40497, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_44_64056, moe_route_13_9272, moe_route_13_9273, moe_route_13_9274
        fused_rms_norm_affine_44_73096 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_44_64057, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_44_64057
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_102_40569 = torch.nn.functional.linear(fused_rms_norm_affine_44_73096, self.model_model_layers_14_mlp_shared_experts_gate_proj_weight_9279, bias=None)
        del fused_rms_norm_affine_44_73096
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_14_40625 = torch.nn.functional.silu(linear_102_40569, inplace=False)
        del linear_102_40569
        fused_rms_norm_affine_44_73136 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_44_64058, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_44_64058
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_103_40649 = torch.nn.functional.linear(fused_rms_norm_affine_44_73136, self.model_model_layers_14_mlp_shared_experts_up_proj_weight_9282, bias=None)
        del fused_rms_norm_affine_44_73136
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_74_40697 = torch.mul(silu_14_40625, linear_103_40649)
        del silu_14_40625, linear_103_40649
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_104_40721 = torch.nn.functional.linear(mul_74_40697, self.model_model_layers_14_mlp_shared_experts_down_proj_weight_9285, bias=None)
        del mul_74_40697
        nnscaler_moe_gmm_13_9278 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_13_40505, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_13_40505
        linear_104_9286 = nnscaler.runtime.adapter.nn.allgather_split(linear_104_40721, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_104_40721
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_73_9287 = torch.add(nnscaler_moe_gmm_13_9278, linear_104_9286, alpha=1)
        del nnscaler_moe_gmm_13_9278, linear_104_9286
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_74_9288 = torch.add(add_72_83704, add_73_9287, alpha=1)
        del add_72_83704, add_73_9287
        # created at IRAdapterGener:local_consumer_multiref
        add_74_83764, add_74_83768 = nnscaler.runtime.function.multiref(add_74_9288, times=2)
        del add_74_9288
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_45_9290 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_74_83764, self.model_model_layers_15_input_layernorm_weight_9289, (2048,), 1e-06)
        del add_74_83764
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_45_64067, fused_rms_norm_affine_45_64068 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_45_9290, times=2)
        del fused_rms_norm_affine_45_9290
        fused_rms_norm_affine_45_73192 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_45_64067, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_45_64067
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_105_40849 = torch.nn.functional.linear(fused_rms_norm_affine_45_73192, self.model_model_layers_15_self_attn_q_proj_weight_9291, bias=None)
        del fused_rms_norm_affine_45_73192
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_75_40905 = torch.Tensor.view(linear_105_40849, size=(8, 256, 16, 192))
        del linear_105_40849
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_120_40929 = torch.transpose(view_75_40905, dim0=1, dim1=2)
        del view_75_40905
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_45_40993, split_45_41001 = torch.functional.split(transpose_120_40929, split_size_or_sections=[128, 64], dim=-1)
        del transpose_120_40929
        fused_rms_norm_affine_45_73256 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_45_64068, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_45_64068
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_106_41033 = torch.nn.functional.linear(fused_rms_norm_affine_45_73256, self.model_model_layers_15_self_attn_kv_a_proj_with_mqa_weight_41025, bias=None)
        del fused_rms_norm_affine_45_73256
        linear_106_9298 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_106_41033, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_106_41033
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_46_9299, split_46_9300 = torch.functional.split(linear_106_9298, split_size_or_sections=[512, 64], dim=-1)
        del linear_106_9298
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_76_9301 = torch.Tensor.view(split_46_9300, size=(8, 2048, 1, 64))
        del split_46_9300
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_121_9302 = torch.transpose(view_76_9301, dim0=1, dim1=2)
        del view_76_9301
        split_46_41073 = nnscaler.runtime.adapter.nn.split_allgather(split_46_9299, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_46_9299
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_46_41153 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_46_41073, self.model_model_layers_15_self_attn_kv_a_layernorm_weight_9303, (512,), 1e-06)
        del split_46_41073
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_107_41169 = torch.nn.functional.linear(fused_rms_norm_affine_46_41153, self.model_model_layers_15_self_attn_kv_b_proj_weight_9305, bias=None)
        del fused_rms_norm_affine_46_41153
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_77_41225 = torch.Tensor.view(linear_107_41169, size=(8, 256, 16, 256))
        del linear_107_41169
        view_77_41217 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_77_41225, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_77_41225
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_122_41241 = torch.transpose(view_77_41217, dim0=1, dim1=2)
        del view_77_41217
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_47_41281, split_47_41289 = torch.functional.split(transpose_122_41241, split_size_or_sections=[128, 128], dim=-1)
        del transpose_122_41241
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_538_9312 = nnscaler.runtime.function.fullslice(self.model_model_layers_15_self_attn_rotary_emb_cos_cached_9311, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_30_9313 = torch.Tensor.to(getitem_538_9312, dtype=torch.bfloat16)
        del getitem_538_9312
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_539_9315 = nnscaler.runtime.function.fullslice(self.model_model_layers_15_self_attn_rotary_emb_sin_cached_9314, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_31_9316 = torch.Tensor.to(getitem_539_9315, dtype=torch.bfloat16)
        del getitem_539_9315
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_540_9317 = nnscaler.runtime.function.fullslice(to_30_9313, unsqueeze_8005)
        del to_30_9313
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_31_9318 = torch.unsqueeze(getitem_540_9317, dim=1)
        del getitem_540_9317
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_31_64073, unsqueeze_31_64074 = nnscaler.runtime.function.multiref(unsqueeze_31_9318, times=2)
        del unsqueeze_31_9318
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_541_9319 = nnscaler.runtime.function.fullslice(to_31_9316, unsqueeze_8005)
        del to_31_9316
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_32_9320 = torch.unsqueeze(getitem_541_9319, dim=1)
        del getitem_541_9319
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_32_64077, unsqueeze_32_64078 = nnscaler.runtime.function.multiref(unsqueeze_32_9320, times=2)
        del unsqueeze_32_9320
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_78_41377 = torch.Tensor.view(split_45_41001, size=(8, 16, 256, 32, 2))
        del split_45_41001
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_123_41417 = torch.transpose(view_78_41377, dim0=4, dim1=3)
        del view_78_41377
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_45_41449 = torch.Tensor.reshape(transpose_123_41417, shape=(8, 16, 256, 64))
        del transpose_123_41417
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_79_9324 = torch.Tensor.view(transpose_121_9302, size=(8, 1, 2048, 32, 2))
        del transpose_121_9302
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_124_9325 = torch.transpose(view_79_9324, dim0=4, dim1=3)
        del view_79_9324
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_46_9326 = torch.Tensor.reshape(transpose_124_9325, shape=(8, 1, 2048, 64))
        del transpose_124_9325
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_46_64085, reshape_46_64086, reshape_46_64087 = nnscaler.runtime.function.multiref(reshape_46_9326, times=3)
        del reshape_46_9326
        unsqueeze_31_73384 = nnscaler.runtime.adapter.chunk(unsqueeze_31_64073, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_31_64073
        # created at IRAdapterGener:local_consumer_multiref
        reshape_45_83859, reshape_45_83863, reshape_45_83867 = nnscaler.runtime.function.multiref(reshape_45_41449, times=3)
        del reshape_45_41449
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_75_41545 = torch.mul(reshape_45_83859, unsqueeze_31_73384)
        del unsqueeze_31_73384, reshape_45_83859
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_551_41593 = nnscaler.runtime.function.fullslice(reshape_45_83863, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_45_83863
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_553_41617 = nnscaler.runtime.function.fullslice(reshape_45_83867, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_45_83867
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_30_41641 = _operator.neg(getitem_553_41617)
        del getitem_553_41617
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_60_41681 = nnscaler.runtime.function.cat(neg_30_41641, getitem_551_41593, dim=-1)
        del getitem_551_41593, neg_30_41641
        unsqueeze_32_73456 = nnscaler.runtime.adapter.chunk(unsqueeze_32_64077, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_32_64077
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_76_41713 = torch.mul(cat_60_41681, unsqueeze_32_73456)
        del cat_60_41681, unsqueeze_32_73456
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_75_41761 = torch.add(mul_75_41545, mul_76_41713, alpha=1)
        del mul_75_41545, mul_76_41713
        unsqueeze_31_73488 = nnscaler.runtime.adapter.chunk(unsqueeze_31_64074, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_31_64074
        reshape_46_73480 = nnscaler.runtime.adapter.nn.split_allgather(reshape_46_64087, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_46_64087
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_77_41801 = torch.mul(reshape_46_73480, unsqueeze_31_73488)
        del unsqueeze_31_73488, reshape_46_73480
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_555_9335 = nnscaler.runtime.function.fullslice(reshape_46_64085, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_46_64085
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_557_9336 = nnscaler.runtime.function.fullslice(reshape_46_64086, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_46_64086
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_31_9337 = _operator.neg(getitem_557_9336)
        del getitem_557_9336
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_61_9338 = nnscaler.runtime.function.cat(neg_31_9337, getitem_555_9335, dim=-1)
        del getitem_555_9335, neg_31_9337
        cat_61_41905 = nnscaler.runtime.adapter.nn.split_allgather(cat_61_9338, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_61_9338
        unsqueeze_32_73512 = nnscaler.runtime.adapter.chunk(unsqueeze_32_64078, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_32_64078
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_78_41913 = torch.mul(cat_61_41905, unsqueeze_32_73512)
        del cat_61_41905, unsqueeze_32_73512
        mul_77_9334 = nnscaler.runtime.adapter.nn.allgather_split(mul_77_41801, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_77_41801
        mul_78_9339 = nnscaler.runtime.adapter.nn.allgather_split(mul_78_41913, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_78_41913
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_76_9340 = torch.add(mul_77_9334, mul_78_9339, alpha=1)
        del mul_77_9334, mul_78_9339
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_62_41961 = nnscaler.runtime.function.cat(split_45_40993, add_75_41761, dim=-1)
        del split_45_40993, add_75_41761
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_15_9342 = torch.Tensor.expand(add_76_9340, size=[-1, 16, -1, -1])
        del add_76_9340
        split_47_41297 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_47_41281, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_47_41281
        expand_15_42001 = nnscaler.runtime.adapter.nn.split_allgather(expand_15_9342, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_15_9342
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_63_42009 = nnscaler.runtime.function.cat(split_47_41297, expand_15_42001, dim=-1)
        del split_47_41297, expand_15_42001
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_15_42025 = torch.nn.functional.pad(split_47_41289, pad=[0, 64], mode='constant', value=0.0)
        del split_47_41289
        cat_62_41953 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_62_41961, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_62_41961
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_125_42057 = torch.transpose(cat_62_41953, dim0=1, dim1=2)
        del cat_62_41953
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_126_42097 = torch.transpose(cat_63_42009, dim0=1, dim1=2)
        del cat_63_42009
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_127_42129 = torch.transpose(pad_15_42025, dim0=1, dim1=2)
        del pad_15_42025
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_15_self_attn_training_7828 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_15_7829 = 0.0 if model_model_layers_15_self_attn_training_7828 else 0.0
        transpose_127_42137 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_127_42129, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_127_42129
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_15_42177 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_125_42057, transpose_126_42097, transpose_127_42137, dropout=ifexpr_15_7829, causal=True, attention_mask=None, query_length=2048)
        del transpose_125_42057, transpose_126_42097, transpose_127_42137
        nnscaler_flash_attention_forward_15_9348 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_15_42177, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_15_42177
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_558_9349 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_15_9348, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_15_9348
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_47_9350 = torch.Tensor.reshape(getitem_558_9349, shape=(8, 2048, 2048))
        del getitem_558_9349
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_15_9351 = torch.Tensor.contiguous(reshape_47_9350)
        del reshape_47_9350
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_108_9353 = torch.nn.functional.linear(contiguous_15_9351, self.model_model_layers_15_self_attn_o_proj_weight_9352, bias=None)
        del contiguous_15_9351
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_77_9354 = torch.add(add_74_83768, linear_108_9353, alpha=1)
        del add_74_83768, linear_108_9353
        # created at IRAdapterGener:local_consumer_multiref
        add_77_83932, add_77_83936 = nnscaler.runtime.function.multiref(add_77_9354, times=2)
        del add_77_9354
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_47_9356 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_77_83932, self.model_model_layers_15_post_attention_layernorm_weight_9355, (2048,), 1e-06)
        del add_77_83932
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_47_64099, fused_rms_norm_affine_47_64100, fused_rms_norm_affine_47_64101, fused_rms_norm_affine_47_64102 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_47_9356, times=4)
        del fused_rms_norm_affine_47_9356
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_15_mlp_gate_training_7831 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_14_9358, moe_route_14_9359, moe_route_14_9360 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_47_64099, self.model_model_layers_15_mlp_gate_weight_9357, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_15_mlp_gate_training_7831, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_47_64099
        moe_route_14_9359 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_14_9359, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_14_9360 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_14_9360, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_47_64100 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_47_64100, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_14_42377 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_47_64100, moe_route_14_9358, moe_route_14_9359, moe_route_14_9360, self.model_model_layers_15_mlp_gate_projs_42353, self.model_model_layers_15_mlp_up_projs_42361, self.model_model_layers_15_mlp_down_projs_42369, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_47_64100, moe_route_14_9358, moe_route_14_9359, moe_route_14_9360
        fused_rms_norm_affine_47_73672 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_47_64101, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_47_64101
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_109_42441 = torch.nn.functional.linear(fused_rms_norm_affine_47_73672, self.model_model_layers_15_mlp_shared_experts_gate_proj_weight_9365, bias=None)
        del fused_rms_norm_affine_47_73672
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_15_42497 = torch.nn.functional.silu(linear_109_42441, inplace=False)
        del linear_109_42441
        fused_rms_norm_affine_47_73712 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_47_64102, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_47_64102
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_110_42521 = torch.nn.functional.linear(fused_rms_norm_affine_47_73712, self.model_model_layers_15_mlp_shared_experts_up_proj_weight_9368, bias=None)
        del fused_rms_norm_affine_47_73712
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_79_42569 = torch.mul(silu_15_42497, linear_110_42521)
        del silu_15_42497, linear_110_42521
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_111_42593 = torch.nn.functional.linear(mul_79_42569, self.model_model_layers_15_mlp_shared_experts_down_proj_weight_9371, bias=None)
        del mul_79_42569
        nnscaler_moe_gmm_14_9364 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_14_42377, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_14_42377
        linear_111_9372 = nnscaler.runtime.adapter.nn.allgather_split(linear_111_42593, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_111_42593
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_78_9373 = torch.add(nnscaler_moe_gmm_14_9364, linear_111_9372, alpha=1)
        del nnscaler_moe_gmm_14_9364, linear_111_9372
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_79_9374 = torch.add(add_77_83936, add_78_9373, alpha=1)
        del add_77_83936, add_78_9373
        # created at IRAdapterGener:local_consumer_multiref
        add_79_83996, add_79_84000 = nnscaler.runtime.function.multiref(add_79_9374, times=2)
        del add_79_9374
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_48_9376 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_79_83996, self.model_model_layers_16_input_layernorm_weight_9375, (2048,), 1e-06)
        del add_79_83996
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_48_64111, fused_rms_norm_affine_48_64112 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_48_9376, times=2)
        del fused_rms_norm_affine_48_9376
        fused_rms_norm_affine_48_73768 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_48_64111, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_48_64111
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_112_42721 = torch.nn.functional.linear(fused_rms_norm_affine_48_73768, self.model_model_layers_16_self_attn_q_proj_weight_9377, bias=None)
        del fused_rms_norm_affine_48_73768
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_80_42777 = torch.Tensor.view(linear_112_42721, size=(8, 256, 16, 192))
        del linear_112_42721
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_128_42801 = torch.transpose(view_80_42777, dim0=1, dim1=2)
        del view_80_42777
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_48_42865, split_48_42873 = torch.functional.split(transpose_128_42801, split_size_or_sections=[128, 64], dim=-1)
        del transpose_128_42801
        fused_rms_norm_affine_48_73832 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_48_64112, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_48_64112
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_113_42905 = torch.nn.functional.linear(fused_rms_norm_affine_48_73832, self.model_model_layers_16_self_attn_kv_a_proj_with_mqa_weight_42897, bias=None)
        del fused_rms_norm_affine_48_73832
        linear_113_9384 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_113_42905, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_113_42905
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_49_9385, split_49_9386 = torch.functional.split(linear_113_9384, split_size_or_sections=[512, 64], dim=-1)
        del linear_113_9384
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_81_9387 = torch.Tensor.view(split_49_9386, size=(8, 2048, 1, 64))
        del split_49_9386
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_129_9388 = torch.transpose(view_81_9387, dim0=1, dim1=2)
        del view_81_9387
        split_49_42945 = nnscaler.runtime.adapter.nn.split_allgather(split_49_9385, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_49_9385
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_49_43025 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_49_42945, self.model_model_layers_16_self_attn_kv_a_layernorm_weight_9389, (512,), 1e-06)
        del split_49_42945
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_114_43041 = torch.nn.functional.linear(fused_rms_norm_affine_49_43025, self.model_model_layers_16_self_attn_kv_b_proj_weight_9391, bias=None)
        del fused_rms_norm_affine_49_43025
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_82_43097 = torch.Tensor.view(linear_114_43041, size=(8, 256, 16, 256))
        del linear_114_43041
        view_82_43089 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_82_43097, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_82_43097
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_130_43113 = torch.transpose(view_82_43089, dim0=1, dim1=2)
        del view_82_43089
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_50_43153, split_50_43161 = torch.functional.split(transpose_130_43113, split_size_or_sections=[128, 128], dim=-1)
        del transpose_130_43113
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_573_9398 = nnscaler.runtime.function.fullslice(self.model_model_layers_16_self_attn_rotary_emb_cos_cached_9397, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_32_9399 = torch.Tensor.to(getitem_573_9398, dtype=torch.bfloat16)
        del getitem_573_9398
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_574_9401 = nnscaler.runtime.function.fullslice(self.model_model_layers_16_self_attn_rotary_emb_sin_cached_9400, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_33_9402 = torch.Tensor.to(getitem_574_9401, dtype=torch.bfloat16)
        del getitem_574_9401
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_575_9403 = nnscaler.runtime.function.fullslice(to_32_9399, unsqueeze_8005)
        del to_32_9399
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_33_9404 = torch.unsqueeze(getitem_575_9403, dim=1)
        del getitem_575_9403
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_33_64117, unsqueeze_33_64118 = nnscaler.runtime.function.multiref(unsqueeze_33_9404, times=2)
        del unsqueeze_33_9404
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_576_9405 = nnscaler.runtime.function.fullslice(to_33_9402, unsqueeze_8005)
        del to_33_9402
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_34_9406 = torch.unsqueeze(getitem_576_9405, dim=1)
        del getitem_576_9405
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_34_64121, unsqueeze_34_64122 = nnscaler.runtime.function.multiref(unsqueeze_34_9406, times=2)
        del unsqueeze_34_9406
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_83_43249 = torch.Tensor.view(split_48_42873, size=(8, 16, 256, 32, 2))
        del split_48_42873
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_131_43289 = torch.transpose(view_83_43249, dim0=4, dim1=3)
        del view_83_43249
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_48_43321 = torch.Tensor.reshape(transpose_131_43289, shape=(8, 16, 256, 64))
        del transpose_131_43289
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_84_9410 = torch.Tensor.view(transpose_129_9388, size=(8, 1, 2048, 32, 2))
        del transpose_129_9388
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_132_9411 = torch.transpose(view_84_9410, dim0=4, dim1=3)
        del view_84_9410
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_49_9412 = torch.Tensor.reshape(transpose_132_9411, shape=(8, 1, 2048, 64))
        del transpose_132_9411
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_49_64129, reshape_49_64130, reshape_49_64131 = nnscaler.runtime.function.multiref(reshape_49_9412, times=3)
        del reshape_49_9412
        unsqueeze_33_73960 = nnscaler.runtime.adapter.chunk(unsqueeze_33_64117, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_33_64117
        # created at IRAdapterGener:local_consumer_multiref
        reshape_48_84091, reshape_48_84095, reshape_48_84099 = nnscaler.runtime.function.multiref(reshape_48_43321, times=3)
        del reshape_48_43321
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_80_43417 = torch.mul(reshape_48_84091, unsqueeze_33_73960)
        del unsqueeze_33_73960, reshape_48_84091
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_586_43465 = nnscaler.runtime.function.fullslice(reshape_48_84095, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_48_84095
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_588_43489 = nnscaler.runtime.function.fullslice(reshape_48_84099, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_48_84099
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_32_43513 = _operator.neg(getitem_588_43489)
        del getitem_588_43489
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_64_43553 = nnscaler.runtime.function.cat(neg_32_43513, getitem_586_43465, dim=-1)
        del getitem_586_43465, neg_32_43513
        unsqueeze_34_74032 = nnscaler.runtime.adapter.chunk(unsqueeze_34_64121, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_34_64121
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_81_43585 = torch.mul(cat_64_43553, unsqueeze_34_74032)
        del cat_64_43553, unsqueeze_34_74032
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_80_43633 = torch.add(mul_80_43417, mul_81_43585, alpha=1)
        del mul_80_43417, mul_81_43585
        unsqueeze_33_74064 = nnscaler.runtime.adapter.chunk(unsqueeze_33_64118, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_33_64118
        reshape_49_74056 = nnscaler.runtime.adapter.nn.split_allgather(reshape_49_64131, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_49_64131
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_82_43673 = torch.mul(reshape_49_74056, unsqueeze_33_74064)
        del unsqueeze_33_74064, reshape_49_74056
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_590_9421 = nnscaler.runtime.function.fullslice(reshape_49_64129, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_49_64129
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_592_9422 = nnscaler.runtime.function.fullslice(reshape_49_64130, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_49_64130
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_33_9423 = _operator.neg(getitem_592_9422)
        del getitem_592_9422
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_65_9424 = nnscaler.runtime.function.cat(neg_33_9423, getitem_590_9421, dim=-1)
        del getitem_590_9421, neg_33_9423
        cat_65_43777 = nnscaler.runtime.adapter.nn.split_allgather(cat_65_9424, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_65_9424
        unsqueeze_34_74088 = nnscaler.runtime.adapter.chunk(unsqueeze_34_64122, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_34_64122
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_83_43785 = torch.mul(cat_65_43777, unsqueeze_34_74088)
        del cat_65_43777, unsqueeze_34_74088
        mul_82_9420 = nnscaler.runtime.adapter.nn.allgather_split(mul_82_43673, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_82_43673
        mul_83_9425 = nnscaler.runtime.adapter.nn.allgather_split(mul_83_43785, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_83_43785
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_81_9426 = torch.add(mul_82_9420, mul_83_9425, alpha=1)
        del mul_82_9420, mul_83_9425
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_66_43833 = nnscaler.runtime.function.cat(split_48_42865, add_80_43633, dim=-1)
        del split_48_42865, add_80_43633
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_16_9428 = torch.Tensor.expand(add_81_9426, size=[-1, 16, -1, -1])
        del add_81_9426
        split_50_43169 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_50_43153, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_50_43153
        expand_16_43873 = nnscaler.runtime.adapter.nn.split_allgather(expand_16_9428, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_16_9428
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_67_43881 = nnscaler.runtime.function.cat(split_50_43169, expand_16_43873, dim=-1)
        del split_50_43169, expand_16_43873
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_16_43897 = torch.nn.functional.pad(split_50_43161, pad=[0, 64], mode='constant', value=0.0)
        del split_50_43161
        cat_66_43825 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_66_43833, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_66_43833
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_133_43929 = torch.transpose(cat_66_43825, dim0=1, dim1=2)
        del cat_66_43825
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_134_43969 = torch.transpose(cat_67_43881, dim0=1, dim1=2)
        del cat_67_43881
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_135_44001 = torch.transpose(pad_16_43897, dim0=1, dim1=2)
        del pad_16_43897
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_16_self_attn_training_7843 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_16_7844 = 0.0 if model_model_layers_16_self_attn_training_7843 else 0.0
        transpose_135_44009 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_135_44001, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_135_44001
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_16_44049 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_133_43929, transpose_134_43969, transpose_135_44009, dropout=ifexpr_16_7844, causal=True, attention_mask=None, query_length=2048)
        del transpose_133_43929, transpose_134_43969, transpose_135_44009
        nnscaler_flash_attention_forward_16_9434 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_16_44049, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_16_44049
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_593_9435 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_16_9434, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_16_9434
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_50_9436 = torch.Tensor.reshape(getitem_593_9435, shape=(8, 2048, 2048))
        del getitem_593_9435
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_16_9437 = torch.Tensor.contiguous(reshape_50_9436)
        del reshape_50_9436
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_115_9439 = torch.nn.functional.linear(contiguous_16_9437, self.model_model_layers_16_self_attn_o_proj_weight_9438, bias=None)
        del contiguous_16_9437
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_82_9440 = torch.add(add_79_84000, linear_115_9439, alpha=1)
        del add_79_84000, linear_115_9439
        # created at IRAdapterGener:local_consumer_multiref
        add_82_84164, add_82_84168 = nnscaler.runtime.function.multiref(add_82_9440, times=2)
        del add_82_9440
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_50_9442 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_82_84164, self.model_model_layers_16_post_attention_layernorm_weight_9441, (2048,), 1e-06)
        del add_82_84164
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_50_64143, fused_rms_norm_affine_50_64144, fused_rms_norm_affine_50_64145, fused_rms_norm_affine_50_64146 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_50_9442, times=4)
        del fused_rms_norm_affine_50_9442
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_16_mlp_gate_training_7846 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_15_9444, moe_route_15_9445, moe_route_15_9446 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_50_64143, self.model_model_layers_16_mlp_gate_weight_9443, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_16_mlp_gate_training_7846, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_50_64143
        moe_route_15_9445 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_15_9445, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_15_9446 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_15_9446, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_50_64144 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_50_64144, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_15_44249 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_50_64144, moe_route_15_9444, moe_route_15_9445, moe_route_15_9446, self.model_model_layers_16_mlp_gate_projs_44225, self.model_model_layers_16_mlp_up_projs_44233, self.model_model_layers_16_mlp_down_projs_44241, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_50_64144, moe_route_15_9444, moe_route_15_9445, moe_route_15_9446
        fused_rms_norm_affine_50_74248 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_50_64145, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_50_64145
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_116_44313 = torch.nn.functional.linear(fused_rms_norm_affine_50_74248, self.model_model_layers_16_mlp_shared_experts_gate_proj_weight_9451, bias=None)
        del fused_rms_norm_affine_50_74248
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_16_44369 = torch.nn.functional.silu(linear_116_44313, inplace=False)
        del linear_116_44313
        fused_rms_norm_affine_50_74288 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_50_64146, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_50_64146
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_117_44393 = torch.nn.functional.linear(fused_rms_norm_affine_50_74288, self.model_model_layers_16_mlp_shared_experts_up_proj_weight_9454, bias=None)
        del fused_rms_norm_affine_50_74288
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_84_44441 = torch.mul(silu_16_44369, linear_117_44393)
        del silu_16_44369, linear_117_44393
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_118_44465 = torch.nn.functional.linear(mul_84_44441, self.model_model_layers_16_mlp_shared_experts_down_proj_weight_9457, bias=None)
        del mul_84_44441
        nnscaler_moe_gmm_15_9450 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_15_44249, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_15_44249
        linear_118_9458 = nnscaler.runtime.adapter.nn.allgather_split(linear_118_44465, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_118_44465
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_83_9459 = torch.add(nnscaler_moe_gmm_15_9450, linear_118_9458, alpha=1)
        del nnscaler_moe_gmm_15_9450, linear_118_9458
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_84_9460 = torch.add(add_82_84168, add_83_9459, alpha=1)
        del add_82_84168, add_83_9459
        # created at IRAdapterGener:local_consumer_multiref
        add_84_84228, add_84_84232 = nnscaler.runtime.function.multiref(add_84_9460, times=2)
        del add_84_9460
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_51_9462 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_84_84228, self.model_model_layers_17_input_layernorm_weight_9461, (2048,), 1e-06)
        del add_84_84228
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_51_64155, fused_rms_norm_affine_51_64156 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_51_9462, times=2)
        del fused_rms_norm_affine_51_9462
        fused_rms_norm_affine_51_74344 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_51_64155, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_51_64155
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_119_44593 = torch.nn.functional.linear(fused_rms_norm_affine_51_74344, self.model_model_layers_17_self_attn_q_proj_weight_9463, bias=None)
        del fused_rms_norm_affine_51_74344
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_85_44649 = torch.Tensor.view(linear_119_44593, size=(8, 256, 16, 192))
        del linear_119_44593
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_136_44673 = torch.transpose(view_85_44649, dim0=1, dim1=2)
        del view_85_44649
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_51_44737, split_51_44745 = torch.functional.split(transpose_136_44673, split_size_or_sections=[128, 64], dim=-1)
        del transpose_136_44673
        fused_rms_norm_affine_51_74408 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_51_64156, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_51_64156
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_120_44777 = torch.nn.functional.linear(fused_rms_norm_affine_51_74408, self.model_model_layers_17_self_attn_kv_a_proj_with_mqa_weight_44769, bias=None)
        del fused_rms_norm_affine_51_74408
        linear_120_9470 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_120_44777, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_120_44777
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_52_9471, split_52_9472 = torch.functional.split(linear_120_9470, split_size_or_sections=[512, 64], dim=-1)
        del linear_120_9470
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_86_9473 = torch.Tensor.view(split_52_9472, size=(8, 2048, 1, 64))
        del split_52_9472
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_137_9474 = torch.transpose(view_86_9473, dim0=1, dim1=2)
        del view_86_9473
        split_52_44817 = nnscaler.runtime.adapter.nn.split_allgather(split_52_9471, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_52_9471
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_52_44897 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_52_44817, self.model_model_layers_17_self_attn_kv_a_layernorm_weight_9475, (512,), 1e-06)
        del split_52_44817
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_121_44913 = torch.nn.functional.linear(fused_rms_norm_affine_52_44897, self.model_model_layers_17_self_attn_kv_b_proj_weight_9477, bias=None)
        del fused_rms_norm_affine_52_44897
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_87_44969 = torch.Tensor.view(linear_121_44913, size=(8, 256, 16, 256))
        del linear_121_44913
        view_87_44961 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_87_44969, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_87_44969
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_138_44985 = torch.transpose(view_87_44961, dim0=1, dim1=2)
        del view_87_44961
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_53_45025, split_53_45033 = torch.functional.split(transpose_138_44985, split_size_or_sections=[128, 128], dim=-1)
        del transpose_138_44985
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_608_9484 = nnscaler.runtime.function.fullslice(self.model_model_layers_17_self_attn_rotary_emb_cos_cached_9483, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_34_9485 = torch.Tensor.to(getitem_608_9484, dtype=torch.bfloat16)
        del getitem_608_9484
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_609_9487 = nnscaler.runtime.function.fullslice(self.model_model_layers_17_self_attn_rotary_emb_sin_cached_9486, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_35_9488 = torch.Tensor.to(getitem_609_9487, dtype=torch.bfloat16)
        del getitem_609_9487
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_610_9489 = nnscaler.runtime.function.fullslice(to_34_9485, unsqueeze_8005)
        del to_34_9485
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_35_9490 = torch.unsqueeze(getitem_610_9489, dim=1)
        del getitem_610_9489
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_35_64161, unsqueeze_35_64162 = nnscaler.runtime.function.multiref(unsqueeze_35_9490, times=2)
        del unsqueeze_35_9490
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_611_9491 = nnscaler.runtime.function.fullslice(to_35_9488, unsqueeze_8005)
        del to_35_9488
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_36_9492 = torch.unsqueeze(getitem_611_9491, dim=1)
        del getitem_611_9491
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_36_64165, unsqueeze_36_64166 = nnscaler.runtime.function.multiref(unsqueeze_36_9492, times=2)
        del unsqueeze_36_9492
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_88_45121 = torch.Tensor.view(split_51_44745, size=(8, 16, 256, 32, 2))
        del split_51_44745
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_139_45161 = torch.transpose(view_88_45121, dim0=4, dim1=3)
        del view_88_45121
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_51_45193 = torch.Tensor.reshape(transpose_139_45161, shape=(8, 16, 256, 64))
        del transpose_139_45161
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_89_9496 = torch.Tensor.view(transpose_137_9474, size=(8, 1, 2048, 32, 2))
        del transpose_137_9474
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_140_9497 = torch.transpose(view_89_9496, dim0=4, dim1=3)
        del view_89_9496
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_52_9498 = torch.Tensor.reshape(transpose_140_9497, shape=(8, 1, 2048, 64))
        del transpose_140_9497
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_52_64173, reshape_52_64174, reshape_52_64175 = nnscaler.runtime.function.multiref(reshape_52_9498, times=3)
        del reshape_52_9498
        unsqueeze_35_74536 = nnscaler.runtime.adapter.chunk(unsqueeze_35_64161, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_35_64161
        # created at IRAdapterGener:local_consumer_multiref
        reshape_51_84323, reshape_51_84327, reshape_51_84331 = nnscaler.runtime.function.multiref(reshape_51_45193, times=3)
        del reshape_51_45193
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_85_45289 = torch.mul(reshape_51_84323, unsqueeze_35_74536)
        del unsqueeze_35_74536, reshape_51_84323
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_621_45337 = nnscaler.runtime.function.fullslice(reshape_51_84327, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_51_84327
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_623_45361 = nnscaler.runtime.function.fullslice(reshape_51_84331, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_51_84331
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_34_45385 = _operator.neg(getitem_623_45361)
        del getitem_623_45361
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_68_45425 = nnscaler.runtime.function.cat(neg_34_45385, getitem_621_45337, dim=-1)
        del getitem_621_45337, neg_34_45385
        unsqueeze_36_74608 = nnscaler.runtime.adapter.chunk(unsqueeze_36_64165, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_36_64165
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_86_45457 = torch.mul(cat_68_45425, unsqueeze_36_74608)
        del cat_68_45425, unsqueeze_36_74608
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_85_45505 = torch.add(mul_85_45289, mul_86_45457, alpha=1)
        del mul_85_45289, mul_86_45457
        unsqueeze_35_74640 = nnscaler.runtime.adapter.chunk(unsqueeze_35_64162, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_35_64162
        reshape_52_74632 = nnscaler.runtime.adapter.nn.split_allgather(reshape_52_64175, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_52_64175
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_87_45545 = torch.mul(reshape_52_74632, unsqueeze_35_74640)
        del unsqueeze_35_74640, reshape_52_74632
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_625_9507 = nnscaler.runtime.function.fullslice(reshape_52_64173, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_52_64173
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_627_9508 = nnscaler.runtime.function.fullslice(reshape_52_64174, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_52_64174
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_35_9509 = _operator.neg(getitem_627_9508)
        del getitem_627_9508
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_69_9510 = nnscaler.runtime.function.cat(neg_35_9509, getitem_625_9507, dim=-1)
        del getitem_625_9507, neg_35_9509
        cat_69_45649 = nnscaler.runtime.adapter.nn.split_allgather(cat_69_9510, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_69_9510
        unsqueeze_36_74664 = nnscaler.runtime.adapter.chunk(unsqueeze_36_64166, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_36_64166
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_88_45657 = torch.mul(cat_69_45649, unsqueeze_36_74664)
        del cat_69_45649, unsqueeze_36_74664
        mul_87_9506 = nnscaler.runtime.adapter.nn.allgather_split(mul_87_45545, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_87_45545
        mul_88_9511 = nnscaler.runtime.adapter.nn.allgather_split(mul_88_45657, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_88_45657
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_86_9512 = torch.add(mul_87_9506, mul_88_9511, alpha=1)
        del mul_87_9506, mul_88_9511
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_70_45705 = nnscaler.runtime.function.cat(split_51_44737, add_85_45505, dim=-1)
        del split_51_44737, add_85_45505
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_17_9514 = torch.Tensor.expand(add_86_9512, size=[-1, 16, -1, -1])
        del add_86_9512
        split_53_45041 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_53_45025, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_53_45025
        expand_17_45745 = nnscaler.runtime.adapter.nn.split_allgather(expand_17_9514, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_17_9514
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_71_45753 = nnscaler.runtime.function.cat(split_53_45041, expand_17_45745, dim=-1)
        del split_53_45041, expand_17_45745
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_17_45769 = torch.nn.functional.pad(split_53_45033, pad=[0, 64], mode='constant', value=0.0)
        del split_53_45033
        cat_70_45697 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_70_45705, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_70_45705
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_141_45801 = torch.transpose(cat_70_45697, dim0=1, dim1=2)
        del cat_70_45697
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_142_45841 = torch.transpose(cat_71_45753, dim0=1, dim1=2)
        del cat_71_45753
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_143_45873 = torch.transpose(pad_17_45769, dim0=1, dim1=2)
        del pad_17_45769
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_17_self_attn_training_7858 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_17_7859 = 0.0 if model_model_layers_17_self_attn_training_7858 else 0.0
        transpose_143_45881 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_143_45873, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_143_45873
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_17_45921 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_141_45801, transpose_142_45841, transpose_143_45881, dropout=ifexpr_17_7859, causal=True, attention_mask=None, query_length=2048)
        del transpose_141_45801, transpose_142_45841, transpose_143_45881
        nnscaler_flash_attention_forward_17_9520 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_17_45921, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_17_45921
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_628_9521 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_17_9520, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_17_9520
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_53_9522 = torch.Tensor.reshape(getitem_628_9521, shape=(8, 2048, 2048))
        del getitem_628_9521
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_17_9523 = torch.Tensor.contiguous(reshape_53_9522)
        del reshape_53_9522
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_122_9525 = torch.nn.functional.linear(contiguous_17_9523, self.model_model_layers_17_self_attn_o_proj_weight_9524, bias=None)
        del contiguous_17_9523
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_87_9526 = torch.add(add_84_84232, linear_122_9525, alpha=1)
        del add_84_84232, linear_122_9525
        # created at IRAdapterGener:local_consumer_multiref
        add_87_84396, add_87_84400 = nnscaler.runtime.function.multiref(add_87_9526, times=2)
        del add_87_9526
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_53_9528 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_87_84396, self.model_model_layers_17_post_attention_layernorm_weight_9527, (2048,), 1e-06)
        del add_87_84396
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_53_64187, fused_rms_norm_affine_53_64188, fused_rms_norm_affine_53_64189, fused_rms_norm_affine_53_64190 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_53_9528, times=4)
        del fused_rms_norm_affine_53_9528
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_17_mlp_gate_training_7861 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_16_9530, moe_route_16_9531, moe_route_16_9532 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_53_64187, self.model_model_layers_17_mlp_gate_weight_9529, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_17_mlp_gate_training_7861, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_53_64187
        moe_route_16_9531 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_16_9531, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_16_9532 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_16_9532, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_53_64188 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_53_64188, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_16_46121 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_53_64188, moe_route_16_9530, moe_route_16_9531, moe_route_16_9532, self.model_model_layers_17_mlp_gate_projs_46097, self.model_model_layers_17_mlp_up_projs_46105, self.model_model_layers_17_mlp_down_projs_46113, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_53_64188, moe_route_16_9530, moe_route_16_9531, moe_route_16_9532
        fused_rms_norm_affine_53_74824 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_53_64189, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_53_64189
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_123_46185 = torch.nn.functional.linear(fused_rms_norm_affine_53_74824, self.model_model_layers_17_mlp_shared_experts_gate_proj_weight_9537, bias=None)
        del fused_rms_norm_affine_53_74824
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_17_46241 = torch.nn.functional.silu(linear_123_46185, inplace=False)
        del linear_123_46185
        fused_rms_norm_affine_53_74864 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_53_64190, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_53_64190
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_124_46265 = torch.nn.functional.linear(fused_rms_norm_affine_53_74864, self.model_model_layers_17_mlp_shared_experts_up_proj_weight_9540, bias=None)
        del fused_rms_norm_affine_53_74864
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_89_46313 = torch.mul(silu_17_46241, linear_124_46265)
        del silu_17_46241, linear_124_46265
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_125_46337 = torch.nn.functional.linear(mul_89_46313, self.model_model_layers_17_mlp_shared_experts_down_proj_weight_9543, bias=None)
        del mul_89_46313
        nnscaler_moe_gmm_16_9536 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_16_46121, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_16_46121
        linear_125_9544 = nnscaler.runtime.adapter.nn.allgather_split(linear_125_46337, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_125_46337
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_88_9545 = torch.add(nnscaler_moe_gmm_16_9536, linear_125_9544, alpha=1)
        del nnscaler_moe_gmm_16_9536, linear_125_9544
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_89_9546 = torch.add(add_87_84400, add_88_9545, alpha=1)
        del add_87_84400, add_88_9545
        # created at IRAdapterGener:local_consumer_multiref
        add_89_84460, add_89_84464 = nnscaler.runtime.function.multiref(add_89_9546, times=2)
        del add_89_9546
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_54_9548 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_89_84460, self.model_model_layers_18_input_layernorm_weight_9547, (2048,), 1e-06)
        del add_89_84460
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_54_64199, fused_rms_norm_affine_54_64200 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_54_9548, times=2)
        del fused_rms_norm_affine_54_9548
        fused_rms_norm_affine_54_74920 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_54_64199, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_54_64199
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_126_46465 = torch.nn.functional.linear(fused_rms_norm_affine_54_74920, self.model_model_layers_18_self_attn_q_proj_weight_9549, bias=None)
        del fused_rms_norm_affine_54_74920
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_90_46521 = torch.Tensor.view(linear_126_46465, size=(8, 256, 16, 192))
        del linear_126_46465
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_144_46545 = torch.transpose(view_90_46521, dim0=1, dim1=2)
        del view_90_46521
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_54_46609, split_54_46617 = torch.functional.split(transpose_144_46545, split_size_or_sections=[128, 64], dim=-1)
        del transpose_144_46545
        fused_rms_norm_affine_54_74984 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_54_64200, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_54_64200
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_127_46649 = torch.nn.functional.linear(fused_rms_norm_affine_54_74984, self.model_model_layers_18_self_attn_kv_a_proj_with_mqa_weight_46641, bias=None)
        del fused_rms_norm_affine_54_74984
        linear_127_9556 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_127_46649, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_127_46649
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_55_9557, split_55_9558 = torch.functional.split(linear_127_9556, split_size_or_sections=[512, 64], dim=-1)
        del linear_127_9556
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_91_9559 = torch.Tensor.view(split_55_9558, size=(8, 2048, 1, 64))
        del split_55_9558
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_145_9560 = torch.transpose(view_91_9559, dim0=1, dim1=2)
        del view_91_9559
        split_55_46689 = nnscaler.runtime.adapter.nn.split_allgather(split_55_9557, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_55_9557
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_55_46769 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_55_46689, self.model_model_layers_18_self_attn_kv_a_layernorm_weight_9561, (512,), 1e-06)
        del split_55_46689
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_128_46785 = torch.nn.functional.linear(fused_rms_norm_affine_55_46769, self.model_model_layers_18_self_attn_kv_b_proj_weight_9563, bias=None)
        del fused_rms_norm_affine_55_46769
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_92_46841 = torch.Tensor.view(linear_128_46785, size=(8, 256, 16, 256))
        del linear_128_46785
        view_92_46833 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_92_46841, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_92_46841
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_146_46857 = torch.transpose(view_92_46833, dim0=1, dim1=2)
        del view_92_46833
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_56_46897, split_56_46905 = torch.functional.split(transpose_146_46857, split_size_or_sections=[128, 128], dim=-1)
        del transpose_146_46857
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_643_9570 = nnscaler.runtime.function.fullslice(self.model_model_layers_18_self_attn_rotary_emb_cos_cached_9569, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_36_9571 = torch.Tensor.to(getitem_643_9570, dtype=torch.bfloat16)
        del getitem_643_9570
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_644_9573 = nnscaler.runtime.function.fullslice(self.model_model_layers_18_self_attn_rotary_emb_sin_cached_9572, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_37_9574 = torch.Tensor.to(getitem_644_9573, dtype=torch.bfloat16)
        del getitem_644_9573
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_645_9575 = nnscaler.runtime.function.fullslice(to_36_9571, unsqueeze_8005)
        del to_36_9571
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_37_9576 = torch.unsqueeze(getitem_645_9575, dim=1)
        del getitem_645_9575
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_37_64205, unsqueeze_37_64206 = nnscaler.runtime.function.multiref(unsqueeze_37_9576, times=2)
        del unsqueeze_37_9576
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_646_9577 = nnscaler.runtime.function.fullslice(to_37_9574, unsqueeze_8005)
        del to_37_9574
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_38_9578 = torch.unsqueeze(getitem_646_9577, dim=1)
        del getitem_646_9577
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_38_64209, unsqueeze_38_64210 = nnscaler.runtime.function.multiref(unsqueeze_38_9578, times=2)
        del unsqueeze_38_9578
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_93_46993 = torch.Tensor.view(split_54_46617, size=(8, 16, 256, 32, 2))
        del split_54_46617
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_147_47033 = torch.transpose(view_93_46993, dim0=4, dim1=3)
        del view_93_46993
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_54_47065 = torch.Tensor.reshape(transpose_147_47033, shape=(8, 16, 256, 64))
        del transpose_147_47033
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_94_9582 = torch.Tensor.view(transpose_145_9560, size=(8, 1, 2048, 32, 2))
        del transpose_145_9560
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_148_9583 = torch.transpose(view_94_9582, dim0=4, dim1=3)
        del view_94_9582
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_55_9584 = torch.Tensor.reshape(transpose_148_9583, shape=(8, 1, 2048, 64))
        del transpose_148_9583
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_55_64217, reshape_55_64218, reshape_55_64219 = nnscaler.runtime.function.multiref(reshape_55_9584, times=3)
        del reshape_55_9584
        unsqueeze_37_75112 = nnscaler.runtime.adapter.chunk(unsqueeze_37_64205, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_37_64205
        # created at IRAdapterGener:local_consumer_multiref
        reshape_54_84555, reshape_54_84559, reshape_54_84563 = nnscaler.runtime.function.multiref(reshape_54_47065, times=3)
        del reshape_54_47065
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_90_47161 = torch.mul(reshape_54_84555, unsqueeze_37_75112)
        del unsqueeze_37_75112, reshape_54_84555
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_656_47209 = nnscaler.runtime.function.fullslice(reshape_54_84559, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_54_84559
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_658_47233 = nnscaler.runtime.function.fullslice(reshape_54_84563, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_54_84563
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_36_47257 = _operator.neg(getitem_658_47233)
        del getitem_658_47233
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_72_47297 = nnscaler.runtime.function.cat(neg_36_47257, getitem_656_47209, dim=-1)
        del getitem_656_47209, neg_36_47257
        unsqueeze_38_75184 = nnscaler.runtime.adapter.chunk(unsqueeze_38_64209, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_38_64209
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_91_47329 = torch.mul(cat_72_47297, unsqueeze_38_75184)
        del cat_72_47297, unsqueeze_38_75184
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_90_47377 = torch.add(mul_90_47161, mul_91_47329, alpha=1)
        del mul_90_47161, mul_91_47329
        unsqueeze_37_75216 = nnscaler.runtime.adapter.chunk(unsqueeze_37_64206, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_37_64206
        reshape_55_75208 = nnscaler.runtime.adapter.nn.split_allgather(reshape_55_64219, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_55_64219
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_92_47417 = torch.mul(reshape_55_75208, unsqueeze_37_75216)
        del unsqueeze_37_75216, reshape_55_75208
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_660_9593 = nnscaler.runtime.function.fullslice(reshape_55_64217, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_55_64217
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_662_9594 = nnscaler.runtime.function.fullslice(reshape_55_64218, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_55_64218
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_37_9595 = _operator.neg(getitem_662_9594)
        del getitem_662_9594
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_73_9596 = nnscaler.runtime.function.cat(neg_37_9595, getitem_660_9593, dim=-1)
        del getitem_660_9593, neg_37_9595
        cat_73_47521 = nnscaler.runtime.adapter.nn.split_allgather(cat_73_9596, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_73_9596
        unsqueeze_38_75240 = nnscaler.runtime.adapter.chunk(unsqueeze_38_64210, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_38_64210
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_93_47529 = torch.mul(cat_73_47521, unsqueeze_38_75240)
        del cat_73_47521, unsqueeze_38_75240
        mul_92_9592 = nnscaler.runtime.adapter.nn.allgather_split(mul_92_47417, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_92_47417
        mul_93_9597 = nnscaler.runtime.adapter.nn.allgather_split(mul_93_47529, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_93_47529
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_91_9598 = torch.add(mul_92_9592, mul_93_9597, alpha=1)
        del mul_92_9592, mul_93_9597
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_74_47577 = nnscaler.runtime.function.cat(split_54_46609, add_90_47377, dim=-1)
        del split_54_46609, add_90_47377
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_18_9600 = torch.Tensor.expand(add_91_9598, size=[-1, 16, -1, -1])
        del add_91_9598
        split_56_46913 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_56_46897, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_56_46897
        expand_18_47617 = nnscaler.runtime.adapter.nn.split_allgather(expand_18_9600, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_18_9600
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_75_47625 = nnscaler.runtime.function.cat(split_56_46913, expand_18_47617, dim=-1)
        del split_56_46913, expand_18_47617
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_18_47641 = torch.nn.functional.pad(split_56_46905, pad=[0, 64], mode='constant', value=0.0)
        del split_56_46905
        cat_74_47569 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_74_47577, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_74_47577
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_149_47673 = torch.transpose(cat_74_47569, dim0=1, dim1=2)
        del cat_74_47569
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_150_47713 = torch.transpose(cat_75_47625, dim0=1, dim1=2)
        del cat_75_47625
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_151_47745 = torch.transpose(pad_18_47641, dim0=1, dim1=2)
        del pad_18_47641
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_18_self_attn_training_7873 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_18_7874 = 0.0 if model_model_layers_18_self_attn_training_7873 else 0.0
        transpose_151_47753 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_151_47745, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_151_47745
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_18_47793 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_149_47673, transpose_150_47713, transpose_151_47753, dropout=ifexpr_18_7874, causal=True, attention_mask=None, query_length=2048)
        del transpose_149_47673, transpose_150_47713, transpose_151_47753
        nnscaler_flash_attention_forward_18_9606 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_18_47793, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_18_47793
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_663_9607 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_18_9606, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_18_9606
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_56_9608 = torch.Tensor.reshape(getitem_663_9607, shape=(8, 2048, 2048))
        del getitem_663_9607
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_18_9609 = torch.Tensor.contiguous(reshape_56_9608)
        del reshape_56_9608
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_129_9611 = torch.nn.functional.linear(contiguous_18_9609, self.model_model_layers_18_self_attn_o_proj_weight_9610, bias=None)
        del contiguous_18_9609
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_92_9612 = torch.add(add_89_84464, linear_129_9611, alpha=1)
        del add_89_84464, linear_129_9611
        # created at IRAdapterGener:local_consumer_multiref
        add_92_84628, add_92_84632 = nnscaler.runtime.function.multiref(add_92_9612, times=2)
        del add_92_9612
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_56_9614 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_92_84628, self.model_model_layers_18_post_attention_layernorm_weight_9613, (2048,), 1e-06)
        del add_92_84628
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_56_64231, fused_rms_norm_affine_56_64232, fused_rms_norm_affine_56_64233, fused_rms_norm_affine_56_64234 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_56_9614, times=4)
        del fused_rms_norm_affine_56_9614
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_18_mlp_gate_training_7876 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_17_9616, moe_route_17_9617, moe_route_17_9618 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_56_64231, self.model_model_layers_18_mlp_gate_weight_9615, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_18_mlp_gate_training_7876, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_56_64231
        moe_route_17_9617 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_17_9617, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_17_9618 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_17_9618, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_56_64232 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_56_64232, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_17_47993 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_56_64232, moe_route_17_9616, moe_route_17_9617, moe_route_17_9618, self.model_model_layers_18_mlp_gate_projs_47969, self.model_model_layers_18_mlp_up_projs_47977, self.model_model_layers_18_mlp_down_projs_47985, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_56_64232, moe_route_17_9616, moe_route_17_9617, moe_route_17_9618
        fused_rms_norm_affine_56_75400 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_56_64233, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_56_64233
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_130_48057 = torch.nn.functional.linear(fused_rms_norm_affine_56_75400, self.model_model_layers_18_mlp_shared_experts_gate_proj_weight_9623, bias=None)
        del fused_rms_norm_affine_56_75400
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_18_48113 = torch.nn.functional.silu(linear_130_48057, inplace=False)
        del linear_130_48057
        fused_rms_norm_affine_56_75440 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_56_64234, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_56_64234
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_131_48137 = torch.nn.functional.linear(fused_rms_norm_affine_56_75440, self.model_model_layers_18_mlp_shared_experts_up_proj_weight_9626, bias=None)
        del fused_rms_norm_affine_56_75440
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_94_48185 = torch.mul(silu_18_48113, linear_131_48137)
        del silu_18_48113, linear_131_48137
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_132_48209 = torch.nn.functional.linear(mul_94_48185, self.model_model_layers_18_mlp_shared_experts_down_proj_weight_9629, bias=None)
        del mul_94_48185
        nnscaler_moe_gmm_17_9622 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_17_47993, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_17_47993
        linear_132_9630 = nnscaler.runtime.adapter.nn.allgather_split(linear_132_48209, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_132_48209
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_93_9631 = torch.add(nnscaler_moe_gmm_17_9622, linear_132_9630, alpha=1)
        del nnscaler_moe_gmm_17_9622, linear_132_9630
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_94_9632 = torch.add(add_92_84632, add_93_9631, alpha=1)
        del add_92_84632, add_93_9631
        # created at IRAdapterGener:local_consumer_multiref
        add_94_84692, add_94_84696 = nnscaler.runtime.function.multiref(add_94_9632, times=2)
        del add_94_9632
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_57_9634 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_94_84692, self.model_model_layers_19_input_layernorm_weight_9633, (2048,), 1e-06)
        del add_94_84692
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_57_64243, fused_rms_norm_affine_57_64244 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_57_9634, times=2)
        del fused_rms_norm_affine_57_9634
        fused_rms_norm_affine_57_75496 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_57_64243, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_57_64243
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_133_48337 = torch.nn.functional.linear(fused_rms_norm_affine_57_75496, self.model_model_layers_19_self_attn_q_proj_weight_9635, bias=None)
        del fused_rms_norm_affine_57_75496
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_95_48393 = torch.Tensor.view(linear_133_48337, size=(8, 256, 16, 192))
        del linear_133_48337
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_152_48417 = torch.transpose(view_95_48393, dim0=1, dim1=2)
        del view_95_48393
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_57_48481, split_57_48489 = torch.functional.split(transpose_152_48417, split_size_or_sections=[128, 64], dim=-1)
        del transpose_152_48417
        fused_rms_norm_affine_57_75560 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_57_64244, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_57_64244
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_134_48521 = torch.nn.functional.linear(fused_rms_norm_affine_57_75560, self.model_model_layers_19_self_attn_kv_a_proj_with_mqa_weight_48513, bias=None)
        del fused_rms_norm_affine_57_75560
        linear_134_9642 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_134_48521, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_134_48521
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_58_9643, split_58_9644 = torch.functional.split(linear_134_9642, split_size_or_sections=[512, 64], dim=-1)
        del linear_134_9642
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_96_9645 = torch.Tensor.view(split_58_9644, size=(8, 2048, 1, 64))
        del split_58_9644
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_153_9646 = torch.transpose(view_96_9645, dim0=1, dim1=2)
        del view_96_9645
        split_58_48561 = nnscaler.runtime.adapter.nn.split_allgather(split_58_9643, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_58_9643
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_58_48641 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_58_48561, self.model_model_layers_19_self_attn_kv_a_layernorm_weight_9647, (512,), 1e-06)
        del split_58_48561
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_135_48657 = torch.nn.functional.linear(fused_rms_norm_affine_58_48641, self.model_model_layers_19_self_attn_kv_b_proj_weight_9649, bias=None)
        del fused_rms_norm_affine_58_48641
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_97_48713 = torch.Tensor.view(linear_135_48657, size=(8, 256, 16, 256))
        del linear_135_48657
        view_97_48705 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_97_48713, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_97_48713
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_154_48729 = torch.transpose(view_97_48705, dim0=1, dim1=2)
        del view_97_48705
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_59_48769, split_59_48777 = torch.functional.split(transpose_154_48729, split_size_or_sections=[128, 128], dim=-1)
        del transpose_154_48729
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_678_9656 = nnscaler.runtime.function.fullslice(self.model_model_layers_19_self_attn_rotary_emb_cos_cached_9655, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_38_9657 = torch.Tensor.to(getitem_678_9656, dtype=torch.bfloat16)
        del getitem_678_9656
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_679_9659 = nnscaler.runtime.function.fullslice(self.model_model_layers_19_self_attn_rotary_emb_sin_cached_9658, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_39_9660 = torch.Tensor.to(getitem_679_9659, dtype=torch.bfloat16)
        del getitem_679_9659
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_680_9661 = nnscaler.runtime.function.fullslice(to_38_9657, unsqueeze_8005)
        del to_38_9657
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_39_9662 = torch.unsqueeze(getitem_680_9661, dim=1)
        del getitem_680_9661
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_39_64249, unsqueeze_39_64250 = nnscaler.runtime.function.multiref(unsqueeze_39_9662, times=2)
        del unsqueeze_39_9662
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_681_9663 = nnscaler.runtime.function.fullslice(to_39_9660, unsqueeze_8005)
        del to_39_9660
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_40_9664 = torch.unsqueeze(getitem_681_9663, dim=1)
        del getitem_681_9663
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_40_64253, unsqueeze_40_64254 = nnscaler.runtime.function.multiref(unsqueeze_40_9664, times=2)
        del unsqueeze_40_9664
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_98_48865 = torch.Tensor.view(split_57_48489, size=(8, 16, 256, 32, 2))
        del split_57_48489
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_155_48905 = torch.transpose(view_98_48865, dim0=4, dim1=3)
        del view_98_48865
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_57_48937 = torch.Tensor.reshape(transpose_155_48905, shape=(8, 16, 256, 64))
        del transpose_155_48905
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_99_9668 = torch.Tensor.view(transpose_153_9646, size=(8, 1, 2048, 32, 2))
        del transpose_153_9646
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_156_9669 = torch.transpose(view_99_9668, dim0=4, dim1=3)
        del view_99_9668
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_58_9670 = torch.Tensor.reshape(transpose_156_9669, shape=(8, 1, 2048, 64))
        del transpose_156_9669
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_58_64261, reshape_58_64262, reshape_58_64263 = nnscaler.runtime.function.multiref(reshape_58_9670, times=3)
        del reshape_58_9670
        unsqueeze_39_75688 = nnscaler.runtime.adapter.chunk(unsqueeze_39_64249, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_39_64249
        # created at IRAdapterGener:local_consumer_multiref
        reshape_57_84787, reshape_57_84791, reshape_57_84795 = nnscaler.runtime.function.multiref(reshape_57_48937, times=3)
        del reshape_57_48937
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_95_49033 = torch.mul(reshape_57_84787, unsqueeze_39_75688)
        del unsqueeze_39_75688, reshape_57_84787
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_691_49081 = nnscaler.runtime.function.fullslice(reshape_57_84791, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_57_84791
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_693_49105 = nnscaler.runtime.function.fullslice(reshape_57_84795, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_57_84795
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_38_49129 = _operator.neg(getitem_693_49105)
        del getitem_693_49105
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_76_49169 = nnscaler.runtime.function.cat(neg_38_49129, getitem_691_49081, dim=-1)
        del getitem_691_49081, neg_38_49129
        unsqueeze_40_75760 = nnscaler.runtime.adapter.chunk(unsqueeze_40_64253, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_40_64253
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_96_49201 = torch.mul(cat_76_49169, unsqueeze_40_75760)
        del cat_76_49169, unsqueeze_40_75760
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_95_49249 = torch.add(mul_95_49033, mul_96_49201, alpha=1)
        del mul_95_49033, mul_96_49201
        unsqueeze_39_75792 = nnscaler.runtime.adapter.chunk(unsqueeze_39_64250, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_39_64250
        reshape_58_75784 = nnscaler.runtime.adapter.nn.split_allgather(reshape_58_64263, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_58_64263
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_97_49289 = torch.mul(reshape_58_75784, unsqueeze_39_75792)
        del unsqueeze_39_75792, reshape_58_75784
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_695_9679 = nnscaler.runtime.function.fullslice(reshape_58_64261, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_58_64261
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_697_9680 = nnscaler.runtime.function.fullslice(reshape_58_64262, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_58_64262
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_39_9681 = _operator.neg(getitem_697_9680)
        del getitem_697_9680
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_77_9682 = nnscaler.runtime.function.cat(neg_39_9681, getitem_695_9679, dim=-1)
        del getitem_695_9679, neg_39_9681
        cat_77_49393 = nnscaler.runtime.adapter.nn.split_allgather(cat_77_9682, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_77_9682
        unsqueeze_40_75816 = nnscaler.runtime.adapter.chunk(unsqueeze_40_64254, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_40_64254
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_98_49401 = torch.mul(cat_77_49393, unsqueeze_40_75816)
        del cat_77_49393, unsqueeze_40_75816
        mul_97_9678 = nnscaler.runtime.adapter.nn.allgather_split(mul_97_49289, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_97_49289
        mul_98_9683 = nnscaler.runtime.adapter.nn.allgather_split(mul_98_49401, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_98_49401
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_96_9684 = torch.add(mul_97_9678, mul_98_9683, alpha=1)
        del mul_97_9678, mul_98_9683
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_78_49449 = nnscaler.runtime.function.cat(split_57_48481, add_95_49249, dim=-1)
        del split_57_48481, add_95_49249
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_19_9686 = torch.Tensor.expand(add_96_9684, size=[-1, 16, -1, -1])
        del add_96_9684
        split_59_48785 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_59_48769, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_59_48769
        expand_19_49489 = nnscaler.runtime.adapter.nn.split_allgather(expand_19_9686, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_19_9686
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_79_49497 = nnscaler.runtime.function.cat(split_59_48785, expand_19_49489, dim=-1)
        del split_59_48785, expand_19_49489
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_19_49513 = torch.nn.functional.pad(split_59_48777, pad=[0, 64], mode='constant', value=0.0)
        del split_59_48777
        cat_78_49441 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_78_49449, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_78_49449
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_157_49545 = torch.transpose(cat_78_49441, dim0=1, dim1=2)
        del cat_78_49441
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_158_49585 = torch.transpose(cat_79_49497, dim0=1, dim1=2)
        del cat_79_49497
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_159_49617 = torch.transpose(pad_19_49513, dim0=1, dim1=2)
        del pad_19_49513
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_19_self_attn_training_7888 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_19_7889 = 0.0 if model_model_layers_19_self_attn_training_7888 else 0.0
        transpose_159_49625 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_159_49617, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_159_49617
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_19_49665 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_157_49545, transpose_158_49585, transpose_159_49625, dropout=ifexpr_19_7889, causal=True, attention_mask=None, query_length=2048)
        del transpose_157_49545, transpose_158_49585, transpose_159_49625
        nnscaler_flash_attention_forward_19_9692 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_19_49665, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_19_49665
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_698_9693 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_19_9692, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_19_9692
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_59_9694 = torch.Tensor.reshape(getitem_698_9693, shape=(8, 2048, 2048))
        del getitem_698_9693
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_19_9695 = torch.Tensor.contiguous(reshape_59_9694)
        del reshape_59_9694
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_136_9697 = torch.nn.functional.linear(contiguous_19_9695, self.model_model_layers_19_self_attn_o_proj_weight_9696, bias=None)
        del contiguous_19_9695
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_97_9698 = torch.add(add_94_84696, linear_136_9697, alpha=1)
        del add_94_84696, linear_136_9697
        # created at IRAdapterGener:local_consumer_multiref
        add_97_84860, add_97_84864 = nnscaler.runtime.function.multiref(add_97_9698, times=2)
        del add_97_9698
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_59_9700 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_97_84860, self.model_model_layers_19_post_attention_layernorm_weight_9699, (2048,), 1e-06)
        del add_97_84860
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_59_64275, fused_rms_norm_affine_59_64276, fused_rms_norm_affine_59_64277, fused_rms_norm_affine_59_64278 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_59_9700, times=4)
        del fused_rms_norm_affine_59_9700
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_19_mlp_gate_training_7891 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_18_9702, moe_route_18_9703, moe_route_18_9704 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_59_64275, self.model_model_layers_19_mlp_gate_weight_9701, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_19_mlp_gate_training_7891, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_59_64275
        moe_route_18_9703 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_18_9703, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_18_9704 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_18_9704, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_59_64276 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_59_64276, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_18_49865 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_59_64276, moe_route_18_9702, moe_route_18_9703, moe_route_18_9704, self.model_model_layers_19_mlp_gate_projs_49841, self.model_model_layers_19_mlp_up_projs_49849, self.model_model_layers_19_mlp_down_projs_49857, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_59_64276, moe_route_18_9702, moe_route_18_9703, moe_route_18_9704
        fused_rms_norm_affine_59_75976 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_59_64277, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_59_64277
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_137_49929 = torch.nn.functional.linear(fused_rms_norm_affine_59_75976, self.model_model_layers_19_mlp_shared_experts_gate_proj_weight_9709, bias=None)
        del fused_rms_norm_affine_59_75976
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_19_49985 = torch.nn.functional.silu(linear_137_49929, inplace=False)
        del linear_137_49929
        fused_rms_norm_affine_59_76016 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_59_64278, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_59_64278
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_138_50009 = torch.nn.functional.linear(fused_rms_norm_affine_59_76016, self.model_model_layers_19_mlp_shared_experts_up_proj_weight_9712, bias=None)
        del fused_rms_norm_affine_59_76016
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_99_50057 = torch.mul(silu_19_49985, linear_138_50009)
        del silu_19_49985, linear_138_50009
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_139_50081 = torch.nn.functional.linear(mul_99_50057, self.model_model_layers_19_mlp_shared_experts_down_proj_weight_9715, bias=None)
        del mul_99_50057
        nnscaler_moe_gmm_18_9708 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_18_49865, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_18_49865
        linear_139_9716 = nnscaler.runtime.adapter.nn.allgather_split(linear_139_50081, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_139_50081
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_98_9717 = torch.add(nnscaler_moe_gmm_18_9708, linear_139_9716, alpha=1)
        del nnscaler_moe_gmm_18_9708, linear_139_9716
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_99_9718 = torch.add(add_97_84864, add_98_9717, alpha=1)
        del add_97_84864, add_98_9717
        # created at IRAdapterGener:local_consumer_multiref
        add_99_84924, add_99_84928 = nnscaler.runtime.function.multiref(add_99_9718, times=2)
        del add_99_9718
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_60_9720 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_99_84924, self.model_model_layers_20_input_layernorm_weight_9719, (2048,), 1e-06)
        del add_99_84924
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_60_64287, fused_rms_norm_affine_60_64288 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_60_9720, times=2)
        del fused_rms_norm_affine_60_9720
        fused_rms_norm_affine_60_76072 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_60_64287, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_60_64287
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_140_50209 = torch.nn.functional.linear(fused_rms_norm_affine_60_76072, self.model_model_layers_20_self_attn_q_proj_weight_9721, bias=None)
        del fused_rms_norm_affine_60_76072
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_100_50265 = torch.Tensor.view(linear_140_50209, size=(8, 256, 16, 192))
        del linear_140_50209
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_160_50289 = torch.transpose(view_100_50265, dim0=1, dim1=2)
        del view_100_50265
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_60_50353, split_60_50361 = torch.functional.split(transpose_160_50289, split_size_or_sections=[128, 64], dim=-1)
        del transpose_160_50289
        fused_rms_norm_affine_60_76136 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_60_64288, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_60_64288
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_141_50393 = torch.nn.functional.linear(fused_rms_norm_affine_60_76136, self.model_model_layers_20_self_attn_kv_a_proj_with_mqa_weight_50385, bias=None)
        del fused_rms_norm_affine_60_76136
        linear_141_9728 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_141_50393, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_141_50393
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_61_9729, split_61_9730 = torch.functional.split(linear_141_9728, split_size_or_sections=[512, 64], dim=-1)
        del linear_141_9728
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_101_9731 = torch.Tensor.view(split_61_9730, size=(8, 2048, 1, 64))
        del split_61_9730
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_161_9732 = torch.transpose(view_101_9731, dim0=1, dim1=2)
        del view_101_9731
        split_61_50433 = nnscaler.runtime.adapter.nn.split_allgather(split_61_9729, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_61_9729
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_61_50513 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_61_50433, self.model_model_layers_20_self_attn_kv_a_layernorm_weight_9733, (512,), 1e-06)
        del split_61_50433
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_142_50529 = torch.nn.functional.linear(fused_rms_norm_affine_61_50513, self.model_model_layers_20_self_attn_kv_b_proj_weight_9735, bias=None)
        del fused_rms_norm_affine_61_50513
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_102_50585 = torch.Tensor.view(linear_142_50529, size=(8, 256, 16, 256))
        del linear_142_50529
        view_102_50577 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_102_50585, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_102_50585
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_162_50601 = torch.transpose(view_102_50577, dim0=1, dim1=2)
        del view_102_50577
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_62_50641, split_62_50649 = torch.functional.split(transpose_162_50601, split_size_or_sections=[128, 128], dim=-1)
        del transpose_162_50601
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_713_9742 = nnscaler.runtime.function.fullslice(self.model_model_layers_20_self_attn_rotary_emb_cos_cached_9741, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_40_9743 = torch.Tensor.to(getitem_713_9742, dtype=torch.bfloat16)
        del getitem_713_9742
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_714_9745 = nnscaler.runtime.function.fullslice(self.model_model_layers_20_self_attn_rotary_emb_sin_cached_9744, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_41_9746 = torch.Tensor.to(getitem_714_9745, dtype=torch.bfloat16)
        del getitem_714_9745
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_715_9747 = nnscaler.runtime.function.fullslice(to_40_9743, unsqueeze_8005)
        del to_40_9743
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_41_9748 = torch.unsqueeze(getitem_715_9747, dim=1)
        del getitem_715_9747
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_41_64293, unsqueeze_41_64294 = nnscaler.runtime.function.multiref(unsqueeze_41_9748, times=2)
        del unsqueeze_41_9748
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_716_9749 = nnscaler.runtime.function.fullslice(to_41_9746, unsqueeze_8005)
        del to_41_9746
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_42_9750 = torch.unsqueeze(getitem_716_9749, dim=1)
        del getitem_716_9749
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_42_64297, unsqueeze_42_64298 = nnscaler.runtime.function.multiref(unsqueeze_42_9750, times=2)
        del unsqueeze_42_9750
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_103_50737 = torch.Tensor.view(split_60_50361, size=(8, 16, 256, 32, 2))
        del split_60_50361
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_163_50777 = torch.transpose(view_103_50737, dim0=4, dim1=3)
        del view_103_50737
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_60_50809 = torch.Tensor.reshape(transpose_163_50777, shape=(8, 16, 256, 64))
        del transpose_163_50777
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_104_9754 = torch.Tensor.view(transpose_161_9732, size=(8, 1, 2048, 32, 2))
        del transpose_161_9732
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_164_9755 = torch.transpose(view_104_9754, dim0=4, dim1=3)
        del view_104_9754
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_61_9756 = torch.Tensor.reshape(transpose_164_9755, shape=(8, 1, 2048, 64))
        del transpose_164_9755
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_61_64305, reshape_61_64306, reshape_61_64307 = nnscaler.runtime.function.multiref(reshape_61_9756, times=3)
        del reshape_61_9756
        unsqueeze_41_76264 = nnscaler.runtime.adapter.chunk(unsqueeze_41_64293, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_41_64293
        # created at IRAdapterGener:local_consumer_multiref
        reshape_60_85019, reshape_60_85023, reshape_60_85027 = nnscaler.runtime.function.multiref(reshape_60_50809, times=3)
        del reshape_60_50809
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_100_50905 = torch.mul(reshape_60_85019, unsqueeze_41_76264)
        del unsqueeze_41_76264, reshape_60_85019
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_726_50953 = nnscaler.runtime.function.fullslice(reshape_60_85023, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_60_85023
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_728_50977 = nnscaler.runtime.function.fullslice(reshape_60_85027, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_60_85027
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_40_51001 = _operator.neg(getitem_728_50977)
        del getitem_728_50977
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_80_51041 = nnscaler.runtime.function.cat(neg_40_51001, getitem_726_50953, dim=-1)
        del getitem_726_50953, neg_40_51001
        unsqueeze_42_76336 = nnscaler.runtime.adapter.chunk(unsqueeze_42_64297, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_42_64297
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_101_51073 = torch.mul(cat_80_51041, unsqueeze_42_76336)
        del cat_80_51041, unsqueeze_42_76336
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_100_51121 = torch.add(mul_100_50905, mul_101_51073, alpha=1)
        del mul_100_50905, mul_101_51073
        unsqueeze_41_76368 = nnscaler.runtime.adapter.chunk(unsqueeze_41_64294, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_41_64294
        reshape_61_76360 = nnscaler.runtime.adapter.nn.split_allgather(reshape_61_64307, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_61_64307
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_102_51161 = torch.mul(reshape_61_76360, unsqueeze_41_76368)
        del unsqueeze_41_76368, reshape_61_76360
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_730_9765 = nnscaler.runtime.function.fullslice(reshape_61_64305, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_61_64305
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_732_9766 = nnscaler.runtime.function.fullslice(reshape_61_64306, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_61_64306
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_41_9767 = _operator.neg(getitem_732_9766)
        del getitem_732_9766
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_81_9768 = nnscaler.runtime.function.cat(neg_41_9767, getitem_730_9765, dim=-1)
        del getitem_730_9765, neg_41_9767
        cat_81_51265 = nnscaler.runtime.adapter.nn.split_allgather(cat_81_9768, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_81_9768
        unsqueeze_42_76392 = nnscaler.runtime.adapter.chunk(unsqueeze_42_64298, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_42_64298
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_103_51273 = torch.mul(cat_81_51265, unsqueeze_42_76392)
        del cat_81_51265, unsqueeze_42_76392
        mul_102_9764 = nnscaler.runtime.adapter.nn.allgather_split(mul_102_51161, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_102_51161
        mul_103_9769 = nnscaler.runtime.adapter.nn.allgather_split(mul_103_51273, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_103_51273
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_101_9770 = torch.add(mul_102_9764, mul_103_9769, alpha=1)
        del mul_102_9764, mul_103_9769
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_82_51321 = nnscaler.runtime.function.cat(split_60_50353, add_100_51121, dim=-1)
        del split_60_50353, add_100_51121
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_20_9772 = torch.Tensor.expand(add_101_9770, size=[-1, 16, -1, -1])
        del add_101_9770
        split_62_50657 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_62_50641, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_62_50641
        expand_20_51361 = nnscaler.runtime.adapter.nn.split_allgather(expand_20_9772, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_20_9772
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_83_51369 = nnscaler.runtime.function.cat(split_62_50657, expand_20_51361, dim=-1)
        del split_62_50657, expand_20_51361
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_20_51385 = torch.nn.functional.pad(split_62_50649, pad=[0, 64], mode='constant', value=0.0)
        del split_62_50649
        cat_82_51313 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_82_51321, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_82_51321
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_165_51417 = torch.transpose(cat_82_51313, dim0=1, dim1=2)
        del cat_82_51313
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_166_51457 = torch.transpose(cat_83_51369, dim0=1, dim1=2)
        del cat_83_51369
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_167_51489 = torch.transpose(pad_20_51385, dim0=1, dim1=2)
        del pad_20_51385
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_20_self_attn_training_7903 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_20_7904 = 0.0 if model_model_layers_20_self_attn_training_7903 else 0.0
        transpose_167_51497 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_167_51489, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_167_51489
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_20_51537 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_165_51417, transpose_166_51457, transpose_167_51497, dropout=ifexpr_20_7904, causal=True, attention_mask=None, query_length=2048)
        del transpose_165_51417, transpose_166_51457, transpose_167_51497
        nnscaler_flash_attention_forward_20_9778 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_20_51537, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_20_51537
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_733_9779 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_20_9778, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_20_9778
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_62_9780 = torch.Tensor.reshape(getitem_733_9779, shape=(8, 2048, 2048))
        del getitem_733_9779
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_20_9781 = torch.Tensor.contiguous(reshape_62_9780)
        del reshape_62_9780
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_143_9783 = torch.nn.functional.linear(contiguous_20_9781, self.model_model_layers_20_self_attn_o_proj_weight_9782, bias=None)
        del contiguous_20_9781
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_102_9784 = torch.add(add_99_84928, linear_143_9783, alpha=1)
        del add_99_84928, linear_143_9783
        # created at IRAdapterGener:local_consumer_multiref
        add_102_85092, add_102_85096 = nnscaler.runtime.function.multiref(add_102_9784, times=2)
        del add_102_9784
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_62_9786 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_102_85092, self.model_model_layers_20_post_attention_layernorm_weight_9785, (2048,), 1e-06)
        del add_102_85092
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_62_64319, fused_rms_norm_affine_62_64320, fused_rms_norm_affine_62_64321, fused_rms_norm_affine_62_64322 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_62_9786, times=4)
        del fused_rms_norm_affine_62_9786
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_20_mlp_gate_training_7906 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_19_9788, moe_route_19_9789, moe_route_19_9790 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_62_64319, self.model_model_layers_20_mlp_gate_weight_9787, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_20_mlp_gate_training_7906, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_62_64319
        moe_route_19_9789 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_19_9789, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_19_9790 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_19_9790, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_62_64320 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_62_64320, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_19_51737 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_62_64320, moe_route_19_9788, moe_route_19_9789, moe_route_19_9790, self.model_model_layers_20_mlp_gate_projs_51713, self.model_model_layers_20_mlp_up_projs_51721, self.model_model_layers_20_mlp_down_projs_51729, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_62_64320, moe_route_19_9788, moe_route_19_9789, moe_route_19_9790
        fused_rms_norm_affine_62_76552 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_62_64321, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_62_64321
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_144_51801 = torch.nn.functional.linear(fused_rms_norm_affine_62_76552, self.model_model_layers_20_mlp_shared_experts_gate_proj_weight_9795, bias=None)
        del fused_rms_norm_affine_62_76552
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_20_51857 = torch.nn.functional.silu(linear_144_51801, inplace=False)
        del linear_144_51801
        fused_rms_norm_affine_62_76592 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_62_64322, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_62_64322
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_145_51881 = torch.nn.functional.linear(fused_rms_norm_affine_62_76592, self.model_model_layers_20_mlp_shared_experts_up_proj_weight_9798, bias=None)
        del fused_rms_norm_affine_62_76592
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_104_51929 = torch.mul(silu_20_51857, linear_145_51881)
        del silu_20_51857, linear_145_51881
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_146_51953 = torch.nn.functional.linear(mul_104_51929, self.model_model_layers_20_mlp_shared_experts_down_proj_weight_9801, bias=None)
        del mul_104_51929
        nnscaler_moe_gmm_19_9794 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_19_51737, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_19_51737
        linear_146_9802 = nnscaler.runtime.adapter.nn.allgather_split(linear_146_51953, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_146_51953
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_103_9803 = torch.add(nnscaler_moe_gmm_19_9794, linear_146_9802, alpha=1)
        del nnscaler_moe_gmm_19_9794, linear_146_9802
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_104_9804 = torch.add(add_102_85096, add_103_9803, alpha=1)
        del add_102_85096, add_103_9803
        # created at IRAdapterGener:local_consumer_multiref
        add_104_85156, add_104_85160 = nnscaler.runtime.function.multiref(add_104_9804, times=2)
        del add_104_9804
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_63_9806 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_104_85156, self.model_model_layers_21_input_layernorm_weight_9805, (2048,), 1e-06)
        del add_104_85156
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_63_64331, fused_rms_norm_affine_63_64332 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_63_9806, times=2)
        del fused_rms_norm_affine_63_9806
        fused_rms_norm_affine_63_76648 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_63_64331, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_63_64331
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_147_52081 = torch.nn.functional.linear(fused_rms_norm_affine_63_76648, self.model_model_layers_21_self_attn_q_proj_weight_9807, bias=None)
        del fused_rms_norm_affine_63_76648
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_105_52137 = torch.Tensor.view(linear_147_52081, size=(8, 256, 16, 192))
        del linear_147_52081
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_168_52161 = torch.transpose(view_105_52137, dim0=1, dim1=2)
        del view_105_52137
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_63_52225, split_63_52233 = torch.functional.split(transpose_168_52161, split_size_or_sections=[128, 64], dim=-1)
        del transpose_168_52161
        fused_rms_norm_affine_63_76712 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_63_64332, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_63_64332
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_148_52265 = torch.nn.functional.linear(fused_rms_norm_affine_63_76712, self.model_model_layers_21_self_attn_kv_a_proj_with_mqa_weight_52257, bias=None)
        del fused_rms_norm_affine_63_76712
        linear_148_9814 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_148_52265, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_148_52265
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_64_9815, split_64_9816 = torch.functional.split(linear_148_9814, split_size_or_sections=[512, 64], dim=-1)
        del linear_148_9814
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_106_9817 = torch.Tensor.view(split_64_9816, size=(8, 2048, 1, 64))
        del split_64_9816
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_169_9818 = torch.transpose(view_106_9817, dim0=1, dim1=2)
        del view_106_9817
        split_64_52305 = nnscaler.runtime.adapter.nn.split_allgather(split_64_9815, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_64_9815
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_64_52385 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_64_52305, self.model_model_layers_21_self_attn_kv_a_layernorm_weight_9819, (512,), 1e-06)
        del split_64_52305
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_149_52401 = torch.nn.functional.linear(fused_rms_norm_affine_64_52385, self.model_model_layers_21_self_attn_kv_b_proj_weight_9821, bias=None)
        del fused_rms_norm_affine_64_52385
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_107_52457 = torch.Tensor.view(linear_149_52401, size=(8, 256, 16, 256))
        del linear_149_52401
        view_107_52449 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_107_52457, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_107_52457
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_170_52473 = torch.transpose(view_107_52449, dim0=1, dim1=2)
        del view_107_52449
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_65_52513, split_65_52521 = torch.functional.split(transpose_170_52473, split_size_or_sections=[128, 128], dim=-1)
        del transpose_170_52473
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_748_9828 = nnscaler.runtime.function.fullslice(self.model_model_layers_21_self_attn_rotary_emb_cos_cached_9827, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_42_9829 = torch.Tensor.to(getitem_748_9828, dtype=torch.bfloat16)
        del getitem_748_9828
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_749_9831 = nnscaler.runtime.function.fullslice(self.model_model_layers_21_self_attn_rotary_emb_sin_cached_9830, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_43_9832 = torch.Tensor.to(getitem_749_9831, dtype=torch.bfloat16)
        del getitem_749_9831
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_750_9833 = nnscaler.runtime.function.fullslice(to_42_9829, unsqueeze_8005)
        del to_42_9829
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_43_9834 = torch.unsqueeze(getitem_750_9833, dim=1)
        del getitem_750_9833
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_43_64337, unsqueeze_43_64338 = nnscaler.runtime.function.multiref(unsqueeze_43_9834, times=2)
        del unsqueeze_43_9834
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_751_9835 = nnscaler.runtime.function.fullslice(to_43_9832, unsqueeze_8005)
        del to_43_9832
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_44_9836 = torch.unsqueeze(getitem_751_9835, dim=1)
        del getitem_751_9835
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_44_64341, unsqueeze_44_64342 = nnscaler.runtime.function.multiref(unsqueeze_44_9836, times=2)
        del unsqueeze_44_9836
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_108_52609 = torch.Tensor.view(split_63_52233, size=(8, 16, 256, 32, 2))
        del split_63_52233
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_171_52649 = torch.transpose(view_108_52609, dim0=4, dim1=3)
        del view_108_52609
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_63_52681 = torch.Tensor.reshape(transpose_171_52649, shape=(8, 16, 256, 64))
        del transpose_171_52649
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_109_9840 = torch.Tensor.view(transpose_169_9818, size=(8, 1, 2048, 32, 2))
        del transpose_169_9818
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_172_9841 = torch.transpose(view_109_9840, dim0=4, dim1=3)
        del view_109_9840
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_64_9842 = torch.Tensor.reshape(transpose_172_9841, shape=(8, 1, 2048, 64))
        del transpose_172_9841
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_64_64349, reshape_64_64350, reshape_64_64351 = nnscaler.runtime.function.multiref(reshape_64_9842, times=3)
        del reshape_64_9842
        unsqueeze_43_76840 = nnscaler.runtime.adapter.chunk(unsqueeze_43_64337, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_43_64337
        # created at IRAdapterGener:local_consumer_multiref
        reshape_63_85251, reshape_63_85255, reshape_63_85259 = nnscaler.runtime.function.multiref(reshape_63_52681, times=3)
        del reshape_63_52681
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_105_52777 = torch.mul(reshape_63_85251, unsqueeze_43_76840)
        del unsqueeze_43_76840, reshape_63_85251
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_761_52825 = nnscaler.runtime.function.fullslice(reshape_63_85255, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_63_85255
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_763_52849 = nnscaler.runtime.function.fullslice(reshape_63_85259, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_63_85259
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_42_52873 = _operator.neg(getitem_763_52849)
        del getitem_763_52849
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_84_52913 = nnscaler.runtime.function.cat(neg_42_52873, getitem_761_52825, dim=-1)
        del getitem_761_52825, neg_42_52873
        unsqueeze_44_76912 = nnscaler.runtime.adapter.chunk(unsqueeze_44_64341, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_44_64341
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_106_52945 = torch.mul(cat_84_52913, unsqueeze_44_76912)
        del cat_84_52913, unsqueeze_44_76912
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_105_52993 = torch.add(mul_105_52777, mul_106_52945, alpha=1)
        del mul_105_52777, mul_106_52945
        unsqueeze_43_76944 = nnscaler.runtime.adapter.chunk(unsqueeze_43_64338, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_43_64338
        reshape_64_76936 = nnscaler.runtime.adapter.nn.split_allgather(reshape_64_64351, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_64_64351
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_107_53033 = torch.mul(reshape_64_76936, unsqueeze_43_76944)
        del unsqueeze_43_76944, reshape_64_76936
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_765_9851 = nnscaler.runtime.function.fullslice(reshape_64_64349, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_64_64349
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_767_9852 = nnscaler.runtime.function.fullslice(reshape_64_64350, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_64_64350
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_43_9853 = _operator.neg(getitem_767_9852)
        del getitem_767_9852
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_85_9854 = nnscaler.runtime.function.cat(neg_43_9853, getitem_765_9851, dim=-1)
        del getitem_765_9851, neg_43_9853
        cat_85_53137 = nnscaler.runtime.adapter.nn.split_allgather(cat_85_9854, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_85_9854
        unsqueeze_44_76968 = nnscaler.runtime.adapter.chunk(unsqueeze_44_64342, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_44_64342
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_108_53145 = torch.mul(cat_85_53137, unsqueeze_44_76968)
        del cat_85_53137, unsqueeze_44_76968
        mul_107_9850 = nnscaler.runtime.adapter.nn.allgather_split(mul_107_53033, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_107_53033
        mul_108_9855 = nnscaler.runtime.adapter.nn.allgather_split(mul_108_53145, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_108_53145
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_106_9856 = torch.add(mul_107_9850, mul_108_9855, alpha=1)
        del mul_107_9850, mul_108_9855
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_86_53193 = nnscaler.runtime.function.cat(split_63_52225, add_105_52993, dim=-1)
        del split_63_52225, add_105_52993
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_21_9858 = torch.Tensor.expand(add_106_9856, size=[-1, 16, -1, -1])
        del add_106_9856
        split_65_52529 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_65_52513, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_65_52513
        expand_21_53233 = nnscaler.runtime.adapter.nn.split_allgather(expand_21_9858, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_21_9858
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_87_53241 = nnscaler.runtime.function.cat(split_65_52529, expand_21_53233, dim=-1)
        del split_65_52529, expand_21_53233
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_21_53257 = torch.nn.functional.pad(split_65_52521, pad=[0, 64], mode='constant', value=0.0)
        del split_65_52521
        cat_86_53185 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_86_53193, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_86_53193
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_173_53289 = torch.transpose(cat_86_53185, dim0=1, dim1=2)
        del cat_86_53185
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_174_53329 = torch.transpose(cat_87_53241, dim0=1, dim1=2)
        del cat_87_53241
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_175_53361 = torch.transpose(pad_21_53257, dim0=1, dim1=2)
        del pad_21_53257
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_21_self_attn_training_7918 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_21_7919 = 0.0 if model_model_layers_21_self_attn_training_7918 else 0.0
        transpose_175_53369 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_175_53361, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_175_53361
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_21_53409 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_173_53289, transpose_174_53329, transpose_175_53369, dropout=ifexpr_21_7919, causal=True, attention_mask=None, query_length=2048)
        del transpose_173_53289, transpose_174_53329, transpose_175_53369
        nnscaler_flash_attention_forward_21_9864 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_21_53409, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_21_53409
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_768_9865 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_21_9864, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_21_9864
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_65_9866 = torch.Tensor.reshape(getitem_768_9865, shape=(8, 2048, 2048))
        del getitem_768_9865
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_21_9867 = torch.Tensor.contiguous(reshape_65_9866)
        del reshape_65_9866
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_150_9869 = torch.nn.functional.linear(contiguous_21_9867, self.model_model_layers_21_self_attn_o_proj_weight_9868, bias=None)
        del contiguous_21_9867
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_107_9870 = torch.add(add_104_85160, linear_150_9869, alpha=1)
        del add_104_85160, linear_150_9869
        # created at IRAdapterGener:local_consumer_multiref
        add_107_85324, add_107_85328 = nnscaler.runtime.function.multiref(add_107_9870, times=2)
        del add_107_9870
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_65_9872 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_107_85324, self.model_model_layers_21_post_attention_layernorm_weight_9871, (2048,), 1e-06)
        del add_107_85324
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_65_64363, fused_rms_norm_affine_65_64364, fused_rms_norm_affine_65_64365, fused_rms_norm_affine_65_64366 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_65_9872, times=4)
        del fused_rms_norm_affine_65_9872
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_21_mlp_gate_training_7921 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_20_9874, moe_route_20_9875, moe_route_20_9876 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_65_64363, self.model_model_layers_21_mlp_gate_weight_9873, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_21_mlp_gate_training_7921, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_65_64363
        moe_route_20_9875 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_20_9875, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_20_9876 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_20_9876, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_65_64364 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_65_64364, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_20_53609 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_65_64364, moe_route_20_9874, moe_route_20_9875, moe_route_20_9876, self.model_model_layers_21_mlp_gate_projs_53585, self.model_model_layers_21_mlp_up_projs_53593, self.model_model_layers_21_mlp_down_projs_53601, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_65_64364, moe_route_20_9874, moe_route_20_9875, moe_route_20_9876
        fused_rms_norm_affine_65_77128 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_65_64365, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_65_64365
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_151_53673 = torch.nn.functional.linear(fused_rms_norm_affine_65_77128, self.model_model_layers_21_mlp_shared_experts_gate_proj_weight_9881, bias=None)
        del fused_rms_norm_affine_65_77128
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_21_53729 = torch.nn.functional.silu(linear_151_53673, inplace=False)
        del linear_151_53673
        fused_rms_norm_affine_65_77168 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_65_64366, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_65_64366
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_152_53753 = torch.nn.functional.linear(fused_rms_norm_affine_65_77168, self.model_model_layers_21_mlp_shared_experts_up_proj_weight_9884, bias=None)
        del fused_rms_norm_affine_65_77168
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_109_53801 = torch.mul(silu_21_53729, linear_152_53753)
        del silu_21_53729, linear_152_53753
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_153_53825 = torch.nn.functional.linear(mul_109_53801, self.model_model_layers_21_mlp_shared_experts_down_proj_weight_9887, bias=None)
        del mul_109_53801
        nnscaler_moe_gmm_20_9880 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_20_53609, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_20_53609
        linear_153_9888 = nnscaler.runtime.adapter.nn.allgather_split(linear_153_53825, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_153_53825
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_108_9889 = torch.add(nnscaler_moe_gmm_20_9880, linear_153_9888, alpha=1)
        del nnscaler_moe_gmm_20_9880, linear_153_9888
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_109_9890 = torch.add(add_107_85328, add_108_9889, alpha=1)
        del add_107_85328, add_108_9889
        # created at IRAdapterGener:local_consumer_multiref
        add_109_85388, add_109_85392 = nnscaler.runtime.function.multiref(add_109_9890, times=2)
        del add_109_9890
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_66_9892 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_109_85388, self.model_model_layers_22_input_layernorm_weight_9891, (2048,), 1e-06)
        del add_109_85388
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_66_64375, fused_rms_norm_affine_66_64376 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_66_9892, times=2)
        del fused_rms_norm_affine_66_9892
        fused_rms_norm_affine_66_77224 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_66_64375, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_66_64375
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_154_53953 = torch.nn.functional.linear(fused_rms_norm_affine_66_77224, self.model_model_layers_22_self_attn_q_proj_weight_9893, bias=None)
        del fused_rms_norm_affine_66_77224
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_110_54009 = torch.Tensor.view(linear_154_53953, size=(8, 256, 16, 192))
        del linear_154_53953
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_176_54033 = torch.transpose(view_110_54009, dim0=1, dim1=2)
        del view_110_54009
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_66_54097, split_66_54105 = torch.functional.split(transpose_176_54033, split_size_or_sections=[128, 64], dim=-1)
        del transpose_176_54033
        fused_rms_norm_affine_66_77288 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_66_64376, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_66_64376
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_155_54137 = torch.nn.functional.linear(fused_rms_norm_affine_66_77288, self.model_model_layers_22_self_attn_kv_a_proj_with_mqa_weight_54129, bias=None)
        del fused_rms_norm_affine_66_77288
        linear_155_9900 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_155_54137, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_155_54137
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_67_9901, split_67_9902 = torch.functional.split(linear_155_9900, split_size_or_sections=[512, 64], dim=-1)
        del linear_155_9900
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_111_9903 = torch.Tensor.view(split_67_9902, size=(8, 2048, 1, 64))
        del split_67_9902
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_177_9904 = torch.transpose(view_111_9903, dim0=1, dim1=2)
        del view_111_9903
        split_67_54177 = nnscaler.runtime.adapter.nn.split_allgather(split_67_9901, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_67_9901
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_67_54257 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_67_54177, self.model_model_layers_22_self_attn_kv_a_layernorm_weight_9905, (512,), 1e-06)
        del split_67_54177
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_156_54273 = torch.nn.functional.linear(fused_rms_norm_affine_67_54257, self.model_model_layers_22_self_attn_kv_b_proj_weight_9907, bias=None)
        del fused_rms_norm_affine_67_54257
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_112_54329 = torch.Tensor.view(linear_156_54273, size=(8, 256, 16, 256))
        del linear_156_54273
        view_112_54321 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_112_54329, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_112_54329
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_178_54345 = torch.transpose(view_112_54321, dim0=1, dim1=2)
        del view_112_54321
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_68_54385, split_68_54393 = torch.functional.split(transpose_178_54345, split_size_or_sections=[128, 128], dim=-1)
        del transpose_178_54345
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_783_9914 = nnscaler.runtime.function.fullslice(self.model_model_layers_22_self_attn_rotary_emb_cos_cached_9913, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_44_9915 = torch.Tensor.to(getitem_783_9914, dtype=torch.bfloat16)
        del getitem_783_9914
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_784_9917 = nnscaler.runtime.function.fullslice(self.model_model_layers_22_self_attn_rotary_emb_sin_cached_9916, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_45_9918 = torch.Tensor.to(getitem_784_9917, dtype=torch.bfloat16)
        del getitem_784_9917
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_785_9919 = nnscaler.runtime.function.fullslice(to_44_9915, unsqueeze_8005)
        del to_44_9915
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_45_9920 = torch.unsqueeze(getitem_785_9919, dim=1)
        del getitem_785_9919
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_45_64381, unsqueeze_45_64382 = nnscaler.runtime.function.multiref(unsqueeze_45_9920, times=2)
        del unsqueeze_45_9920
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_786_9921 = nnscaler.runtime.function.fullslice(to_45_9918, unsqueeze_8005)
        del to_45_9918
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_46_9922 = torch.unsqueeze(getitem_786_9921, dim=1)
        del getitem_786_9921
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_46_64385, unsqueeze_46_64386 = nnscaler.runtime.function.multiref(unsqueeze_46_9922, times=2)
        del unsqueeze_46_9922
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_113_54481 = torch.Tensor.view(split_66_54105, size=(8, 16, 256, 32, 2))
        del split_66_54105
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_179_54521 = torch.transpose(view_113_54481, dim0=4, dim1=3)
        del view_113_54481
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_66_54553 = torch.Tensor.reshape(transpose_179_54521, shape=(8, 16, 256, 64))
        del transpose_179_54521
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_114_9926 = torch.Tensor.view(transpose_177_9904, size=(8, 1, 2048, 32, 2))
        del transpose_177_9904
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_180_9927 = torch.transpose(view_114_9926, dim0=4, dim1=3)
        del view_114_9926
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_67_9928 = torch.Tensor.reshape(transpose_180_9927, shape=(8, 1, 2048, 64))
        del transpose_180_9927
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_67_64393, reshape_67_64394, reshape_67_64395 = nnscaler.runtime.function.multiref(reshape_67_9928, times=3)
        del reshape_67_9928
        unsqueeze_45_77416 = nnscaler.runtime.adapter.chunk(unsqueeze_45_64381, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_45_64381
        # created at IRAdapterGener:local_consumer_multiref
        reshape_66_85483, reshape_66_85487, reshape_66_85491 = nnscaler.runtime.function.multiref(reshape_66_54553, times=3)
        del reshape_66_54553
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_110_54649 = torch.mul(reshape_66_85483, unsqueeze_45_77416)
        del unsqueeze_45_77416, reshape_66_85483
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_796_54697 = nnscaler.runtime.function.fullslice(reshape_66_85487, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_66_85487
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_798_54721 = nnscaler.runtime.function.fullslice(reshape_66_85491, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_66_85491
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_44_54745 = _operator.neg(getitem_798_54721)
        del getitem_798_54721
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_88_54785 = nnscaler.runtime.function.cat(neg_44_54745, getitem_796_54697, dim=-1)
        del getitem_796_54697, neg_44_54745
        unsqueeze_46_77488 = nnscaler.runtime.adapter.chunk(unsqueeze_46_64385, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_46_64385
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_111_54817 = torch.mul(cat_88_54785, unsqueeze_46_77488)
        del cat_88_54785, unsqueeze_46_77488
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_110_54865 = torch.add(mul_110_54649, mul_111_54817, alpha=1)
        del mul_110_54649, mul_111_54817
        unsqueeze_45_77520 = nnscaler.runtime.adapter.chunk(unsqueeze_45_64382, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_45_64382
        reshape_67_77512 = nnscaler.runtime.adapter.nn.split_allgather(reshape_67_64395, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_67_64395
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_112_54905 = torch.mul(reshape_67_77512, unsqueeze_45_77520)
        del unsqueeze_45_77520, reshape_67_77512
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_800_9937 = nnscaler.runtime.function.fullslice(reshape_67_64393, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_67_64393
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_802_9938 = nnscaler.runtime.function.fullslice(reshape_67_64394, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_67_64394
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_45_9939 = _operator.neg(getitem_802_9938)
        del getitem_802_9938
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_89_9940 = nnscaler.runtime.function.cat(neg_45_9939, getitem_800_9937, dim=-1)
        del getitem_800_9937, neg_45_9939
        cat_89_55009 = nnscaler.runtime.adapter.nn.split_allgather(cat_89_9940, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_89_9940
        unsqueeze_46_77544 = nnscaler.runtime.adapter.chunk(unsqueeze_46_64386, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_46_64386
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_113_55017 = torch.mul(cat_89_55009, unsqueeze_46_77544)
        del cat_89_55009, unsqueeze_46_77544
        mul_112_9936 = nnscaler.runtime.adapter.nn.allgather_split(mul_112_54905, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_112_54905
        mul_113_9941 = nnscaler.runtime.adapter.nn.allgather_split(mul_113_55017, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_113_55017
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_111_9942 = torch.add(mul_112_9936, mul_113_9941, alpha=1)
        del mul_112_9936, mul_113_9941
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_90_55065 = nnscaler.runtime.function.cat(split_66_54097, add_110_54865, dim=-1)
        del split_66_54097, add_110_54865
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_22_9944 = torch.Tensor.expand(add_111_9942, size=[-1, 16, -1, -1])
        del add_111_9942
        split_68_54401 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_68_54385, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_68_54385
        expand_22_55105 = nnscaler.runtime.adapter.nn.split_allgather(expand_22_9944, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_22_9944
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_91_55113 = nnscaler.runtime.function.cat(split_68_54401, expand_22_55105, dim=-1)
        del split_68_54401, expand_22_55105
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_22_55129 = torch.nn.functional.pad(split_68_54393, pad=[0, 64], mode='constant', value=0.0)
        del split_68_54393
        cat_90_55057 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_90_55065, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_90_55065
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_181_55161 = torch.transpose(cat_90_55057, dim0=1, dim1=2)
        del cat_90_55057
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_182_55201 = torch.transpose(cat_91_55113, dim0=1, dim1=2)
        del cat_91_55113
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_183_55233 = torch.transpose(pad_22_55129, dim0=1, dim1=2)
        del pad_22_55129
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_22_self_attn_training_7933 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_22_7934 = 0.0 if model_model_layers_22_self_attn_training_7933 else 0.0
        transpose_183_55241 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_183_55233, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_183_55233
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_22_55281 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_181_55161, transpose_182_55201, transpose_183_55241, dropout=ifexpr_22_7934, causal=True, attention_mask=None, query_length=2048)
        del transpose_181_55161, transpose_182_55201, transpose_183_55241
        nnscaler_flash_attention_forward_22_9950 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_22_55281, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_22_55281
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_803_9951 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_22_9950, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_22_9950
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_68_9952 = torch.Tensor.reshape(getitem_803_9951, shape=(8, 2048, 2048))
        del getitem_803_9951
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_22_9953 = torch.Tensor.contiguous(reshape_68_9952)
        del reshape_68_9952
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_157_9955 = torch.nn.functional.linear(contiguous_22_9953, self.model_model_layers_22_self_attn_o_proj_weight_9954, bias=None)
        del contiguous_22_9953
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_112_9956 = torch.add(add_109_85392, linear_157_9955, alpha=1)
        del add_109_85392, linear_157_9955
        # created at IRAdapterGener:local_consumer_multiref
        add_112_85556, add_112_85560 = nnscaler.runtime.function.multiref(add_112_9956, times=2)
        del add_112_9956
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_68_9958 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_112_85556, self.model_model_layers_22_post_attention_layernorm_weight_9957, (2048,), 1e-06)
        del add_112_85556
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_68_64407, fused_rms_norm_affine_68_64408, fused_rms_norm_affine_68_64409, fused_rms_norm_affine_68_64410 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_68_9958, times=4)
        del fused_rms_norm_affine_68_9958
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_22_mlp_gate_training_7936 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_21_9960, moe_route_21_9961, moe_route_21_9962 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_68_64407, self.model_model_layers_22_mlp_gate_weight_9959, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_22_mlp_gate_training_7936, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_68_64407
        moe_route_21_9961 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_21_9961, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_21_9962 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_21_9962, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_68_64408 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_68_64408, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_21_55481 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_68_64408, moe_route_21_9960, moe_route_21_9961, moe_route_21_9962, self.model_model_layers_22_mlp_gate_projs_55457, self.model_model_layers_22_mlp_up_projs_55465, self.model_model_layers_22_mlp_down_projs_55473, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_68_64408, moe_route_21_9960, moe_route_21_9961, moe_route_21_9962
        fused_rms_norm_affine_68_77704 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_68_64409, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_68_64409
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_158_55545 = torch.nn.functional.linear(fused_rms_norm_affine_68_77704, self.model_model_layers_22_mlp_shared_experts_gate_proj_weight_9967, bias=None)
        del fused_rms_norm_affine_68_77704
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_22_55601 = torch.nn.functional.silu(linear_158_55545, inplace=False)
        del linear_158_55545
        fused_rms_norm_affine_68_77744 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_68_64410, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_68_64410
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_159_55625 = torch.nn.functional.linear(fused_rms_norm_affine_68_77744, self.model_model_layers_22_mlp_shared_experts_up_proj_weight_9970, bias=None)
        del fused_rms_norm_affine_68_77744
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_114_55673 = torch.mul(silu_22_55601, linear_159_55625)
        del silu_22_55601, linear_159_55625
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_160_55697 = torch.nn.functional.linear(mul_114_55673, self.model_model_layers_22_mlp_shared_experts_down_proj_weight_9973, bias=None)
        del mul_114_55673
        nnscaler_moe_gmm_21_9966 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_21_55481, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_21_55481
        linear_160_9974 = nnscaler.runtime.adapter.nn.allgather_split(linear_160_55697, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_160_55697
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_113_9975 = torch.add(nnscaler_moe_gmm_21_9966, linear_160_9974, alpha=1)
        del nnscaler_moe_gmm_21_9966, linear_160_9974
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_114_9976 = torch.add(add_112_85560, add_113_9975, alpha=1)
        del add_112_85560, add_113_9975
        # created at IRAdapterGener:local_consumer_multiref
        add_114_85620, add_114_85624 = nnscaler.runtime.function.multiref(add_114_9976, times=2)
        del add_114_9976
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_69_9978 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_114_85620, self.model_model_layers_23_input_layernorm_weight_9977, (2048,), 1e-06)
        del add_114_85620
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_69_64419, fused_rms_norm_affine_69_64420 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_69_9978, times=2)
        del fused_rms_norm_affine_69_9978
        fused_rms_norm_affine_69_77800 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_69_64419, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_69_64419
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_161_55825 = torch.nn.functional.linear(fused_rms_norm_affine_69_77800, self.model_model_layers_23_self_attn_q_proj_weight_9979, bias=None)
        del fused_rms_norm_affine_69_77800
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_115_55881 = torch.Tensor.view(linear_161_55825, size=(8, 256, 16, 192))
        del linear_161_55825
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_184_55905 = torch.transpose(view_115_55881, dim0=1, dim1=2)
        del view_115_55881
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_69_55969, split_69_55977 = torch.functional.split(transpose_184_55905, split_size_or_sections=[128, 64], dim=-1)
        del transpose_184_55905
        fused_rms_norm_affine_69_77864 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_69_64420, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_69_64420
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_162_56009 = torch.nn.functional.linear(fused_rms_norm_affine_69_77864, self.model_model_layers_23_self_attn_kv_a_proj_with_mqa_weight_56001, bias=None)
        del fused_rms_norm_affine_69_77864
        linear_162_9986 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_162_56009, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_162_56009
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_70_9987, split_70_9988 = torch.functional.split(linear_162_9986, split_size_or_sections=[512, 64], dim=-1)
        del linear_162_9986
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_116_9989 = torch.Tensor.view(split_70_9988, size=(8, 2048, 1, 64))
        del split_70_9988
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_185_9990 = torch.transpose(view_116_9989, dim0=1, dim1=2)
        del view_116_9989
        split_70_56049 = nnscaler.runtime.adapter.nn.split_allgather(split_70_9987, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_70_9987
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_70_56129 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_70_56049, self.model_model_layers_23_self_attn_kv_a_layernorm_weight_9991, (512,), 1e-06)
        del split_70_56049
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_163_56145 = torch.nn.functional.linear(fused_rms_norm_affine_70_56129, self.model_model_layers_23_self_attn_kv_b_proj_weight_9993, bias=None)
        del fused_rms_norm_affine_70_56129
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_117_56201 = torch.Tensor.view(linear_163_56145, size=(8, 256, 16, 256))
        del linear_163_56145
        view_117_56193 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_117_56201, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_117_56201
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_186_56217 = torch.transpose(view_117_56193, dim0=1, dim1=2)
        del view_117_56193
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_71_56257, split_71_56265 = torch.functional.split(transpose_186_56217, split_size_or_sections=[128, 128], dim=-1)
        del transpose_186_56217
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_818_10000 = nnscaler.runtime.function.fullslice(self.model_model_layers_23_self_attn_rotary_emb_cos_cached_9999, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_46_10001 = torch.Tensor.to(getitem_818_10000, dtype=torch.bfloat16)
        del getitem_818_10000
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_819_10003 = nnscaler.runtime.function.fullslice(self.model_model_layers_23_self_attn_rotary_emb_sin_cached_10002, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_47_10004 = torch.Tensor.to(getitem_819_10003, dtype=torch.bfloat16)
        del getitem_819_10003
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_820_10005 = nnscaler.runtime.function.fullslice(to_46_10001, unsqueeze_8005)
        del to_46_10001
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_47_10006 = torch.unsqueeze(getitem_820_10005, dim=1)
        del getitem_820_10005
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_47_64425, unsqueeze_47_64426 = nnscaler.runtime.function.multiref(unsqueeze_47_10006, times=2)
        del unsqueeze_47_10006
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_821_10007 = nnscaler.runtime.function.fullslice(to_47_10004, unsqueeze_8005)
        del to_47_10004
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_48_10008 = torch.unsqueeze(getitem_821_10007, dim=1)
        del getitem_821_10007
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_48_64429, unsqueeze_48_64430 = nnscaler.runtime.function.multiref(unsqueeze_48_10008, times=2)
        del unsqueeze_48_10008
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_118_56353 = torch.Tensor.view(split_69_55977, size=(8, 16, 256, 32, 2))
        del split_69_55977
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_187_56393 = torch.transpose(view_118_56353, dim0=4, dim1=3)
        del view_118_56353
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_69_56425 = torch.Tensor.reshape(transpose_187_56393, shape=(8, 16, 256, 64))
        del transpose_187_56393
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_119_10012 = torch.Tensor.view(transpose_185_9990, size=(8, 1, 2048, 32, 2))
        del transpose_185_9990
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_188_10013 = torch.transpose(view_119_10012, dim0=4, dim1=3)
        del view_119_10012
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_70_10014 = torch.Tensor.reshape(transpose_188_10013, shape=(8, 1, 2048, 64))
        del transpose_188_10013
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_70_64437, reshape_70_64438, reshape_70_64439 = nnscaler.runtime.function.multiref(reshape_70_10014, times=3)
        del reshape_70_10014
        unsqueeze_47_77992 = nnscaler.runtime.adapter.chunk(unsqueeze_47_64425, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_47_64425
        # created at IRAdapterGener:local_consumer_multiref
        reshape_69_85715, reshape_69_85719, reshape_69_85723 = nnscaler.runtime.function.multiref(reshape_69_56425, times=3)
        del reshape_69_56425
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_115_56521 = torch.mul(reshape_69_85715, unsqueeze_47_77992)
        del unsqueeze_47_77992, reshape_69_85715
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_831_56569 = nnscaler.runtime.function.fullslice(reshape_69_85719, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_69_85719
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_833_56593 = nnscaler.runtime.function.fullslice(reshape_69_85723, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_69_85723
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_46_56617 = _operator.neg(getitem_833_56593)
        del getitem_833_56593
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_92_56657 = nnscaler.runtime.function.cat(neg_46_56617, getitem_831_56569, dim=-1)
        del getitem_831_56569, neg_46_56617
        unsqueeze_48_78064 = nnscaler.runtime.adapter.chunk(unsqueeze_48_64429, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_48_64429
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_116_56689 = torch.mul(cat_92_56657, unsqueeze_48_78064)
        del cat_92_56657, unsqueeze_48_78064
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_115_56737 = torch.add(mul_115_56521, mul_116_56689, alpha=1)
        del mul_115_56521, mul_116_56689
        unsqueeze_47_78096 = nnscaler.runtime.adapter.chunk(unsqueeze_47_64426, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_47_64426
        reshape_70_78088 = nnscaler.runtime.adapter.nn.split_allgather(reshape_70_64439, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_70_64439
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_117_56777 = torch.mul(reshape_70_78088, unsqueeze_47_78096)
        del unsqueeze_47_78096, reshape_70_78088
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_835_10023 = nnscaler.runtime.function.fullslice(reshape_70_64437, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_70_64437
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_837_10024 = nnscaler.runtime.function.fullslice(reshape_70_64438, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_70_64438
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_47_10025 = _operator.neg(getitem_837_10024)
        del getitem_837_10024
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_93_10026 = nnscaler.runtime.function.cat(neg_47_10025, getitem_835_10023, dim=-1)
        del getitem_835_10023, neg_47_10025
        cat_93_56881 = nnscaler.runtime.adapter.nn.split_allgather(cat_93_10026, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_93_10026
        unsqueeze_48_78120 = nnscaler.runtime.adapter.chunk(unsqueeze_48_64430, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_48_64430
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_118_56889 = torch.mul(cat_93_56881, unsqueeze_48_78120)
        del cat_93_56881, unsqueeze_48_78120
        mul_117_10022 = nnscaler.runtime.adapter.nn.allgather_split(mul_117_56777, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_117_56777
        mul_118_10027 = nnscaler.runtime.adapter.nn.allgather_split(mul_118_56889, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_118_56889
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_116_10028 = torch.add(mul_117_10022, mul_118_10027, alpha=1)
        del mul_117_10022, mul_118_10027
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_94_56937 = nnscaler.runtime.function.cat(split_69_55969, add_115_56737, dim=-1)
        del split_69_55969, add_115_56737
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_23_10030 = torch.Tensor.expand(add_116_10028, size=[-1, 16, -1, -1])
        del add_116_10028
        split_71_56273 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_71_56257, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_71_56257
        expand_23_56977 = nnscaler.runtime.adapter.nn.split_allgather(expand_23_10030, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_23_10030
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_95_56985 = nnscaler.runtime.function.cat(split_71_56273, expand_23_56977, dim=-1)
        del split_71_56273, expand_23_56977
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_23_57001 = torch.nn.functional.pad(split_71_56265, pad=[0, 64], mode='constant', value=0.0)
        del split_71_56265
        cat_94_56929 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_94_56937, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_94_56937
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_189_57033 = torch.transpose(cat_94_56929, dim0=1, dim1=2)
        del cat_94_56929
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_190_57073 = torch.transpose(cat_95_56985, dim0=1, dim1=2)
        del cat_95_56985
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_191_57105 = torch.transpose(pad_23_57001, dim0=1, dim1=2)
        del pad_23_57001
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_23_self_attn_training_7948 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_23_7949 = 0.0 if model_model_layers_23_self_attn_training_7948 else 0.0
        transpose_191_57113 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_191_57105, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_191_57105
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_23_57153 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_189_57033, transpose_190_57073, transpose_191_57113, dropout=ifexpr_23_7949, causal=True, attention_mask=None, query_length=2048)
        del transpose_189_57033, transpose_190_57073, transpose_191_57113
        nnscaler_flash_attention_forward_23_10036 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_23_57153, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_23_57153
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_838_10037 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_23_10036, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_23_10036
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_71_10038 = torch.Tensor.reshape(getitem_838_10037, shape=(8, 2048, 2048))
        del getitem_838_10037
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_23_10039 = torch.Tensor.contiguous(reshape_71_10038)
        del reshape_71_10038
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_164_10041 = torch.nn.functional.linear(contiguous_23_10039, self.model_model_layers_23_self_attn_o_proj_weight_10040, bias=None)
        del contiguous_23_10039
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_117_10042 = torch.add(add_114_85624, linear_164_10041, alpha=1)
        del add_114_85624, linear_164_10041
        # created at IRAdapterGener:local_consumer_multiref
        add_117_85788, add_117_85792 = nnscaler.runtime.function.multiref(add_117_10042, times=2)
        del add_117_10042
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_71_10044 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_117_85788, self.model_model_layers_23_post_attention_layernorm_weight_10043, (2048,), 1e-06)
        del add_117_85788
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_71_64451, fused_rms_norm_affine_71_64452, fused_rms_norm_affine_71_64453, fused_rms_norm_affine_71_64454 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_71_10044, times=4)
        del fused_rms_norm_affine_71_10044
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_23_mlp_gate_training_7951 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_22_10046, moe_route_22_10047, moe_route_22_10048 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_71_64451, self.model_model_layers_23_mlp_gate_weight_10045, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_23_mlp_gate_training_7951, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_71_64451
        moe_route_22_10047 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_22_10047, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_22_10048 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_22_10048, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_71_64452 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_71_64452, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_22_57353 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_71_64452, moe_route_22_10046, moe_route_22_10047, moe_route_22_10048, self.model_model_layers_23_mlp_gate_projs_57329, self.model_model_layers_23_mlp_up_projs_57337, self.model_model_layers_23_mlp_down_projs_57345, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_71_64452, moe_route_22_10046, moe_route_22_10047, moe_route_22_10048
        fused_rms_norm_affine_71_78280 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_71_64453, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_71_64453
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_165_57417 = torch.nn.functional.linear(fused_rms_norm_affine_71_78280, self.model_model_layers_23_mlp_shared_experts_gate_proj_weight_10053, bias=None)
        del fused_rms_norm_affine_71_78280
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_23_57473 = torch.nn.functional.silu(linear_165_57417, inplace=False)
        del linear_165_57417
        fused_rms_norm_affine_71_78320 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_71_64454, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_71_64454
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_166_57497 = torch.nn.functional.linear(fused_rms_norm_affine_71_78320, self.model_model_layers_23_mlp_shared_experts_up_proj_weight_10056, bias=None)
        del fused_rms_norm_affine_71_78320
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_119_57545 = torch.mul(silu_23_57473, linear_166_57497)
        del silu_23_57473, linear_166_57497
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_167_57569 = torch.nn.functional.linear(mul_119_57545, self.model_model_layers_23_mlp_shared_experts_down_proj_weight_10059, bias=None)
        del mul_119_57545
        nnscaler_moe_gmm_22_10052 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_22_57353, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_22_57353
        linear_167_10060 = nnscaler.runtime.adapter.nn.allgather_split(linear_167_57569, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_167_57569
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_118_10061 = torch.add(nnscaler_moe_gmm_22_10052, linear_167_10060, alpha=1)
        del nnscaler_moe_gmm_22_10052, linear_167_10060
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_119_10062 = torch.add(add_117_85792, add_118_10061, alpha=1)
        del add_117_85792, add_118_10061
        # created at IRAdapterGener:local_consumer_multiref
        add_119_85852, add_119_85856 = nnscaler.runtime.function.multiref(add_119_10062, times=2)
        del add_119_10062
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_72_10064 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_119_85852, self.model_model_layers_24_input_layernorm_weight_10063, (2048,), 1e-06)
        del add_119_85852
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_72_64463, fused_rms_norm_affine_72_64464 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_72_10064, times=2)
        del fused_rms_norm_affine_72_10064
        fused_rms_norm_affine_72_78376 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_72_64463, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_72_64463
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_168_57697 = torch.nn.functional.linear(fused_rms_norm_affine_72_78376, self.model_model_layers_24_self_attn_q_proj_weight_10065, bias=None)
        del fused_rms_norm_affine_72_78376
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_120_57753 = torch.Tensor.view(linear_168_57697, size=(8, 256, 16, 192))
        del linear_168_57697
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_192_57777 = torch.transpose(view_120_57753, dim0=1, dim1=2)
        del view_120_57753
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_72_57841, split_72_57849 = torch.functional.split(transpose_192_57777, split_size_or_sections=[128, 64], dim=-1)
        del transpose_192_57777
        fused_rms_norm_affine_72_78440 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_72_64464, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_72_64464
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_169_57881 = torch.nn.functional.linear(fused_rms_norm_affine_72_78440, self.model_model_layers_24_self_attn_kv_a_proj_with_mqa_weight_57873, bias=None)
        del fused_rms_norm_affine_72_78440
        linear_169_10072 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_169_57881, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_169_57881
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_73_10073, split_73_10074 = torch.functional.split(linear_169_10072, split_size_or_sections=[512, 64], dim=-1)
        del linear_169_10072
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_121_10075 = torch.Tensor.view(split_73_10074, size=(8, 2048, 1, 64))
        del split_73_10074
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_193_10076 = torch.transpose(view_121_10075, dim0=1, dim1=2)
        del view_121_10075
        split_73_57921 = nnscaler.runtime.adapter.nn.split_allgather(split_73_10073, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_73_10073
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_73_58001 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_73_57921, self.model_model_layers_24_self_attn_kv_a_layernorm_weight_10077, (512,), 1e-06)
        del split_73_57921
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_170_58017 = torch.nn.functional.linear(fused_rms_norm_affine_73_58001, self.model_model_layers_24_self_attn_kv_b_proj_weight_10079, bias=None)
        del fused_rms_norm_affine_73_58001
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_122_58073 = torch.Tensor.view(linear_170_58017, size=(8, 256, 16, 256))
        del linear_170_58017
        view_122_58065 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_122_58073, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_122_58073
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_194_58089 = torch.transpose(view_122_58065, dim0=1, dim1=2)
        del view_122_58065
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_74_58129, split_74_58137 = torch.functional.split(transpose_194_58089, split_size_or_sections=[128, 128], dim=-1)
        del transpose_194_58089
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_853_10086 = nnscaler.runtime.function.fullslice(self.model_model_layers_24_self_attn_rotary_emb_cos_cached_10085, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_48_10087 = torch.Tensor.to(getitem_853_10086, dtype=torch.bfloat16)
        del getitem_853_10086
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_854_10089 = nnscaler.runtime.function.fullslice(self.model_model_layers_24_self_attn_rotary_emb_sin_cached_10088, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_49_10090 = torch.Tensor.to(getitem_854_10089, dtype=torch.bfloat16)
        del getitem_854_10089
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_855_10091 = nnscaler.runtime.function.fullslice(to_48_10087, unsqueeze_8005)
        del to_48_10087
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_49_10092 = torch.unsqueeze(getitem_855_10091, dim=1)
        del getitem_855_10091
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_49_64469, unsqueeze_49_64470 = nnscaler.runtime.function.multiref(unsqueeze_49_10092, times=2)
        del unsqueeze_49_10092
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_856_10093 = nnscaler.runtime.function.fullslice(to_49_10090, unsqueeze_8005)
        del to_49_10090
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_50_10094 = torch.unsqueeze(getitem_856_10093, dim=1)
        del getitem_856_10093
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_50_64473, unsqueeze_50_64474 = nnscaler.runtime.function.multiref(unsqueeze_50_10094, times=2)
        del unsqueeze_50_10094
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_123_58225 = torch.Tensor.view(split_72_57849, size=(8, 16, 256, 32, 2))
        del split_72_57849
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_195_58265 = torch.transpose(view_123_58225, dim0=4, dim1=3)
        del view_123_58225
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_72_58297 = torch.Tensor.reshape(transpose_195_58265, shape=(8, 16, 256, 64))
        del transpose_195_58265
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_124_10098 = torch.Tensor.view(transpose_193_10076, size=(8, 1, 2048, 32, 2))
        del transpose_193_10076
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_196_10099 = torch.transpose(view_124_10098, dim0=4, dim1=3)
        del view_124_10098
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_73_10100 = torch.Tensor.reshape(transpose_196_10099, shape=(8, 1, 2048, 64))
        del transpose_196_10099
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_73_64481, reshape_73_64482, reshape_73_64483 = nnscaler.runtime.function.multiref(reshape_73_10100, times=3)
        del reshape_73_10100
        unsqueeze_49_78568 = nnscaler.runtime.adapter.chunk(unsqueeze_49_64469, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_49_64469
        # created at IRAdapterGener:local_consumer_multiref
        reshape_72_85947, reshape_72_85951, reshape_72_85955 = nnscaler.runtime.function.multiref(reshape_72_58297, times=3)
        del reshape_72_58297
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_120_58393 = torch.mul(reshape_72_85947, unsqueeze_49_78568)
        del unsqueeze_49_78568, reshape_72_85947
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_866_58441 = nnscaler.runtime.function.fullslice(reshape_72_85951, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_72_85951
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_868_58465 = nnscaler.runtime.function.fullslice(reshape_72_85955, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_72_85955
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_48_58489 = _operator.neg(getitem_868_58465)
        del getitem_868_58465
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_96_58529 = nnscaler.runtime.function.cat(neg_48_58489, getitem_866_58441, dim=-1)
        del getitem_866_58441, neg_48_58489
        unsqueeze_50_78640 = nnscaler.runtime.adapter.chunk(unsqueeze_50_64473, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_50_64473
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_121_58561 = torch.mul(cat_96_58529, unsqueeze_50_78640)
        del cat_96_58529, unsqueeze_50_78640
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_120_58609 = torch.add(mul_120_58393, mul_121_58561, alpha=1)
        del mul_120_58393, mul_121_58561
        unsqueeze_49_78672 = nnscaler.runtime.adapter.chunk(unsqueeze_49_64470, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_49_64470
        reshape_73_78664 = nnscaler.runtime.adapter.nn.split_allgather(reshape_73_64483, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_73_64483
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_122_58649 = torch.mul(reshape_73_78664, unsqueeze_49_78672)
        del unsqueeze_49_78672, reshape_73_78664
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_870_10109 = nnscaler.runtime.function.fullslice(reshape_73_64481, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_73_64481
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_872_10110 = nnscaler.runtime.function.fullslice(reshape_73_64482, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_73_64482
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_49_10111 = _operator.neg(getitem_872_10110)
        del getitem_872_10110
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_97_10112 = nnscaler.runtime.function.cat(neg_49_10111, getitem_870_10109, dim=-1)
        del getitem_870_10109, neg_49_10111
        cat_97_58753 = nnscaler.runtime.adapter.nn.split_allgather(cat_97_10112, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_97_10112
        unsqueeze_50_78696 = nnscaler.runtime.adapter.chunk(unsqueeze_50_64474, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_50_64474
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_123_58761 = torch.mul(cat_97_58753, unsqueeze_50_78696)
        del cat_97_58753, unsqueeze_50_78696
        mul_122_10108 = nnscaler.runtime.adapter.nn.allgather_split(mul_122_58649, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_122_58649
        mul_123_10113 = nnscaler.runtime.adapter.nn.allgather_split(mul_123_58761, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_123_58761
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_121_10114 = torch.add(mul_122_10108, mul_123_10113, alpha=1)
        del mul_122_10108, mul_123_10113
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_98_58809 = nnscaler.runtime.function.cat(split_72_57841, add_120_58609, dim=-1)
        del split_72_57841, add_120_58609
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_24_10116 = torch.Tensor.expand(add_121_10114, size=[-1, 16, -1, -1])
        del add_121_10114
        split_74_58145 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_74_58129, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_74_58129
        expand_24_58849 = nnscaler.runtime.adapter.nn.split_allgather(expand_24_10116, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_24_10116
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_99_58857 = nnscaler.runtime.function.cat(split_74_58145, expand_24_58849, dim=-1)
        del split_74_58145, expand_24_58849
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_24_58873 = torch.nn.functional.pad(split_74_58137, pad=[0, 64], mode='constant', value=0.0)
        del split_74_58137
        cat_98_58801 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_98_58809, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_98_58809
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_197_58905 = torch.transpose(cat_98_58801, dim0=1, dim1=2)
        del cat_98_58801
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_198_58945 = torch.transpose(cat_99_58857, dim0=1, dim1=2)
        del cat_99_58857
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_199_58977 = torch.transpose(pad_24_58873, dim0=1, dim1=2)
        del pad_24_58873
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_24_self_attn_training_7963 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_24_7964 = 0.0 if model_model_layers_24_self_attn_training_7963 else 0.0
        transpose_199_58985 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_199_58977, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_199_58977
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_24_59025 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_197_58905, transpose_198_58945, transpose_199_58985, dropout=ifexpr_24_7964, causal=True, attention_mask=None, query_length=2048)
        del transpose_197_58905, transpose_198_58945, transpose_199_58985
        nnscaler_flash_attention_forward_24_10122 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_24_59025, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_24_59025
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_873_10123 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_24_10122, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_24_10122
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_74_10124 = torch.Tensor.reshape(getitem_873_10123, shape=(8, 2048, 2048))
        del getitem_873_10123
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_24_10125 = torch.Tensor.contiguous(reshape_74_10124)
        del reshape_74_10124
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_171_10127 = torch.nn.functional.linear(contiguous_24_10125, self.model_model_layers_24_self_attn_o_proj_weight_10126, bias=None)
        del contiguous_24_10125
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_122_10128 = torch.add(add_119_85856, linear_171_10127, alpha=1)
        del add_119_85856, linear_171_10127
        # created at IRAdapterGener:local_consumer_multiref
        add_122_86020, add_122_86024 = nnscaler.runtime.function.multiref(add_122_10128, times=2)
        del add_122_10128
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_74_10130 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_122_86020, self.model_model_layers_24_post_attention_layernorm_weight_10129, (2048,), 1e-06)
        del add_122_86020
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_74_64495, fused_rms_norm_affine_74_64496, fused_rms_norm_affine_74_64497, fused_rms_norm_affine_74_64498 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_74_10130, times=4)
        del fused_rms_norm_affine_74_10130
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_24_mlp_gate_training_7966 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_23_10132, moe_route_23_10133, moe_route_23_10134 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_74_64495, self.model_model_layers_24_mlp_gate_weight_10131, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_24_mlp_gate_training_7966, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_74_64495
        moe_route_23_10133 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_23_10133, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_23_10134 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_23_10134, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_74_64496 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_74_64496, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_23_59225 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_74_64496, moe_route_23_10132, moe_route_23_10133, moe_route_23_10134, self.model_model_layers_24_mlp_gate_projs_59201, self.model_model_layers_24_mlp_up_projs_59209, self.model_model_layers_24_mlp_down_projs_59217, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_74_64496, moe_route_23_10132, moe_route_23_10133, moe_route_23_10134
        fused_rms_norm_affine_74_78856 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_74_64497, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_74_64497
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_172_59289 = torch.nn.functional.linear(fused_rms_norm_affine_74_78856, self.model_model_layers_24_mlp_shared_experts_gate_proj_weight_10139, bias=None)
        del fused_rms_norm_affine_74_78856
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_24_59345 = torch.nn.functional.silu(linear_172_59289, inplace=False)
        del linear_172_59289
        fused_rms_norm_affine_74_78896 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_74_64498, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_74_64498
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_173_59369 = torch.nn.functional.linear(fused_rms_norm_affine_74_78896, self.model_model_layers_24_mlp_shared_experts_up_proj_weight_10142, bias=None)
        del fused_rms_norm_affine_74_78896
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_124_59417 = torch.mul(silu_24_59345, linear_173_59369)
        del silu_24_59345, linear_173_59369
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_174_59441 = torch.nn.functional.linear(mul_124_59417, self.model_model_layers_24_mlp_shared_experts_down_proj_weight_10145, bias=None)
        del mul_124_59417
        nnscaler_moe_gmm_23_10138 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_23_59225, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_23_59225
        linear_174_10146 = nnscaler.runtime.adapter.nn.allgather_split(linear_174_59441, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_174_59441
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_123_10147 = torch.add(nnscaler_moe_gmm_23_10138, linear_174_10146, alpha=1)
        del nnscaler_moe_gmm_23_10138, linear_174_10146
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_124_10148 = torch.add(add_122_86024, add_123_10147, alpha=1)
        del add_122_86024, add_123_10147
        # created at IRAdapterGener:local_consumer_multiref
        add_124_86084, add_124_86088 = nnscaler.runtime.function.multiref(add_124_10148, times=2)
        del add_124_10148
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_75_10150 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_124_86084, self.model_model_layers_25_input_layernorm_weight_10149, (2048,), 1e-06)
        del add_124_86084
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_75_64507, fused_rms_norm_affine_75_64508 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_75_10150, times=2)
        del fused_rms_norm_affine_75_10150
        fused_rms_norm_affine_75_78952 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_75_64507, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_75_64507
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_175_59569 = torch.nn.functional.linear(fused_rms_norm_affine_75_78952, self.model_model_layers_25_self_attn_q_proj_weight_10151, bias=None)
        del fused_rms_norm_affine_75_78952
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_125_59625 = torch.Tensor.view(linear_175_59569, size=(8, 256, 16, 192))
        del linear_175_59569
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_200_59649 = torch.transpose(view_125_59625, dim0=1, dim1=2)
        del view_125_59625
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_75_59713, split_75_59721 = torch.functional.split(transpose_200_59649, split_size_or_sections=[128, 64], dim=-1)
        del transpose_200_59649
        fused_rms_norm_affine_75_79016 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_75_64508, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_75_64508
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_176_59753 = torch.nn.functional.linear(fused_rms_norm_affine_75_79016, self.model_model_layers_25_self_attn_kv_a_proj_with_mqa_weight_59745, bias=None)
        del fused_rms_norm_affine_75_79016
        linear_176_10158 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_176_59753, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_176_59753
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_76_10159, split_76_10160 = torch.functional.split(linear_176_10158, split_size_or_sections=[512, 64], dim=-1)
        del linear_176_10158
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_126_10161 = torch.Tensor.view(split_76_10160, size=(8, 2048, 1, 64))
        del split_76_10160
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_201_10162 = torch.transpose(view_126_10161, dim0=1, dim1=2)
        del view_126_10161
        split_76_59793 = nnscaler.runtime.adapter.nn.split_allgather(split_76_10159, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_76_10159
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_76_59873 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_76_59793, self.model_model_layers_25_self_attn_kv_a_layernorm_weight_10163, (512,), 1e-06)
        del split_76_59793
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_177_59889 = torch.nn.functional.linear(fused_rms_norm_affine_76_59873, self.model_model_layers_25_self_attn_kv_b_proj_weight_10165, bias=None)
        del fused_rms_norm_affine_76_59873
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_127_59945 = torch.Tensor.view(linear_177_59889, size=(8, 256, 16, 256))
        del linear_177_59889
        view_127_59937 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_127_59945, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_127_59945
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_202_59961 = torch.transpose(view_127_59937, dim0=1, dim1=2)
        del view_127_59937
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_77_60001, split_77_60009 = torch.functional.split(transpose_202_59961, split_size_or_sections=[128, 128], dim=-1)
        del transpose_202_59961
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_888_10172 = nnscaler.runtime.function.fullslice(self.model_model_layers_25_self_attn_rotary_emb_cos_cached_10171, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_50_10173 = torch.Tensor.to(getitem_888_10172, dtype=torch.bfloat16)
        del getitem_888_10172
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_889_10175 = nnscaler.runtime.function.fullslice(self.model_model_layers_25_self_attn_rotary_emb_sin_cached_10174, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_51_10176 = torch.Tensor.to(getitem_889_10175, dtype=torch.bfloat16)
        del getitem_889_10175
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_890_10177 = nnscaler.runtime.function.fullslice(to_50_10173, unsqueeze_8005)
        del to_50_10173
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_51_10178 = torch.unsqueeze(getitem_890_10177, dim=1)
        del getitem_890_10177
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_51_64513, unsqueeze_51_64514 = nnscaler.runtime.function.multiref(unsqueeze_51_10178, times=2)
        del unsqueeze_51_10178
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_891_10179 = nnscaler.runtime.function.fullslice(to_51_10176, unsqueeze_8005)
        del to_51_10176
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_52_10180 = torch.unsqueeze(getitem_891_10179, dim=1)
        del getitem_891_10179
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_52_64517, unsqueeze_52_64518 = nnscaler.runtime.function.multiref(unsqueeze_52_10180, times=2)
        del unsqueeze_52_10180
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_128_60097 = torch.Tensor.view(split_75_59721, size=(8, 16, 256, 32, 2))
        del split_75_59721
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_203_60137 = torch.transpose(view_128_60097, dim0=4, dim1=3)
        del view_128_60097
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_75_60169 = torch.Tensor.reshape(transpose_203_60137, shape=(8, 16, 256, 64))
        del transpose_203_60137
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_129_10184 = torch.Tensor.view(transpose_201_10162, size=(8, 1, 2048, 32, 2))
        del transpose_201_10162
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_204_10185 = torch.transpose(view_129_10184, dim0=4, dim1=3)
        del view_129_10184
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_76_10186 = torch.Tensor.reshape(transpose_204_10185, shape=(8, 1, 2048, 64))
        del transpose_204_10185
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_76_64525, reshape_76_64526, reshape_76_64527 = nnscaler.runtime.function.multiref(reshape_76_10186, times=3)
        del reshape_76_10186
        unsqueeze_51_79144 = nnscaler.runtime.adapter.chunk(unsqueeze_51_64513, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_51_64513
        # created at IRAdapterGener:local_consumer_multiref
        reshape_75_86179, reshape_75_86183, reshape_75_86187 = nnscaler.runtime.function.multiref(reshape_75_60169, times=3)
        del reshape_75_60169
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_125_60265 = torch.mul(reshape_75_86179, unsqueeze_51_79144)
        del unsqueeze_51_79144, reshape_75_86179
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_901_60313 = nnscaler.runtime.function.fullslice(reshape_75_86183, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_75_86183
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_903_60337 = nnscaler.runtime.function.fullslice(reshape_75_86187, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_75_86187
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_50_60361 = _operator.neg(getitem_903_60337)
        del getitem_903_60337
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_100_60401 = nnscaler.runtime.function.cat(neg_50_60361, getitem_901_60313, dim=-1)
        del getitem_901_60313, neg_50_60361
        unsqueeze_52_79216 = nnscaler.runtime.adapter.chunk(unsqueeze_52_64517, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_52_64517
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_126_60433 = torch.mul(cat_100_60401, unsqueeze_52_79216)
        del cat_100_60401, unsqueeze_52_79216
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_125_60481 = torch.add(mul_125_60265, mul_126_60433, alpha=1)
        del mul_125_60265, mul_126_60433
        unsqueeze_51_79248 = nnscaler.runtime.adapter.chunk(unsqueeze_51_64514, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_51_64514
        reshape_76_79240 = nnscaler.runtime.adapter.nn.split_allgather(reshape_76_64527, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_76_64527
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_127_60521 = torch.mul(reshape_76_79240, unsqueeze_51_79248)
        del unsqueeze_51_79248, reshape_76_79240
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_905_10195 = nnscaler.runtime.function.fullslice(reshape_76_64525, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_76_64525
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_907_10196 = nnscaler.runtime.function.fullslice(reshape_76_64526, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_76_64526
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_51_10197 = _operator.neg(getitem_907_10196)
        del getitem_907_10196
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_101_10198 = nnscaler.runtime.function.cat(neg_51_10197, getitem_905_10195, dim=-1)
        del getitem_905_10195, neg_51_10197
        cat_101_60625 = nnscaler.runtime.adapter.nn.split_allgather(cat_101_10198, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_101_10198
        unsqueeze_52_79272 = nnscaler.runtime.adapter.chunk(unsqueeze_52_64518, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_52_64518
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_128_60633 = torch.mul(cat_101_60625, unsqueeze_52_79272)
        del cat_101_60625, unsqueeze_52_79272
        mul_127_10194 = nnscaler.runtime.adapter.nn.allgather_split(mul_127_60521, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_127_60521
        mul_128_10199 = nnscaler.runtime.adapter.nn.allgather_split(mul_128_60633, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_128_60633
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_126_10200 = torch.add(mul_127_10194, mul_128_10199, alpha=1)
        del mul_127_10194, mul_128_10199
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_102_60681 = nnscaler.runtime.function.cat(split_75_59713, add_125_60481, dim=-1)
        del split_75_59713, add_125_60481
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_25_10202 = torch.Tensor.expand(add_126_10200, size=[-1, 16, -1, -1])
        del add_126_10200
        split_77_60017 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_77_60001, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_77_60001
        expand_25_60721 = nnscaler.runtime.adapter.nn.split_allgather(expand_25_10202, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_25_10202
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_103_60729 = nnscaler.runtime.function.cat(split_77_60017, expand_25_60721, dim=-1)
        del split_77_60017, expand_25_60721
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_25_60745 = torch.nn.functional.pad(split_77_60009, pad=[0, 64], mode='constant', value=0.0)
        del split_77_60009
        cat_102_60673 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_102_60681, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_102_60681
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_205_60777 = torch.transpose(cat_102_60673, dim0=1, dim1=2)
        del cat_102_60673
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_206_60817 = torch.transpose(cat_103_60729, dim0=1, dim1=2)
        del cat_103_60729
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_207_60849 = torch.transpose(pad_25_60745, dim0=1, dim1=2)
        del pad_25_60745
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_25_self_attn_training_7978 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_25_7979 = 0.0 if model_model_layers_25_self_attn_training_7978 else 0.0
        transpose_207_60857 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_207_60849, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_207_60849
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_25_60897 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_205_60777, transpose_206_60817, transpose_207_60857, dropout=ifexpr_25_7979, causal=True, attention_mask=None, query_length=2048)
        del transpose_205_60777, transpose_206_60817, transpose_207_60857
        nnscaler_flash_attention_forward_25_10208 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_25_60897, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_25_60897
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_908_10209 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_25_10208, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_25_10208
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_77_10210 = torch.Tensor.reshape(getitem_908_10209, shape=(8, 2048, 2048))
        del getitem_908_10209
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_25_10211 = torch.Tensor.contiguous(reshape_77_10210)
        del reshape_77_10210
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_178_10213 = torch.nn.functional.linear(contiguous_25_10211, self.model_model_layers_25_self_attn_o_proj_weight_10212, bias=None)
        del contiguous_25_10211
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_127_10214 = torch.add(add_124_86088, linear_178_10213, alpha=1)
        del add_124_86088, linear_178_10213
        # created at IRAdapterGener:local_consumer_multiref
        add_127_86252, add_127_86256 = nnscaler.runtime.function.multiref(add_127_10214, times=2)
        del add_127_10214
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_77_10216 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_127_86252, self.model_model_layers_25_post_attention_layernorm_weight_10215, (2048,), 1e-06)
        del add_127_86252
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_77_64539, fused_rms_norm_affine_77_64540, fused_rms_norm_affine_77_64541, fused_rms_norm_affine_77_64542 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_77_10216, times=4)
        del fused_rms_norm_affine_77_10216
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_25_mlp_gate_training_7981 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_24_10218, moe_route_24_10219, moe_route_24_10220 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_77_64539, self.model_model_layers_25_mlp_gate_weight_10217, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_25_mlp_gate_training_7981, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_77_64539
        moe_route_24_10219 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_24_10219, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_24_10220 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_24_10220, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_77_64540 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_77_64540, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_24_61097 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_77_64540, moe_route_24_10218, moe_route_24_10219, moe_route_24_10220, self.model_model_layers_25_mlp_gate_projs_61073, self.model_model_layers_25_mlp_up_projs_61081, self.model_model_layers_25_mlp_down_projs_61089, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_77_64540, moe_route_24_10218, moe_route_24_10219, moe_route_24_10220
        fused_rms_norm_affine_77_79432 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_77_64541, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_77_64541
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_179_61161 = torch.nn.functional.linear(fused_rms_norm_affine_77_79432, self.model_model_layers_25_mlp_shared_experts_gate_proj_weight_10225, bias=None)
        del fused_rms_norm_affine_77_79432
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_25_61217 = torch.nn.functional.silu(linear_179_61161, inplace=False)
        del linear_179_61161
        fused_rms_norm_affine_77_79472 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_77_64542, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_77_64542
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_180_61241 = torch.nn.functional.linear(fused_rms_norm_affine_77_79472, self.model_model_layers_25_mlp_shared_experts_up_proj_weight_10228, bias=None)
        del fused_rms_norm_affine_77_79472
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_129_61289 = torch.mul(silu_25_61217, linear_180_61241)
        del silu_25_61217, linear_180_61241
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_181_61313 = torch.nn.functional.linear(mul_129_61289, self.model_model_layers_25_mlp_shared_experts_down_proj_weight_10231, bias=None)
        del mul_129_61289
        nnscaler_moe_gmm_24_10224 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_24_61097, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_24_61097
        linear_181_10232 = nnscaler.runtime.adapter.nn.allgather_split(linear_181_61313, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_181_61313
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_128_10233 = torch.add(nnscaler_moe_gmm_24_10224, linear_181_10232, alpha=1)
        del nnscaler_moe_gmm_24_10224, linear_181_10232
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_129_10234 = torch.add(add_127_86256, add_128_10233, alpha=1)
        del add_127_86256, add_128_10233
        # created at IRAdapterGener:local_consumer_multiref
        add_129_86316, add_129_86320 = nnscaler.runtime.function.multiref(add_129_10234, times=2)
        del add_129_10234
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_78_10236 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_129_86316, self.model_model_layers_26_input_layernorm_weight_10235, (2048,), 1e-06)
        del add_129_86316
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_78_64551, fused_rms_norm_affine_78_64552 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_78_10236, times=2)
        del fused_rms_norm_affine_78_10236
        fused_rms_norm_affine_78_79528 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_78_64551, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_78_64551
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 119, in forward,  q = self.q_proj(hidden_states)
        linear_182_61441 = torch.nn.functional.linear(fused_rms_norm_affine_78_79528, self.model_model_layers_26_self_attn_q_proj_weight_10237, bias=None)
        del fused_rms_norm_affine_78_79528
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        view_130_61497 = torch.Tensor.view(linear_182_61441, size=(8, 256, 16, 192))
        del linear_182_61441
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 122, in forward,  q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        transpose_208_61521 = torch.transpose(view_130_61497, dim0=1, dim1=2)
        del view_130_61497
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 123, in forward,  q_nope, q_pe = torch.split(
        split_78_61585, split_78_61593 = torch.functional.split(transpose_208_61521, split_size_or_sections=[128, 64], dim=-1)
        del transpose_208_61521
        fused_rms_norm_affine_78_79592 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_78_64552, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_78_64552
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 130, in forward,  compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # start signal
        linear_183_61625 = torch.nn.functional.linear(fused_rms_norm_affine_78_79592, self.model_model_layers_26_self_attn_kv_a_proj_with_mqa_weight_61617, bias=None)
        del fused_rms_norm_affine_78_79592
        linear_183_10244 = nnscaler.runtime.adapter.nn.allreduce_identity(linear_183_61625, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_183_61625
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 131, in forward,  compressed_kv, k_pe = torch.split(
        split_79_10245, split_79_10246 = torch.functional.split(linear_183_10244, split_size_or_sections=[512, 64], dim=-1)
        del linear_183_10244
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        view_131_10247 = torch.Tensor.view(split_79_10246, size=(8, 2048, 1, 64))
        del split_79_10246
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 134, in forward,  k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        transpose_209_10248 = torch.transpose(view_131_10247, dim0=1, dim1=2)
        del view_131_10247
        split_79_61665 = nnscaler.runtime.adapter.nn.split_allgather(split_79_10245, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_79_10245
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_79_61745 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(split_79_61665, self.model_model_layers_26_self_attn_kv_a_layernorm_weight_10249, (512,), 1e-06)
        del split_79_61665
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        linear_184_61761 = torch.nn.functional.linear(fused_rms_norm_affine_79_61745, self.model_model_layers_26_self_attn_kv_b_proj_weight_10251, bias=None)
        del fused_rms_norm_affine_79_61745
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 136, in forward,  self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        view_132_61817 = torch.Tensor.view(linear_184_61761, size=(8, 256, 16, 256))
        del linear_184_61761
        view_132_61809 = nnscaler.runtime.adapter.nn.alltoall_alltoall(view_132_61817, idim=1, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_132_61817
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 135, in forward,  kv = (
        transpose_210_61833 = torch.transpose(view_132_61809, dim0=1, dim1=2)
        del view_132_61809
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 141, in forward,  k_nope, value_states = torch.split(
        split_80_61873, split_80_61881 = torch.functional.split(transpose_210_61833, split_size_or_sections=[128, 128], dim=-1)
        del transpose_210_61833
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 152, in forward,  self.cos_cached[:seq_len].to(dtype=x.dtype),
        getitem_923_10258 = nnscaler.runtime.function.fullslice(self.model_model_layers_26_self_attn_rotary_emb_cos_cached_10257, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_52_10259 = torch.Tensor.to(getitem_923_10258, dtype=torch.bfloat16)
        del getitem_923_10258
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 153, in forward,  self.sin_cached[:seq_len].to(dtype=x.dtype),
        getitem_924_10261 = nnscaler.runtime.function.fullslice(self.model_model_layers_26_self_attn_rotary_emb_sin_cached_10260, slice(None, 2048, None))
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 151, in forward,  return (
        to_53_10262 = torch.Tensor.to(getitem_924_10261, dtype=torch.bfloat16)
        del getitem_924_10261
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        getitem_925_10263 = nnscaler.runtime.function.fullslice(to_52_10259, unsqueeze_8005)
        del to_52_10259
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 360, in apply_rotary_pos_emb,  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_53_10264 = torch.unsqueeze(getitem_925_10263, dim=1)
        del getitem_925_10263
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_53_64557, unsqueeze_53_64558 = nnscaler.runtime.function.multiref(unsqueeze_53_10264, times=2)
        del unsqueeze_53_10264
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        getitem_926_10265 = nnscaler.runtime.function.fullslice(to_53_10262, unsqueeze_8005)
        del unsqueeze_8005, to_53_10262
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 361, in apply_rotary_pos_emb,  sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        unsqueeze_54_10266 = torch.unsqueeze(getitem_926_10265, dim=1)
        del getitem_926_10265
        # create at IRAdapterGener:autoref, comment before transformation: activation
        unsqueeze_54_64561, unsqueeze_54_64562 = nnscaler.runtime.function.multiref(unsqueeze_54_10266, times=2)
        del unsqueeze_54_10266
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_133_61969 = torch.Tensor.view(split_78_61593, size=(8, 16, 256, 32, 2))
        del split_78_61593
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_211_62009 = torch.transpose(view_133_61969, dim0=4, dim1=3)
        del view_133_61969
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 364, in apply_rotary_pos_emb,  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_78_62041 = torch.Tensor.reshape(transpose_211_62009, shape=(8, 16, 256, 64))
        del transpose_211_62009
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        view_134_10270 = torch.Tensor.view(transpose_209_10248, size=(8, 1, 2048, 32, 2))
        del transpose_209_10248
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        transpose_212_10271 = torch.transpose(view_134_10270, dim0=4, dim1=3)
        del view_134_10270
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 367, in apply_rotary_pos_emb,  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
        reshape_79_10272 = torch.Tensor.reshape(transpose_212_10271, shape=(8, 1, 2048, 64))
        del transpose_212_10271
        # create at IRAdapterGener:autoref, comment before transformation: activation
        reshape_79_64569, reshape_79_64570, reshape_79_64571 = nnscaler.runtime.function.multiref(reshape_79_10272, times=3)
        del reshape_79_10272
        unsqueeze_53_79720 = nnscaler.runtime.adapter.chunk(unsqueeze_53_64557, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_53_64557
        # created at IRAdapterGener:local_consumer_multiref
        reshape_78_86411, reshape_78_86415, reshape_78_86419 = nnscaler.runtime.function.multiref(reshape_78_62041, times=3)
        del reshape_78_62041
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_130_62137 = torch.mul(reshape_78_86411, unsqueeze_53_79720)
        del unsqueeze_53_79720, reshape_78_86411
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_936_62185 = nnscaler.runtime.function.fullslice(reshape_78_86415, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_78_86415
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_938_62209 = nnscaler.runtime.function.fullslice(reshape_78_86419, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_78_86419
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_52_62233 = _operator.neg(getitem_938_62209)
        del getitem_938_62209
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_104_62273 = nnscaler.runtime.function.cat(neg_52_62233, getitem_936_62185, dim=-1)
        del getitem_936_62185, neg_52_62233
        unsqueeze_54_79792 = nnscaler.runtime.adapter.chunk(unsqueeze_54_64561, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_54_64561
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_131_62305 = torch.mul(cat_104_62273, unsqueeze_54_79792)
        del cat_104_62273, unsqueeze_54_79792
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 369, in apply_rotary_pos_emb,  q_embed = (q * cos) + (rotate_half(q) * sin)
        add_130_62353 = torch.add(mul_130_62137, mul_131_62305, alpha=1)
        del mul_130_62137, mul_131_62305
        unsqueeze_53_79824 = nnscaler.runtime.adapter.chunk(unsqueeze_53_64558, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_53_64558
        reshape_79_79816 = nnscaler.runtime.adapter.nn.split_allgather(reshape_79_64571, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del reshape_79_64571
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_132_62393 = torch.mul(reshape_79_79816, unsqueeze_53_79824)
        del unsqueeze_53_79824, reshape_79_79816
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 333, in rotate_half,  x1 = x[..., : x.shape[-1] // 2]
        getitem_940_10281 = nnscaler.runtime.function.fullslice(reshape_79_64569, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 32, None))
        del reshape_79_64569
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 334, in rotate_half,  x2 = x[..., x.shape[-1] // 2 :]
        getitem_942_10282 = nnscaler.runtime.function.fullslice(reshape_79_64570, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(32, None, None))
        del reshape_79_64570
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        neg_53_10283 = _operator.neg(getitem_942_10282)
        del getitem_942_10282
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 335, in rotate_half,  return torch.cat((-x2, x1), dim=-1)
        cat_105_10284 = nnscaler.runtime.function.cat(neg_53_10283, getitem_940_10281, dim=-1)
        del getitem_940_10281, neg_53_10283
        cat_105_62497 = nnscaler.runtime.adapter.nn.split_allgather(cat_105_10284, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_105_10284
        unsqueeze_54_79848 = nnscaler.runtime.adapter.chunk(unsqueeze_54_64562, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del unsqueeze_54_64562
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_133_62505 = torch.mul(cat_105_62497, unsqueeze_54_79848)
        del cat_105_62497, unsqueeze_54_79848
        mul_132_10280 = nnscaler.runtime.adapter.nn.allgather_split(mul_132_62393, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_132_62393
        mul_133_10285 = nnscaler.runtime.adapter.nn.allgather_split(mul_133_62505, dim=3, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del mul_133_62505
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 370, in apply_rotary_pos_emb,  k_embed = (k * cos) + (rotate_half(k) * sin)
        add_131_10286 = torch.add(mul_132_10280, mul_133_10285, alpha=1)
        del mul_132_10280, mul_133_10285
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 156, in forward,  query_states = torch.cat([q_nope, q_pe], dim=-1)
        cat_106_62553 = nnscaler.runtime.function.cat(split_78_61585, add_130_62353, dim=-1)
        del split_78_61585, add_130_62353
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        expand_26_10288 = torch.Tensor.expand(add_131_10286, size=[-1, 16, -1, -1])
        del add_131_10286
        split_80_61889 = nnscaler.runtime.adapter.nn.alltoall_alltoall(split_80_61873, idim=0, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del split_80_61873
        expand_26_62593 = nnscaler.runtime.adapter.nn.split_allgather(expand_26_10288, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del expand_26_10288
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 161, in forward,  key_states = torch.cat([k_nope, k_pe.expand(-1, k_nope.size(1), -1, -1)], dim=-1)
        cat_107_62601 = nnscaler.runtime.function.cat(split_80_61889, expand_26_62593, dim=-1)
        del split_80_61889, expand_26_62593
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 164, in forward,  value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
        pad_26_62617 = torch.nn.functional.pad(split_80_61881, pad=[0, 64], mode='constant', value=0.0)
        del split_80_61881
        cat_106_62545 = nnscaler.runtime.adapter.nn.alltoall_alltoall(cat_106_62553, idim=2, odim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del cat_106_62553
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 174, in forward,  query_states = query_states.transpose(1, 2)
        transpose_213_62649 = torch.transpose(cat_106_62545, dim0=1, dim1=2)
        del cat_106_62545
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 175, in forward,  key_states = key_states.transpose(1, 2)
        transpose_214_62689 = torch.transpose(cat_107_62601, dim0=1, dim1=2)
        del cat_107_62601
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 176, in forward,  value_states = value_states.transpose(1, 2)
        transpose_215_62721 = torch.transpose(pad_26_62617, dim0=1, dim1=2)
        del pad_26_62617
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        model_model_layers_26_self_attn_training_7993 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 178, in forward,  dropout_rate = self.attention_dropout if self.training else 0.0
        ifexpr_26_7994 = 0.0 if model_model_layers_26_self_attn_training_7993 else 0.0
        transpose_215_62729 = nnscaler.runtime.adapter.nn.alltoall_alltoall(transpose_215_62721, idim=0, odim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del transpose_215_62721
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 216, in forward,  attn_output = nnscaler_flash_attention_forward(
        nnscaler_flash_attention_forward_26_62769 = modeling.modeling_deepseek_modifier.nnscaler_flash_attention_forward(transpose_213_62649, transpose_214_62689, transpose_215_62729, dropout=ifexpr_26_7994, causal=True, attention_mask=None, query_length=2048)
        del transpose_213_62649, transpose_214_62689, transpose_215_62729
        nnscaler_flash_attention_forward_26_10294 = nnscaler.runtime.adapter.nn.allgather_split(nnscaler_flash_attention_forward_26_62769, dim=2, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_flash_attention_forward_26_62769
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 220, in forward,  attn_output = attn_output[:, :, :, : self.v_head_dim]
        getitem_943_10295 = nnscaler.runtime.function.fullslice(nnscaler_flash_attention_forward_26_10294, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 128, None))
        del nnscaler_flash_attention_forward_26_10294
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        reshape_80_10296 = torch.Tensor.reshape(getitem_943_10295, shape=(8, 2048, 2048))
        del getitem_943_10295
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 222, in forward,  attn_output = attn_output.reshape(
        contiguous_26_10297 = torch.Tensor.contiguous(reshape_80_10296)
        del reshape_80_10296
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 225, in forward,  attn_output = self.o_proj(attn_output)
        linear_185_10299 = torch.nn.functional.linear(contiguous_26_10297, self.model_model_layers_26_self_attn_o_proj_weight_10298, bias=None)
        del contiguous_26_10297
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1268, in forward,  hidden_states = residual + hidden_states
        add_132_10300 = torch.add(add_129_86320, linear_185_10299, alpha=1)
        del add_129_86320, linear_185_10299
        # created at IRAdapterGener:local_consumer_multiref
        add_132_86484, add_132_86488 = nnscaler.runtime.function.multiref(add_132_10300, times=2)
        del add_132_10300
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_80_10302 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_132_86484, self.model_model_layers_26_post_attention_layernorm_weight_10301, (2048,), 1e-06)
        del add_132_86484
        # create at IRAdapterGener:autoref, comment before transformation: activation
        fused_rms_norm_affine_80_64583, fused_rms_norm_affine_80_64584, fused_rms_norm_affine_80_64585, fused_rms_norm_affine_80_64586 = nnscaler.runtime.function.multiref(fused_rms_norm_affine_80_10302, times=4)
        del fused_rms_norm_affine_80_10302
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 70, in moe_gate_fwd,  self.topk_group, self.training, self.alpha, self.norm_topk_prob, self.routed_scaling_factor, self.seq_aux
        model_model_layers_26_mlp_gate_training_7996 = self.training
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 68, in moe_gate_fwd,  topk_idx, topk_weight, aux_loss = moe_route(
        moe_route_25_10304, moe_route_25_10305, moe_route_25_10306 = modeling.modeling_deepseek_modifier.moe_route(fused_rms_norm_affine_80_64583, self.model_model_layers_26_mlp_gate_weight_10303, topk_method='greedy', top_k=6, n_group=1, n_routed_experts=64, topk_group=1, training=model_model_layers_26_mlp_gate_training_7996, alpha=0.001, norm_topk_prob=False, routed_scaling_factor=1.0, seq_aux=True)
        del fused_rms_norm_affine_80_64583
        moe_route_25_10305 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_25_10305, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        moe_route_25_10306 = nnscaler.runtime.adapter.nn.identity_allreduce(moe_route_25_10306, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        fused_rms_norm_affine_80_64584 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_80_64584, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 81, in moe_fwd,  y = nnscaler_moe_gmm(
        nnscaler_moe_gmm_25_62969 = modeling.modeling_deepseek_modifier.nnscaler_moe_gmm(fused_rms_norm_affine_80_64584, moe_route_25_10304, moe_route_25_10305, moe_route_25_10306, self.model_model_layers_26_mlp_gate_projs_62945, self.model_model_layers_26_mlp_up_projs_62953, self.model_model_layers_26_mlp_down_projs_62961, n_routed_experts=64, local_expert_start=48, local_expert_end=56)
        del fused_rms_norm_affine_80_64584, moe_route_25_10304, moe_route_25_10305, moe_route_25_10306
        fused_rms_norm_affine_80_80008 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_80_64585, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_80_64585
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_186_63033 = torch.nn.functional.linear(fused_rms_norm_affine_80_80008, self.model_model_layers_26_mlp_shared_experts_gate_proj_weight_10311, bias=None)
        del fused_rms_norm_affine_80_80008
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        silu_26_63089 = torch.nn.functional.silu(linear_186_63033, inplace=False)
        del linear_186_63033
        fused_rms_norm_affine_80_80048 = nnscaler.runtime.adapter.nn.split_allgather(fused_rms_norm_affine_80_64586, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del fused_rms_norm_affine_80_64586
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_187_63113 = torch.nn.functional.linear(fused_rms_norm_affine_80_80048, self.model_model_layers_26_mlp_shared_experts_up_proj_weight_10314, bias=None)
        del fused_rms_norm_affine_80_80048
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_134_63161 = torch.mul(silu_26_63089, linear_187_63113)
        del silu_26_63089, linear_187_63113
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 389, in forward,  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_188_63185 = torch.nn.functional.linear(mul_134_63161, self.model_model_layers_26_mlp_shared_experts_down_proj_weight_10317, bias=None)
        del mul_134_63161
        nnscaler_moe_gmm_25_10310 = nnscaler.runtime.adapter.nn.allreduce_identity(nnscaler_moe_gmm_25_62969, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nnscaler_moe_gmm_25_62969
        linear_188_10318 = nnscaler.runtime.adapter.nn.allgather_split(linear_188_63185, dim=1, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_188_63185
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 90, in moe_fwd,  y = y + self.shared_experts(identity)
        add_133_10319 = torch.add(nnscaler_moe_gmm_25_10310, linear_188_10318, alpha=1)
        del nnscaler_moe_gmm_25_10310, linear_188_10318
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1274, in forward,  hidden_states = residual + hidden_states
        add_134_10320 = torch.add(add_132_86488, add_133_10319, alpha=1)
        del add_132_86488, add_133_10319
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek_modifier.py", line 58, in rmsnorm_fwd,  return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
        fused_rms_norm_affine_81_10322 = apex.normalization.fused_layer_norm.fused_rms_norm_affine(add_134_10320, self.model_model_norm_weight_10321, (2048,), 1e-06)
        del add_134_10320
        fused_rms_norm_affine_81_10322 = nnscaler.runtime.adapter.nn.identity_allreduce(fused_rms_norm_affine_81_10322, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1689, in forward,  logits = self.lm_head(hidden_states)
        linear_189_63353 = torch.nn.functional.linear(fused_rms_norm_affine_81_10322, self.model_lm_head_weight_63345, bias=None)
        del fused_rms_norm_affine_81_10322
        linear_189_63305 = nnscaler.runtime.adapter.nn.alltoall_alltoall(linear_189_63353, idim=2, odim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del linear_189_63353
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/modeling/modeling_deepseek.py", line 1690, in forward,  logits = logits.float()
        float_1_63361 = torch.Tensor.float(linear_189_63305)
        del linear_189_63305
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/train.py", line 130, in forward,  logits = outputs[0].view(-1, outputs[0].size(-1))
        view_135_63385 = torch.Tensor.view(float_1_63361, size=(-1, 102400))
        del float_1_63361
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/train.py", line 131, in forward,  labels = samples['target'].view(-1)
        getitem_947_10327 = _operator.getitem(samples_10336, 'target')
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/train.py", line 131, in forward,  labels = samples['target'].view(-1)
        view_136_10328 = torch.Tensor.view(getitem_947_10327, size=(-1,))
        del getitem_947_10327
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/train.py", line 132, in forward,  normalized_logits = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
        log_softmax_63401 = torch.nn.functional.log_softmax(view_135_63385, dim=-1, _stacklevel=3, dtype=torch.float32)
        del view_135_63385
        view_136_63409 = nnscaler.runtime.adapter.chunk(view_136_10328, dim=0, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del view_136_10328
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/train.py", line 133, in forward,  loss = torch.nn.functional.nll_loss(normalized_logits, labels, reduction='sum', ignore_index=IGNORE_IDX)
        nll_loss_63417 = torch.nn.functional.nll_loss(log_softmax_63401, view_136_63409, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='sum')
        del log_softmax_63401, view_136_63409
        nll_loss_8001 = nnscaler.runtime.adapter.nn.allreduce_identity(nll_loss_63417, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        # create at IRAdapterGener:autoref, comment before transformation: activation
        nll_loss_80160 = nnscaler.runtime.function.multiref(nll_loss_63417, times=1)
        del nll_loss_63417
        nll_loss_64593 = nnscaler.runtime.adapter.all_reduce(nll_loss_80160, ranks=[0, 1, 2, 3, 4, 5, 6, 7])
        del nll_loss_80160
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/train.py", line 134, in forward,  return loss, loss.data, samples['ntokens'], samples['nsentences']
        getattr_1061_8002 = builtins.getattr(nll_loss_64593, 'data')
        del nll_loss_64593
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/train.py", line 134, in forward,  return loss, loss.data, samples['ntokens'], samples['nsentences']
        getitem_948_7999 = _operator.getitem(samples_10336, 'ntokens')
        # File "/home/aiscuser/MagicCube/examples/deepseek_coder_v2_lite/train.py", line 134, in forward,  return loss, loss.data, samples['ntokens'], samples['nsentences']
        getitem_949_8000 = _operator.getitem(samples_10336, 'nsentences')
        return nll_loss_8001, getattr_1061_8002, getitem_948_7999, getitem_949_8000
    
    def reducer150596(self):
        self.wreducer150596.sync_grads()
        return 
    
    def _forward_impl(self, *args, **kwargs):
        raise NotImplementedError("Code of forward is not generated. You should use module.train_step/module.infer_step instead.")


########## Generated Schedule Code ###########
import torch
import nnscaler

def _train_step(model, dataloader_10337):
    _ = None
    model.zero_grad()
    samples_10336 = next(*(dataloader_10337, ))
    nll_loss_8001, getattr_1061_8002, getitem_948_7999, getitem_949_8000 = nnscaler.runtime.executor.fexecute('segment160545', model.segment160545, *(samples_10336, ), requires_grad=True)
    _ = nnscaler.runtime.executor.backward('segment160545', (), (nll_loss_8001, ), (None, ))
    nll_loss_8001 = nll_loss_8001.detach()
    _ = nnscaler.runtime.executor.aexecute(model.reducer150596, *(), requires_grad=False)
    return nll_loss_8001, getattr_1061_8002, getitem_948_7999, getitem_949_8000


def _infer_step(model, dataloader_10337):
    _ = None
    samples_10336 = next(*(dataloader_10337, ))
    nll_loss_8001, getattr_1061_8002, getitem_948_7999, getitem_949_8000 = nnscaler.runtime.executor.fexecute('segment160545', model.segment160545, *(samples_10336, ), requires_grad=False)
    return nll_loss_8001, getattr_1061_8002, getitem_948_7999, getitem_949_8000
