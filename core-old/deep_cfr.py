# deep_cfr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
import os
import collections
import subprocess
import time
import warnings
import pokers as pkrs
from src.core.model import (PokerNetwork, SizingAnchorNet, StrategySizingNet,
                             QValueNetwork, SizingQNetwork, encode_state, encode_state_with_position,
                             VERBOSE, set_verbose)
from src.utils.config import cfg_get, cfg_reservoir_flag
from src.utils.settings import STRICT_CHECKING
from src.utils.logging import log_game_error
from src.core.checkpointing import _resolve_model_save_path, _strip_legacy_bucket_head_keys














from .buffers import (
    PrioritizedMemory,
    PolicyGradientMemory,
    AdvantageBuffer,
    StrategyBuffer,
    SizingAdvantageBuffer,
    SizingStrategyBuffer,
    QValueBuffer,
    SizingQBuffer,
)

from src.core.sizing import (
    regret_matching_anchors,
    sample_waugh_sizing,
    credit_assignment,
    compute_sizing_heat_weights,
)


















class DeepCFRAgent:
    def __init__(self, player_id=0, num_players=None, memory_size=None,
                 device='cpu', pg_lr=None, entropy_bonus=None,
                 pg_memory_size=None):
        self.player_id = player_id
        self.num_players = num_players or cfg_get('num_players', 6)
        self.num_trainable_players = self.num_players
        self.device = device

        self.num_actions = cfg_get('num_actions', 4)
        self.use_multi_agent = cfg_get('use_multi_agent_advantage', False)

        base_input_size = 52 + 52 + 5 + 1 + self.num_players + self.num_players + self.num_players * 4 + 1 + 1 + 4 + 5
        if self.use_multi_agent:
            input_size = base_input_size + self.num_players
        else:
            input_size = base_input_size

        self.min_bet_size = cfg_get('min_bet_size', 0.1)
        self.max_bet_size = cfg_get('max_bet_size', 3.0)
        self.big_blind = float(cfg_get('big_blind', 2.0))
        hidden_size = cfg_get('hidden_size', 256)

        # One trainable advantage network: every seat shares the observed player's policy.
        self.advantage_net = PokerNetwork(
            input_size=input_size, hidden_size=hidden_size, num_actions=self.num_actions
        ).to(device)
        # Compatibility alias for older callers/checkpoints; it deliberately has one entry.
        self.advantage_nets = nn.ModuleList([self.advantage_net])
        self.optimizer = optim.AdamW(
            self.advantage_net.parameters(),
            lr=cfg_get('advantage_lr', 1e-4),
            weight_decay=float(cfg_get('advantage_weight_decay', 1e-5)),
        )
        self.advantage_optimizers = [self.optimizer]

        _memory_size = memory_size or cfg_get('memory_size', 300000)
        _adv_mem = cfg_get('advantage_memory_size', 16384)
        _strat_mem = cfg_get('strategy_memory_size', 300000)
        self.advantage_batch_size = cfg_get('advantage_batch_size', 256)
        self.strategy_batch_size = cfg_get('strategy_batch_size', 128)
        self.advantage_epochs = cfg_get('advantage_epochs', 1)
        self.strategy_epochs = cfg_get('strategy_epochs', 3)
        # One persistent replay buffer for the observed player.
        self.advantage_buffer = AdvantageBuffer(
            _adv_mem, state_dim=input_size, num_actions=self.num_actions
        )
        self.advantage_buffers = [self.advantage_buffer]

        self.strategy_net = PokerNetwork(
            input_size=input_size, hidden_size=hidden_size, num_actions=self.num_actions
        ).to(device)
        self.strategy_optimizer = optim.AdamW(self.strategy_net.parameters(),
                                               lr=cfg_get('strategy_lr', 5e-5),
                                               weight_decay=float(cfg_get('strategy_weight_decay', 1e-5)))
        self.strategy_buffer = StrategyBuffer(_strat_mem, state_dim=input_size, num_actions=self.num_actions)
        self.advantage_buffer_reservoir = cfg_reservoir_flag(
            'advantage_buffer_reservoir'
        )
        self.strategy_buffer_reservoir = cfg_reservoir_flag(
            'strategy_buffer_reservoir'
        )
        self.sizing_advantage_buffer_reservoir = cfg_reservoir_flag(
            'sizing_advantage_buffer_reservoir'
        )
        self.sizing_strategy_buffer_reservoir = cfg_reservoir_flag(
            'sizing_strategy_buffer_reservoir'
        )
        # Сохранение 4 CFR replay-буферов в чекпоинт — безусловно (не зависит от sizing_q).
        self.save_replay_buffers_in_checkpoint = bool(
            cfg_get('save_replay_buffers_in_checkpoint', False)
        )

        # Sizing-сети: фиксированная сетка сайзингов (Bug #50)
        sizing_hidden = cfg_get('sizing_hidden_size', 128)
        self.num_anchors = int(cfg_get('num_anchors', 15))
        self.anchors = list(cfg_get('fixed_sizing_grid', cfg_get('anchor_sizes', [
            0.10, 0.25, 0.33, 0.50, 0.66, 0.75,
            1.00, 1.25, 1.50, 1.75, 2.00,
            2.25, 2.50, 2.75, 3.00,
        ])))
        if len(self.anchors) != self.num_anchors:
            raise ValueError("len(fixed_sizing_grid) должен совпадать с num_anchors")
        self.anchors = [float(np.clip(a, self.min_bet_size, self.max_bet_size)) for a in self.anchors]
        self.anchors.sort()
        self.anchors_arr = np.array(self.anchors, dtype=np.float32)
        self.sizing_min_prob_start = float(cfg_get('sizing_min_prob_start', 0.05))
        self.sizing_min_prob_end = float(cfg_get('sizing_min_prob_end', cfg_get('sizing_min_prob', 0.0125)))
        self.sizing_min_prob_decay_iterations = int(cfg_get('sizing_min_prob_decay_iterations', 1000))
        self.sizing_bucket_min_prob_start = float(cfg_get('sizing_bucket_min_prob_start', 0.0))
        self.sizing_bucket_min_prob_end = float(cfg_get('sizing_bucket_min_prob_end', 0.0))
        self.sizing_bucket_explore_prob = float(cfg_get('sizing_bucket_explore_prob', 0.2))
        self.sizing_importance_weight_clip = float(cfg_get('sizing_importance_weight_clip', 10.0))
        self.sizing_bucket_min_prob_decay_iterations = int(
            cfg_get('sizing_bucket_min_prob_decay_iterations', self.sizing_min_prob_decay_iterations)
        )
        self.sizing_bucket_groups = self._resolve_sizing_bucket_groups(cfg_get('sizing_bucket_groups', [
            [0.10, 0.25, 0.33, 0.50, 0.66],
            [0.75, 1.00, 1.25, 1.50, 1.75],
            [2.00, 2.25, 2.50, 2.75, 3.00],
        ]))
        self.sizing_bucket_loss_weight = float(cfg_get('sizing_bucket_loss_weight', 0.3))
        self.sizing_cfr_mode = bool(cfg_get('sizing_cfr_mode', False))
        self.sizing_availability_on_input = cfg_get('sizing_availability_on_input', False)

        # Heat weighting params

        self.sizing_heat_temperature = float(cfg_get('sizing_heat_temperature', 0.35))
        self.sizing_heat_min_advantage = float(cfg_get('sizing_heat_min_advantage', 0.03))
        self.sizing_heat_top_p = float(cfg_get('sizing_heat_top_p', 0.90))

        # Probe params (для cold-start coverage)
        self.sizing_probe_uniform_weight = float(cfg_get('sizing_probe_uniform_weight', 0.40))
        self.sizing_probe_anchor_weight = float(cfg_get('sizing_probe_anchor_weight', 0.40))
        self.sizing_probe_slot_jitter_weight = float(cfg_get('sizing_probe_slot_jitter_weight', 0.20))
        self.sizing_probe_jitter = float(cfg_get('sizing_probe_jitter', 0.10))

        # One-step lookahead params (Bug #64)
        self.sizing_lookahead_enabled = cfg_get('sizing_lookahead_enabled', True)
        self.sizing_lookahead_prob = float(cfg_get('sizing_lookahead_prob', 0.10))
        self.sizing_lookahead_max_anchors = int(cfg_get('sizing_lookahead_max_anchors', 5))

        if self.sizing_availability_on_input:
            sizing_input_size = input_size + 2 * self.num_anchors
        else:
            sizing_input_size = input_size
        self.sizing_input_size = sizing_input_size

        # One trainable sizing-advantage network shared by the observed player and opponents.
        self.advantage_sizing_net = SizingAnchorNet(
            input_size=sizing_input_size, hidden_size=sizing_hidden,
            num_sizes=self.num_anchors,
        ).to(device)
        self.advantage_sizing_nets = nn.ModuleList([self.advantage_sizing_net])
        self.sizing_target_net = SizingAnchorNet(
            input_size=sizing_input_size, hidden_size=sizing_hidden,
            num_sizes=self.num_anchors,
        ).to(device)
        self.sizing_target_net.load_state_dict(self.advantage_sizing_net.state_dict())
        self.sizing_target_net.eval()
        for p in self.sizing_target_net.parameters():
            p.requires_grad = False
        self.strategy_sizing_net = StrategySizingNet(
            input_size=sizing_input_size, hidden_size=sizing_hidden,
            num_sizes=self.num_anchors
        ).to(device)

        _pg_memory_size = pg_memory_size or cfg_get('pg_memory_size', 10000)
        self.pg_memory = PolicyGradientMemory(_pg_memory_size, state_dim=input_size)
        self.pg_lr = pg_lr or cfg_get('pg_lr', 1e-4)
        _sizing_lr = cfg_get('sizing_anchor_lr', cfg_get('sizing_lr', self.pg_lr))
        _strategy_sizing_lr = cfg_get('strategy_sizing_lr', 5e-5)

        self.sizing_optimizer = optim.AdamW(
            self.advantage_sizing_net.parameters(), lr=_sizing_lr,
            weight_decay=float(cfg_get('sizing_anchor_weight_decay', 1e-5)),
        )
        self.sizing_optimizers = [self.sizing_optimizer]
        self.strategy_sizing_optimizer = optim.AdamW(
            self.strategy_sizing_net.parameters(), lr=_strategy_sizing_lr,
            weight_decay=float(cfg_get('sizing_strategy_weight_decay', 1e-5))
        )
        self.value_optimizer = None
        self.entropy_bonus = entropy_bonus or cfg_get('entropy_bonus', 0.01)
        self.pg_memory_size = _pg_memory_size
        self.strategy_sizing_train_steps = int(cfg_get('strategy_sizing_train_steps', 20))
        self.sizing_strategy_buffer = SizingStrategyBuffer(
            cfg_get('sizing_strategy_buffer_size', 300000),
            state_dim=sizing_input_size,
            num_anchors=self.num_anchors,
        )

        # One persistent sizing-advantage replay buffer for the observed player.
        if self.sizing_cfr_mode:
            self.sizing_advantage_buffer = SizingAdvantageBuffer(
                cfg_get('sizing_advantage_buffer_size', 16384),
                state_dim=sizing_input_size,
                num_anchors=self.num_anchors,
            )
            self.sizing_advantage_buffers = [self.sizing_advantage_buffer]
        else:
            self.sizing_advantage_buffers = None
            self.sizing_advantage_buffer = None

        # --- EMA target smoothing для sizing strategy buffer ---
        self.sizing_target_ema_enabled = cfg_get('sizing_target_ema_enabled', True)
        self.sizing_target_ema_beta = cfg_get('sizing_target_ema_beta', 0.9)
        self.sizing_target_ema_min_mass = cfg_get('sizing_target_ema_min_mass', 1e-8)
        self.sizing_target_ema_warmup_steps = cfg_get('sizing_target_ema_warmup_steps', 5000)
        self.sizing_target_ema_beta_start = cfg_get('sizing_target_ema_beta_start', 0.0)
        self.sizing_target_ema_beta_end = cfg_get('sizing_target_ema_beta_end', 0.9)
        self.sizing_target_ema_max_size = cfg_get('sizing_target_ema_max_size', 200000)
        self.sizing_target_ema = collections.OrderedDict()

        # --- Sparse target для strategy sizing ---
        self.sizing_target_sparsify_enabled = cfg_get('sizing_target_sparsify_enabled', True)
        self.sizing_target_top_buckets = cfg_get('sizing_target_top_buckets', 2)
        self.sizing_target_top_per_bucket = cfg_get('sizing_target_top_per_bucket', 1)
        self.sizing_sparsify_fallback_count = 0
        self.sizing_sparsify_total_count = 0

        # --- Inference mode ---
        self.sizing_inference_mode = cfg_get('sizing_inference_mode', 'mean')
        self.inference_min_action_prob = float(cfg_get('inference_min_action_prob', 0.0))
        self.sizing_inference_min_prob = float(cfg_get('sizing_inference_min_prob', 0.0))

        self.pg_baseline_mean = 0.0
        self.pg_baseline_var = 1.0
        self.pg_baseline_momentum = 0.99

        self.iteration_count = 0

        self.max_regret_seen = 1.0

        self.discount_alpha = cfg_get('discount_alpha', 2.0)
        self.discount_gamma = cfg_get('discount_gamma', 2.0)
        self.advantage_accumulation = cfg_get('advantage_accumulation', 'dcfr_plus')

        self.target_net = PokerNetwork(
            input_size=input_size, hidden_size=hidden_size, num_actions=self.num_actions
        ).to(device)
        self.target_net.load_state_dict(self.advantage_net.state_dict())
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.target_nets = nn.ModuleList([self.target_net])

        self.use_q_baseline = cfg_get('use_q_baseline', True)
        self.q_reward_scale = cfg_get('q_reward_scale', 'bb')
        self.q_reward_scale_value = cfg_get('q_reward_scale_value', None)
        self.q_target_update_interval = cfg_get('q_target_update_interval', 100)
        self.q_bootstrap_policy_mix_enabled = cfg_get('q_bootstrap_policy_mix_enabled', False)
        mix_val = float(cfg_get('q_bootstrap_policy_uniform_mix', 0.10))
        self.q_bootstrap_policy_uniform_mix = float(np.clip(mix_val, 0.0, 1.0))
        self.q_grad_clip_max_norm = float(cfg_get('q_grad_clip_max_norm', 1.0))
        self.q_terminal_balanced_loss_enabled = cfg_get('q_terminal_balanced_loss_enabled', False)
        self.q_terminal_loss_alpha = float(cfg_get('q_terminal_loss_alpha', 0.5))
        self.q_reward_propagation = cfg_get('q_reward_propagation', 'none')
        self.q_target_norm = cfg_get('q_target_norm', 'none')
        self.q_bootstrap_per_player_enabled = cfg_get('q_bootstrap_per_player_enabled', False)

        if self.use_q_baseline:
            q_hidden = cfg_get('q_hidden_size', 128)
            self._q_input_size = input_size
            self._q_hidden = q_hidden
            self._q_lr = float(cfg_get('q_lr', 1e-3))
            self.q_net = QValueNetwork(input_size, q_hidden, self.num_actions).to(device)
            self.q_optimizer = optim.Adam(
                self.q_net.parameters(), lr=self._q_lr
            )
            self.q_buffer = QValueBuffer(
                cfg_get('q_buffer_size', 300000),
                state_dim=input_size,
                num_actions=self.num_actions
            )
            self.q_target_net = QValueNetwork(input_size, q_hidden, self.num_actions).to(device)
            self.q_target_net.load_state_dict(self.q_net.state_dict())
            self.q_target_net.eval()
            for p in self.q_target_net.parameters():
                p.requires_grad = False
        else:
            self.q_net = None
            self.q_optimizer = None
            self.q_buffer = None
            self.q_target_net = None
            self._q_input_size = input_size
            self._q_hidden = cfg_get('q_hidden_size', 128)
            self._q_lr = float(cfg_get('q_lr', 1e-3))
        self.q_loaded_from_checkpoint = False

        # #96: advantage-regret scale
        self.advantage_regret_norm = cfg_get('advantage_regret_norm', 'none')
        _arc = cfg_get('advantage_regret_clip', None)
        self.advantage_regret_clip = float(_arc) if _arc is not None else None
        self.advantage_loss = cfg_get('advantage_loss', 'mse')
        self.advantage_huber_delta = float(cfg_get('advantage_huber_delta', 1.0))
        self.advantage_reward_scale = float(cfg_get('advantage_reward_scale', 1.0))

        self.sizing_q_enabled = cfg_get('sizing_q_enabled', False)
        if self.sizing_q_enabled:
            sq_hidden = cfg_get('sizing_q_hidden_size', 128)
            sq_embed = cfg_get('sizing_q_size_embed_dim', 16)
            self.sizing_q_net = SizingQNetwork(
                state_dim=sizing_input_size, hidden_size=sq_hidden, size_embed_dim=sq_embed
            ).to(device)
            self.sizing_q_optimizer = optim.AdamW(
                self.sizing_q_net.parameters(), lr=cfg_get('sizing_q_lr', 1e-4),
                weight_decay=float(cfg_get('sizing_q_weight_decay', 1e-5))
            )
            self.sizing_q_buffer = SizingQBuffer(
                cfg_get('sizing_q_buffer_size', 100000), state_dim=sizing_input_size,
                num_anchors=self.num_anchors
            )
            self.sizing_candidate_sizes = cfg_get(
                'sizing_candidate_sizes', [0.82, 1.55, 2.27]
            )
            self.sizing_anchor_sizes = cfg_get('sizing_anchor_sizes', [0.82, 1.55, 2.27])
            self.sizing_policy_samples_per_state = cfg_get('sizing_policy_samples_per_state', 4)
            self.sizing_awr_temperature = cfg_get('sizing_awr_temperature', 0.5)
            self.sizing_awr_uniform_mix = cfg_get('sizing_awr_uniform_mix', 0.2)
            self.sizing_reinforce_weight = cfg_get('sizing_reinforce_weight', 0.0)
            self.sizing_q_target_normalization = cfg_get('sizing_q_target_normalization', 'pot')
            self.sizing_q_train_steps = cfg_get('sizing_q_train_steps_per_iteration', 2)
            self.sizing_probe_prob_start = cfg_get('sizing_probe_prob_start', 0.30)
            self.sizing_probe_prob_end = cfg_get('sizing_probe_prob_end', 0.05)
            self.sizing_probe_decay_iterations = cfg_get('sizing_probe_decay_iterations', 1000)
            self.sizing_q_min_samples_per_anchor = cfg_get('sizing_q_min_samples_per_anchor', 128)
            self.sizing_q_min_total_samples = cfg_get('sizing_q_min_total_samples', 5000)
            self.sizing_q_bootstrap_scale_fix = cfg_get('sizing_q_bootstrap_scale_fix', False)
            self.sizing_q_stratified_sampling = cfg_get('sizing_q_stratified_sampling', False)
            self.sizing_q_target_diagnostics_enabled = cfg_get('sizing_q_target_diagnostics_enabled', False)
            self.sizing_q_bootstrap_target_net = cfg_get('sizing_q_bootstrap_target_net', False)
            self.sizing_q_train_zscore = cfg_get('sizing_q_train_zscore', False)
            self.sizing_q_zscore_eps = float(cfg_get('sizing_q_zscore_eps', 1e-6))
            self.sizing_q_loss = cfg_get('sizing_q_loss', 'mse')
            self.sizing_q_huber_delta = float(cfg_get('sizing_q_huber_delta', 1.0))
            self.sizing_q_src2_loss_weight_enabled = cfg_get('sizing_q_src2_loss_weight_enabled', False)
            self.sizing_q_src2_weight_start = float(cfg_get('sizing_q_src2_weight_start', 1.0))
            self.sizing_q_src2_weight_end = float(cfg_get('sizing_q_src2_weight_end', 1.0))
            self.sizing_q_src2_weight_decay_iterations = int(cfg_get('sizing_q_src2_weight_decay_iterations', 400))
            self.save_q_buffer_in_checkpoint = cfg_get('save_q_buffer_in_checkpoint', False)
            self.sizing_q_legal_anchor_mask_enabled = cfg_get('sizing_q_legal_anchor_mask_enabled', False)
            self.sizing_q_mask_exclude_kinds = cfg_get('sizing_q_mask_exclude_kinds', ['ALL_IN', 'MIN_RAISE'])
            self.sizing_q_kind_filter_enabled = cfg_get('sizing_q_kind_filter_enabled', False)
            self.sizing_q_kind_filter_mode = cfg_get('sizing_q_kind_filter_mode', 'drop')
            self.sizing_q_kind_filter_downweight = float(cfg_get('sizing_q_kind_filter_downweight', 0.25))
            self.sizing_q_replay_clear_once = cfg_get('sizing_q_replay_clear_once', False)
            self._sizing_q_replay_clear_once_done = False
            self.sizing_q_mask_diagnostics_enabled = cfg_get('sizing_q_mask_diagnostics_enabled', False)
            self.sizing_q_mask_dry_run = cfg_get('sizing_q_mask_dry_run', False)
            self.sizing_q_replay_clear_diagnostics_enabled = cfg_get('sizing_q_replay_clear_diagnostics_enabled', False)
            self.sizing_q_selected_credit_enabled = cfg_get('sizing_q_selected_credit_enabled', False)
            self.sizing_anchor_availability_enabled = cfg_get('sizing_anchor_availability_enabled', False)
            self.sizing_allin_boundary_enabled = cfg_get('sizing_allin_boundary_enabled', False)
            self.sizing_preflop_disable_buckets = cfg_get('sizing_preflop_disable_buckets', False)
            self.sizing_bucket_head_enabled = cfg_get('sizing_bucket_head_enabled', True)
        else:
            self.sizing_q_net = None
            self.sizing_q_optimizer = None
            self.sizing_q_buffer = None
            self.sizing_candidate_sizes = cfg_get('sizing_candidate_sizes', [0.82, 1.55, 2.27])
            self.sizing_anchor_sizes = cfg_get('sizing_anchor_sizes', [0.82, 1.55, 2.27])
            self.sizing_policy_samples_per_state = cfg_get('sizing_policy_samples_per_state', 4)
            self.sizing_awr_temperature = cfg_get('sizing_awr_temperature', 0.5)
            self.sizing_awr_uniform_mix = cfg_get('sizing_awr_uniform_mix', 0.2)
            self.sizing_reinforce_weight = 1.0
            self.sizing_q_target_normalization = cfg_get('sizing_q_target_normalization', 'pot')
            self.sizing_q_train_steps = cfg_get('sizing_q_train_steps_per_iteration', 2)
            self.sizing_probe_prob_start = cfg_get('sizing_probe_prob_start', 0.30)
            self.sizing_probe_prob_end = cfg_get('sizing_probe_prob_end', 0.05)
            self.sizing_probe_decay_iterations = cfg_get('sizing_probe_decay_iterations', 1000)
            self.sizing_q_min_samples_per_anchor = cfg_get('sizing_q_min_samples_per_anchor', 128)
            self.sizing_q_bootstrap_scale_fix = False
            self.sizing_q_stratified_sampling = False
            self.sizing_q_target_diagnostics_enabled = False
            self.sizing_q_bootstrap_target_net = False
            self.sizing_q_train_zscore = False
            self.sizing_q_zscore_eps = 1e-6
            self.sizing_q_loss = 'mse'
            self.sizing_q_huber_delta = 1.0
            self.sizing_q_src2_loss_weight_enabled = False
            self.sizing_q_src2_weight_start = 1.0
            self.sizing_q_src2_weight_end = 1.0
            self.sizing_q_src2_weight_decay_iterations = 400
            self.save_q_buffer_in_checkpoint = False
            self.sizing_q_legal_anchor_mask_enabled = False
            self.sizing_q_mask_exclude_kinds = ['ALL_IN', 'MIN_RAISE']
            self.sizing_q_kind_filter_enabled = False
            self.sizing_q_kind_filter_mode = 'drop'
            self.sizing_q_kind_filter_downweight = 0.25
            self.sizing_q_replay_clear_once = False
            self._sizing_q_replay_clear_once_done = False
            self.sizing_q_mask_diagnostics_enabled = False
            self.sizing_q_mask_dry_run = False
            self.sizing_q_replay_clear_diagnostics_enabled = False
            self.sizing_q_selected_credit_enabled = False
            self.sizing_anchor_availability_enabled = False
            self.sizing_allin_boundary_enabled = False
            self.sizing_preflop_disable_buckets = False
            self.sizing_bucket_head_enabled = True

        self.sizing_q_min_total_samples = cfg_get('sizing_q_min_total_samples', 5000)
        self.sizing_q_target_diag = {}
        self.sizing_q_kind_filter_diag = {
            'normal_kept': 0,
            'min_raise_dropped': 0,
            'all_in_dropped': 0,
            'unknown_dropped': 0,
            'downweighted': 0,
            'zero_weight_batches': 0,
            'cleared_once': 0,
        }
        self.sizing_q_mask_diag = {}
        self.sizing_availability_diag = {
            'calls': 0,
            'avail_mean_sum': 0.0,
            'boundary_present': 0,
            'boundary_allin_idx_sum': 0,
            'boundary_allin_idx_sqsum': 0.0,
            'allin_selected': 0,
            'allin_total_selections': 0,
            'preflop_selections': 0,
            'postflop_selections': 0,
            'eff_idx_mismatches': 0,
            'eff_idx_total': 0,
        }
        self.raise_funnel_diag = {
            'nodes_seen': 0,
            'raise_legal_count': 0,
            'action_strategy_raise_mass_sum': 0.0,
            'sampling_policy_raise_mass_sum': 0.0,
            'sampled_raise_count': 0,
            'apply_raise_ok_count': 0,
            'q_buffer_raise_add_count': 0,
            'hero_os': {'raise_legal': 0, 'raise_sampled': 0, 'raise_apply_ok': 0, 'q_buffer_raise_add': 0},
            'hero_full': {'raise_legal': 0, 'raise_sampled': 0, 'raise_apply_ok': 0, 'q_buffer_raise_add': 0},
            'opponent': {'raise_legal': 0, 'raise_sampled': 0, 'raise_apply_ok': 0, 'q_buffer_raise_add': 0},
        }
        self.sizing_q_insert_kind_diag = {
            'by_source': {},
            'by_callsite': {},
        }
        self.sizing_q_replay_clear_diag = {
            'before_clear': None,
            'cleared': False,
            'refill_snapshots': [],
        }
        # Bug #74 диаг: инструментируем путь записи source=2 (lookahead), чтобы понять,
        # почему target == 0.0. Бегущие агрегаты [n, sum, sumsq, min, max] по каждой метрике.
        self.sizing_lookahead_diag = {
            'counts': {
                'attempts': 0,
                'applied_ok': 0,
                'added': 0,
                'final_state': 0,
                'nonterminal': 0,
                'bootstrap_debug_missing': 0,
                'vk_near_zero': 0,
                'target_near_zero': 0,
                'using_target_net': 0,
                'q_input_has_nan': 0,
                'next_is_hero': 0,
            },
            'stats': {},
        }
        self._last_bootstrap_debug = None
        self.sizing_path_diag = {
            'hero_os_raise': np.zeros(self.num_anchors, dtype=np.int64),
            'hero_full_raise': np.zeros(self.num_anchors, dtype=np.int64),
            'opponent_raise': np.zeros(self.num_anchors, dtype=np.int64),
        }
        self.sizing_selected_effective_diag = {
            'hero_os_raise': np.zeros((self.num_anchors, self.num_anchors), dtype=np.int64),
            'hero_full_raise': np.zeros((self.num_anchors, self.num_anchors), dtype=np.int64),
            'opponent_raise': np.zeros((self.num_anchors, self.num_anchors), dtype=np.int64),
        }
        self.sizing_selected_effective_kind_diag = {
            key: {
                'selected_kind': {'NORMAL': 0, 'MIN_RAISE': 0, 'ALL_IN': 0},
                'bucket_kind': {'NORMAL': 0, 'MIN_RAISE': 0, 'ALL_IN': 0},
                'bucket_relation': {'below_effective': 0, 'equal_effective': 0, 'above_effective': 0},
            }
            for key in self.sizing_selected_effective_diag
        }
        self.hybrid_os_enabled = cfg_get('hybrid_os_enabled', False)
        self.hybrid_os_min_active_opponents = cfg_get('hybrid_os_min_active_opponents', 2)
        self.hybrid_os_epsilon_start = cfg_get('hybrid_os_epsilon_start', 0.25)
        self.hybrid_os_epsilon_end = cfg_get('hybrid_os_epsilon_end', 0.05)
        self.hybrid_os_epsilon_decay_iterations = cfg_get('hybrid_os_epsilon_decay_iterations', 1000)
        self.hybrid_os_importance_weight_clip = cfg_get('hybrid_os_importance_weight_clip', 10.0)
        self.hybrid_os_regret_clip = cfg_get('hybrid_os_regret_clip', None)
        self.hybrid_os_pg_min_raise_sample_prob = cfg_get('hybrid_os_pg_min_raise_sample_prob', 0.15)
        self.hybrid_os_q_warmup_iterations = cfg_get('hybrid_os_q_warmup_iterations', 50)
        self.hybrid_os_min_q_buffer_size = cfg_get('hybrid_os_min_q_buffer_size', 5000)
        self.hybrid_os_allow_loaded_q_without_buffer = cfg_get('hybrid_os_allow_loaded_q_without_buffer', True)
        self.hybrid_os_loaded_q_min_iteration = cfg_get('hybrid_os_loaded_q_min_iteration', 5000)
        self.os_force_full_traversal = cfg_get('os_force_full_traversal', False)
        self.pg_train_steps_per_iteration = cfg_get('pg_train_steps_per_iteration', 5)
        self.pg_freshness_window = cfg_get('pg_freshness_window', 5)
        self.q_train_steps_per_iteration = cfg_get('q_train_steps_per_iteration', 2)
        self.q_train_steps_with_os = cfg_get('q_train_steps_with_os', 5)
        self.q_refit_enabled = cfg_get('q_refit_enabled', False)
        self.q_refit_steps = cfg_get('q_refit_steps', 200)
        self.q_refit_batch_size = cfg_get('q_refit_batch_size', 2048)
        self.q_refit_sync_every = cfg_get('q_refit_sync_every', 50)

        self.last_advantage_train_steps = 0
        self.last_advantage_effective_batch_size = 0
        self._traversal_random_agent = None

        self.reset_traversal_stats()

    def _adv_net(self, player_id):
        return self.advantage_net

    def _adv_buffer(self, player_id):
        return self.advantage_buffer

    def _target_net(self, player_id):
        return self.target_net

    def _adv_optimizer(self, player_id):
        return self.optimizer

    def _encode_state(self, state, player_id=0):
        if self.use_multi_agent:
            return encode_state_with_position(state, player_id)
        return encode_state(state, player_id)

    def _encode_state_for_sizing(self, state, player_id):
        """Bug #94: расширенный энкод для sizing-сетей — добавляет векторы доступности.

        При sizing_availability_on_input=False: возвращает base-энкод (input_size),
        размерность не меняется, cold-start отсутствует.
        При sizing_availability_on_input=True: возвращает base + 2*num_anchors
        (векторы доступности), sizing_input_size = input_size + 2*num_anchors,
        sizing-сети переобучаются с нуля.
        """
        if not self.sizing_availability_on_input:
            return self._encode_state(state, player_id)
        base = self._encode_state(state, player_id)
        avail, allin_idx = self._anchor_availability(state)
        extra = np.zeros(2 * self.num_anchors, dtype=np.float32)
        extra[:self.num_anchors] = avail.astype(np.float32)
        if allin_idx >= 0:
            extra[self.num_anchors + allin_idx] = 1.0
        return np.concatenate([base, extra])

    def reset_traversal_stats(self):
        self.traversal_nodes = 0
        self.traversal_terminal_nodes = 0
        self.traversal_max_depth_observed = 0
        self.traversal_max_depth_hits = 0
        self.traversal_traversing_decision_nodes = 0
        self.traversal_opponent_decision_nodes = 0

        self.os_nodes = 0
        self.os_traversing_nodes = 0
        self.postflop_multiway_nodes = 0
        self.os_sampled_raise_nodes = 0
        self.os_sampled_non_raise_nodes = 0

        self.os_importance_weight_sum = 0.0
        self.os_importance_weight_max = 0.0
        self.os_importance_weight_count = 0

        self.os_q_correction_abs_sum = 0.0
        self.os_q_correction_count = 0
        self.os_q_baseline_abs_error_sum = 0.0
        self.os_q_baseline_abs_error_max = 0.0
        self.os_q_baseline_abs_error_count = 0

        self.os_regret_abs_sum = 0.0
        self.os_regret_sq_sum = 0.0
        self.os_regret_count = 0
        self.os_max_abs_regret = 0.0
        self.os_regret_clip_count = 0

        for key in self.sizing_path_diag:
            self.sizing_path_diag[key].fill(0)

        for matrix in self.sizing_selected_effective_diag.values():
            matrix.fill(0)

        for diag in self.sizing_selected_effective_kind_diag.values():
            for group in diag.values():
                for key in group:
                    group[key] = 0

        persistent_sizing_q_kind_filter_keys = {'cleared_once'}
        for key in self.sizing_q_kind_filter_diag:
            if key not in persistent_sizing_q_kind_filter_keys:
                self.sizing_q_kind_filter_diag[key] = 0

        self.sizing_q_mask_diag.clear()
        for key in self.raise_funnel_diag:
            if isinstance(self.raise_funnel_diag[key], dict):
                for subkey in self.raise_funnel_diag[key]:
                    self.raise_funnel_diag[key][subkey] = 0
            else:
                self.raise_funnel_diag[key] = 0
        self.sizing_q_insert_kind_diag['by_source'].clear()
        self.sizing_q_insert_kind_diag['by_callsite'].clear()
        self.sizing_q_replay_clear_diag['refill_snapshots'].clear()

    def get_traversal_stats(self):
        base = {
            'nodes': self.traversal_nodes,
            'terminal_nodes': self.traversal_terminal_nodes,
            'max_depth': self.traversal_max_depth_observed,
            'max_depth_hits': self.traversal_max_depth_hits,
            'traversing_decision_nodes': self.traversal_traversing_decision_nodes,
            'opponent_decision_nodes': self.traversal_opponent_decision_nodes,
        }

        os_stats = {}
        os_stats['os_nodes'] = self.os_nodes
        os_stats['os_traversing_nodes'] = self.os_traversing_nodes
        os_stats['postflop_multiway_nodes'] = self.postflop_multiway_nodes
        os_stats['os_sampled_raise_nodes'] = self.os_sampled_raise_nodes
        os_stats['os_sampled_non_raise_nodes'] = self.os_sampled_non_raise_nodes

        os_stats['os_importance_weight_mean'] = (
            self.os_importance_weight_sum / self.os_importance_weight_count
            if self.os_importance_weight_count > 0 else 0.0
        )
        os_stats['os_importance_weight_max'] = self.os_importance_weight_max

        os_stats['os_q_correction_abs_mean'] = (
            self.os_q_correction_abs_sum / self.os_q_correction_count
            if self.os_q_correction_count > 0 else 0.0
        )
        os_stats['os_q_baseline_abs_error_mean'] = (
            self.os_q_baseline_abs_error_sum / self.os_q_baseline_abs_error_count
            if self.os_q_baseline_abs_error_count > 0 else 0.0
        )
        os_stats['os_q_baseline_abs_error_max'] = self.os_q_baseline_abs_error_max

        if self.os_regret_count > 0:
            os_regret_mean = self.os_regret_abs_sum / self.os_regret_count
            os_regret_var = self.os_regret_sq_sum / self.os_regret_count - os_regret_mean ** 2
            os_stats['os_mean_abs_regret'] = os_regret_mean
            os_stats['os_regret_std'] = math.sqrt(max(os_regret_var, 0.0))
        else:
            os_stats['os_mean_abs_regret'] = 0.0
            os_stats['os_regret_std'] = 0.0
        os_stats['os_max_abs_regret'] = self.os_max_abs_regret
        os_stats['os_regret_clip_count'] = self.os_regret_clip_count

        os_stats['os_sampled_raise_ratio'] = (
            self.os_sampled_raise_nodes / self.os_traversing_nodes
            if self.os_traversing_nodes > 0 else 0.0
        )

        base['hybrid_os'] = os_stats
        return base

    def _update_pg_baseline(self, reward_tensors):
        batch_mean = reward_tensors.mean().item()
        batch_var = reward_tensors.var().item() if len(reward_tensors) > 1 else 0.0
        momentum = self.pg_baseline_momentum
        self.pg_baseline_mean = momentum * self.pg_baseline_mean + (1 - momentum) * batch_mean
        self.pg_baseline_var = momentum * self.pg_baseline_var + (1 - momentum) * batch_var

    def prepare_iteration(self, iteration, traversing_player):
        """Apply only the independent per-buffer reservoir policy."""
        if not self.advantage_buffer_reservoir:
            self.advantage_buffer.clear()
        if not self.strategy_buffer_reservoir:
            self.strategy_buffer.clear()
        if not self.sizing_advantage_buffer_reservoir and self.sizing_advantage_buffer is not None:
            self.sizing_advantage_buffer.clear()
        if not self.sizing_strategy_buffer_reservoir:
            self.sizing_strategy_buffer.clear()

    def _get_min_raise_increment(self, state):
        """Bug #58: единый расчёт min-raise для среды pokers."""
        if hasattr(state, 'last_raise_increment') and state.last_raise_increment and float(state.last_raise_increment) > 0:
            return max(1.0, float(state.last_raise_increment))
        if hasattr(state, 'bb') and state.bb and float(state.bb) > 0:
            return max(1.0, float(state.bb))
        return 1.0

    def _anchor_availability(self, state):
        """Bug #94: доступность анкеров по размеру стека/мин-рейзу.

        Returns (avail[K] bool, allin_idx int).
        avail[i] = True если anchor_i легален (NORMAL или ALLIN_BOUNDARY).
        allin_idx — первый анкер, превышающий remaining (граничная оллин-кнопка),
        или -1 если стек покрывает все анкеры.
        """
        import pokers as pkrs
        pot = max(1.0, float(state.pot))
        ps = state.players_state[state.current_player]
        call_amount = max(0.0, float(state.min_bet) - float(ps.bet_chips))
        remaining = max(0.0, float(ps.stake) - call_amount)
        min_raise = self._get_min_raise_increment(state)

        chips = self.anchors_arr * pot
        avail = np.zeros(self.num_anchors, dtype=bool)
        normal = (chips >= min_raise) & (chips <= remaining)
        avail |= normal

        over = np.where(chips > remaining)[0]
        allin_idx = int(over[0]) if len(over) > 0 else -1
        can_raise = (hasattr(state, 'legal_actions')
                     and isinstance(state.legal_actions, (list, tuple))
                     and pkrs.ActionEnum.Raise in state.legal_actions)
        if allin_idx >= 0 and remaining > 0 and can_raise:
            avail[allin_idx] = True
        return avail, allin_idx

    def _compute_sizing_anchor_kind_mask(self, state):
        mask = np.ones(self.num_anchors, dtype=bool)
        kind_by_anchor = []
        excluded = set(self.sizing_q_mask_exclude_kinds)
        for idx, anchor in enumerate(self.anchors_arr):
            _, _, kind = self._resolve_effective_sizing(state, float(anchor))
            kind_by_anchor.append(kind)
            if kind in excluded:
                mask[idx] = False
        fallback_all_true = False
        if not np.any(mask):
            mask = np.ones(self.num_anchors, dtype=bool)
            fallback_all_true = True
        return mask, kind_by_anchor, fallback_all_true

    def _legal_sizing_anchor_mask(self, state):
        if not self.sizing_q_legal_anchor_mask_enabled:
            return np.ones(self.num_anchors, dtype=bool)
        mask, _, _ = self._compute_sizing_anchor_kind_mask(state)
        return mask

    def _apply_sizing_anchor_mask(self, probs, state, callsite='unknown'):
        original = np.asarray(probs, dtype=np.float32)
        diagnostics_enabled = self.sizing_q_mask_diagnostics_enabled
        mask_enabled = self.sizing_q_legal_anchor_mask_enabled
        dry_run = self.sizing_q_mask_dry_run

        if not diagnostics_enabled and not mask_enabled and not dry_run:
            return original

        counterfactual_mask, kind_by_anchor, fallback_all_true = self._compute_sizing_anchor_kind_mask(state)
        post_mask = np.asarray(original, dtype=np.float64).copy()
        post_mask[~counterfactual_mask] = 0.0
        total = float(post_mask.sum())
        if total > 1e-8:
            post_mask = (post_mask / total).astype(np.float32)

        if diagnostics_enabled:
            callsite_diag = self.sizing_q_mask_diag.setdefault(callsite, {
                'calls': 0,
                'fallback_all_true': 0,
                'pre_entropy_sum': 0.0,
                'post_entropy_sum': 0.0,
                'pre_top1_idx_hist': {},
                'post_top1_idx_hist': {},
                'pre_top1_mass_sum': 0.0,
                'post_top1_mass_sum': 0.0,
                'pre_top2_mass_sum': 0.0,
                'post_top2_mass_sum': 0.0,
                'removed_mass_total_sum': 0.0,
                'removed_mass_by_kind': {'NORMAL': 0.0, 'MIN_RAISE': 0.0, 'ALL_IN': 0.0},
                'kept_mass_total_sum': 0.0,
            })
            callsite_diag['calls'] += 1
            if fallback_all_true:
                callsite_diag['fallback_all_true'] += 1

            sorted_original = np.sort(original)[::-1]
            pre_entropy = float(-np.sum(original[original > 0] * np.log(original[original > 0] + np.finfo(float).eps)))
            callsite_diag['pre_entropy_sum'] += pre_entropy
            callsite_diag['pre_top1_mass_sum'] += float(sorted_original[0])
            callsite_diag['pre_top2_mass_sum'] += float(sorted_original[0] + sorted_original[1])
            pre_top1_idx = int(np.argmax(original))
            callsite_diag['pre_top1_idx_hist'][str(pre_top1_idx)] = callsite_diag['pre_top1_idx_hist'].get(str(pre_top1_idx), 0) + 1

            sorted_post = np.sort(post_mask)[::-1] if total > 1e-8 else np.zeros_like(original)
            if total > 1e-8:
                post_entropy = float(-np.sum(post_mask[post_mask > 0] * np.log(post_mask[post_mask > 0] + np.finfo(float).eps)))
                callsite_diag['post_entropy_sum'] += post_entropy
                callsite_diag['post_top1_mass_sum'] += float(sorted_post[0])
                callsite_diag['post_top2_mass_sum'] += float(sorted_post[0] + sorted_post[1]) if len(sorted_post) > 1 else float(sorted_post[0])
                post_top1_idx = int(np.argmax(post_mask))
                callsite_diag['post_top1_idx_hist'][str(post_top1_idx)] = callsite_diag['post_top1_idx_hist'].get(str(post_top1_idx), 0) + 1

            removed = original - post_mask if total > 1e-8 else original.copy()
            if total > 1e-8:
                callsite_diag['removed_mass_total_sum'] += float(np.sum(np.maximum(removed, 0)))
                callsite_diag['kept_mass_total_sum'] += float(np.sum(post_mask))
                for i, kind in enumerate(kind_by_anchor):
                    if kind in callsite_diag['removed_mass_by_kind'] and removed[i] > 1e-8:
                        callsite_diag['removed_mass_by_kind'][kind] += float(removed[i])

        if dry_run:
            return original

        if mask_enabled:
            if total > 1e-8:
                return post_mask
            return original

        return original

    def _update_insert_kind_diag(self, source, kind, callsite):
        diag = self.sizing_q_insert_kind_diag
        src_key = str(source)
        diag['by_source'].setdefault(src_key, {})
        diag['by_source'][src_key][kind] = diag['by_source'][src_key].get(kind, 0) + 1
        diag['by_callsite'].setdefault(callsite, {})
        diag['by_callsite'][callsite][kind] = diag['by_callsite'][callsite].get(kind, 0) + 1

    def _maybe_clear_sizing_q_replay_once(self):
        if not self.sizing_q_replay_clear_once:
            return
        if self._sizing_q_replay_clear_once_done:
            return
        if self.sizing_q_buffer is None:
            return

        if self.sizing_q_replay_clear_diagnostics_enabled and self.sizing_q_buffer is not None:
            buf = self.sizing_q_buffer
            n = buf._size
            src_counts = {}
            anchor_counts = {}
            for i in range(n):
                s = int(buf._sources[i])
                src_counts[str(s)] = src_counts.get(str(s), 0) + 1
                aidx = int(buf._anchor_indices[i]) if hasattr(buf, '_anchor_indices') else -1
                if aidx >= 0:
                    anchor_counts[str(aidx)] = anchor_counts.get(str(aidx), 0) + 1
            self.sizing_q_replay_clear_diag['before_clear'] = {
                'total': n,
                'by_source': src_counts,
                'by_anchor': anchor_counts,
            }

        self.sizing_q_buffer.clear()
        self._sizing_q_replay_clear_once_done = True
        self.sizing_q_kind_filter_diag['cleared_once'] += 1
        self.sizing_q_replay_clear_diag['cleared'] = True

    def _resolve_effective_sizing(self, state, anchor):
        """Bug #58: преобразовать raw anchor в effective multiplier с учётом ограничений среды.

        Устраняет silent mismatch: алгоритм выбирает anchor (0.10 или 3.00),
        а среда молча применяет другое значение (min-raise или all-in).

        Returns: (effective_multiplier, effective_chips, kind)
            kind ∈ {"NORMAL", "MIN_RAISE", "ALL_IN"}
        """
        pot = max(1.0, float(state.pot))
        mult = max(self.min_bet_size, min(self.max_bet_size, float(anchor)))
        raw = pot * mult

        player_state = state.players_state[state.current_player]
        call_amount = max(0.0, float(state.min_bet) - float(player_state.bet_chips))
        remaining = max(0.0, float(player_state.stake) - call_amount)
        min_raise = self._get_min_raise_increment(state)

        effective = raw
        kind = "NORMAL"

        if remaining < min_raise:
            effective = remaining
            kind = "ALL_IN" if effective > 0 else "NORMAL"
        else:
            if effective > remaining:
                effective = remaining
                kind = "ALL_IN"
            if effective < min_raise:
                effective = min_raise
                kind = "MIN_RAISE"

        effective = round(effective, 2)
        effective_mult = effective / pot
        return effective_mult, effective, kind

    def action_type_to_pokers_action(self, action_type, state, bet_size_multiplier=None):
        try:
            if action_type == 0:
                if pkrs.ActionEnum.Fold in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Fold)
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check)
                if pkrs.ActionEnum.Call in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Call)
                return pkrs.Action(pkrs.ActionEnum.Fold)

            elif action_type == 1:
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check)
                if pkrs.ActionEnum.Call in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Call)
                if pkrs.ActionEnum.Fold in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Fold)
                return pkrs.Action(pkrs.ActionEnum.Check)

            elif action_type == 2:
                if pkrs.ActionEnum.Call in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Call)
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check)
                if pkrs.ActionEnum.Fold in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Fold)
                return pkrs.Action(pkrs.ActionEnum.Call)

            elif action_type == 3:
                if pkrs.ActionEnum.Raise not in state.legal_actions:
                    if pkrs.ActionEnum.Call in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    if pkrs.ActionEnum.Check in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Check)
                    return pkrs.Action(pkrs.ActionEnum.Fold)

                _, additional, _ = self._resolve_effective_sizing(state, bet_size_multiplier or 1.0)
                if additional <= 0:
                    if pkrs.ActionEnum.Call in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    if pkrs.ActionEnum.Check in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Check)
                    return pkrs.Action(pkrs.ActionEnum.Fold)

                return pkrs.Action(pkrs.ActionEnum.Raise, additional)

            else:
                if pkrs.ActionEnum.Call in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Call)
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check)
                return pkrs.Action(pkrs.ActionEnum.Fold)

        except Exception as e:
            if VERBOSE:
                print(f"DeepCFRAgent ERROR action_type_to_pokers_action: {e}")
            if pkrs.ActionEnum.Call in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Call)
            if pkrs.ActionEnum.Check in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Check)
            return pkrs.Action(pkrs.ActionEnum.Fold)

    def get_legal_action_types(self, state):
        legal_action_types = []

        if pkrs.ActionEnum.Fold in state.legal_actions:
            legal_action_types.append(0)

        if pkrs.ActionEnum.Check in state.legal_actions:
            legal_action_types.append(1)

        if pkrs.ActionEnum.Call in state.legal_actions:
            legal_action_types.append(2)

        if pkrs.ActionEnum.Raise in state.legal_actions:
            legal_action_types.append(3)

        return legal_action_types

    def get_legal_action_mask(self, state):
        mask = np.zeros(self.num_actions, dtype=np.float32)
        if pkrs.ActionEnum.Fold in state.legal_actions:
            mask[0] = 1.0
        if pkrs.ActionEnum.Check in state.legal_actions:
            mask[1] = 1.0
        if pkrs.ActionEnum.Call in state.legal_actions:
            mask[2] = 1.0
        if pkrs.ActionEnum.Raise in state.legal_actions:
            mask[3] = 1.0
        return mask


    def _is_postflop(self, state):
        """True если стадия Flop(1), Turn(2), River(3) или Showdown(4)."""
        return int(state.stage) > 0

    def _count_active_opponents(self, state, traversing_player):
        """Количество активных оппонентов (не traversing_player, active=True)."""
        return sum(
            1 for i, ps in enumerate(state.players_state)
            if i != traversing_player and ps.active
        )

    def _should_use_outcome_sampling(self, state, traversing_player):
        """True если узел должен обрабатываться через Outcome Sampling."""
        if self.os_force_full_traversal:
            return False
        if not self.hybrid_os_enabled:
            return False
        if not self._is_postflop(state):
            return False
        if self._count_active_opponents(state, traversing_player) < self.hybrid_os_min_active_opponents:
            return False
        if not self.use_q_baseline:
            return False

        q_ready_from_buffer = (
            self.iteration_count >= self.hybrid_os_q_warmup_iterations
            and len(self.q_buffer) >= self.hybrid_os_min_q_buffer_size
        )

        q_ready_from_checkpoint = (
            self.hybrid_os_allow_loaded_q_without_buffer
            and self.q_loaded_from_checkpoint
            and self.iteration_count >= self.hybrid_os_loaded_q_min_iteration
        )

        if not (q_ready_from_buffer or q_ready_from_checkpoint):
            return False

        return True

    def _compute_regret_matching_strategy(self, advantages, legal_mask):
        """Regret matching: positive regrets * mask, нормализация."""
        pos = np.maximum(advantages, 0) * legal_mask
        s = pos.sum()
        if s > 0:
            return pos / s
        return legal_mask / legal_mask.sum()

    def _get_os_epsilon(self, iteration):
        """Anneal epsilon от start к end по decay_iterations."""
        if iteration >= self.hybrid_os_epsilon_decay_iterations:
            return self.hybrid_os_epsilon_end
        progress = iteration / self.hybrid_os_epsilon_decay_iterations
        return self.hybrid_os_epsilon_start + progress * (self.hybrid_os_epsilon_end - self.hybrid_os_epsilon_start)

    def _build_sampling_policy(self, strategy, legal_mask, iteration):
        """Sampling policy mu = (1-eps)*strategy + eps*uniform_legal + Raise floor."""
        eps = self._get_os_epsilon(iteration)
        legal_actions = [a for a in range(self.num_actions) if legal_mask[a] > 0]
        n_legal = len(legal_actions)
        if n_legal == 0:
            return strategy.copy()

        uniform_legal = legal_mask / n_legal
        mu = (1.0 - eps) * strategy + eps * uniform_legal

        if 3 in legal_actions:
            mu[3] = max(mu[3], self.hybrid_os_pg_min_raise_sample_prob)

        for a in legal_actions:
            mu[a] = max(mu[a], 1e-6)

        mu_sum = mu.sum()
        if mu_sum > 0:
            mu = mu / mu_sum
        else:
            mu = legal_mask / legal_mask.sum()

        return mu

    def _current_sizing_min_prob(self, iteration):
        decay_iters = max(int(self.sizing_min_prob_decay_iterations), 1)
        progress = min(max(float(iteration) / decay_iters, 0.0), 1.0)
        return float(
            self.sizing_min_prob_start
            + (self.sizing_min_prob_end - self.sizing_min_prob_start) * progress
        )

    def _resolve_sizing_bucket_groups(self, bucket_groups):
        """Преобразовать заданные sizing-группы в индексы fixed grid."""
        resolved = []
        used_indices = set()
        anchors_np = np.asarray(self.anchors, dtype=np.float32)
        for group in bucket_groups:
            indices = []
            for bucket_size in group:
                distances = np.abs(anchors_np - float(bucket_size))
                idx = int(np.argmin(distances))
                if float(distances[idx]) > 1e-4:
                    raise ValueError(f"sizing_bucket_groups содержит размер вне fixed_sizing_grid: {bucket_size}")
                indices.append(idx)
                used_indices.add(idx)
            if not indices:
                raise ValueError("sizing_bucket_groups не может содержать пустые группы")
            resolved.append(sorted(set(indices)))
        missing = sorted(set(range(self.num_anchors)) - used_indices)
        if missing:
            resolved.append(missing)
        return resolved

    @staticmethod
    def _bucketize(value, edges):
        """Дискретизировать value в bucket-индекс по границам edges."""
        return int(np.searchsorted(edges, value))

    def _get_current_ema_beta(self, iteration):
        """Warmup: линейный рост beta от start до end за warmup_steps итераций."""
        warmup = max(int(self.sizing_target_ema_warmup_steps), 1)
        progress = min(max(float(iteration) / warmup, 0.0), 1.0)
        return float(
            self.sizing_target_ema_beta_start
            + progress * (self.sizing_target_ema_beta_end - self.sizing_target_ema_beta_start)
        )

    def _get_sizing_target_bucket_key(self, state, encoded_state):
        """Вернуть стабильный грубый ключ абстракции для EMA sizing target.

        Признаки: stage, current_player, button, pot_bb_bucket,
        eff_stack_bb_bucket, spr_bucket, has_raise.
        """
        _ = encoded_state  # не используется — все признаки из state
        bb = float(state.bb) if hasattr(state, 'bb') and float(state.bb) > 0 else 1.0
        pot = float(state.pot)
        pot_bb = pot / max(bb, 1.0)

        # SPR временно отключён (test)
        active_stacks = [
            float(p.stake)
            for p in state.players_state
            if getattr(p, 'active', True)
        ]
        effective_stack = (
            min(active_stacks)
            if active_stacks
            else float(state.players_state[state.current_player].stake)
        )
        eff_stack_bb = effective_stack / max(bb, 1.0)
        spr = 0.0

        has_raise = int(pkrs.ActionEnum.Raise in state.legal_actions)

        pot_edges = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0, float('inf'))
        stack_edges = (0.0, 20.0, 40.0, 60.0, 100.0, 200.0, float('inf'))
        spr_edges = (0.0, 1.0, 2.0, 4.0, 8.0, 13.0, float('inf'))

        return (
            int(state.stage),
            int(state.current_player),
            int(state.button),
            self._bucketize(pot_bb, pot_edges),
            self._bucketize(eff_stack_bb, stack_edges),
            self._bucketize(spr, spr_edges),
            has_raise,
        )

    def _smooth_sizing_target_by_bucket(self, state, encoded_state, weights, iteration):
        """EMA-сглаживание Q-derived sizing target по state bucket.

        LRU eviction при превышении sizing_target_ema_max_size.
        """
        if not self.sizing_target_ema_enabled:
            return np.asarray(weights, dtype=np.float32)

        weights = np.asarray(weights, dtype=np.float32).copy()
        weights = np.maximum(weights, 0.0)
        total = float(weights.sum())
        if total <= 1e-8:
            weights[:] = 1.0 / max(int(self.num_anchors), 1)
        else:
            weights = weights / total

        key = self._get_sizing_target_bucket_key(state, encoded_state)

        if key in self.sizing_target_ema:
            old = self.sizing_target_ema[key]
            self.sizing_target_ema.move_to_end(key)
        else:
            old = None

        max_size = max(int(self.sizing_target_ema_max_size), 1)
        while len(self.sizing_target_ema) >= max_size:
            self.sizing_target_ema.popitem(last=False)

        if old is None:
            smoothed = weights
        else:
            beta = self._get_current_ema_beta(iteration)
            smoothed = beta * old + (1.0 - beta) * weights
            stotal = float(smoothed.sum())
            if stotal > 1e-8:
                smoothed = smoothed / stotal

        self.sizing_target_ema[key] = smoothed.astype(np.float32)
        return smoothed.astype(np.float32)

    def _sparsify_sizing_target_by_buckets(self, weights):
        """Top-2 bucket / top-1 sizing внутри bucket для снижения шума.

        Возвращает sparse target. При нулевой массе инкрементирует fallback-счётчик
        и возвращает исходное распределение.
        """
        if not self.sizing_target_sparsify_enabled:
            return np.asarray(weights, dtype=np.float32)

        weights = np.asarray(weights, dtype=np.float32).copy()
        weights = np.maximum(weights, 0.0)
        total = float(weights.sum())
        if total <= 1e-8:
            weights[:] = 1.0 / max(int(self.num_anchors), 1)
        else:
            weights = weights / total

        self.sizing_sparsify_total_count += 1

        if not self.sizing_bucket_groups:
            return weights

        group_masses = []
        for group_idx, group in enumerate(self.sizing_bucket_groups):
            mass = float(weights[group].sum())
            group_masses.append((mass, group_idx, group))

        group_masses.sort(reverse=True, key=lambda x: x[0])
        top_k = min(int(self.sizing_target_top_buckets), len(group_masses))
        selected_groups = group_masses[:top_k]

        sparse = np.zeros_like(weights)

        for mass, group_idx, group in selected_groups:
            if mass <= 1e-8:
                continue

            group_weights = weights[group]
            local_order = np.argsort(group_weights)[::-1]
            keep_n = min(int(self.sizing_target_top_per_bucket), len(group))

            for local_idx in local_order[:keep_n]:
                anchor_idx = group[local_idx]
                sparse[anchor_idx] = weights[anchor_idx]

        stotal = float(sparse.sum())
        if stotal <= 1e-8:
            self.sizing_sparsify_fallback_count += 1
            return weights

        sparse = sparse / stotal
        return sparse.astype(np.float32)

    def _current_sizing_bucket_floor(self, iteration):
        if not self.sizing_bucket_groups:
            return 0.0
        decay_iters = max(int(self.sizing_bucket_min_prob_decay_iterations), 1)
        progress = min(max(float(iteration) / decay_iters, 0.0), 1.0)
        floor = float(
            self.sizing_bucket_min_prob_start
            + (self.sizing_bucket_min_prob_end - self.sizing_bucket_min_prob_start) * progress
        )
        max_feasible_floor = 1.0 / len(self.sizing_bucket_groups)
        return float(np.clip(floor, 0.0, max_feasible_floor))

    def _apply_sizing_bucket_floor(self, slot_weights, iteration):
        """Гарантировать минимальную probability mass для каждой sizing-группы."""
        weights = np.asarray(slot_weights, dtype=np.float32).copy()
        if weights.ndim != 1 or weights.size != self.num_anchors:
            raise ValueError("slot_weights должен быть одномерным массивом размера num_anchors")
        total = float(np.maximum(weights, 0.0).sum())
        if total <= 1e-8:
            weights = np.ones(self.num_anchors, dtype=np.float32) / self.num_anchors
        else:
            weights = np.maximum(weights, 0.0) / total

        bucket_floor = self._current_sizing_bucket_floor(iteration)
        if bucket_floor <= 0.0 or not self.sizing_bucket_groups:
            return weights.astype(np.float32)

        group_masses = np.array([float(weights[group].sum()) for group in self.sizing_bucket_groups], dtype=np.float32)
        residual_mass = max(1.0 - bucket_floor * len(self.sizing_bucket_groups), 0.0)
        group_total = float(group_masses.sum())
        if group_total <= 1e-8:
            target_group_masses = np.full(
                len(self.sizing_bucket_groups),
                1.0 / len(self.sizing_bucket_groups),
                dtype=np.float32,
            )
        else:
            target_group_masses = bucket_floor + residual_mass * (group_masses / group_total)

        adjusted = np.zeros_like(weights)
        for target_mass, group in zip(target_group_masses, self.sizing_bucket_groups):
            group_weights = weights[group]
            group_total = float(group_weights.sum())
            if group_total <= 1e-8:
                adjusted[group] = float(target_mass) / len(group)
            else:
                adjusted[group] = group_weights / group_total * float(target_mass)
        adjusted = adjusted / max(float(adjusted.sum()), 1e-8)
        return adjusted.astype(np.float32)

    def _current_sizing_probe_prob(self, iteration):
        if not self.sizing_q_enabled:
            return 0.0
        decay_iters = max(int(self.sizing_probe_decay_iterations), 1)
        progress = min(max(float(iteration) / decay_iters, 0.0), 1.0)
        return float(
            self.sizing_probe_prob_start
            + (self.sizing_probe_prob_end - self.sizing_probe_prob_start) * progress
        )

    def _sizing_q_has_anchor_coverage(self):
        if not self.sizing_q_enabled or self.sizing_q_net is None:
            return False
        min_count = int(self.sizing_q_min_samples_per_anchor)
        if min_count <= 0:
            return True
        counts = self.sizing_q_buffer.anchor_counts(len(self.sizing_anchor_sizes))
        return bool(np.all(counts >= min_count))

    def _add_sizing_q_sample(self, state_arr, bet_size, target_value, iteration, is_probe=False, is_lookahead=False, anchor_idx=-1,
                              selected_anchor_idx=-1, effective_anchor_idx=None, selected_kind='UNKNOWN'):
        if not self.sizing_q_enabled or self.sizing_q_net is None or self.sizing_q_buffer is None:
            return
        norm_size = (float(bet_size) - self.min_bet_size) / (self.max_bet_size - self.min_bet_size)
        self.sizing_q_buffer.add(
            state_arr,
            norm_size,
            target_value,
            iteration,
            source=2 if is_lookahead else (1 if is_probe else 0),
            anchor_idx=anchor_idx,
            selected_anchor_idx=selected_anchor_idx,
            effective_anchor_idx=effective_anchor_idx,
            selected_kind=selected_kind,
        )

    def _sizing_q_is_ready(self):
        """Проверяет, достаточно ли Q-сеть обучена для использования.

        Bug #49v4: усиленный критерий — buffer >= min_total_samples
        и anchor coverage, а не просто >256 семплов.
        Пока Q не ready — Phase 0 warm-up: candidates = anchors, sizing сети не обучаются.

        Bug #57: version-based O(1) кэш — пересчёт только при изменении буфера.
        """
        if not self.sizing_q_enabled or self.sizing_q_net is None or self.sizing_q_buffer is None:
            return False
        buf_ver = self.sizing_q_buffer._version
        if hasattr(self, '_sqr_version') and self._sqr_version == buf_ver:
            return self._sqr_cached
        if len(self.sizing_q_buffer) < self.sizing_q_min_total_samples:
            result = False
        elif not self._sizing_q_has_anchor_coverage():
            result = False
        else:
            result = True
        self._sqr_version = buf_ver
        self._sqr_cached = result
        return result

    def _evaluate_sizing_q_for_sizes(self, state_tensor, slot_sizes_np):
        """Q(state, size_i) по фиксированной сетке.

        Args:
            state_tensor: Tensor [1, state_dim]
            slot_sizes_np: np.array [K] — фиксированные размеры
        Returns:
            q_vals: np.array [K] — Q-values per size
        """
        if not self.sizing_q_enabled or self.sizing_q_net is None:
            return np.zeros(len(slot_sizes_np), dtype=np.float32)
        K = len(slot_sizes_np)
        sizes_t = torch.from_numpy(slot_sizes_np).float().to(self.device)
        norm_sizes = (sizes_t - self.min_bet_size) / (self.max_bet_size - self.min_bet_size)
        state_expanded = state_tensor.expand(K, -1)
        self.sizing_q_net.eval()
        with torch.no_grad():
            q_vals = self.sizing_q_net(state_expanded, norm_sizes).cpu().numpy()
        return q_vals

    def _bootstrap_sizing_value(self, state, traversing_player):
        """Оценка value состояния через q_net для sizing lookahead.

        Использует тот же подход, что train_q_network для TD-target:
        strategy — с перспективы current_player (через advantage_net);
        Q — с перспективы traversing_player (через q_net).
        V(s) = sum(strategy[a] * Q(s, a)).

        Bug #69: при sizing_q_bootstrap_target_net=True использует замороженные
        target-сети (target_net + q_target_net) для стабильного bootstrap,
        разрывая deadly-triad положительную обратную связь.
        """
        current_player = state.current_player
        encoded = self._encode_state(state, current_player)
        state_t = torch.from_numpy(encoded.astype(np.float32)).unsqueeze(0).to(self.device)
        legal_mask = self.get_legal_action_mask(state)
        mask_t = torch.from_numpy(legal_mask.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            if self.sizing_q_bootstrap_target_net:
                adv = self._target_net(current_player)(state_t)[:, :self.num_actions]
            else:
                adv = self._adv_net(current_player)(state_t)[:, :self.num_actions]
            pos = torch.clamp(adv, min=0) * mask_t
            pos_sum = pos.sum(dim=1, keepdim=True)
            strategy = torch.where(
                pos_sum > 0,
                pos / pos_sum,
                mask_t / mask_t.sum(dim=1, keepdim=True).clamp(min=1),
            )
            q_encoded = self._encode_state(state, traversing_player)
            q_t = torch.from_numpy(q_encoded.astype(np.float32)).unsqueeze(0).to(self.device)
            q_net_used = self.q_target_net if self.sizing_q_bootstrap_target_net else self.q_net
            q_vals = q_net_used(q_t)
            v = float((strategy * q_vals).sum().item())

            # Bug #74 диаг v4: микро-разведка, почему q_vals == 0 на post-raise.
            # base_out проверяет гипотезу мёртвых ReLU (выход base == 0 -> q_head(0)=bias),
            # canary проверяет «живость» сети на заведомо ненулевом входе (вектор единиц).
            if self.sizing_q_target_diagnostics_enabled:
                base_out = q_net_used.base(q_t)
                canary = q_net_used(torch.ones_like(q_t))
                self._last_bootstrap_debug = {
                    'strategy_sum': float(strategy.sum().item()),
                    'legal_sum': float(mask_t.sum().item()),
                    'q_vals': q_vals.detach().reshape(-1).cpu().numpy().copy(),
                    'using_target_net': bool(self.sizing_q_bootstrap_target_net),
                    'q_input_abs_sum': float(q_t.abs().sum().item()),
                    'q_input_min': float(q_t.min().item()),
                    'q_input_max': float(q_t.max().item()),
                    'q_input_has_nan': bool(torch.isnan(q_t).any().item()),
                    'base_out_abs_sum': float(base_out.abs().sum().item()),
                    'q_canary_abs_sum': float(canary.abs().sum().item()),
                    'current_player': int(current_player),
                    'traversing_player': int(traversing_player),
                    'next_is_hero': bool(int(current_player) == int(traversing_player)),
                }

        if self.sizing_q_bootstrap_scale_fix:
            v *= self._get_q_reward_unit()

        return v

    def _perform_sizing_lookahead(self, state, state_arr, iteration, traversing_player, ev, eff_idx):
        """One-step lookahead: записывает sizing Q-samples для несыгранных анкоров (Bug #64).

        Для raise-legal узла с вероятностью sizing_lookahead_prob выбирает
        случайную подвыборку анкоров (кроме уже сыгранного eff_idx) и для каждого:
        - apply_action(Raise, size_k) — один шаг вперёд;
        - если терминал: прямой reward;
        - иначе: bootstrap через q_net (_bootstrap_sizing_value);
        - target = _normalize_sizing_q_target(v_k, ev, pot);
        - _add_sizing_q_sample с is_lookahead=True (source=2).
        """
        if self.sizing_lookahead_max_anchors <= 0:
            return
        if not self.sizing_lookahead_enabled:
            return
        if not self.sizing_q_enabled or self.sizing_q_net is None:
            return
        if random.random() >= self.sizing_lookahead_prob:
            return
        if pkrs.ActionEnum.Raise not in state.legal_actions:
            return

        candidate_indices = [i for i in range(self.num_anchors) if i != eff_idx]
        n_choose = min(self.sizing_lookahead_max_anchors, len(candidate_indices))
        if n_choose <= 0:
            return
        selected_indices = random.sample(candidate_indices, n_choose)

        diag_on = self.sizing_q_target_diagnostics_enabled

        for anchor_idx in selected_indices:
            if diag_on:
                self.sizing_lookahead_diag['counts']['attempts'] += 1
            size_k = float(self.anchors_arr[anchor_idx])
            pokers_action = self.action_type_to_pokers_action(3, state, size_k)
            if pokers_action is None:
                continue

            try:
                s_next = state.apply_action(pokers_action)
            except Exception:
                continue

            if s_next.status != pkrs.StateStatus.Ok:
                continue

            if diag_on:
                self.sizing_lookahead_diag['counts']['applied_ok'] += 1
                self._last_bootstrap_debug = None

            if s_next.final_state:
                v_k = s_next.players_state[traversing_player].reward
            else:
                v_k = self._bootstrap_sizing_value(s_next, traversing_player)

            pre_clip, target_k = self._compute_sizing_q_target_values(v_k, state.pot)

            if diag_on:
                c = self.sizing_lookahead_diag['counts']
                eps = 1e-8
                c['added'] += 1
                if s_next.final_state:
                    c['final_state'] += 1
                else:
                    c['nonterminal'] += 1
                if abs(float(v_k)) <= eps:
                    c['vk_near_zero'] += 1
                if abs(float(target_k)) <= eps:
                    c['target_near_zero'] += 1

                self._upd_la_stat('v_k', v_k)
                self._upd_la_stat('pre_clip', pre_clip)
                self._upd_la_stat('target', target_k)
                self._upd_la_stat('pot', state.pot)

                dbg = self._last_bootstrap_debug
                if dbg is not None:
                    self._upd_la_stat('strategy_sum', dbg['strategy_sum'])
                    self._upd_la_stat('legal_sum', dbg['legal_sum'])
                    qv = dbg['q_vals']
                    for ai in range(len(qv)):
                        self._upd_la_stat(f'q_a{ai}', qv[ai])
                    # v4: вход в q_net, выход base (мёртвые ReLU?), canary (живость сети).
                    self._upd_la_stat('q_input_abs_sum', dbg['q_input_abs_sum'])
                    self._upd_la_stat('q_input_min', dbg['q_input_min'])
                    self._upd_la_stat('q_input_max', dbg['q_input_max'])
                    self._upd_la_stat('base_out_abs_sum', dbg['base_out_abs_sum'])
                    self._upd_la_stat('q_canary_abs_sum', dbg['q_canary_abs_sum'])
                    if dbg['using_target_net']:
                        c['using_target_net'] += 1
                    if dbg['q_input_has_nan']:
                        c['q_input_has_nan'] += 1
                    if dbg['next_is_hero']:
                        c['next_is_hero'] += 1
                elif not s_next.final_state:
                    # Нонтерминал, но bootstrap не оставил debug → его где-то пропустило.
                    c['bootstrap_debug_missing'] += 1

            self._record_sizing_q_target_diag(
                source=2, anchor_idx=anchor_idx, raw_v3=v_k, pot=state.pot,
                pre_clip_target=pre_clip, post_clip_target=target_k,
            )
            self._add_sizing_q_sample(state_arr, size_k, target_k, iteration,
                                      is_lookahead=True, anchor_idx=anchor_idx)
            self._update_insert_kind_diag(2, 'UNKNOWN', 'lookahead')

    def _hierarchical_sizing(self, state_tensor, iteration, player_id=None, state=None, use_q=True):
        """Иерархический sizing: сначала bucket (small/medium/large), потом анкер.

        state_tensor: предварительно заэнкожен с перспективы вызывающей стороны.
        player_id: индекс игрока (0..num_trainable-1). Если None — из state.current_player.
        state: опционально для legal-anchor mask, preflop detection, и re-encode.

        Returns: (sampled_bet_size, full_15_probs, anchor_idx)
        """
        num_buckets = min(len(self.sizing_bucket_groups), 3)
        bucket_min_prob = max(self._current_sizing_bucket_floor(iteration) / max(num_buckets, 1), 1e-6)
        anchor_min_prob = self._current_sizing_min_prob(iteration)

        if state is not None:
            sizing_encoded = self._encode_state_for_sizing(state, int(state.current_player))
            state_tensor = torch.from_numpy(sizing_encoded.astype(np.float32)).to(self.device)

        if player_id is None:
            player_id = int(state.current_player) if state is not None else 0
        sizing_net = self.advantage_sizing_net

        is_preflop = (state is not None and int(state.stage) == 0)
        use_flat = (
            (self.sizing_preflop_disable_buckets and is_preflop)
            or not self.sizing_bucket_head_enabled
        )

        if use_flat:
            with torch.inference_mode():
                anchor_logits_t = sizing_net(state_tensor.unsqueeze(0))
            anchor_logits = anchor_logits_t[0].cpu().numpy()
            full_probs = regret_matching_anchors(anchor_logits, min_prob=anchor_min_prob)
        elif self.sizing_cfr_mode:
            with torch.inference_mode():
                anchor_logits_t = sizing_net(state_tensor.unsqueeze(0))
            anchor_logits = anchor_logits_t[0].cpu().numpy()
            full_probs = regret_matching_anchors(anchor_logits, min_prob=anchor_min_prob)
        elif use_q and self.sizing_q_enabled and self._sizing_q_is_ready():
            q_vals = self._evaluate_sizing_q_for_sizes(state_tensor, self.anchors_arr)
            bucket_q = np.array([float(np.max(q_vals[indices])) for indices in self.sizing_bucket_groups], dtype=np.float32)
            bucket_probs = compute_sizing_heat_weights(
                bucket_q, temperature=self.sizing_heat_temperature,
                min_advantage=self.sizing_heat_min_advantage, top_p=self.sizing_heat_top_p)
            bucket_probs = np.maximum(bucket_probs, bucket_min_prob)
            bucket_probs = bucket_probs / bucket_probs.sum()
            full_probs = np.zeros(self.num_anchors, dtype=np.float32)
            for b, indices in enumerate(self.sizing_bucket_groups):
                local_q = q_vals[indices]
                local_probs = compute_sizing_heat_weights(
                    local_q, temperature=self.sizing_heat_temperature,
                    min_advantage=self.sizing_heat_min_advantage, top_p=self.sizing_heat_top_p)
                full_probs[indices] = bucket_probs[b] * local_probs
        else:
            with torch.inference_mode():
                anchor_logits_t = sizing_net(state_tensor.unsqueeze(0))
            anchor_logits = anchor_logits_t[0].cpu().numpy()
            full_probs = regret_matching_anchors(anchor_logits, min_prob=anchor_min_prob)

        total = float(full_probs.sum())
        if total > 1e-8:
            full_probs = full_probs / total
        else:
            full_probs = np.ones(self.num_anchors, dtype=np.float32) / self.num_anchors

        if state is not None and self.sizing_anchor_availability_enabled:
            avail, allin_idx = self._anchor_availability(state)
            full_probs[~avail] = 0.0
            total_avail = float(full_probs.sum())
            if total_avail > 1e-8:
                full_probs = full_probs / total_avail
            else:
                full_probs = np.ones(self.num_anchors, dtype=np.float32) / self.num_anchors
            self.sizing_availability_diag['calls'] += 1
            self.sizing_availability_diag['avail_mean_sum'] += float(avail.sum())
            if allin_idx >= 0:
                self.sizing_availability_diag['boundary_present'] += 1
                self.sizing_availability_diag['boundary_allin_idx_sum'] += allin_idx
                self.sizing_availability_diag['boundary_allin_idx_sqsum'] += float(allin_idx * allin_idx)
        elif state is not None:
            full_probs = self._apply_sizing_anchor_mask(full_probs, state, callsite='hierarchical')

        full_probs = np.clip(full_probs.astype(np.float64), 0.0, None)
        total = max(full_probs.sum(), 1e-12)
        full_probs /= total

        sampled_anchor_idx = int(np.random.choice(self.num_anchors, p=full_probs))
        sampled_bet_size = float(self.anchors[sampled_anchor_idx])

        if state is not None:
            if is_preflop:
                self.sizing_availability_diag['preflop_selections'] += 1
            else:
                self.sizing_availability_diag['postflop_selections'] += 1
            self.sizing_availability_diag['allin_total_selections'] += 1
            if self.sizing_anchor_availability_enabled:
                _, boundary_idx = self._anchor_availability(state)
                if sampled_anchor_idx == boundary_idx:
                    self.sizing_availability_diag['allin_selected'] += 1

        return sampled_bet_size, full_probs.astype(np.float32), sampled_anchor_idx

    def _sample_traversal_sizing_anchors(self, slot_weights, state=None, iteration=0):
        """One branch per available bucket: argmax, with uniform exploration."""
        weights = np.asarray(slot_weights, dtype=np.float64)
        if weights.shape != (self.num_anchors,):
            raise ValueError("slot_weights must have one value per sizing anchor")
        if state is not None and self.sizing_anchor_availability_enabled:
            available_mask, _ = self._anchor_availability(state)
        elif state is not None:
            available_mask = self._legal_sizing_anchor_mask(state)
        else:
            available_mask = weights > 0.0

        explore = float(np.clip(self.sizing_bucket_explore_prob, 0.0, 1.0))
        samples = []
        for indices in self.sizing_bucket_groups[:3]:
            available = np.asarray([idx for idx in indices if available_mask[idx]], dtype=np.int64)
            if available.size == 0:
                continue
            explored = random.random() < explore
            if explored:
                selected = int(np.random.choice(available))
                importance_weight = min(float(available.size), self.sizing_importance_weight_clip)
            else:
                selected = int(available[int(np.argmax(weights[available]))])
                importance_weight = 1.0
            samples.append((selected, importance_weight))
        return samples

    def _sample_probe_size(self, current_slot_sizes_np):
        """Exploratory probe: смесь uniform, anchors, slot jitter."""
        probe_uniform = random.random()
        if probe_uniform < self.sizing_probe_uniform_weight:
            return float(np.random.uniform(self.min_bet_size, self.max_bet_size))
        elif probe_uniform < self.sizing_probe_uniform_weight + self.sizing_probe_anchor_weight:
            return float(np.random.choice(self.anchors))
        else:
            if len(current_slot_sizes_np) > 0:
                base = float(np.random.choice(current_slot_sizes_np))
                jitter = float(np.random.uniform(-self.sizing_probe_jitter, self.sizing_probe_jitter))
                return float(np.clip(base + jitter, self.min_bet_size, self.max_bet_size))
            return float(np.random.uniform(self.min_bet_size, self.max_bet_size))

    def cfr_traverse(self, state, iteration, random_agents, depth=0):
        # Bug #50: удалён legacy-путь с floating sizing.
        raise NotImplementedError(
            "cfr_traverse() — legacy Phase 1 путь. Используйте train_self_play_multi."
        )

    def _cfr_traverse_multi_outcome_node(self, state, iteration, traversing_player, depth):
        """Outcome Sampling traversing node: сэмплирует одну ветку, regrets через Q-baseline."""
        self.os_nodes += 1
        self.os_traversing_nodes += 1
        if self._is_postflop(state):
            self.postflop_multiway_nodes += 1

        encoded_state = self._encode_state(state, traversing_player)
        state_tensor = torch.from_numpy(encoded_state.astype(np.float32, copy=False)).to(self.device)
        sizing_encoded = self._encode_state_for_sizing(state, traversing_player)
        sizing_state_tensor = torch.from_numpy(sizing_encoded.astype(np.float32)).unsqueeze(0).to(self.device)

        # Раздельные forward'ы: action из advantage_net.
        with torch.inference_mode():
            action_logits = self._adv_net(traversing_player)(state_tensor.unsqueeze(0))
            advantages = action_logits[0].cpu().numpy()

        legal_mask = self.get_legal_action_mask(state)
        legal_action_types = [a for a in range(self.num_actions) if legal_mask[a] > 0]

        strategy = self._compute_regret_matching_strategy(advantages, legal_mask)

        mu = self._build_sampling_policy(strategy, legal_mask, iteration)

        legal_probs = np.array([mu[a] for a in legal_action_types])
        legal_probs = legal_probs / legal_probs.sum()

        action_idx = np.random.choice(len(legal_action_types), p=legal_probs)
        sampled_action = legal_action_types[action_idx]
        sampled_mu = mu[sampled_action]

        sampled_bet_size = None
        slot_weights_np = None
        sizing_anchor_idx = -1

        if 3 in legal_action_types:
            self.raise_funnel_diag['hero_os']['raise_legal'] += 1
            slot_sizes_np = self.anchors_arr

            if self._sizing_q_is_ready():
                q_vals = self._evaluate_sizing_q_for_sizes(sizing_state_tensor, slot_sizes_np)
                bucket_q = np.array([float(np.max(q_vals[indices])) for indices in self.sizing_bucket_groups], dtype=np.float32)
                bucket_probs = compute_sizing_heat_weights(
                    bucket_q, temperature=self.sizing_heat_temperature,
                    min_advantage=self.sizing_heat_min_advantage, top_p=self.sizing_heat_top_p)
                num_b = len(self.sizing_bucket_groups)
                bucket_floor = max(self._current_sizing_bucket_floor(iteration) / max(num_b, 1), 1e-6)
                bucket_probs = np.maximum(bucket_probs, bucket_floor)
                bucket_probs = bucket_probs / bucket_probs.sum()
                full_probs = np.zeros(self.num_anchors, dtype=np.float32)
                for b, indices in enumerate(self.sizing_bucket_groups):
                    local_q = q_vals[indices]
                    local_probs = compute_sizing_heat_weights(
                        local_q, temperature=self.sizing_heat_temperature,
                        min_advantage=self.sizing_heat_min_advantage, top_p=self.sizing_heat_top_p)
                    full_probs[indices] = bucket_probs[b] * local_probs
                total = float(full_probs.sum())
                if total > 1e-8:
                    full_probs = full_probs / total
                else:
                    full_probs = np.ones(self.num_anchors, dtype=np.float32) / self.num_anchors
                if self.sizing_anchor_availability_enabled:
                    avail, _ = self._anchor_availability(state)
                    full_probs[~avail] = 0.0
                    total_avail = float(full_probs.sum())
                    if total_avail > 1e-8:
                        full_probs = full_probs / total_avail
                else:
                    full_probs = self._apply_sizing_anchor_mask(full_probs, state, callsite='os_q_ready')
                slot_weights_np = full_probs.astype(np.float32)
            else:
                _, slot_weights_np, _ = self._hierarchical_sizing(state_tensor, iteration, traversing_player, state=state)

        if sampled_action == 3 and slot_weights_np is not None:
            self.raise_funnel_diag['hero_os']['raise_sampled'] += 1
            slot_weights_np = np.clip(slot_weights_np.astype(np.float64), 0.0, None)
            total = max(slot_weights_np.sum(), 1e-12)
            slot_weights_np /= total
            slot_weights_np[-1] = max(1.0 - slot_weights_np[:-1].sum(), 0.0)
            sizing_anchor_idx = int(np.random.choice(self.num_anchors, p=slot_weights_np))
            sampled_bet_size = float(self.anchors[sizing_anchor_idx])
            bet_size_multiplier = sampled_bet_size
            self.os_sampled_raise_nodes += 1
            if sizing_anchor_idx >= 0:
                self.sizing_path_diag['hero_os_raise'][sizing_anchor_idx] += 1
        else:
            bet_size_multiplier = float(np.mean(self.anchors))
            self.os_sampled_non_raise_nodes += 1

        try:
            pokers_action = self.action_type_to_pokers_action(sampled_action, state, bet_size_multiplier)
            new_state = state.apply_action(pokers_action)

            if new_state.status != pkrs.StateStatus.Ok:
                log_file = log_game_error(state, pokers_action, f"State status not OK ({new_state.status})")
                if STRICT_CHECKING:
                    raise ValueError(f"State status not OK ({new_state.status}) during OS traversal. Details logged to {log_file}")
                elif VERBOSE:
                    print(f"WARNING: Invalid action {sampled_action} at depth {depth}. Status: {new_state.status}")
                return 0
        except Exception as e:
            if VERBOSE:
                print(f"ERROR in OS traversal for action {sampled_action}: {e}")
            if STRICT_CHECKING:
                raise
            return 0

        v_sampled = self.cfr_traverse_multi(new_state, iteration, traversing_player, depth + 1)

        if sampled_action == 3:
            self.raise_funnel_diag['hero_os']['raise_apply_ok'] += 1

        with torch.inference_mode():
            q_values = self.q_net(state_tensor.unsqueeze(0))[0].cpu().numpy()

        weight = min(1.0 / sampled_mu, self.hybrid_os_importance_weight_clip)

        self.os_importance_weight_sum += weight
        self.os_importance_weight_max = max(self.os_importance_weight_max, weight)
        self.os_importance_weight_count += 1

        correction = weight * (v_sampled - q_values[sampled_action])

        self.os_q_correction_abs_sum += abs(correction)
        self.os_q_correction_count += 1

        q_baseline_error = abs(q_values[sampled_action] - v_sampled)
        self.os_q_baseline_abs_error_sum += q_baseline_error
        self.os_q_baseline_abs_error_max = max(self.os_q_baseline_abs_error_max, q_baseline_error)
        self.os_q_baseline_abs_error_count += 1

        value_hat = np.copy(q_values)
        value_hat[sampled_action] = q_values[sampled_action] + correction

        ev_hat = sum(strategy[a] * value_hat[a] for a in legal_action_types)

        cf_regrets = np.zeros(self.num_actions, dtype=np.float32)
        for a in legal_action_types:
            cf_regrets[a] = value_hat[a] - ev_hat

        if self.hybrid_os_regret_clip is not None:
            clip_val = self.hybrid_os_regret_clip
            clipped_count = int(np.sum(np.abs(cf_regrets) > clip_val))
            self.os_regret_clip_count += clipped_count
            cf_regrets = np.clip(cf_regrets, -clip_val, clip_val)

        abs_regrets = np.abs(cf_regrets)
        self.os_regret_abs_sum += abs_regrets.sum()
        self.os_regret_sq_sum += (cf_regrets ** 2).sum()
        self.os_regret_count += len(legal_action_types)
        self.os_max_abs_regret = max(self.os_max_abs_regret, float(abs_regrets.max()))

        _state_arr = np.asarray(encoded_state, dtype=np.float32)
        _sizing_state_arr = self._encode_state_for_sizing(state, traversing_player).astype(np.float32)
        self._adv_buffer(traversing_player).add(_state_arr, cf_regrets, legal_mask, iteration)

        strategy_bet_size = sampled_bet_size if sampled_action == 3 and sampled_bet_size is not None else 0.0
        strategy_full = np.zeros(self.num_actions, dtype=np.float32)
        for a in legal_action_types:
            strategy_full[a] = strategy[a]
        self.strategy_buffer.add(_state_arr, strategy_full, legal_mask, iteration, strategy_bet_size)

        # Bug #93: при sizing_cfr_mode пишем сыгранную regret-matched стратегию
        # (strategy_sizing_net усреднит её — симметрия с action-стороной)
        write_strategy_sizing = (
            slot_weights_np is not None
            and (self.sizing_cfr_mode or self._sizing_q_is_ready())
        )
        if write_strategy_sizing:
            target = self._smooth_sizing_target_by_bucket(state, encoded_state, slot_weights_np, iteration)
            target = self._sparsify_sizing_target_by_buckets(target)
            self.sizing_strategy_buffer.add(_sizing_state_arr, target, iteration)

        # Bug #93/#94: per-anchor CFR-regret для advantage_sizing_net (OS-заземление)
        if self.sizing_cfr_mode and sampled_action == 3 and sampled_bet_size is not None:
            if self.sizing_allin_boundary_enabled and sizing_anchor_idx >= 0:
                eff_idx_cfr = sizing_anchor_idx
            else:
                eff_mult_cfr, _, _ = self._resolve_effective_sizing(state, sampled_bet_size)
                eff_idx_cfr = int(np.argmin(np.abs(self.anchors_arr - eff_mult_cfr)))
            q_anchor = self._evaluate_sizing_q_for_sizes(sizing_state_tensor, self.anchors_arr)
            sz_mu = max(float(slot_weights_np[sizing_anchor_idx]), 1e-6)
            sz_weight = min(1.0 / sz_mu, self.hybrid_os_importance_weight_clip)
            _, v_grounded = self._compute_sizing_q_target_values(v_sampled, state.pot)
            q_anchor[eff_idx_cfr] = q_anchor[eff_idx_cfr] + sz_weight * (v_grounded - q_anchor[eff_idx_cfr])
            ev_sizing = float((slot_weights_np * q_anchor).sum())
            sizing_cf_regrets = q_anchor - ev_sizing
            if self.sizing_anchor_availability_enabled:
                avail_mask, _ = self._anchor_availability(state)
                mask = avail_mask.astype(np.float32)
            else:
                mask = self._legal_sizing_anchor_mask(state).astype(np.float32)
            if self.sizing_advantage_buffer is not None:
                self.sizing_advantage_buffer.add(_sizing_state_arr, sizing_cf_regrets, mask, iteration)

        if sampled_action == 3 and sampled_bet_size is not None and self.sizing_q_enabled and self.sizing_q_net is not None:
            sizing_regret = float(value_hat[3]) - float(ev_hat)
            raw_v3 = sizing_regret + ev_hat
            pre_clip, sizing_target = self._compute_sizing_q_target_values(raw_v3, state.pot)
            eff_mult, _, selected_kind = self._resolve_effective_sizing(state, sampled_bet_size)
            eff_idx_diag = int(np.argmin(np.abs(self.anchors_arr - eff_mult)))
            self._record_selected_effective_sizing(
                'hero_os_raise',
                state,
                sizing_anchor_idx,
                eff_mult,
                selected_kind,
            )
            self._record_sizing_q_target_diag(
                source=0, anchor_idx=eff_idx_diag, raw_v3=raw_v3, pot=state.pot,
                pre_clip_target=pre_clip, post_clip_target=sizing_target,
            )
            if self.sizing_allin_boundary_enabled and sizing_anchor_idx >= 0:
                credit_size = sampled_bet_size
                credit_idx = int(sizing_anchor_idx)
            elif self.sizing_q_selected_credit_enabled and sizing_anchor_idx >= 0:
                credit_size = sampled_bet_size
                credit_idx = int(sizing_anchor_idx)
            else:
                credit_size = eff_mult
                credit_idx = eff_idx_diag
            self._add_sizing_q_sample(_sizing_state_arr, credit_size, sizing_target, iteration, is_probe=False,
                                       anchor_idx=credit_idx,
                                       selected_anchor_idx=sizing_anchor_idx,
                                       effective_anchor_idx=eff_idx_diag,
                                       selected_kind=selected_kind)
            self._update_insert_kind_diag(0, selected_kind, 'hero_os_raise')

        if self.use_q_baseline:
            if new_state.final_state:
                next_q_encoded = np.zeros_like(encoded_state)
                next_policy_encoded = np.zeros_like(encoded_state)
                next_mask = np.zeros(self.num_actions, dtype=np.float32)
                is_terminal = True
                reward = new_state.players_state[traversing_player].reward
                next_is_hero = False
            else:
                next_q_encoded = self._encode_state(new_state, traversing_player)
                next_policy_encoded = self._encode_state(new_state, new_state.current_player)
                next_mask = self.get_legal_action_mask(new_state)
                is_terminal = False
                reward = 0.0
                next_is_hero = int(new_state.current_player) == int(traversing_player)

            if not new_state.final_state and self.q_reward_propagation == 'mc':
                reward = v_sampled
                is_terminal = True

            if self.q_target_norm == 'pot_relative':
                reward = reward / max(float(state.pot), 1.0)
            elif self.q_target_norm == 'pot_stack_relative':
                denom = max(float(state.pot) + float(state.players_state[traversing_player].stake), 1.0)
                reward = reward / denom

            self.q_buffer.add(
                state=encoded_state.astype(np.float32),
                action=sampled_action,
                reward=reward,
                next_state=next_q_encoded.astype(np.float32),
                next_policy_state=next_policy_encoded.astype(np.float32),
                next_mask=next_mask,
                is_terminal=is_terminal,
                next_is_hero=next_is_hero,
                next_player_id=int(new_state.current_player) if not is_terminal else -1,
            )
            if sampled_action == 3:
                self.raise_funnel_diag['hero_os']['q_buffer_raise_add'] += 1

        return ev_hat

    def cfr_traverse_multi(self, state, iteration, traversing_player, depth=0, random_agent=None):
        if depth == 0 and random_agent is not None:
            self._traversal_random_agent = random_agent

        self.traversal_nodes += 1
        self.traversal_max_depth_observed = max(self.traversal_max_depth_observed, depth)

        max_depth = 200
        if depth > max_depth:
            self.traversal_max_depth_hits += 1
            if VERBOSE:
                print(f"WARNING: Max recursion depth reached ({max_depth}). Returning zero value.")
            return 0

        if state.final_state:
            self.traversal_terminal_nodes += 1
            return state.players_state[traversing_player].reward

        current_player = state.current_player

        if self._traversal_random_agent is not None and current_player == self.num_players - 1:
            try:
                action = self._traversal_random_agent.choose_action(state)
            except Exception:
                if VERBOSE:
                    print(f"ERROR: RandomAgent choose_action failed at depth {depth}")
                return 0
            new_state = state.apply_action(action)
            if new_state.status != pkrs.StateStatus.Ok:
                if VERBOSE:
                    print(f"WARNING: RandomAgent invalid action at depth {depth}. Status: {new_state.status}")
                return 0
            return self.cfr_traverse_multi(new_state, iteration, traversing_player, depth=depth + 1)

        legal_mask = self.get_legal_action_mask(state)
        legal_action_types = [a for a in range(self.num_actions) if legal_mask[a] > 0]

        if not legal_action_types:
            if VERBOSE:
                print(f"WARNING: No legal actions for player {current_player} at depth {depth}")
            return 0

        if current_player == traversing_player:
            self.traversal_traversing_decision_nodes += 1

            if self._should_use_outcome_sampling(state, traversing_player):
                return self._cfr_traverse_multi_outcome_node(state, iteration, traversing_player, depth)

            encoded_state = self._encode_state(state, traversing_player)
            state_tensor = torch.from_numpy(encoded_state.astype(np.float32, copy=False)).to(self.device)
            sizing_encoded = self._encode_state_for_sizing(state, traversing_player)
            sizing_state_tensor = torch.from_numpy(sizing_encoded.astype(np.float32)).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                action_logits = self._adv_net(traversing_player)(state_tensor.unsqueeze(0))
                advantages = action_logits[0].cpu().numpy()

            sampled_bet_size = None
            bet_size_multiplier = None
            slot_weights_np = None
            is_probe = False
            sizing_anchor_idx = -1

            if 3 in legal_action_types:
                self.raise_funnel_diag['hero_full']['raise_legal'] += 1
                slot_sizes_np = self.anchors_arr

                probe_prob = self._current_sizing_probe_prob(iteration)
                if self.sizing_q_enabled and random.random() < probe_prob:
                    sampled_bet_size = self._sample_probe_size(slot_sizes_np)
                    is_probe = True
                    sizing_anchor_idx = -1
                    slot_weights_np = np.ones(self.num_anchors, dtype=np.float32) / self.num_anchors
                else:
                    sampled_bet_size, slot_weights_np, sizing_anchor_idx = self._hierarchical_sizing(state_tensor, iteration, traversing_player, state=state)

                if sizing_anchor_idx >= 0:
                    self.sizing_path_diag['hero_full_raise'][sizing_anchor_idx] += 1

                self.raise_funnel_diag['hero_full']['raise_sampled'] += 1
                bet_size_multiplier = sampled_bet_size
            else:
                bet_size_multiplier = float(np.mean(self.anchors))

            pos_advantages = np.maximum(advantages, 0)
            pos_masked = pos_advantages * legal_mask
            adv_sum = pos_masked.sum()
            if adv_sum > 0:
                strategy = pos_masked / adv_sum
            else:
                strategy = legal_mask / legal_mask.sum()

            action_values = np.zeros(self.num_actions)
            sizing_traversal_values = []
            sizing_traversal_samples = []
            if self.sizing_cfr_mode and not is_probe and slot_weights_np is not None and 3 in legal_action_types:
                sizing_traversal_samples = self._sample_traversal_sizing_anchors(
                    slot_weights_np, state=state, iteration=iteration)
            for action_type in legal_action_types:
                try:
                    if action_type == 3 and sizing_traversal_samples:
                        for traversal_anchor_idx, conditional_mu in sizing_traversal_samples:
                            traversal_size = float(self.anchors[traversal_anchor_idx])
                            pokers_action = self.action_type_to_pokers_action(action_type, state, traversal_size)
                            new_state = state.apply_action(pokers_action)
                            if new_state.status != pkrs.StateStatus.Ok:
                                continue
                            branch_value = self.cfr_traverse_multi(
                                new_state, iteration, traversing_player, depth + 1)
                            sizing_traversal_values.append(
                                (traversal_anchor_idx, conditional_mu, branch_value))
                            self.raise_funnel_diag['hero_full']['raise_apply_ok'] += 1
                        if sizing_traversal_values:
                            action_values[action_type] = float(np.mean(
                                [value for _, _, value in sizing_traversal_values]))
                        continue
                    if action_type == 3:
                        pokers_action = self.action_type_to_pokers_action(action_type, state, bet_size_multiplier)
                    else:
                        pokers_action = self.action_type_to_pokers_action(action_type, state)

                    new_state = state.apply_action(pokers_action)

                    if new_state.status != pkrs.StateStatus.Ok:
                        log_file = log_game_error(state, pokers_action, f"State status not OK ({new_state.status})")
                        if STRICT_CHECKING:
                            raise ValueError(f"State status not OK ({new_state.status}) during CFR traversal. Details logged to {log_file}")
                        elif VERBOSE:
                            print(f"WARNING: Invalid action {action_type} at depth {depth}. Status: {new_state.status}")
                            print(f"Details logged to {log_file}")
                        continue

                    action_values[action_type] = self.cfr_traverse_multi(new_state, iteration, traversing_player, depth + 1)
                    if action_type == 3:
                        self.raise_funnel_diag['hero_full']['raise_apply_ok'] += 1
                except Exception as e:
                    if VERBOSE:
                        print(f"ERROR in traversal for action {action_type}: {e}")
                    action_values[action_type] = 0
                    if STRICT_CHECKING:
                        raise

            ev = sum(strategy[a] * action_values[a] for a in legal_action_types)

            _state_arr = np.asarray(encoded_state, dtype=np.float32)
            _sizing_state_arr = self._encode_state_for_sizing(state, traversing_player).astype(np.float32)
            cf_regrets = np.zeros(self.num_actions, dtype=np.float32)
            for a in legal_action_types:
                cf_regrets[a] = action_values[a] - ev

            # #96: advantage-regret scale normalization
            if self.advantage_regret_norm == 'pot_stack':
                denom = max(float(state.pot) + float(state.players_state[traversing_player].stake), 1.0)
                cf_regrets = cf_regrets / denom
            elif self.advantage_regret_norm == 'per_node_max':
                max_abs_val = max(abs(action_values[a]) for a in legal_action_types)
                denom = max(max_abs_val, 1.0)
                cf_regrets = cf_regrets / denom
            if self.advantage_regret_clip is not None:
                cf_regrets = np.clip(cf_regrets, -self.advantage_regret_clip, self.advantage_regret_clip)
            cf_regrets = cf_regrets.astype(np.float32)
            if self.advantage_reward_scale != 1.0:
                cf_regrets = (cf_regrets / self.advantage_reward_scale).astype(np.float32)

            self._adv_buffer(traversing_player).add(_state_arr, cf_regrets, legal_mask, iteration)

            # Bug #93: при sizing_cfr_mode пишем сыгранную regret-matched стратегию
            write_strategy_sizing = (
                slot_weights_np is not None
                and (self.sizing_cfr_mode or self._sizing_q_is_ready())
            )
            if write_strategy_sizing:
                target = self._smooth_sizing_target_by_bucket(state, encoded_state, slot_weights_np, iteration)
                target = self._sparsify_sizing_target_by_buckets(target)
                self.sizing_strategy_buffer.add(_sizing_state_arr, target, iteration)

            # Bug #93/#94: per-anchor CFR-regret для advantage_sizing_net (заземление через action_values[3])
            if self.sizing_cfr_mode and sampled_bet_size is not None and 3 in legal_action_types:
                if self.sizing_allin_boundary_enabled and sizing_anchor_idx >= 0:
                    eff_idx_cfr = sizing_anchor_idx
                else:
                    eff_mult, _, _ = self._resolve_effective_sizing(state, sampled_bet_size)
                    eff_idx_cfr = int(np.argmin(np.abs(self.anchors_arr - eff_mult)))
                q_anchor = self._evaluate_sizing_q_for_sizes(sizing_state_tensor, self.anchors_arr)
                if sizing_traversal_values:
                    credited_effective = set()
                    for selected_idx, conditional_mu, branch_value in sizing_traversal_values:
                        if self.sizing_allin_boundary_enabled:
                            effective_idx = int(selected_idx)
                        else:
                            effective_mult, _, _ = self._resolve_effective_sizing(
                                state, float(self.anchors[selected_idx]))
                            effective_idx = int(np.argmin(np.abs(self.anchors_arr - effective_mult)))
                        if effective_idx in credited_effective:
                            continue
                        credited_effective.add(effective_idx)
                        _, grounded = self._compute_sizing_q_target_values(branch_value, state.pot)
                        weight = min(1.0 / max(float(conditional_mu), 1e-6),
                                     self.hybrid_os_importance_weight_clip)
                        q_anchor[effective_idx] += weight * (grounded - q_anchor[effective_idx])
                else:
                    _, v_grounded = self._compute_sizing_q_target_values(action_values[3], state.pot)
                    if sizing_anchor_idx >= 0 and slot_weights_np is not None:
                        sz_mu = max(float(slot_weights_np[sizing_anchor_idx]), 1e-6)
                        sz_weight = min(1.0 / sz_mu, self.hybrid_os_importance_weight_clip)
                        q_anchor[eff_idx_cfr] = q_anchor[eff_idx_cfr] + sz_weight * (v_grounded - q_anchor[eff_idx_cfr])
                    else:
                        q_anchor[eff_idx_cfr] = v_grounded
                ev_sizing = float((slot_weights_np * q_anchor).sum())
                sizing_cf_regrets = q_anchor - ev_sizing
                if self.sizing_anchor_availability_enabled:
                    avail_mask, _ = self._anchor_availability(state)
                    mask = avail_mask.astype(np.float32)
                else:
                    mask = self._legal_sizing_anchor_mask(state).astype(np.float32)
                if self.sizing_advantage_buffer is not None:
                    self.sizing_advantage_buffer.add(_sizing_state_arr, sizing_cf_regrets, mask, iteration)

            # Store Q sample
            if sampled_bet_size is not None and 3 in legal_action_types and self.sizing_q_enabled and self.sizing_q_net is not None:
                sizing_regret = float(action_values[3]) - float(ev)
                raw_v3 = sizing_regret + ev
                pre_clip, sizing_target = self._compute_sizing_q_target_values(raw_v3, state.pot)
                eff_mult, _, selected_kind = self._resolve_effective_sizing(state, sampled_bet_size)
                eff_idx_diag = int(np.argmin(np.abs(self.anchors_arr - eff_mult)))
                self._record_selected_effective_sizing(
                    'hero_full_raise',
                    state,
                    sizing_anchor_idx,
                    eff_mult,
                    selected_kind,
                )
                source_s1 = 1 if is_probe else 0
                self._record_sizing_q_target_diag(
                    source=source_s1, anchor_idx=eff_idx_diag, raw_v3=raw_v3, pot=state.pot,
                    pre_clip_target=pre_clip, post_clip_target=sizing_target,
                )
                if self.sizing_allin_boundary_enabled and sizing_anchor_idx >= 0:
                    credit_size = sampled_bet_size
                    credit_idx = int(sizing_anchor_idx)
                elif self.sizing_q_selected_credit_enabled and sizing_anchor_idx >= 0:
                    credit_size = sampled_bet_size
                    credit_idx = int(sizing_anchor_idx)
                else:
                    credit_size = eff_mult
                    credit_idx = eff_idx_diag
                self._add_sizing_q_sample(_sizing_state_arr, credit_size, sizing_target, iteration,
                                          is_probe=is_probe, anchor_idx=credit_idx,
                                          selected_anchor_idx=sizing_anchor_idx,
                                          effective_anchor_idx=eff_idx_diag,
                                          selected_kind=selected_kind)
                self._update_insert_kind_diag(source_s1, selected_kind, 'hero_full_raise')
                self.raise_funnel_diag['hero_full']['q_buffer_raise_add'] += 1

            # One-step lookahead: Q-targets для несыгранных sizing'ов (Bug #64)
            if self.sizing_q_enabled and self.sizing_q_net is not None and 3 in legal_action_types:
                _lookahead_eff_idx = eff_idx_diag if sampled_bet_size is not None else -1
                self._perform_sizing_lookahead(
                    state, _sizing_state_arr, iteration, traversing_player, ev,
                    eff_idx=_lookahead_eff_idx,
                )

            strategy_bet_size = sampled_bet_size if sampled_bet_size is not None else bet_size_multiplier if 3 in legal_action_types else 0.0
            strategy_full = np.zeros(self.num_actions, dtype=np.float32)
            for a in legal_action_types:
                strategy_full[a] = strategy[a]
            self.strategy_buffer.add(_state_arr, strategy_full, legal_mask, iteration, strategy_bet_size)

            return ev

        else:
            self.traversal_opponent_decision_nodes += 1
            try:
                opp_mask = legal_mask
                opp_legal = legal_action_types

                opp_encoded = self._encode_state(state, current_player)
                opp_state_tensor = torch.from_numpy(
                    opp_encoded.astype(np.float32, copy=False)
                ).unsqueeze(0).to(self.device)

                q_encoded = self._encode_state(state, traversing_player)
                q_state_tensor = torch.from_numpy(
                    q_encoded.astype(np.float32, copy=False)
                ).unsqueeze(0).to(self.device)

                q_values = None
                with torch.inference_mode():
                    opp_advantages = self._adv_net(current_player)(opp_state_tensor)[0].cpu().numpy()

                    if self.use_q_baseline:
                        q_values = self.q_net(q_state_tensor)[0].cpu().numpy()

                opp_pos = np.maximum(opp_advantages, 0) * opp_mask
                opp_sum = opp_pos.sum()
                if opp_sum > 0:
                    opp_strategy = opp_pos / opp_sum
                else:
                    opp_strategy = opp_mask / opp_mask.sum()

                legal_probs = np.array([opp_strategy[a] for a in opp_legal])
                if np.sum(legal_probs) > 0:
                    legal_probs = legal_probs / np.sum(legal_probs)
                else:
                    legal_probs = np.ones(len(opp_legal)) / len(opp_legal)

                action_idx = np.random.choice(len(opp_legal), p=legal_probs)
                action_type = opp_legal[action_idx]

                if action_type == 3:
                    self.raise_funnel_diag['opponent']['raise_legal'] += 1
                    opp_bet_size, _, opp_anchor_idx = self._hierarchical_sizing(opp_state_tensor, iteration, int(state.current_player), state=state, use_q=False)
                    if opp_anchor_idx >= 0:
                        self.sizing_path_diag['opponent_raise'][opp_anchor_idx] += 1
                    self.raise_funnel_diag['opponent']['raise_sampled'] += 1
                    eff_mult, _, selected_kind = self._resolve_effective_sizing(state, opp_bet_size)
                    self._record_selected_effective_sizing(
                        'opponent_raise',
                        state,
                        opp_anchor_idx,
                        eff_mult,
                        selected_kind,
                    )
                    pokers_action = self.action_type_to_pokers_action(action_type, state, opp_bet_size)
                else:
                    pokers_action = self.action_type_to_pokers_action(action_type, state)

                new_state = state.apply_action(pokers_action)

                if new_state.status != pkrs.StateStatus.Ok:
                    log_file = log_game_error(state, pokers_action, f"State status not OK ({new_state.status})")
                    if STRICT_CHECKING:
                        raise ValueError(f"State status not OK ({new_state.status}) from advantage_net. Details logged to {log_file}")
                    if VERBOSE:
                        print(f"WARNING: advantage_net invalid action at depth {depth}. Status: {new_state.status}")
                        print(f"Details logged to {log_file}")
                    return 0

                v_sampled = self.cfr_traverse_multi(new_state, iteration, traversing_player, depth + 1)

                if action_type == 3:
                    self.raise_funnel_diag['opponent']['raise_apply_ok'] += 1

                if self.use_q_baseline:
                    baseline = sum(opp_strategy[a] * q_values[a] for a in opp_legal)
                    adjusted_value = baseline + (v_sampled - q_values[action_type])

                    if new_state.final_state:
                        next_q_encoded = np.zeros_like(q_encoded)
                        next_policy_encoded = np.zeros_like(q_encoded)
                        next_mask = np.zeros(self.num_actions, dtype=np.float32)
                        is_terminal = True
                        reward = new_state.players_state[traversing_player].reward
                        next_is_hero = False
                    else:
                        next_q_encoded = self._encode_state(new_state, traversing_player)
                        next_policy_encoded = self._encode_state(new_state, new_state.current_player)
                        next_mask = self.get_legal_action_mask(new_state)
                        is_terminal = False
                        reward = 0.0
                        next_is_hero = int(new_state.current_player) == int(traversing_player)

                    if not new_state.final_state and self.q_reward_propagation == 'mc':
                        reward = v_sampled
                        is_terminal = True

                    if self.q_target_norm == 'pot_relative':
                        reward = reward / max(float(state.pot), 1.0)
                    elif self.q_target_norm == 'pot_stack_relative':
                        denom = max(float(state.pot) + float(state.players_state[traversing_player].stake), 1.0)
                        reward = reward / denom

                    self.q_buffer.add(
                        state=q_encoded.astype(np.float32),
                        action=action_type,
                        reward=reward,
                        next_state=next_q_encoded.astype(np.float32),
                        next_policy_state=next_policy_encoded.astype(np.float32),
                        next_mask=next_mask,
                        is_terminal=is_terminal,
                        next_is_hero=next_is_hero,
                        next_player_id=int(new_state.current_player) if not is_terminal else -1,
                    )
                    if action_type == 3:
                        self.raise_funnel_diag['opponent']['q_buffer_raise_add'] += 1
                else:
                    adjusted_value = v_sampled

                return adjusted_value
            except Exception as e:
                if VERBOSE:
                    print(f"ERROR in advantage_net traversal: {e}")
                if STRICT_CHECKING:
                    raise
                return 0

    def train_advantage_network(self, player_id=None, batch_size=256, epochs=3, beta_start=0.4, beta_end=1.0):
        """LEGACY: не используется в train_self_play_multi. Используйте train_advantage_network_multi()."""
        raise NotImplementedError(
            "train_advantage_network() — legacy-метод (PER + per-action формат). "
            "Используйте train_advantage_network_multi() для DCFR+ с full-vector AdvantageBuffer."
        )

    def train_advantage_network_multi(self, batch_size=None, epochs=None, player_id=0):
        if batch_size is None:
            batch_size = self.advantage_batch_size
        if epochs is None:
            epochs = self.advantage_epochs
        buf = self._adv_buffer(player_id)
        n = len(buf)
        if n == 0:
            self.last_advantage_train_steps = 0
            self.last_advantage_effective_batch_size = 0
            return 0.0

        effective_batch_size = min(batch_size, n)
        self.last_advantage_effective_batch_size = effective_batch_size

        profile_enabled = bool(cfg_get('training_profiling_enabled', True))
        profile_started = time.perf_counter()
        sample_started = time.perf_counter()
        samples = buf.sample(n)
        sample_seconds = time.perf_counter() - sample_started
        if samples is None:
            self.last_advantage_train_steps = 0
            self.last_advantage_effective_batch_size = 0
            return 0.0
        all_states, all_regrets, all_masks, all_iterations = samples

        materialize_started = time.perf_counter()
        perm = np.random.permutation(len(all_states))
        all_states = all_states[perm]
        all_regrets = all_regrets[perm]
        all_masks = all_masks[perm]
        all_iterations = all_iterations[perm]
        materialize_seconds = time.perf_counter() - materialize_started

        net = self._adv_net(player_id)
        net.train()
        total_loss = 0
        steps = 0
        t = self.iteration_count
        max_abs_target = 0.0
        max_abs_pred = 0.0

        all_target_chunks = []
        all_error_chunks = []

        epoch_seconds = []
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            ep_perm = np.random.permutation(len(all_states))

            for start in range(0, len(all_states), effective_batch_size):
                sel = ep_perm[start:start + effective_batch_size]

                state_tensors = torch.from_numpy(all_states[sel]).to(self.device)
                regret_tensors = torch.from_numpy(all_regrets[sel]).to(self.device)
                mask_tensors = torch.from_numpy(all_masks[sel]).to(self.device)
                iter_tensors = torch.from_numpy(all_iterations[sel]).float().to(self.device)

                with torch.no_grad():
                    prev_logits = self._target_net(player_id)(state_tensors)
                    prev_pred = prev_logits[:, :self.num_actions]
                    prev_clamped = torch.clamp(prev_pred, min=0)

                is_fresh = (iter_tensors == t).unsqueeze(1).float()

                if t > 1:
                    if self.advantage_accumulation == 'plain':
                        fresh_target = prev_pred + regret_tensors
                        old_target = prev_pred
                    else:
                        discount = (t - 1) ** self.discount_alpha / ((t - 1) ** self.discount_alpha + 1.5)
                        fresh_target = prev_clamped * discount + regret_tensors
                        old_target = prev_clamped
                    bootstrap_target = is_fresh * fresh_target + (1.0 - is_fresh) * old_target
                else:
                    bootstrap_target = is_fresh * regret_tensors + (1.0 - is_fresh) * prev_clamped

                batch_max_target = bootstrap_target.abs().max().item()
                max_abs_target = max(max_abs_target, batch_max_target)

                action_logits = net(state_tensors)
                predicted = action_logits[:, :self.num_actions]
                batch_max_pred = predicted.abs().max().item()
                max_abs_pred = max(max_abs_pred, batch_max_pred)

                if epoch == 0:
                    mask_flat = mask_tensors.bool()
                    batch_target = bootstrap_target.detach()[mask_flat].cpu()
                    batch_pred = predicted.detach()[mask_flat].cpu()
                    batch_error = (batch_pred - batch_target).abs()
                    all_target_chunks.append(batch_target)
                    all_error_chunks.append(batch_error)

                predicted_masked = predicted * mask_tensors
                target_masked = bootstrap_target * mask_tensors

                if self.advantage_loss == 'huber':
                    loss = F.smooth_l1_loss(predicted_masked, target_masked.detach(), beta=self.advantage_huber_delta)
                else:
                    loss = F.mse_loss(predicted_masked, target_masked.detach())

                optimizer = self._adv_optimizer(player_id)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                steps += 1

            if epoch == 0:
                all_targets = torch.cat(all_target_chunks)
                all_errors = torch.cat(all_error_chunks)

            epoch_seconds.append(time.perf_counter() - epoch_started)

        self.last_advantage_train_steps = steps
        self._target_net(player_id).load_state_dict(net.state_dict())

        diagnostics_started = time.perf_counter()
        if all_targets is not None and all_targets.numel() > 0:
            targets_abs_flat = torch.flatten(all_targets.abs())
            errors_flat = torch.flatten(all_errors)
            self.last_advantage_target_stats = {
                'target_abs_max': float(targets_abs_flat.max().item()),
                'target_abs_p99': float(
                    torch.quantile(targets_abs_flat, 0.99).item()),
                'target_abs_median': float(targets_abs_flat.median().item()),
                'error_abs_max': float(errors_flat.max().item()),
                'error_abs_p99': float(
                    torch.quantile(errors_flat, 0.99).item()),
                'error_abs_median': float(errors_flat.median().item()),
                'frac_error_gt_1': float(
                    (errors_flat > 1.0).float().mean().item()),
            }
        else:
            self.last_advantage_target_stats = None

        diagnostics_seconds = time.perf_counter() - diagnostics_started
        expected_steps = epochs * math.ceil(n / effective_batch_size)
        self.last_advantage_profile = {"samples": n, "batch": effective_batch_size, "epochs": epochs, "expected_steps": expected_steps, "actual_steps": steps, "sample_seconds": sample_seconds, "materialize_seconds": materialize_seconds, "epoch_seconds": epoch_seconds, "diagnostics_seconds": diagnostics_seconds, "total_seconds": time.perf_counter() - profile_started}
        if profile_enabled:
            epoch0 = epoch_seconds[0] if epoch_seconds else 0.0
            remaining = sum(epoch_seconds[1:])
            print(f"  [AdvProfile] samples={n} batch={effective_batch_size} epochs={epochs} steps={steps}/{expected_steps} sample={sample_seconds:.3f}s materialize={materialize_seconds:.3f}s epoch0={epoch0:.3f}s epochs1+={remaining:.3f}s diagnostics={diagnostics_seconds:.3f}s total={self.last_advantage_profile['total_seconds']:.3f}s")

        self._log_advantage_magnitude(player_id, max_abs_target, max_abs_pred)

        return total_loss / max(steps, 1)

    def _log_advantage_magnitude(self, player_id, max_target, max_pred):
        if self.advantage_accumulation != 'plain':
            return
        t = self.iteration_count
        if t % 10 == 0 or max_target > 100:
            print(f"  [N4-monitor] iter={t} p{player_id} "
                  f"max|bootstrap_target|={max_target:.2f} max|adv_net_out|={max_pred:.2f}")



    def _compute_sizing_q_target_values(self, raw_v3, pot):
        """Вычисляет pre-clip и post-clip target для диагностики (Bug #68).

        Возвращает (pre_clip_target, post_clip_target).
        """
        pot_val = max(float(pot), 1.0)
        if self.sizing_q_target_normalization == 'raw':
            pre_clip = float(raw_v3)
        elif self.sizing_q_target_normalization == 'pot':
            pre_clip = float(raw_v3) / pot_val
        else:
            pre_clip = float(raw_v3) / pot_val
        post_clip = float(np.clip(pre_clip, -5.0, 5.0))
        return pre_clip, post_clip

    def _record_selected_effective_sizing(self, path, state, selected_idx, effective_mult, selected_kind):
        matrix = self.sizing_selected_effective_diag.get(path)
        kind_diag = self.sizing_selected_effective_kind_diag.get(path)
        if matrix is None or kind_diag is None:
            return
        if selected_kind not in kind_diag['selected_kind']:
            return
        if selected_idx < 0 or selected_idx >= self.num_anchors:
            return

        effective_mult = float(effective_mult)
        bucket_idx = int(np.argmin(np.abs(self.anchors_arr - effective_mult)))
        matrix[int(selected_idx), bucket_idx] += 1
        kind_diag['selected_kind'][selected_kind] += 1

        bucket_anchor = float(self.anchors_arr[bucket_idx])
        _, _, bucket_kind = self._resolve_effective_sizing(state, bucket_anchor)
        if bucket_kind in kind_diag['bucket_kind']:
            kind_diag['bucket_kind'][bucket_kind] += 1

        eps = 1e-6
        if bucket_anchor < effective_mult - eps:
            kind_diag['bucket_relation']['below_effective'] += 1
        elif bucket_anchor > effective_mult + eps:
            kind_diag['bucket_relation']['above_effective'] += 1
        else:
            kind_diag['bucket_relation']['equal_effective'] += 1

    def _upd_la_stat(self, key, val):
        """Bug #74 диаг: обновляет бегущие [n, sum, sumsq, min, max] для метрики."""
        val = float(val)
        s = self.sizing_lookahead_diag['stats'].get(key)
        if s is None:
            self.sizing_lookahead_diag['stats'][key] = [1, val, val * val, val, val]
        else:
            s[0] += 1
            s[1] += val
            s[2] += val * val
            s[3] = min(s[3], val)
            s[4] = max(s[4], val)

    def _record_sizing_q_target_diag(self, source, anchor_idx, raw_v3, pot,
                                      pre_clip_target, post_clip_target):
        """Записывает диагностику target formation (Bug #68)."""
        if not self.sizing_q_target_diagnostics_enabled:
            return

        source = int(source)
        anchor_idx = int(anchor_idx)
        key = (source, anchor_idx)

        if key not in self.sizing_q_target_diag:
            self.sizing_q_target_diag[key] = {
                'count': 0,
                'clip_high': 0,
                'clip_low': 0,
                'raw_min': float('inf'),
                'raw_max': float('-inf'),
                'pre_min': float('inf'),
                'pre_max': float('-inf'),
                'post_min': float('inf'),
                'post_max': float('-inf'),
            }

        stat = self.sizing_q_target_diag[key]
        raw = float(raw_v3)
        pre = float(pre_clip_target)
        post = float(post_clip_target)

        stat['count'] += 1
        stat['raw_min'] = min(stat['raw_min'], raw)
        stat['raw_max'] = max(stat['raw_max'], raw)
        stat['pre_min'] = min(stat['pre_min'], pre)
        stat['pre_max'] = max(stat['pre_max'], pre)
        stat['post_min'] = min(stat['post_min'], post)
        stat['post_max'] = max(stat['post_max'], post)

        if pre >= 5.0:
            stat['clip_high'] += 1
        if pre <= -5.0:
            stat['clip_low'] += 1

    def _normalize_sizing_q_target(self, raw_v3, ev_hat, pot):
        """Bug #48: target SizingQ = pot-normalized value, НЕ pre-subtracted relative advantage.

        Раньше использовался (raw_v3 - ev_hat) / pot — двойной baseline:
        Q-target уже содержал относительное преимущество, а AWR ещё раз
        вычитал mean(Q) по candidates. Это системно занижало Q для крупных
        sizing и приводило к монотонно убывающей Q-поверхности.

        Теперь Q учит pot-normalized абсолютное value: raw_v3 / pot.
        Relative comparison делается только один раз — внутри AWR через
        q_baseline = mean(Q) по candidates per state.
        """
        _, post_clip = self._compute_sizing_q_target_values(raw_v3, pot)
        return post_clip

    def train_sizing_anchor_network(self, batch_size=256, train_steps=1, player_id=0):
        """Bug #50: Q-derived advantage training над фиксированной сеткой.

        Bug #93: при sizing_cfr_mode учит CFR-regret'ы из SizingAdvantageBuffer
        (симметрия с action-advantage training).
        """
        pid = 0
        # Sizing advantage follows the action-advantage loss configuration.
        use_huber = self.advantage_loss == 'huber'
        # Bug #93: CFR-режим — учим настоящие regret'ы
        if self.sizing_cfr_mode:
            buf = self.sizing_advantage_buffer
            if buf is None:
                return 0.0
            n = len(buf)
            if n == 0:
                return 0.0
            effective_batch = min(batch_size, n)

            sizing_net = self.advantage_sizing_net
            sizing_net.train()
            total_loss = 0.0
            completed = 0

            for _ in range(train_steps):
                samples = buf.sample(effective_batch)
                if samples is None:
                    continue
                states, regrets, masks, iterations = samples
                state_tensors = torch.from_numpy(states.copy()).to(self.device)
                regrets_t = torch.from_numpy(regrets.copy()).to(self.device)
                masks_t = torch.from_numpy(masks.copy()).to(self.device)
                iterations_t = torch.from_numpy(iterations.copy()).to(self.device)

                slot_logits = sizing_net(state_tensors)
                with torch.no_grad():
                    previous = self.sizing_target_net(state_tensors)
                    previous = torch.clamp(previous, min=0.0)
                    t = max(int(self.iteration_count), 1)
                    discount = ((t - 1) ** self.discount_alpha /
                                (((t - 1) ** self.discount_alpha) + 1.5)) if t > 1 else 0.0
                    fresh = (iterations_t == t).unsqueeze(1).float()
                    accumulated = previous * discount + regrets_t
                    target = (fresh * accumulated + (1.0 - fresh) * previous) * masks_t

                if use_huber:
                    per_sample_loss = F.smooth_l1_loss(slot_logits, target, reduction='none',
                                                       beta=self.sizing_q_huber_delta)
                else:
                    per_sample_loss = F.mse_loss(slot_logits, target, reduction='none')
                weighted = (per_sample_loss * masks_t).sum()
                denom = masks_t.sum().clamp_min(1)
                loss_anchor = weighted / denom

                loss = loss_anchor
                self.last_advantage_sizing_anchor_loss = float(loss_anchor.item())
                self.last_advantage_sizing_bucket_loss = 0.0

                self.sizing_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sizing_net.parameters(), max_norm=1.0)
                self.sizing_optimizer.step()
                total_loss += loss.item()
                completed += 1

            if completed:
                self.sizing_target_net.load_state_dict(sizing_net.state_dict())
                self.sizing_target_net.eval()
            return total_loss / max(completed, 1)

        # Оригинальный путь: Q-derived training (default)
        if not self._sizing_q_is_ready():
            return 0.0

        n = len(self.sizing_strategy_buffer)
        if n == 0:
            return 0.0
        effective_batch = min(batch_size, n)

        sizing_net = self.advantage_sizing_net
        sizing_net.train()
        total_loss = 0.0
        completed = 0
        fixed_sizes_t = torch.tensor(self.anchors, dtype=torch.float32, device=self.device)
        norm_sizes_t = (fixed_sizes_t - self.min_bet_size) / (self.max_bet_size - self.min_bet_size)
        K = len(self.anchors)

        for _ in range(train_steps):
            samples = self.sizing_strategy_buffer.sample(effective_batch)
            if samples is None:
                continue
            states = samples[0]
            state_tensors = torch.from_numpy(states.copy()).to(self.device)
            B = state_tensors.shape[0]

            slot_logits = sizing_net(state_tensors)

            state_expanded = state_tensors.unsqueeze(1).expand(-1, K, -1).reshape(-1, state_tensors.shape[-1])
            sizes_expanded = norm_sizes_t.unsqueeze(0).expand(B, -1).reshape(-1)

            self.sizing_q_net.eval()
            with torch.no_grad():
                q_vals = self.sizing_q_net(state_expanded, sizes_expanded).reshape(B, K)
                baseline = q_vals.mean(dim=1, keepdim=True)
                target_adv = q_vals - baseline

            if use_huber:
                loss_anchor = F.smooth_l1_loss(slot_logits, target_adv, beta=self.sizing_q_huber_delta)
            else:
                loss_anchor = F.mse_loss(slot_logits, target_adv)

            loss = loss_anchor
            self.last_advantage_sizing_bucket_loss = 0.0

            self.last_advantage_sizing_anchor_loss = float(loss_anchor.item())

            self.sizing_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sizing_net.parameters(), max_norm=1.0)
            self.sizing_optimizer.step()
            total_loss += loss.item()
            completed += 1

        return total_loss / max(completed, 1)

    def train_strategy_network(self, batch_size=None, epochs=None):
        if batch_size is None:
            batch_size = self.strategy_batch_size
        if epochs is None:
            epochs = self.strategy_epochs
        n = len(self.strategy_buffer)
        if n == 0:
            return 0

        effective_batch_size = min(batch_size, n)
        self.strategy_net.train()
        T = max(self.iteration_count, 1)
        total_loss = 0

        for _ in range(epochs):
            samples = self.strategy_buffer.sample(effective_batch_size)
            if samples is None:
                return 0
            states, policies, masks, iterations, bet_sizes = samples

            state_tensors = torch.from_numpy(states.copy()).to(self.device)
            policy_tensors = torch.from_numpy(policies.copy()).to(self.device)
            mask_tensors = torch.from_numpy(masks.copy()).to(self.device)
            iteration_tensors = torch.from_numpy(iterations.copy()).to(self.device)

            weights = torch.pow(
                torch.clamp(iteration_tensors / T, min=1e-6),
                self.discount_gamma)

            # Action-policy: DCFR+ weighted MSE — per-sample loss (вес один раз)
            action_logits = self.strategy_net(state_tensors)
            masked_logits = torch.where(
                mask_tensors == 1,
                action_logits[:, :self.num_actions],
                torch.tensor(-1e20, device=self.device))
            predicted_policies = F.softmax(masked_logits, dim=1)

            per_sample_loss = (
                (predicted_policies - policy_tensors) ** 2 * mask_tensors
            ).sum(dim=1)
            action_loss = (per_sample_loss * weights).mean()

            self.strategy_optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), max_norm=0.5)
            self.strategy_optimizer.step()

            total_loss += action_loss.item()

        return total_loss / epochs

    def train_strategy_sizing_anchor_network(self, batch_size=128, train_steps=20):
        """Bug #50: KL-distill slot weights на strategy_sizing_net.

        Target weights = Q-derived heat weights из sizing_strategy_buffer.
        Только Q-ready данные.
        """
        if not self._sizing_q_is_ready():
            return 0.0
        n = len(self.sizing_strategy_buffer)
        if n == 0:
            return 0.0
        effective_batch = min(batch_size, n)
        self.strategy_sizing_net.train()
        total_loss = 0.0
        completed_steps = 0
        t = max(self.iteration_count, 1)
        for _ in range(train_steps):
            samples = self.sizing_strategy_buffer.sample(effective_batch)
            if samples is None:
                continue
            states, target_weights, iterations = samples
            state_tensors = torch.from_numpy(states.copy()).to(self.device)
            target_weights_t = torch.from_numpy(target_weights.copy()).to(self.device)
            target_weights_t = target_weights_t.clamp_min(1e-6)
            target_weights_t = target_weights_t / target_weights_t.sum(dim=1, keepdim=True)
            iter_tensors = torch.from_numpy(iterations.copy()).float().to(self.device)
            iter_weights = torch.pow(torch.clamp(iter_tensors / t, min=1e-6), self.discount_gamma)

            slot_logits, scalar_bet = self.strategy_sizing_net(state_tensors)

            pred_log_probs = F.log_softmax(slot_logits, dim=1)
            kl_per_sample = F.kl_div(pred_log_probs, target_weights_t, reduction='none').sum(dim=1)

            anchor_kl_weighted = (kl_per_sample * iter_weights).mean()

            loss = anchor_kl_weighted
            self.last_strategy_sizing_bucket_loss = 0.0

            self.last_strategy_sizing_anchor_loss = float(anchor_kl_weighted.item())

            self.strategy_sizing_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strategy_sizing_net.parameters(), max_norm=0.5)
            self.strategy_sizing_optimizer.step()
            total_loss += loss.item()
            completed_steps += 1
        return total_loss / max(completed_steps, 1)

    def _get_q_reward_unit(self):
        """Возвращает единицу нормализации reward для q_net."""
        if self.q_reward_scale_value is not None:
            return max(float(self.q_reward_scale_value), 1e-8)

        if self.q_reward_scale == 'bb':
            return max(float(self.big_blind), 1e-8)

        if self.q_reward_scale == 'none':
            return 1.0

        raise ValueError(f"Неизвестный q_reward_scale: {self.q_reward_scale}")

    def _sync_q_target_net(self):
        """Синхронизирует target-сеть q_net для стабильного TD-bootstrap."""
        if self.q_net is None or self.q_target_net is None:
            return
        self.q_target_net.load_state_dict(self.q_net.state_dict())
        self.q_target_net.eval()

    def _train_q_network_refit(self):
        """Референсный режим: переобучение Q-сети с нуля каждую итерацию.

        По образцу DeepPDCFR QValueTrainer.train_model:
        - re-init сети + Adam с нуля каждый вызов
        - target-sync каждые self.q_refit_sync_every шагов
        - best-model selection по минимальному loss внутри итерации
        - восстановление best-state в конце

        При q_reward_propagation=mc все сэмплы terminal → target=rewards,
        bootstrap=0 → target-sync инертен. Refit сводится к чистому
        per-iter regression fit + best-model.
        """
        if self.q_net is None or self.q_buffer is None:
            return 0.0
        if len(self.q_buffer) < self.q_refit_batch_size:
            return 0.0

        self.q_net = QValueNetwork(self._q_input_size, self._q_hidden, self.num_actions).to(self.device)
        self.q_target_net = QValueNetwork(self._q_input_size, self._q_hidden, self.num_actions).to(self.device)
        self.q_target_net.load_state_dict(self.q_net.state_dict())
        self.q_target_net.eval()
        for p in self.q_target_net.parameters():
            p.requires_grad = False
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=self._q_lr)

        self.q_net.train()
        best_loss = float('inf')
        best_state = None
        total_loss = 0.0

        do_diag = (self.iteration_count % 10 == 0)
        if do_diag:
            grad_norms = []
            fit_ratios = []

        for step in range(self.q_refit_steps):
            samples = self.q_buffer.sample(self.q_refit_batch_size)
            if samples is None:
                break

            states, actions, rewards, next_states, next_policy_states, next_masks, terminals, _next_is_hero_batch = samples
            next_player_ids = np.full(len(actions), -1, dtype=np.int32)

            states_t = torch.from_numpy(states.copy()).to(self.device)
            actions_t = torch.from_numpy(actions.copy()).long().to(self.device)
            reward_unit = self._get_q_reward_unit()
            rewards_t = torch.from_numpy(rewards.copy()).to(self.device) / reward_unit
            next_states_t = torch.from_numpy(next_states.copy()).to(self.device)
            next_policy_states_t = torch.from_numpy(next_policy_states.copy()).to(self.device)
            next_masks_t = torch.from_numpy(next_masks.copy()).to(self.device)
            terminals_t = torch.from_numpy(terminals.copy()).to(self.device)
            next_player_ids_t = torch.from_numpy(next_player_ids.copy()).long().to(self.device)

            with torch.no_grad():
                uniform_strategy = next_masks_t / next_masks_t.sum(dim=1, keepdim=True).clamp(min=1)
                if self.q_bootstrap_per_player_enabled and self.use_multi_agent:
                    next_strategy = uniform_strategy.clone()
                    for p in range(self.num_trainable_players):
                        p_mask = next_player_ids_t == p
                        if not p_mask.any():
                            continue
                        adv_p = self.advantage_nets[p](next_policy_states_t[p_mask])
                        pos_p = torch.clamp(adv_p[:, :self.num_actions], min=0) * next_masks_t[p_mask]
                        sum_p = pos_p.sum(dim=1, keepdim=True)
                        next_strategy[p_mask] = torch.where(sum_p > 0, pos_p / sum_p, uniform_strategy[p_mask])
                    backup_strategy = self._apply_q_bootstrap_policy_mix(next_strategy, uniform_strategy)
                else:
                    next_adv = self.advantage_net(next_policy_states_t)
                    next_pos = torch.clamp(next_adv[:, :self.num_actions], min=0) * next_masks_t
                    next_sum = next_pos.sum(dim=1, keepdim=True)
                    next_strategy = torch.where(next_sum > 0, next_pos / next_sum, uniform_strategy)
                    backup_strategy = self._apply_q_bootstrap_policy_mix(next_strategy, uniform_strategy)
                next_q = self.q_target_net(next_states_t)
                next_v = (backup_strategy * next_q).sum(dim=1)
                targets = self._td_target(rewards_t, terminals_t, next_v)

            current_q = self.q_net(states_t)
            q_selected = current_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(q_selected, targets.detach())

            self.q_optimizer.zero_grad()
            loss.backward()
            raw_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.q_grad_clip_max_norm)
            self.q_optimizer.step()

            step_loss = loss.item()
            total_loss += step_loss

            if step % self.q_refit_sync_every == 0:
                self.q_target_net.load_state_dict(self.q_net.state_dict())

            if step_loss < best_loss:
                best_loss = step_loss
                best_state = {k: v.cpu().clone() for k, v in self.q_net.state_dict().items()}

            if do_diag:
                grad_norms.append(raw_norm.item() if isinstance(raw_norm, torch.Tensor) else float(raw_norm))
                with torch.no_grad():
                    term_mask = terminals_t.bool()
                    if term_mask.any():
                        qpred_term = q_selected[term_mask].abs().mean().item()
                        target_term = targets[term_mask].abs().mean().item()
                        fit_ratios.append(qpred_term / max(target_term, 1e-8))

        if best_state is not None:
            self.q_net.load_state_dict(best_state)
            self.q_net.eval()

        steps_done = step + 1 if 'step' in dir() else 0
        avg_loss = total_loss / max(steps_done, 1)

        if do_diag and grad_norms:
            fit = np.mean(fit_ratios) if fit_ratios else 0.0
            diag_line = f'[Q-REFIT] iter={self.iteration_count} steps={steps_done} raw_grad_mean={np.mean(grad_norms):.2f} raw_grad_max={np.max(grad_norms):.2f} best_loss={best_loss:.4f} avg_loss={avg_loss:.4f} fit_ratio={fit:.4f}'
            print(diag_line)
            if getattr(self, 'qdiag_log_path', None):
                os.makedirs(os.path.dirname(self.qdiag_log_path), exist_ok=True)
                with open(self.qdiag_log_path, 'a', encoding='utf-8') as f:
                    f.write(diag_line + '\n')

        return avg_loss

    def _apply_q_bootstrap_policy_mix(self, next_strategy, uniform_strategy):
        if not self.q_bootstrap_policy_mix_enabled:
            return next_strategy
        mix = self.q_bootstrap_policy_uniform_mix
        return (1.0 - mix) * next_strategy + mix * uniform_strategy

    @staticmethod
    def _td_target(rewards, terminals, next_values):
        """One-step TD target; next_values must already contain the bootstrap."""
        return rewards + (1.0 - terminals) * next_values

    def train_q_network(self, batch_size=256, epochs=2):
        """TD-обучение Q-сети для variance reduction (VR-DeepDCFR+).

        Q-сеть предсказывает ожидаемый reward traversing_player.
        state / next_state — perspective traversing_player.
        next_policy_state — perspective current_player для bootstrap strategy.
        Control variate:
          adjusted = Σ σ(a)·Q(s,a) + (v_sampled - Q(s,â))
        Несмещённость при ЛЮБОМ Q, хорошее Q → низкая variance.
        """
        if self.q_net is None or self.q_buffer is None:
            return None

        if self.q_refit_enabled:
            return self._train_q_network_refit()

        if len(self.q_buffer) < batch_size:
            return 0.0

        self.q_net.train()
        total_loss = 0.0

        do_diag = (self.iteration_count % 10 == 0)
        if do_diag:
            grad_norms = []
            term_fracs = []
            qpred_terms = []
            qpred_boots = []
            target_terms = []

        for _ in range(epochs):
            samples = self.q_buffer.sample(batch_size)
            if samples is None:
                return 0.0

            states, actions, rewards, next_states, next_policy_states, next_masks, terminals, _next_is_hero_batch = samples
            next_player_ids = np.full(len(actions), -1, dtype=np.int32)

            states_t = torch.from_numpy(states.copy()).to(self.device)
            actions_t = torch.from_numpy(actions.copy()).long().to(self.device)
            reward_unit = self._get_q_reward_unit()
            rewards_t = torch.from_numpy(rewards.copy()).to(self.device) / reward_unit
            next_states_t = torch.from_numpy(next_states.copy()).to(self.device)
            next_policy_states_t = torch.from_numpy(next_policy_states.copy()).to(self.device)
            next_masks_t = torch.from_numpy(next_masks.copy()).to(self.device)
            terminals_t = torch.from_numpy(terminals.copy()).to(self.device)
            next_player_ids_t = torch.from_numpy(next_player_ids.copy()).long().to(self.device)

            with torch.no_grad():
                uniform_strategy = next_masks_t / next_masks_t.sum(dim=1, keepdim=True).clamp(min=1)
                if self.q_bootstrap_per_player_enabled and self.use_multi_agent:
                    next_strategy = uniform_strategy.clone()
                    for p in range(self.num_trainable_players):
                        p_mask = next_player_ids_t == p
                        if not p_mask.any():
                            continue
                        adv_p = self.advantage_nets[p](next_policy_states_t[p_mask])
                        pos_p = torch.clamp(adv_p[:, :self.num_actions], min=0) * next_masks_t[p_mask]
                        sum_p = pos_p.sum(dim=1, keepdim=True)
                        next_strategy[p_mask] = torch.where(sum_p > 0, pos_p / sum_p, uniform_strategy[p_mask])
                    backup_strategy = self._apply_q_bootstrap_policy_mix(next_strategy, uniform_strategy)
                else:
                    next_adv = self.advantage_net(next_policy_states_t)
                    next_pos = torch.clamp(next_adv[:, :self.num_actions], min=0) * next_masks_t
                    next_sum = next_pos.sum(dim=1, keepdim=True)
                    next_strategy = torch.where(next_sum > 0, next_pos / next_sum, uniform_strategy)
                    backup_strategy = self._apply_q_bootstrap_policy_mix(next_strategy, uniform_strategy)
                next_q = self.q_target_net(next_states_t)
                next_v = (backup_strategy * next_q).sum(dim=1)
                targets = self._td_target(rewards_t, terminals_t, next_v)

            current_q = self.q_net(states_t)
            q_selected = current_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            if self.q_terminal_balanced_loss_enabled:
                term_mask_bal = terminals_t.bool()
                if term_mask_bal.any() and (~term_mask_bal).any():
                    loss = self.q_terminal_loss_alpha * F.mse_loss(q_selected[term_mask_bal], targets[term_mask_bal].detach()) \
                         + (1.0 - self.q_terminal_loss_alpha) * F.mse_loss(q_selected[~term_mask_bal], targets[~term_mask_bal].detach())
                else:
                    loss = F.mse_loss(q_selected, targets.detach())
            else:
                loss = F.mse_loss(q_selected, targets.detach())

            self.q_optimizer.zero_grad()
            loss.backward()
            raw_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.q_grad_clip_max_norm)
            self.q_optimizer.step()

            total_loss += loss.item()

            if do_diag:
                grad_norms.append(raw_norm.item() if isinstance(raw_norm, torch.Tensor) else float(raw_norm))
                term_fracs.append(terminals_t.float().mean().item())
                with torch.no_grad():
                    term_mask = terminals_t.bool()
                    boot_mask = ~term_mask
                    qpred_terms.append(q_selected[term_mask].abs().mean().item() if term_mask.any() else 0.0)
                    qpred_boots.append(q_selected[boot_mask].abs().mean().item() if boot_mask.any() else 0.0)
                    target_terms.append(targets[term_mask].abs().mean().item() if term_mask.any() else 0.0)

        if do_diag and grad_norms:
            targ_mean = np.mean(target_terms)
            fit = np.mean(qpred_terms) / max(targ_mean, 1e-8)
            diag_line = f'[Q-DIAG] iter={self.iteration_count} raw_grad_mean={np.mean(grad_norms):.2f} raw_grad_max={np.max(grad_norms):.2f} term_frac={np.mean(term_fracs):.3f} qpred_term={np.mean(qpred_terms):.3f} qpred_boot={np.mean(qpred_boots):.3f} target_term={targ_mean:.3f} fit_ratio={fit:.4f} q_loss={total_loss / epochs:.4f}'
            print(diag_line)
            if getattr(self, 'qdiag_log_path', None):
                os.makedirs(os.path.dirname(self.qdiag_log_path), exist_ok=True)
                with open(self.qdiag_log_path, 'a', encoding='utf-8') as f:
                    f.write(diag_line + '\n')

        if self.iteration_count % self.q_target_update_interval == 0:
            self._sync_q_target_net()

        return total_loss / epochs

    def _current_sizing_q_src2_weight(self):
        decay = max(int(self.sizing_q_src2_weight_decay_iterations), 1)
        progress = min(max(float(self.iteration_count), 0.0) / float(decay), 1.0)
        return self.sizing_q_src2_weight_start + progress * (
            self.sizing_q_src2_weight_end - self.sizing_q_src2_weight_start
        )

    def train_sizing_q_network(self, batch_size=256, epochs=None):
        """Обучение SizingQNetwork: Q(state, size) ≈ value(size|state).

        Bug #48: MSE-regression на (state, normalized_size, target_value).
        target = raw_v3 / pot (pot-normalized) или raw_v3.
        Q-сеть не меняет policy — она поставляет candidate advantage для AWR.
        Bug #70: опционально z-score targets + per-sample src2 loss weight.
        Bug #74: kind-filter для non-NORMAL samples, replay clear once.
        """
        if not self.sizing_q_enabled or self.sizing_q_net is None or self.sizing_q_buffer is None:
            return 0.0
        if epochs is None:
            epochs = self.sizing_q_train_steps

        self._maybe_clear_sizing_q_replay_once()

        if len(self.sizing_q_buffer) < batch_size:
            return 0.0

        self.sizing_q_net.train()
        total_loss = 0.0
        updates = 0
        grad_norms_preclip = []
        use_huber = self.sizing_q_loss == 'huber'

        return_kinds = self.sizing_q_kind_filter_enabled

        for _ in range(epochs):
            if self.sizing_q_stratified_sampling:
                samples = self.sizing_q_buffer.sample_stratified(
                    batch_size, num_anchors=self.num_anchors, return_sources=True,
                    return_selected_kinds=return_kinds)
            else:
                samples = self.sizing_q_buffer.sample(batch_size, return_sources=True,
                                                       return_selected_kinds=return_kinds)
            if samples is None:
                return 0.0

            if return_kinds:
                states, norm_sizes, targets, sources, selected_kind_ids = samples
            else:
                states, norm_sizes, targets, sources = samples
                selected_kind_ids = None

            states_t = torch.from_numpy(states.copy()).to(self.device)
            sizes_t = torch.from_numpy(norm_sizes.copy()).to(self.device)
            targets_t = torch.from_numpy(targets.copy()).to(self.device)
            sources_t = torch.from_numpy(sources.copy()).to(self.device)

            targets_used = targets_t
            if self.sizing_q_train_zscore:
                mu = targets_t.mean()
                sigma = targets_t.std(unbiased=False).clamp_min(self.sizing_q_zscore_eps)
                targets_used = (targets_t - mu) / sigma

            q_pred = self.sizing_q_net(states_t, sizes_t)

            if self.sizing_q_src2_loss_weight_enabled or self.sizing_q_kind_filter_enabled:
                if use_huber:
                    per_sample_loss = F.smooth_l1_loss(q_pred, targets_used, reduction='none', beta=self.sizing_q_huber_delta)
                else:
                    per_sample_loss = F.mse_loss(q_pred, targets_used, reduction='none')
                weights = torch.ones_like(per_sample_loss)

                if self.sizing_q_src2_loss_weight_enabled:
                    weights[sources_t == 2] = self._current_sizing_q_src2_weight()

                if self.sizing_q_kind_filter_enabled:
                    kind_ids_t = torch.from_numpy(selected_kind_ids.copy()).to(self.device)
                    normal_id = SizingQBuffer.SELECTED_KIND_TO_ID['NORMAL']
                    non_normal = kind_ids_t != normal_id
                    n_non_normal = int(non_normal.sum().item())

                    if self.sizing_q_kind_filter_mode == 'drop':
                        self.sizing_q_kind_filter_diag['normal_kept'] += int((kind_ids_t == normal_id).sum().item())
                        if n_non_normal > 0:
                            self.sizing_q_kind_filter_diag['all_in_dropped'] += int((kind_ids_t == SizingQBuffer.SELECTED_KIND_TO_ID['ALL_IN']).sum().item())
                            self.sizing_q_kind_filter_diag['min_raise_dropped'] += int((kind_ids_t == SizingQBuffer.SELECTED_KIND_TO_ID['MIN_RAISE']).sum().item())
                            self.sizing_q_kind_filter_diag['unknown_dropped'] += int((kind_ids_t == SizingQBuffer.SELECTED_KIND_TO_ID['UNKNOWN']).sum().item())
                        weights[non_normal] = 0.0
                    elif self.sizing_q_kind_filter_mode == 'downweight':
                        weights[non_normal] *= float(self.sizing_q_kind_filter_downweight)
                        self.sizing_q_kind_filter_diag['downweighted'] += n_non_normal
                    else:
                        raise ValueError(f"Unknown sizing_q_kind_filter_mode: {self.sizing_q_kind_filter_mode}")

                weight_sum = weights.sum()
                if float(weight_sum.item()) <= 1e-8:
                    self.sizing_q_kind_filter_diag['zero_weight_batches'] += 1
                    continue

                loss = (weights * per_sample_loss).sum() / weight_sum
            else:
                if use_huber:
                    loss = F.smooth_l1_loss(q_pred, targets_used, beta=self.sizing_q_huber_delta)
                else:
                    loss = F.mse_loss(q_pred, targets_used)

            self.sizing_q_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sizing_q_net.parameters(), max_norm=1.0)
            self.sizing_q_optimizer.step()

            total_loss += loss.item()
            updates += 1

        return total_loss / updates if updates > 0 else 0.0

    def get_policy_distribution(self, state, player_id=None):
        """Return masked action and legal sizing probabilities without sampling.

        This is the frozen-blueprint inference API used by paired evaluation
        and later by the depth-limited solver. It never touches global RNG.
        """
        if player_id is None:
            player_id = int(state.current_player)
        legal_mask = self.get_legal_action_mask(state)
        if float(legal_mask.sum()) <= 0.0:
            raise ValueError("state has no legal action")

        state_tensor = torch.from_numpy(
            self._encode_state(state, player_id).astype(np.float32, copy=False)
        ).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(
            legal_mask.astype(np.float32, copy=False)
        ).unsqueeze(0).to(self.device)
        sizing_state_tensor = torch.from_numpy(
            self._encode_state_for_sizing(state, player_id).astype(np.float32, copy=False)
        ).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.strategy_net(state_tensor)
            masked_logits = torch.where(
                mask_tensor == 1,
                logits[:, :self.num_actions],
                torch.tensor(-1e20, device=self.device),
            )
            action_probs = F.softmax(masked_logits, dim=1)[0].cpu().numpy()
            slot_logits, _scalar_bet = self.strategy_sizing_net(sizing_state_tensor)
            sizing_probs = F.softmax(slot_logits, dim=1)[0].cpu().numpy()

        action_probs = np.asarray(action_probs, dtype=np.float64) * legal_mask
        action_total = float(action_probs.sum())
        if action_total <= 0.0:
            action_probs = legal_mask.astype(np.float64) / float(legal_mask.sum())
        else:
            action_probs /= action_total

        sizing_probs = self._apply_sizing_anchor_mask(sizing_probs, state, callsite='get_policy_distribution')
        if self.sizing_anchor_availability_enabled:
            available, _ = self._anchor_availability(state)
            sizing_probs = np.asarray(sizing_probs, dtype=np.float64)
            sizing_probs[~available] = 0.0
        sizing_total = float(np.sum(sizing_probs))
        if sizing_total <= 0.0:
            sizing_probs = np.full(self.num_anchors, 1.0 / self.num_anchors, dtype=np.float64)
        else:
            sizing_probs = np.asarray(sizing_probs, dtype=np.float64) / sizing_total

        return action_probs, sizing_probs

    def choose_action(self, state, player_id=None, deterministic=False):
        if player_id is None:
            player_id = int(state.current_player)
        legal_mask = self.get_legal_action_mask(state)
        legal_action_types = [a for a in range(self.num_actions) if legal_mask[a] > 0]

        if not legal_action_types:
            if pkrs.ActionEnum.Call in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Call)
            elif pkrs.ActionEnum.Check in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Check)
            else:
                return pkrs.Action(pkrs.ActionEnum.Fold)

        state_tensor = torch.from_numpy(self._encode_state(state, player_id).astype(np.float32, copy=False)).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(legal_mask.astype(np.float32, copy=False)).unsqueeze(0).to(self.device)

        sizing_encoded = self._encode_state_for_sizing(state, player_id)
        sizing_state_tensor = torch.from_numpy(sizing_encoded.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.strategy_net(state_tensor)
            masked_logits = torch.where(mask_tensor == 1, logits[:, :self.num_actions], torch.tensor(-1e20, device=self.device))
            probs = F.softmax(masked_logits, dim=1)[0].cpu().numpy()
            # #101: отсечение действий с вероятностью < порога
            if self.inference_min_action_prob > 0:
                keep = probs >= self.inference_min_action_prob
                if keep.sum() > 0:
                    probs = probs * keep
                    probs = probs / probs.sum()
            slot_logits, scalar_bet = self.strategy_sizing_net(sizing_state_tensor)
            sizing_probs = F.softmax(slot_logits, dim=1)[0].cpu().numpy()
            # #101: отсечение анкеров с вероятностью < порога
            if self.sizing_inference_min_prob > 0:
                keep = sizing_probs >= self.sizing_inference_min_prob
                if keep.sum() > 0:
                    sizing_probs = sizing_probs * keep
                    sizing_probs = sizing_probs / sizing_probs.sum()

            mode = self.sizing_inference_mode
            if mode == 'argmax':
                sizing_probs_mod = self._apply_sizing_anchor_mask(sizing_probs, state, callsite='choose_argmax')
                if self.sizing_anchor_availability_enabled:
                    avail, _ = self._anchor_availability(state)
                    sizing_probs_mod[~avail] = 0.0
                    total_avail = float(sizing_probs_mod.sum())
                    if total_avail > 1e-8:
                        sizing_probs_mod = sizing_probs_mod / total_avail
                anchor_idx = int(np.argmax(sizing_probs_mod))
                bet_size_multiplier = self.anchors_arr[anchor_idx]
            elif mode in {'hierarchical_rm', 'top_bucket'}:
                # Derive group mass from anchor probabilities; never read a bucket head.
                # top_bucket remains a backward-compatible alias for old configs.
                sparse_probs = self._sparsify_sizing_target_by_buckets(sizing_probs)
                sparse_probs = self._apply_sizing_anchor_mask(sparse_probs, state, callsite='choose_hierarchical_rm')
                if self.sizing_anchor_availability_enabled:
                    avail, _ = self._anchor_availability(state)
                    sparse_probs[~avail] = 0.0
                    total_avail = float(sparse_probs.sum())
                    if total_avail > 1e-8:
                        sparse_probs = sparse_probs / total_avail
                if deterministic:
                    anchor_idx = int(np.argmax(sparse_probs))
                    bet_size_multiplier = self.anchors_arr[anchor_idx]
                else:
                    sparse_probs = np.clip(sparse_probs.astype(np.float64), 0.0, None)
                    total = max(sparse_probs.sum(), 1e-12)
                    sparse_probs /= total
                    sparse_probs[-1] = max(1.0 - sparse_probs[:-1].sum(), 0.0)
                    bet_size_multiplier = float(np.random.choice(self.anchors, p=sparse_probs))
            elif mode == 'hierarchical':
                bucket_probs = F.softmax(bucket_logits, dim=1)[0].cpu().numpy()
                anchor_probs = sizing_probs
                full_probs = np.zeros(self.num_anchors, dtype=np.float64)
                for b, indices in enumerate(self.sizing_bucket_groups):
                    if bucket_probs[b] <= 1e-8:
                        continue
                    local = anchor_probs[indices]
                    local_sum = float(local.sum())
                    if local_sum > 1e-8:
                        full_probs[indices] = bucket_probs[b] * (local / local_sum)
                total = float(full_probs.sum())
                if total > 1e-8:
                    full_probs /= total
                else:
                    full_probs[:] = 1.0 / self.num_anchors
                sizing_probs_mod = self._apply_sizing_anchor_mask(full_probs, state, callsite='choose_hierarchical')
                if self.sizing_anchor_availability_enabled:
                    avail, _ = self._anchor_availability(state)
                    sizing_probs_mod[~avail] = 0.0
                    total_avail = float(sizing_probs_mod.sum())
                    if total_avail > 1e-8:
                        sizing_probs_mod = sizing_probs_mod / total_avail
                if deterministic:
                    bet_size_multiplier = float((sizing_probs_mod * self.anchors_arr).sum())
                else:
                    sizing_probs_mod = np.clip(sizing_probs_mod.astype(np.float64), 0.0, None)
                    total = max(sizing_probs_mod.sum(), 1e-12)
                    sizing_probs_mod /= total
                    sizing_probs_mod[-1] = max(1.0 - sizing_probs_mod[:-1].sum(), 0.0)
                    bet_size_multiplier = float(np.random.choice(self.anchors, p=sizing_probs_mod))
            elif deterministic:
                sizing_probs_mod = self._apply_sizing_anchor_mask(sizing_probs, state, callsite='choose_deterministic_mean')
                if self.sizing_anchor_availability_enabled:
                    avail, _ = self._anchor_availability(state)
                    sizing_probs_mod[~avail] = 0.0
                    total_avail = float(sizing_probs_mod.sum())
                    if total_avail > 1e-8:
                        sizing_probs_mod = sizing_probs_mod / total_avail
                bet_size_multiplier = float((sizing_probs_mod * self.anchors_arr).sum())
            else:
                sizing_probs_mod = self._apply_sizing_anchor_mask(sizing_probs, state, callsite='choose_stochastic')
                if self.sizing_anchor_availability_enabled:
                    avail, _ = self._anchor_availability(state)
                    sizing_probs_mod[~avail] = 0.0
                    total_avail = float(sizing_probs_mod.sum())
                    if total_avail > 1e-8:
                        sizing_probs_mod = sizing_probs_mod / total_avail
                sizing_probs_mod = np.clip(sizing_probs_mod.astype(np.float64), 0.0, None)
                total = max(sizing_probs_mod.sum(), 1e-12)
                sizing_probs_mod /= total
                sizing_probs_mod[-1] = max(1.0 - sizing_probs_mod[:-1].sum(), 0.0)
                bet_size_multiplier = float(np.random.choice(self.anchors, p=sizing_probs_mod))

            bet_size_multiplier = max(self.min_bet_size, min(self.max_bet_size, bet_size_multiplier))

        if deterministic:
            best_idx = int(np.argmax([probs[a] for a in legal_action_types]))
            action_type = legal_action_types[best_idx]
        else:
            legal_probs = np.array([probs[a] for a in legal_action_types])
            if np.sum(legal_probs) > 0:
                legal_probs = legal_probs / np.sum(legal_probs)
            else:
                legal_probs = np.ones(len(legal_action_types)) / len(legal_action_types)
            action_idx = np.random.choice(len(legal_action_types), p=legal_probs)
            action_type = legal_action_types[action_idx]

        if action_type == 3:
            return self.action_type_to_pokers_action(action_type, state, bet_size_multiplier)
        else:
            return self.action_type_to_pokers_action(action_type, state)

    def _build_checkpoint(self, seed=None, extra=None):
        """Единый словарь чекпоинта — все точки сохранения используют его."""
        config_dict = {
            'advantage_lr': self.optimizer.defaults['lr'],
            'strategy_lr': self.strategy_optimizer.defaults['lr'],
            'pg_lr': self.pg_lr,
            'sizing_anchor_lr': self.sizing_optimizers[0].defaults['lr'],
            'strategy_sizing_lr': self.strategy_sizing_optimizer.defaults['lr'],
            'strategy_sizing_train_steps': self.strategy_sizing_train_steps,
            'hidden_size': self.advantage_net.base[0].out_features,
            'sizing_hidden_size': self.advantage_sizing_nets[0].base[0].out_features,
            'num_actions': self.num_actions,
            'num_anchors': self.num_anchors,
            'anchor_sizes': self.anchors,
            'sizing_min_prob_start': self.sizing_min_prob_start,
            'sizing_min_prob_end': self.sizing_min_prob_end,
            'sizing_min_prob_decay_iterations': self.sizing_min_prob_decay_iterations,
            'sizing_bucket_min_prob_start': self.sizing_bucket_min_prob_start,
            'sizing_bucket_min_prob_end': self.sizing_bucket_min_prob_end,
            'sizing_bucket_min_prob_decay_iterations': self.sizing_bucket_min_prob_decay_iterations,
            'sizing_bucket_groups': [[self.anchors[idx] for idx in group] for group in self.sizing_bucket_groups],
            'sizing_anchor_availability_enabled': self.sizing_anchor_availability_enabled,
            'sizing_allin_boundary_enabled': self.sizing_allin_boundary_enabled,
            'sizing_preflop_disable_buckets': self.sizing_preflop_disable_buckets,
            'sizing_bucket_head_enabled': self.sizing_bucket_head_enabled,
            'sizing_availability_on_input': self.sizing_availability_on_input,
            'sizing_cfr_mode': self.sizing_cfr_mode,

            'memory_size': self.advantage_buffer.capacity,
            'pg_memory_size': self.pg_memory_size,
            'entropy_bonus': self.entropy_bonus,
            'min_bet_size': self.min_bet_size,
            'max_bet_size': self.max_bet_size,
            'discount_alpha': self.discount_alpha,
            'discount_gamma': self.discount_gamma,
            'device': self.device,
            'num_players': self.num_players,
            'q_lr': self.q_optimizer.defaults['lr'] if self.q_optimizer is not None else float(cfg_get('q_lr', 1e-3)),
            'q_bootstrap_policy_mix_enabled': self.q_bootstrap_policy_mix_enabled,
            'q_bootstrap_policy_uniform_mix': self.q_bootstrap_policy_uniform_mix,
            'q_grad_clip_max_norm': self.q_grad_clip_max_norm,
            'q_terminal_balanced_loss_enabled': self.q_terminal_balanced_loss_enabled,
            'q_terminal_loss_alpha': self.q_terminal_loss_alpha,
            'q_bootstrap_per_player_enabled': self.q_bootstrap_per_player_enabled,
            'advantage_regret_norm': self.advantage_regret_norm,
            'advantage_regret_clip': self.advantage_regret_clip,
            'advantage_loss': self.advantage_loss,
            'advantage_huber_delta': self.advantage_huber_delta,
            'advantage_accumulation': self.advantage_accumulation,
            'advantage_reward_scale': self.advantage_reward_scale,
            'advantage_buffer_reservoir': self.advantage_buffer_reservoir,
            'strategy_buffer_reservoir': self.strategy_buffer_reservoir,
            'sizing_advantage_buffer_reservoir': self.sizing_advantage_buffer_reservoir,
            'sizing_strategy_buffer_reservoir': self.sizing_strategy_buffer_reservoir,
            'save_replay_buffers_in_checkpoint': self.save_replay_buffers_in_checkpoint,
        }
        if self.sizing_q_enabled:
            config_dict['sizing_q_enabled'] = True
            config_dict['sizing_q_lr'] = self.sizing_q_optimizer.defaults['lr']
            config_dict['sizing_q_target_normalization'] = self.sizing_q_target_normalization
            config_dict['sizing_candidate_sizes'] = self.sizing_candidate_sizes
            config_dict['sizing_anchor_sizes'] = self.sizing_anchor_sizes
            config_dict['sizing_awr_temperature'] = self.sizing_awr_temperature
            config_dict['sizing_awr_uniform_mix'] = self.sizing_awr_uniform_mix
            config_dict['sizing_reinforce_weight'] = self.sizing_reinforce_weight
            config_dict['sizing_probe_prob_start'] = self.sizing_probe_prob_start
            config_dict['sizing_probe_prob_end'] = self.sizing_probe_prob_end
            config_dict['sizing_probe_decay_iterations'] = self.sizing_probe_decay_iterations
            config_dict['sizing_q_min_samples_per_anchor'] = self.sizing_q_min_samples_per_anchor
            config_dict['sizing_q_bootstrap_scale_fix'] = self.sizing_q_bootstrap_scale_fix
            config_dict['sizing_q_stratified_sampling'] = self.sizing_q_stratified_sampling
            config_dict['sizing_q_target_diagnostics_enabled'] = self.sizing_q_target_diagnostics_enabled
            config_dict['sizing_q_bootstrap_target_net'] = self.sizing_q_bootstrap_target_net
            config_dict['sizing_q_loss'] = self.sizing_q_loss
            config_dict['sizing_q_huber_delta'] = self.sizing_q_huber_delta
            config_dict['sizing_anchor_weight_decay'] = cfg_get('sizing_anchor_weight_decay', 1e-5)
            config_dict['sizing_strategy_weight_decay'] = cfg_get('sizing_strategy_weight_decay', 1e-5)
            config_dict['sizing_q_weight_decay'] = cfg_get('sizing_q_weight_decay', 1e-5)
            config_dict['sizing_q_train_zscore'] = self.sizing_q_train_zscore
            config_dict['sizing_q_zscore_eps'] = self.sizing_q_zscore_eps
            config_dict['sizing_q_src2_loss_weight_enabled'] = self.sizing_q_src2_loss_weight_enabled
            config_dict['sizing_q_src2_weight_start'] = self.sizing_q_src2_weight_start
            config_dict['sizing_q_src2_weight_end'] = self.sizing_q_src2_weight_end
            config_dict['sizing_q_src2_weight_decay_iterations'] = self.sizing_q_src2_weight_decay_iterations
            config_dict['save_q_buffer_in_checkpoint'] = self.save_q_buffer_in_checkpoint
            config_dict['sizing_q_legal_anchor_mask_enabled'] = self.sizing_q_legal_anchor_mask_enabled
            config_dict['sizing_q_mask_exclude_kinds'] = self.sizing_q_mask_exclude_kinds
            config_dict['sizing_q_kind_filter_enabled'] = self.sizing_q_kind_filter_enabled
            config_dict['sizing_q_kind_filter_mode'] = self.sizing_q_kind_filter_mode
            config_dict['sizing_q_kind_filter_downweight'] = self.sizing_q_kind_filter_downweight
            config_dict['sizing_q_replay_clear_once'] = self.sizing_q_replay_clear_once
            config_dict['sizing_q_mask_diagnostics_enabled'] = self.sizing_q_mask_diagnostics_enabled
            config_dict['sizing_q_mask_dry_run'] = self.sizing_q_mask_dry_run
            config_dict['sizing_q_replay_clear_diagnostics_enabled'] = self.sizing_q_replay_clear_diagnostics_enabled
            config_dict['sizing_q_selected_credit_enabled'] = self.sizing_q_selected_credit_enabled
        git_hash = ''
        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if os.path.isdir(os.path.join(repo_root, '.git')):
                result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                        capture_output=True, text=True, cwd=repo_root, timeout=5)
                git_hash = result.stdout.strip()
        except Exception:
            pass
        checkpoint = {
            'iteration': self.iteration_count,
            'strategy_net': self.strategy_net.state_dict(),
            'advantage_sizing_nets': [net.state_dict() for net in self.advantage_sizing_nets],
            'advantage_sizing_net': self.advantage_sizing_nets[0].state_dict(),  # обратная совместимость
            'sizing_target_net': self.sizing_target_net.state_dict(),
            'strategy_sizing_net': self.strategy_sizing_net.state_dict(),
            'strategy_optimizer': self.strategy_optimizer.state_dict(),
            'sizing_optimizers': [opt.state_dict() for opt in self.sizing_optimizers],
            'sizing_optimizer': self.sizing_optimizers[0].state_dict(),  # обратная совместимость
            'strategy_sizing_optimizer': self.strategy_sizing_optimizer.state_dict(),
            'q_net': self.q_net.state_dict() if self.q_net is not None else None,
            'q_target_net': self.q_target_net.state_dict() if self.q_target_net is not None else None,
            'q_optimizer': self.q_optimizer.state_dict() if self.q_optimizer is not None else None,
            'min_bet_size': self.min_bet_size,
            'max_bet_size': self.max_bet_size,
            'pg_lr': self.pg_lr,
            'entropy_bonus': self.entropy_bonus,
            'pg_memory_size': self.pg_memory_size,
            'pg_baseline_mean': self.pg_baseline_mean,
            'pg_baseline_var': self.pg_baseline_var,
            'max_regret_seen': self.max_regret_seen,
            'anchors': list(self.anchors),
            'seed': seed,
            'config': config_dict,
            'git_hash': git_hash,
            'checkpoint_format_version': 4,
            'use_multi_agent_advantage': self.use_multi_agent,
        }
        if self.use_multi_agent:
            checkpoint['advantage_nets'] = [net.state_dict() for net in self.advantage_nets]
            checkpoint['target_nets'] = [net.state_dict() for net in self.target_nets]
            checkpoint['advantage_optimizers'] = [opt.state_dict() for opt in self.advantage_optimizers]
            checkpoint['advantage_net'] = self.advantage_nets[0].state_dict()
            checkpoint['target_net'] = self.target_nets[0].state_dict()
            checkpoint['advantage_optimizer'] = self.advantage_optimizers[0].state_dict()
        else:
            checkpoint['advantage_net'] = self.advantage_net.state_dict()
            checkpoint['target_net'] = self.target_net.state_dict()
            checkpoint['advantage_optimizer'] = self.optimizer.state_dict()
        sizing_q_runtime_config = {
            'sizing_q_legal_anchor_mask_enabled': self.sizing_q_legal_anchor_mask_enabled,
            'sizing_q_mask_exclude_kinds': self.sizing_q_mask_exclude_kinds,
            'sizing_q_kind_filter_enabled': self.sizing_q_kind_filter_enabled,
            'sizing_q_kind_filter_mode': self.sizing_q_kind_filter_mode,
            'sizing_q_kind_filter_downweight': self.sizing_q_kind_filter_downweight,
            'sizing_q_replay_clear_once': self.sizing_q_replay_clear_once,
            'sizing_q_mask_diagnostics_enabled': self.sizing_q_mask_diagnostics_enabled,
            'sizing_q_mask_dry_run': self.sizing_q_mask_dry_run,
            'sizing_q_replay_clear_diagnostics_enabled': self.sizing_q_replay_clear_diagnostics_enabled,
            'sizing_q_selected_credit_enabled': self.sizing_q_selected_credit_enabled,
            'sizing_anchor_availability_enabled': self.sizing_anchor_availability_enabled,
            'sizing_allin_boundary_enabled': self.sizing_allin_boundary_enabled,
            'sizing_preflop_disable_buckets': self.sizing_preflop_disable_buckets,
            'sizing_bucket_head_enabled': self.sizing_bucket_head_enabled,
            'sizing_availability_on_input': self.sizing_availability_on_input,
        }
        checkpoint.update(sizing_q_runtime_config)
        config_dict.update(sizing_q_runtime_config)
        if self.sizing_q_enabled and self.sizing_q_net is not None:
            checkpoint['sizing_q_net'] = self.sizing_q_net.state_dict()
            checkpoint['sizing_q_optimizer'] = self.sizing_q_optimizer.state_dict()
            checkpoint['sizing_q_enabled'] = True
            checkpoint['sizing_q_buffer_states'] = self.sizing_q_buffer._states[:self.sizing_q_buffer._size].copy()
            checkpoint['sizing_q_buffer_norm_sizes'] = self.sizing_q_buffer._norm_sizes[:self.sizing_q_buffer._size].copy()
            checkpoint['sizing_q_buffer_targets'] = self.sizing_q_buffer._targets[:self.sizing_q_buffer._size].copy()
            checkpoint['sizing_q_buffer_iterations'] = self.sizing_q_buffer._iterations[:self.sizing_q_buffer._size].copy()
            checkpoint['sizing_q_buffer_sources'] = self.sizing_q_buffer._sources[:self.sizing_q_buffer._size].copy()
            checkpoint['sizing_q_buffer_anchor_indices'] = self.sizing_q_buffer._anchor_indices[:self.sizing_q_buffer._size].copy()
            checkpoint['sizing_q_buffer_selected_anchor_indices'] = self.sizing_q_buffer._selected_anchor_indices[:self.sizing_q_buffer._size].copy()
            checkpoint['sizing_q_buffer_effective_anchor_indices'] = self.sizing_q_buffer._effective_anchor_indices[:self.sizing_q_buffer._size].copy()
            checkpoint['sizing_q_buffer_selected_kind_ids'] = self.sizing_q_buffer._selected_kind_ids[:self.sizing_q_buffer._size].copy()
        if self.save_replay_buffers_in_checkpoint:
            self._save_replay_buffers_into(checkpoint)
        if self.save_q_buffer_in_checkpoint and self.q_buffer is not None:
            q_size = len(self.q_buffer)
            checkpoint['q_buffer_states'] = self.q_buffer._states[:q_size].copy()
            checkpoint['q_buffer_actions'] = self.q_buffer._actions[:q_size].copy()
            checkpoint['q_buffer_rewards'] = self.q_buffer._rewards[:q_size].copy()
            checkpoint['q_buffer_next_states'] = self.q_buffer._next_states[:q_size].copy()
            checkpoint['q_buffer_next_policy_states'] = self.q_buffer._next_policy_states[:q_size].copy()
            checkpoint['q_buffer_next_masks'] = self.q_buffer._next_masks[:q_size].copy()
            checkpoint['q_buffer_terminals'] = self.q_buffer._terminals[:q_size].copy()
            checkpoint['q_buffer_next_is_hero'] = self.q_buffer._next_is_hero[:q_size].copy()
            checkpoint['q_buffer_next_player_ids'] = self.q_buffer._next_player_ids[:q_size].copy()
        checkpoint['sizing_path_diag'] = {
            key: value.copy()
            for key, value in self.sizing_path_diag.items()
        }
        checkpoint['sizing_selected_effective_diag'] = {
            key: value.copy()
            for key, value in self.sizing_selected_effective_diag.items()
        }
        checkpoint['sizing_selected_effective_kind_diag'] = {
            path: {
                group_name: dict(group_values)
                for group_name, group_values in groups.items()
            }
            for path, groups in self.sizing_selected_effective_kind_diag.items()
        }
        checkpoint['sizing_q_kind_filter_diag'] = dict(self.sizing_q_kind_filter_diag)
        checkpoint['sizing_q_mask_diag'] = {
            callsite: dict(metrics)
            for callsite, metrics in self.sizing_q_mask_diag.items()
        }
        checkpoint['sizing_availability_diag'] = dict(self.sizing_availability_diag)
        checkpoint['raise_funnel_diag'] = {}
        for path, value in self.raise_funnel_diag.items():
            if isinstance(value, dict):
                checkpoint['raise_funnel_diag'][path] = dict(value)
            else:
                checkpoint['raise_funnel_diag'][path] = value
        checkpoint['sizing_q_insert_kind_diag'] = {
            'by_source': dict(self.sizing_q_insert_kind_diag.get('by_source', {})),
            'by_callsite': dict(self.sizing_q_insert_kind_diag.get('by_callsite', {})),
        }
        checkpoint['sizing_q_replay_clear_diag'] = {
            'before_clear': self.sizing_q_replay_clear_diag.get('before_clear'),
            'cleared': self.sizing_q_replay_clear_diag.get('cleared', False),
            'refill_snapshots': list(self.sizing_q_replay_clear_diag.get('refill_snapshots', [])),
        }
        if self.sizing_q_buffer is not None:
            checkpoint['sizing_q_selected_effective_target_summary'] = (
                self.sizing_q_buffer.selected_effective_target_summary(self.num_anchors)
            )
        if self.sizing_q_target_diagnostics_enabled:
            checkpoint['sizing_lookahead_diag'] = self.sizing_lookahead_diag
        if self.sizing_q_target_diagnostics_enabled and self.sizing_q_target_diag:
            checkpoint['sizing_q_target_diag'] = {
                f"{source}:{anchor_idx}": stat
                for (source, anchor_idx), stat in self.sizing_q_target_diag.items()
            }
        if self.sizing_target_ema_enabled and self.sizing_target_ema:
            checkpoint['sizing_target_ema'] = {
                k if isinstance(k, (int, str, tuple)) else str(k): v
                for k, v in self.sizing_target_ema.items()
            }
        if extra:
            extra = dict(extra)
            # Не даём extra['config'] перетереть полный config_dict: мержим,
            # extra-ключи приоритетнее. Иначе в чекпоинте оставалось ~19 ключей
            # и whitelist-восстановление на ресюме молча падало на config.yaml.
            extra_config = extra.pop('config', None)
            if isinstance(extra_config, dict):
                merged_config = dict(checkpoint.get('config') or {})
                merged_config.update(extra_config)
                checkpoint['config'] = merged_config
            elif extra_config is not None:
                checkpoint['config'] = extra_config
            checkpoint.update(extra)
        return checkpoint

    # Описание 4 CFR replay-буферов для save/load: (префикс в чекпоинте, атрибут агента,
    # список массивов буфера). _cur_id сохраняется отдельно — без него reservoir-сэмплинг поедет.
    _REPLAY_BUFFER_SPECS = (
        ('advantage_buffer', 'advantage_buffer', ('_states', '_regrets', '_masks', '_iterations')),
        ('strategy_buffer', 'strategy_buffer', ('_states', '_policies', '_masks', '_iterations', '_bet_sizes')),
        ('sizing_strategy_buffer', 'sizing_strategy_buffer', ('_states', '_probs', '_iterations')),
        ('sizing_advantage_buffer', 'sizing_advantage_buffer', ('_states', '_regrets', '_masks', '_iterations')),
    )

    def _save_replay_buffers_into(self, checkpoint):
        """Записать занятую часть CFR replay-буферов в чекпоинт."""
        for prefix, attr, fields in self._REPLAY_BUFFER_SPECS:
            buf = getattr(self, attr, None)
            if buf is None:
                continue
            n = len(buf)
            for field in fields:
                arr = getattr(buf, field, None)
                if arr is None:
                    continue
                checkpoint[f'{prefix}{field}'] = arr[:n].copy()
            checkpoint[f'{prefix}_cur_id'] = int(buf._cur_id)
            checkpoint[f'{prefix}_state_dim'] = int(buf._states.shape[1])

    def _load_replay_buffers_from(self, checkpoint):
        """Восстановить CFR replay-буферы из чекпоинта по наличию ключей."""
        for prefix, attr, fields in self._REPLAY_BUFFER_SPECS:
            states_key = f'{prefix}_states'
            if states_key not in checkpoint:
                continue
            buf = getattr(self, attr, None)
            if buf is None:
                continue
            saved_states = checkpoint[states_key]
            ckpt_dim = int(checkpoint.get(f'{prefix}_state_dim', saved_states.shape[1]))
            if ckpt_dim != int(buf._states.shape[1]):
                print(f"  WARNING: {prefix} state_dim mismatch "
                      f"(ckpt={ckpt_dim} vs agent={buf._states.shape[1]}), холодный старт")
                continue
            n = min(int(saved_states.shape[0]), buf.capacity)
            ok = True
            for field in fields:
                arr = getattr(buf, field, None)
                saved = checkpoint.get(f'{prefix}{field}')
                if arr is None or saved is None:
                    continue
                if int(saved.shape[0]) < n:
                    print(f"  WARNING: {prefix}{field} короче ожидаемого, холодный старт")
                    ok = False
                    break
                arr[:n] = saved[:n]
            if not ok:
                buf.clear()
                continue
            cur_id = int(checkpoint.get(f'{prefix}_cur_id', n))
            buf._cur_id = max(cur_id, n)
            if hasattr(buf, '_size'):
                buf._size = n
            print(f"  INFO: {prefix} восстановлен ({n} записей, _cur_id={buf._cur_id})")

    def _load_checkpoint(self, path):
        """Загрузка чекпоинта нового формата (с раздельными sizing-сетями).

        Старые чекпоинты (до выделения SizingNetwork) несовместимы по дизайну —
        бросаем явный ValueError, чтобы пользователь не получил тихий мусор.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Fail-fast: формат должен содержать strategy_sizing_net; advantage_sizing_net опционален (лайт-чекпоинты)
        if 'strategy_sizing_net' not in checkpoint:
            raise ValueError(
                f"Несовместимый чекпоинт '{path}': отсутствует strategy_sizing_net. "
                f"Старые чекпоинты с sizing_head внутри PokerNetwork не поддерживаются."
            )

        self.iteration_count = checkpoint.get('iteration', 0)

        if 'advantage_nets' in checkpoint:
            for i, sd in enumerate(checkpoint['advantage_nets']):
                if i < len(self.advantage_nets):
                    self.advantage_nets[i].load_state_dict(_strip_legacy_bucket_head_keys(sd), strict=False)
            self.advantage_net.load_state_dict(self.advantage_nets[0].state_dict(), strict=False)
            print("  INFO: advantage_nets (v3 multi-agent) загружены")
        elif 'advantage_net' in checkpoint:
            old_sd = checkpoint['advantage_net']
            self.advantage_nets[0].load_state_dict(old_sd, strict=False)
            self.advantage_net.load_state_dict(old_sd, strict=False)
            if self.use_multi_agent:
                for i in range(1, self.num_trainable_players):
                    self.advantage_nets[i].load_state_dict(old_sd, strict=False)
                print("  INFO: advantage_net (v2) мигрирован на все multi-agent сети")
            else:
                print("  INFO: advantage_net (v2) загружен")

        self.strategy_net.load_state_dict(checkpoint['strategy_net'], strict=False)
        sizing_dim_mismatch = False
        sizing_sd_list = None
        if 'advantage_sizing_nets' in checkpoint:
            sizing_sd_list = checkpoint['advantage_sizing_nets']
        elif 'advantage_sizing_net' in checkpoint:
            sizing_sd_list = [checkpoint['advantage_sizing_net']] * self.num_trainable_players

        if sizing_sd_list is not None:
            ckpt_format = checkpoint.get('checkpoint_format_version', 1)
            ckpt_sizing_dim = sizing_sd_list[0]['base.0.weight'].shape[1]
            sizing_dim_mismatch = (ckpt_sizing_dim != self.sizing_input_size)
            if ckpt_format < 4 or sizing_dim_mismatch:
                if ckpt_format < 4:
                    reason = "формат чекпоинта < 4"
                else:
                    reason = f"dim mismatch: ckpt={ckpt_sizing_dim} vs agent={self.sizing_input_size}"
                print(f"  INFO: sizing-сети переинициализируются ({reason})")
                device = self.device
                sizing_hidden = self.advantage_sizing_nets[0].base[0].out_features
                self.advantage_sizing_nets = nn.ModuleList([
                    SizingAnchorNet(input_size=self.sizing_input_size, hidden_size=sizing_hidden,
                                    num_sizes=self.num_anchors).to(device)
                    for _ in range(self.num_trainable_players)
                ])
                self.advantage_sizing_net = self.advantage_sizing_nets[0]
                self.strategy_sizing_net = StrategySizingNet(
                    input_size=self.sizing_input_size, hidden_size=sizing_hidden,
                    num_sizes=self.num_anchors
                ).to(device)
            else:
                for i, sd in enumerate(sizing_sd_list):
                    if i < len(self.advantage_sizing_nets):
                        self.advantage_sizing_nets[i].load_state_dict(_strip_legacy_bucket_head_keys(sd), strict=False)
                self.advantage_sizing_net.load_state_dict(self.advantage_sizing_nets[0].state_dict(), strict=False)
                self.strategy_sizing_net.load_state_dict(_strip_legacy_bucket_head_keys(checkpoint['strategy_sizing_net']), strict=False)
                self.strategy_sizing_net_loaded = True
        else:
            self.strategy_sizing_net.load_state_dict(_strip_legacy_bucket_head_keys(checkpoint['strategy_sizing_net']), strict=False)
            self.strategy_sizing_net_loaded = True
            print("  INFO: лайт-чекпоинт (strategy_sizing_net загружен, advantage_sizing_net пропущен)")

        if 'target_nets' in checkpoint:
            for i, sd in enumerate(checkpoint['target_nets']):
                if i < len(self.target_nets):
                    self.target_nets[i].load_state_dict(_strip_legacy_bucket_head_keys(sd), strict=False)
            self.target_net.load_state_dict(self.target_nets[0].state_dict(), strict=False)
        elif 'target_net' in checkpoint:
            old_sd = checkpoint['target_net']
            self.target_nets[0].load_state_dict(old_sd, strict=False)
            self.target_net.load_state_dict(old_sd, strict=False)
            if self.use_multi_agent:
                for i in range(1, self.num_trainable_players):
                    self.target_nets[i].load_state_dict(old_sd, strict=False)
        else:
            if self.use_multi_agent:
                for i in range(self.num_trainable_players):
                    self.target_nets[i].load_state_dict(self.advantage_net.state_dict())
            else:
                self.target_net.load_state_dict(self.advantage_net.state_dict())

        # Restore the frozen sizing bootstrap target explicitly. Older checkpoints
        # did not save it, so preserve their behavior by syncing from the loaded
        # live sizing network. Keep the target in eval mode and non-trainable.
        sizing_target_state = checkpoint.get('sizing_target_net')
        if sizing_target_state is not None and not sizing_dim_mismatch:
            self.sizing_target_net.load_state_dict(_strip_legacy_bucket_head_keys(sizing_target_state), strict=False)
        else:
            self.sizing_target_net.load_state_dict(self.advantage_sizing_net.state_dict())
        self.sizing_target_net.eval()
        for parameter in self.sizing_target_net.parameters():
            parameter.requires_grad = False

        if 'q_net' in checkpoint and checkpoint['q_net'] is not None:
            if self.q_net is not None:
                self.q_net.load_state_dict(checkpoint['q_net'], strict=False)
                self.q_loaded_from_checkpoint = True
                print("  INFO: q_net загружен из чекпоинта")
            else:
                self.q_loaded_from_checkpoint = False
        else:
            self.q_loaded_from_checkpoint = False
            if self.q_net is not None:
                print("  INFO: q_net отсутствует в чекпоинте, холодный старт")

        if 'q_target_net' in checkpoint and checkpoint['q_target_net'] is not None:
            if self.q_target_net is not None:
                self.q_target_net.load_state_dict(checkpoint['q_target_net'], strict=False)
                print("  INFO: q_target_net загружен из чекпоинта")
        else:
            self._sync_q_target_net()
            if self.q_target_net is not None:
                print("  INFO: q_target_net отсутствует в чекпоинте, синхронизирован из q_net")
            print("  INFO: q_target_net отсутствует в чекпоинте, синхронизирован из q_net")

        if 'sizing_q_net' in checkpoint and self.sizing_q_enabled and self.sizing_q_net is not None:
            sq_dim_mismatch = sizing_dim_mismatch
            if not sq_dim_mismatch and ckpt_format >= 4:
                ckpt_sq_state_dim = checkpoint['sizing_q_net']['base.0.weight'].shape[1]
                agent_sq_state_dim = self.sizing_q_net.base[0].in_features
                sq_dim_mismatch = (ckpt_sq_state_dim != agent_sq_state_dim)
            if ckpt_format < 4 or sq_dim_mismatch:
                reason = "формат < 4" if ckpt_format < 4 else f"dim mismatch"
                print(f"  INFO: sizing_q_net переинициализация ({reason})")
                sq_hidden = self.sizing_q_net.base[0].out_features
                sq_embed = self.sizing_q_net.size_embed.out_features
                self.sizing_q_net = SizingQNetwork(
                    state_dim=self.sizing_input_size, hidden_size=sq_hidden, size_embed_dim=sq_embed
                ).to(self.device)
            else:
                self.sizing_q_net.load_state_dict(checkpoint['sizing_q_net'], strict=False)
            if 'sizing_q_optimizer' in checkpoint and not sq_dim_mismatch:
                try:
                    self.sizing_q_optimizer.load_state_dict(checkpoint['sizing_q_optimizer'])
                except (ValueError, KeyError):
                    print("  WARNING: sizing_q_optimizer несовместим, холодный старт")
            print("  INFO: sizing_q_net загружен из чекпоинта")
            if 'sizing_q_buffer_states' in checkpoint and not sq_dim_mismatch and ckpt_format >= 4:
                buf = self.sizing_q_buffer
                n = checkpoint['sizing_q_buffer_states'].shape[0]
                n = min(n, buf.capacity)
                buf._states[:n] = checkpoint['sizing_q_buffer_states'][:n]
                buf._norm_sizes[:n] = checkpoint['sizing_q_buffer_norm_sizes'][:n]
                buf._targets[:n] = checkpoint['sizing_q_buffer_targets'][:n]
                buf._iterations[:n] = checkpoint['sizing_q_buffer_iterations'][:n]
                if 'sizing_q_buffer_sources' in checkpoint:
                    buf._sources[:n] = checkpoint['sizing_q_buffer_sources'][:n]
                if 'sizing_q_buffer_anchor_indices' in checkpoint:
                    buf._anchor_indices[:n] = checkpoint['sizing_q_buffer_anchor_indices'][:n]
                if 'sizing_q_buffer_selected_anchor_indices' in checkpoint:
                    buf._selected_anchor_indices[:n] = checkpoint['sizing_q_buffer_selected_anchor_indices'][:n]
                if 'sizing_q_buffer_effective_anchor_indices' in checkpoint:
                    buf._effective_anchor_indices[:n] = checkpoint['sizing_q_buffer_effective_anchor_indices'][:n]
                if 'sizing_q_buffer_selected_kind_ids' in checkpoint:
                    buf._selected_kind_ids[:n] = checkpoint['sizing_q_buffer_selected_kind_ids'][:n]
                buf._position = n % buf.capacity
                buf._size = n
                buf._anchor_counts[:] = 0
                for i in range(n):
                    aidx = int(buf._anchor_indices[i])
                    if aidx >= 0:
                        buf._anchor_counts[aidx] += 1
                buf._version = n
                print(f"  INFO: sizing_q_buffer восстановлен ({n} записей)")
        elif self.sizing_q_enabled:
            print("  INFO: sizing_q_net отсутствует в чекпоинте, холодный старт")

        if 'q_buffer_states' in checkpoint and self.q_buffer is not None:
            buf = self.q_buffer
            n = checkpoint['q_buffer_states'].shape[0]
            n = min(n, buf.capacity)
            buf._states[:n] = checkpoint['q_buffer_states'][:n]
            buf._actions[:n] = checkpoint['q_buffer_actions'][:n]
            buf._rewards[:n] = checkpoint['q_buffer_rewards'][:n]
            buf._next_states[:n] = checkpoint['q_buffer_next_states'][:n]
            buf._next_policy_states[:n] = checkpoint['q_buffer_next_policy_states'][:n]
            buf._next_masks[:n] = checkpoint['q_buffer_next_masks'][:n]
            buf._terminals[:n] = checkpoint['q_buffer_terminals'][:n]
            buf._next_is_hero[:n] = checkpoint['q_buffer_next_is_hero'][:n]
            buf._next_player_ids[:n] = checkpoint['q_buffer_next_player_ids'][:n]
            buf._position = n % buf.capacity
            buf._size = n
            print(f"  INFO: q_buffer восстановлен ({n} записей)")

        self._load_replay_buffers_from(checkpoint)

        if 'sizing_target_ema' in checkpoint and self.sizing_target_ema_enabled:
            self.sizing_target_ema = collections.OrderedDict()
            for k, v in checkpoint['sizing_target_ema'].items():
                v_arr = np.asarray(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v
                self.sizing_target_ema[k] = v_arr
            print(f"  INFO: sizing_target_ema восстановлен ({len(self.sizing_target_ema)} ключей)")

        if 'advantage_optimizers' in checkpoint:
            for i, sd in enumerate(checkpoint['advantage_optimizers']):
                if i < len(self.advantage_optimizers):
                    try:
                        self.advantage_optimizers[i].load_state_dict(sd)
                    except (ValueError, KeyError):
                        print(f"  WARNING: advantage_optimizers[{i}] несовместим, холодный старт")
            self.optimizer.load_state_dict(self.advantage_optimizers[0].state_dict())
            print("  INFO: advantage_optimizers (v3 multi-agent) загружены")
        elif 'advantage_optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['advantage_optimizer'])
            except (ValueError, KeyError):
                print("  WARNING: advantage_optimizer несовместим, холодный старт")
            if self.use_multi_agent:
                for i in range(1, self.num_trainable_players):
                    try:
                        self.advantage_optimizers[i].load_state_dict(checkpoint['advantage_optimizer'])
                    except (ValueError, KeyError):
                        pass
        else:
            print("  INFO: advantage_optimizer отсутствует в чекпоинте, холодный старт")

        optimizer_keys = [
            ('strategy_optimizer', self.strategy_optimizer),
            ('strategy_sizing_optimizer', self.strategy_sizing_optimizer),
        ]

        # sizing_optimizers: сначала список (новый формат), затем отдельные ключи (старый)
        if 'sizing_optimizers' in checkpoint and isinstance(checkpoint['sizing_optimizers'], list):
            sizing_list = checkpoint['sizing_optimizers']
            for idx, opt in enumerate(self.sizing_optimizers):
                if idx < len(sizing_list):
                    try:
                        opt.load_state_dict(sizing_list[idx])
                    except (ValueError, KeyError):
                        print(f"  WARNING: sizing_optimizers[{idx}] несовместим, холодный старт")
                else:
                    print(f"  INFO: sizing_optimizers[{idx}] отсутствует в чекпоинте, холодный старт")
        else:
            for idx, opt in enumerate(self.sizing_optimizers):
                optimizer_keys.append((f'sizing_optimizers_{idx}', opt))
            if 'sizing_optimizer' in checkpoint and 'sizing_optimizers_0' not in checkpoint:
                optimizer_keys.append(('sizing_optimizer', self.sizing_optimizers[0]))
        if self.q_optimizer is not None:
            optimizer_keys.append(('q_optimizer', self.q_optimizer))
        for key, opt in optimizer_keys:
            if key in checkpoint:
                try:
                    opt.load_state_dict(checkpoint[key])
                except (ValueError, KeyError):
                    print(f"  WARNING: {key} несовместим, холодный старт")
            else:
                print(f"  INFO: {key} отсутствует в чекпоинте, холодный старт")

        if 'anchors' in checkpoint and checkpoint['anchors'] is not None:
            self.anchors = list(checkpoint['anchors'])
            self.num_anchors = len(self.anchors)
        checkpoint_config = checkpoint.get('config', {}) or {}
        for key in ['min_bet_size', 'max_bet_size', 'pg_lr', 'entropy_bonus',
                    'pg_memory_size', 'pg_baseline_mean', 'pg_baseline_var',
                    'max_regret_seen',
                    'sizing_min_prob_start', 'sizing_min_prob_end',
                    'sizing_min_prob_decay_iterations',
                    'sizing_bucket_min_prob_start', 'sizing_bucket_min_prob_end',
                    'sizing_bucket_min_prob_decay_iterations',
                    'sizing_q_bootstrap_scale_fix', 'sizing_q_stratified_sampling',
                    'sizing_q_target_diagnostics_enabled',
                    'sizing_q_bootstrap_target_net',
                    'sizing_q_train_zscore',
                    'sizing_q_zscore_eps',
                    'sizing_q_src2_loss_weight_enabled',
                     'sizing_q_src2_weight_start',
                     'sizing_q_src2_weight_end',
                     'sizing_q_src2_weight_decay_iterations',
                     'save_q_buffer_in_checkpoint',
                     'save_replay_buffers_in_checkpoint',
                     'q_bootstrap_policy_mix_enabled',
                     'q_bootstrap_policy_uniform_mix',
                     'sizing_q_legal_anchor_mask_enabled',
                     'sizing_q_mask_exclude_kinds',
                     'sizing_q_kind_filter_enabled',
                     'sizing_q_kind_filter_mode',
                     'sizing_q_kind_filter_downweight',
                     'sizing_q_replay_clear_once',
                     'sizing_q_mask_diagnostics_enabled',
                     'sizing_q_mask_dry_run',
                     'sizing_q_replay_clear_diagnostics_enabled',
                     'sizing_q_selected_credit_enabled',
                     'sizing_anchor_availability_enabled',
                     'sizing_allin_boundary_enabled',
                      'sizing_preflop_disable_buckets',
                      'sizing_bucket_head_enabled',
                      'sizing_availability_on_input',
                      'q_bootstrap_per_player_enabled',
                      'use_multi_agent_advantage',
                       'advantage_regret_norm',
                       'advantage_regret_clip',
                       'advantage_loss',
                       'advantage_huber_delta',
                       'advantage_accumulation',
                        'advantage_reward_scale',
                        'discount_alpha',
                        'advantage_buffer_reservoir',
                        'strategy_buffer_reservoir',
                        'sizing_advantage_buffer_reservoir',
                        'sizing_strategy_buffer_reservoir']:

            if key in checkpoint and checkpoint[key] is not None:
                setattr(self, key, checkpoint[key])
            elif key in checkpoint_config and checkpoint_config[key] is not None:
                setattr(self, key, checkpoint_config[key])
        if 'sizing_q_kind_filter_diag' in checkpoint:
            loaded = checkpoint['sizing_q_kind_filter_diag']
            for key in self.sizing_q_kind_filter_diag:
                if key in loaded:
                    self.sizing_q_kind_filter_diag[key] = loaded[key]
        if 'sizing_q_target_diag' in checkpoint:
            raw_diag = checkpoint['sizing_q_target_diag']
            self.sizing_q_target_diag = {}
            for k, v in raw_diag.items():
                parts = k.split(':')
                if len(parts) == 2:
                    try:
                        key_tuple = (int(parts[0]), int(parts[1]))
                        self.sizing_q_target_diag[key_tuple] = v
                    except (ValueError, IndexError):
                        pass
        return checkpoint

    def save_model(self, path_prefix, seed=None):
        model_path = _resolve_model_save_path(path_prefix, self.iteration_count)
        checkpoint = self._build_checkpoint(seed=seed)
        torch.save(checkpoint, model_path)
        light_path = model_path.replace('.pt', '_light.pt')
        torch.save({
            'iteration': self.iteration_count,
            'strategy_net': self.strategy_net.state_dict(),
            'strategy_sizing_net': self.strategy_sizing_net.state_dict(),
            'num_anchors': self.num_anchors,
            'anchors': list(self.anchors),
            'min_bet_size': self.min_bet_size,
            'max_bet_size': self.max_bet_size,
            'use_multi_agent_advantage': self.use_multi_agent,
            'config': {
                'sizing_hidden_size': cfg_get('sizing_hidden_size', 128),
                'sizing_bucket_groups': [[self.anchors[idx] for idx in group]
                                         for group in self.sizing_bucket_groups],
                'sizing_cfr_mode': self.sizing_cfr_mode,
            },
        }, light_path)
        print(f"  Полный чекпоинт: {model_path}")
        print(f"  Лёгкий (для игры): {light_path}")

    def load_model(self, path):
        checkpoint = self._load_checkpoint(path)
        if 'seed' in checkpoint and checkpoint['seed'] is not None:
            print(f"  Чекпоинт обучен с seed={checkpoint['seed']}")
        if 'config' in checkpoint:
            cfg = checkpoint['config']
            print(f"  Конфиг: lr_adv={cfg.get('advantage_lr')}, lr_strat={cfg.get('strategy_lr')}, "
                  f"pg_lr={cfg.get('pg_lr')}, hidden={cfg.get('hidden_size')}, "
                  f"pg_mem={cfg.get('pg_memory_size')}, entropy={cfg.get('entropy_bonus')}")
        if 'git_hash' in checkpoint and checkpoint['git_hash']:
            print(f"  Git: {checkpoint['git_hash']}")

    def total_advantage_memory_size(self):
        return len(self.advantage_buffer)
