# Strategy Opponent Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перевести frozen checkpoint opponents в traversal с regret-matching по `advantage_net` на нашу уже существующую runtime policy `strategy_net -> masked softmax`, не меняя traverser-логику.

**Architecture:** Full checkpoint остается единственным источником весов. Traversing player продолжает считать regrets через `advantage_net`; non-traversing checkpoint opponents получают замороженные `strategy_net` и выбирают sampled action через ту же masked-softmax формулу, что использует `DeepCFRAgent.get_policy_distribution()`. Загрузка checkpoint должна читать файл не чаще одного раза за запуск, а warm-up с RandomAgent не должен заранее настраивать neural opponent pool.

**Tech Stack:** Python, PyTorch, NumPy, pytest, pokers.

**Spec:** Диалог 2026-08-26: frozen checkpoint opponents должны играть через нашу дистиллированную `strategy_net` policy для более сглаженной historical/average policy; `advantage_net` остается источником regrets для обучаемого traverser. Это переносит только принцип выбора opponent policy, а не авторские формулы regret scaling/loss/replay.

## Global Constraints

- Общаться и писать новые комментарии на русском.
- Не удалять работающую логику без явной необходимости.
- Сохранять формат full checkpoint: `multi_checkpoint_iter_N.pt` содержит `advantage_net`, `advantage_target_net`, `strategy_net`, optimizer и optional replay buffers.
- Не возвращать `opponent_advantage_iter_*.pt`; scanner full checkpoint остается текущим.
- Traverser не должен использовать `strategy_net` для расчета regrets.
- Не переносить авторские формулы `sqrt(iteration)`, prioritized replay, `SmoothL1/Huber`, per-node regret normalization или bet-sizing scaling. Этот план меняет только источник политики checkpoint-оппонентов.
- Для checkpoint-оппонентов использовать формулу нашего проекта: `strategy_net(state) -> masked softmax по legal_action_mask -> np.random.choice`.
- Поведение должно быть проверено тестами до и после изменения.

---

### Task 1: Загрузчик strategy-весов из full checkpoint

**Files:**
- Modify: `src/training/train.py`
- Test: `tests/test_checkpoint_lifecycle.py`

**Interfaces:**
- Consumes: `_heavy_checkpoints(directory) -> dict[int, Path]`, `_opponent_checkpoint_paths(...) -> list[Path]`.
- Produces: `_load_full_checkpoint_strategy_state(path: Path) -> dict[str, torch.Tensor]`.

- [ ] **Step 1: Write the failing test**

Add next to `test_opponent_pool_caches_full_checkpoint_advantage_state` in `tests/test_checkpoint_lifecycle.py`:

```python
def test_full_checkpoint_strategy_loader_requires_strategy_net(tmp_path):
    full_path = tmp_path / "multi_checkpoint_iter_1000.pt"
    torch.save(
        {
            "action_space_version": "six_fixed_v2",
            "advantage_net": {"wrong": torch.tensor([1.0])},
        },
        full_path,
    )

    with pytest.raises(ValueError, match="strategy_net"):
        train_mod._load_full_checkpoint_strategy_state(full_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_checkpoint_lifecycle.py::test_full_checkpoint_strategy_loader_requires_strategy_net -v`

Expected: FAIL with `AttributeError` because `_load_full_checkpoint_strategy_state` does not exist.

- [ ] **Step 3: Implement minimal loader**

In `src/training/train.py`, replace `_load_full_checkpoint_advantage_state` with:

```python
def _load_full_checkpoint_strategy_state(path: Path) -> dict[str, torch.Tensor]:
    """Извлекает strategy-веса из full checkpoint и проверяет игровой контракт."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Некорректный full checkpoint: {path}")
    version = checkpoint.get("action_space_version")
    if version != ACTION_SPACE_VERSION:
        raise ValueError(
            f"В full checkpoint {path} action-space '{version}', "
            f"ожидался '{ACTION_SPACE_VERSION}'"
        )
    state_dict = checkpoint.get("strategy_net")
    if not isinstance(state_dict, dict):
        raise ValueError(f"В full checkpoint {path} отсутствует strategy_net")
    return state_dict
```

Do not keep `_load_full_checkpoint_advantage_state` unless a remaining test still imports it; update tests instead.

- [ ] **Step 4: Run loader test**

Run: `pytest tests/test_checkpoint_lifecycle.py::test_full_checkpoint_strategy_loader_requires_strategy_net -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/train.py tests/test_checkpoint_lifecycle.py
git commit -m "test: require strategy state for opponent checkpoints"
```

---

### Task 2: Strategy-net frozen opponents in DeepCFRAgent

**Files:**
- Modify: `src/core/deep_cfr.py`
- Test: `tests/test_training_regressions.py`

**Interfaces:**
- Consumes: `PokerNetwork(input_size, hidden_size, NUM_ACTIONS)`, `get_legal_action_mask(state)`, `_encode_state(state, player_id)`.
- Produces: `set_opponent_strategy_states(state_dicts, traversing_player) -> None`.
- Produces: `clear_opponent_strategy_states() -> None`.
- Preserves: `set_opponent_advantage_states(...)` and `clear_opponent_advantage_states()` for backward-compatible tests until Task 4 removes old usage.

- [ ] **Step 1: Write failing unit test for our masked strategy policy**

Add to `tests/test_training_regressions.py`:

```python
def test_checkpoint_opponent_strategy_uses_strategy_net_not_advantage_net(monkeypatch):
    import numpy as np
    import torch

    from src.core.deep_cfr import DeepCFRAgent

    class DummyState:
        final_state = False
        current_player = 1
        pot = 10.0
        min_bet = 2.0
        button = 0
        from_action = None

        def __init__(self):
            player = type("Player", (), {})()
            player.active = True
            player.bet_chips = 0.0
            player.pot_chips = 0.0
            player.stake = 200.0
            player.reward = 0.0
            self.players_state = [player, player]

        def apply_action(self, action):
            next_state = DummyState()
            next_state.final_state = True
            next_state.players_state[0].reward = 7.0
            next_state.status = train_mod.pkrs.StateStatus.Ok
            return next_state

    agent = DeepCFRAgent(player_id=0, num_players=2, device="cpu")
    agent._encode_state = lambda _state, _player_id: np.zeros(agent.input_size, dtype=np.float32)
    agent.get_legal_action_mask = lambda _state: np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    agent.action_type_to_pokers_action = lambda _slot, _state: object()

    with torch.no_grad():
        for parameter in agent.advantage_net.parameters():
            parameter.zero_()
        for parameter in agent.strategy_net.parameters():
            parameter.zero_()
        agent.strategy_net.policy_head.bias[:2] = torch.tensor([-20.0, 20.0])

    agent.set_opponent_strategy_states([agent.strategy_net.state_dict()], traversing_player=0)
    monkeypatch.setattr(agent.np.random, "choice", lambda legal_slots, p: int(legal_slots[int(np.argmax(p))]))

    agent.cfr_traverse_multi(DummyState(), iteration=1001, traversing_player=0)

    assert agent.action_decision_count == 1
```

If `PokerNetwork` does not expose `policy_head`, adapt only the bias-setting lines to the actual strategy output head name from `src/core/model.py`. Do not change the behavioral assertion.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training_regressions.py::test_checkpoint_opponent_strategy_uses_strategy_net_not_advantage_net -v`

Expected: FAIL with missing `set_opponent_strategy_states`.

- [ ] **Step 3: Implement strategy opponent storage**

In `DeepCFRAgent.__init__`, add:

```python
self._opponent_strategy_nets: dict[int, PokerNetwork] = {}
```

Add methods near `set_opponent_advantage_states`:

```python
def set_opponent_strategy_states(self, state_dicts, traversing_player):
    """Устанавливает замороженные strategy-сети оппонентов для одной итерации."""
    opponent_ids = [
        player_id for player_id in range(self.num_players)
        if player_id != int(traversing_player)
    ]
    if len(state_dicts) != len(opponent_ids):
        raise ValueError("Число состояний оппонентов не совпадает с числом мест за столом")

    networks: dict[int, PokerNetwork] = {}
    for player_id, state_dict in zip(opponent_ids, state_dicts, strict=True):
        network = PokerNetwork(self.input_size, self.strategy_net.base[0].out_features, NUM_ACTIONS)
        network.load_state_dict(state_dict, strict=True)
        network.to(self.device)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        networks[player_id] = network
    self._opponent_strategy_nets = networks

def clear_opponent_strategy_states(self):
    """Очищает strategy-сети checkpoint-оппонентов."""
    self._opponent_strategy_nets = {}
```

Update `clear_opponent_advantage_states` to clear both old and new stores:

```python
def clear_opponent_advantage_states(self):
    """Возвращает оппонентов к общей advantage-сети для обратной совместимости."""
    self._opponent_advantage_nets = {}
    self._opponent_strategy_nets = {}
```

- [ ] **Step 4: Reuse our masked softmax for strategy policies**

Add a static helper near `_regret_matching`:

```python
@staticmethod
def _masked_softmax(logits, mask):
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=logits.device)
    if mask_t.ndim == 1:
        mask_t = mask_t.unsqueeze(0)
    masked_logits = torch.where(mask_t > 0.0, logits, torch.full_like(logits, -1e20))
    return F.softmax(masked_logits, dim=1)
```

Update `get_policy_distribution` to call `_masked_softmax`:

```python
with torch.inference_mode():
    logits = self.strategy_net(state_t)
    probabilities = self._masked_softmax(logits, mask_t)[0].cpu().numpy()
```

This helper is the only strategy formula used by checkpoint opponents. Do not add regret scaling, Huber, prioritized replay, `sqrt(iteration)`, or any other formula from the reference repository in this task.

- [ ] **Step 5: Use our strategy opponent policy before advantage fallback**

In `_cfr_traverse_multi`, replace the non-traverser policy block:

```python
opponent_strategy_net = self._opponent_strategy_nets.get(current_player)
encoded = self._encode_state(state, current_player)
state_t = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)
with torch.inference_mode():
    if opponent_strategy_net is not None:
        logits = opponent_strategy_net(state_t)
        strategy = self._masked_softmax(logits, mask)[0].cpu().numpy()
    else:
        opponent_net = self._opponent_advantage_nets.get(current_player, self.advantage_net)
        advantages = opponent_net(state_t)[0].cpu().numpy()
        strategy = self._regret_matching(advantages, mask)
```

Keep traverser branch unchanged.

- [ ] **Step 6: Run strategy opponent test**

Run: `pytest tests/test_training_regressions.py::test_checkpoint_opponent_strategy_uses_strategy_net_not_advantage_net -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/core/deep_cfr.py tests/test_training_regressions.py
git commit -m "feat: add strategy-net checkpoint opponents"
```

---

### Task 3: Wire training opponent pool to strategy states

**Files:**
- Modify: `src/training/train.py`
- Test: `tests/test_checkpoint_lifecycle.py`

**Interfaces:**
- Consumes: `_load_full_checkpoint_strategy_state(path: Path) -> dict[str, torch.Tensor]`.
- Consumes: `DeepCFRAgent.set_opponent_strategy_states(state_dicts, traversing_player)`.
- Produces: `_configure_opponent_pool(..., state_cache: dict[Path, dict[str, torch.Tensor]] | None = None) -> list[Path]` using strategy states.

- [ ] **Step 1: Replace cache test**

Replace `test_opponent_pool_caches_full_checkpoint_advantage_state` with:

```python
def test_opponent_pool_caches_full_checkpoint_strategy_state(tmp_path, monkeypatch):
    full_path = tmp_path / "multi_checkpoint_iter_1000.pt"
    torch.save(
        {
            "action_space_version": "six_fixed_v2",
            "strategy_net": {"weight": torch.tensor([1.0])},
            "advantage_net": {"weight": torch.tensor([2.0])},
            "advantage_buffer": ["large buffer"],
        },
        full_path,
    )
    loaded_paths = []
    original_load = train_mod.torch.load

    def tracked_load(path, *args, **kwargs):
        loaded_paths.append(Path(path))
        return original_load(path, *args, **kwargs)

    class OpponentPoolAgent:
        num_players = 6

        def clear_opponent_advantage_states(self):
            raise AssertionError("Full checkpoint должен быть найден")

        def set_opponent_strategy_states(self, states, traversing_player):
            assert len(states) == 5
            assert traversing_player == 0
            assert all(state["weight"].item() == 1.0 for state in states)

    monkeypatch.setattr(train_mod.torch, "load", tracked_load)
    state_cache = {}

    train_mod._configure_opponent_pool(
        OpponentPoolAgent(), 1001, tmp_path, traversing_player=0, state_cache=state_cache
    )
    train_mod._configure_opponent_pool(
        OpponentPoolAgent(), 1001, tmp_path, traversing_player=0, state_cache=state_cache
    )

    assert loaded_paths == [full_path]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_checkpoint_lifecycle.py::test_opponent_pool_caches_full_checkpoint_strategy_state -v`

Expected: FAIL because `_configure_opponent_pool` still loads `advantage_net` and calls `set_opponent_advantage_states`.

- [ ] **Step 3: Update `_configure_opponent_pool`**

In `src/training/train.py`, update cache variable names locally and call strategy API:

```python
for path in dict.fromkeys(paths):
    if path not in state_cache:
        state_cache[path] = _load_full_checkpoint_strategy_state(path)
agent.set_opponent_strategy_states(
    [state_cache[path] for path in paths],
    traversing_player=traversing_player,
)
```

Keep the existing `state_cache` parameter name to avoid a broad call-site rename.

- [ ] **Step 4: Run checkpoint lifecycle tests**

Run: `pytest tests/test_checkpoint_lifecycle.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/train.py tests/test_checkpoint_lifecycle.py
git commit -m "feat: load strategy states for opponent pool"
```

---

### Task 4: Avoid neural pool setup during RandomAgent warm-up

**Files:**
- Modify: `src/training/train.py`
- Test: `tests/test_training_regressions.py`

**Interfaces:**
- Consumes: `cfg_get("checkpoint_save_every", 1000)`.
- Produces: training loop that calls `_configure_opponent_pool` only when `random_opponents == 0`.

- [ ] **Step 1: Write failing warm-up test**

Add to `tests/test_training_regressions.py`:

```python
def test_training_skips_opponent_pool_setup_during_random_warmup(monkeypatch, tmp_path):
    calls = []

    class Agent:
        num_players = 6
        iteration_count = 0
        optimizer = type("Optimizer", (), {"param_groups": [{"lr": 1e-4}]})()
        advantage_buffer = []
        strategy_buffer = []

        def load_model(self, _path):
            pass

        def prepare_iteration(self, iteration, traversing_player):
            pass

        def reset_traversal_stats(self):
            pass

        def cfr_traverse_multi(self, *_args, **_kwargs):
            pass

        def train_advantage_network_multi(self):
            return 0.0

        def train_strategy_network(self):
            return 0.0

        def get_traversal_stats(self):
            return {
                "nodes": 0,
                "terminal_nodes": 0,
                "max_depth": 0,
                "recorded_nodes": 0,
                "buffer_skip_ratio": 0.0,
            }

    monkeypatch.setattr(train_mod, "DeepCFRAgent", lambda *args, **kwargs: Agent())
    monkeypatch.setattr(train_mod, "_new_hand", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(train_mod, "_create_writer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_mod, "_configure_opponent_pool", lambda *args, **kwargs: calls.append(args) or [])
    monkeypatch.setattr(train_mod, "_print_opponent_checkpoints", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_mod, "cfg_get", lambda key, default=None: 1000 if key == "checkpoint_save_every" else default)

    train_mod.train_deep_cfr(
        num_iterations=1,
        traversals_per_iteration=0,
        save_dir=tmp_path,
        log_dir=tmp_path / "logs",
        seed=1,
    )

    assert calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training_regressions.py::test_training_skips_opponent_pool_setup_during_random_warmup -v`

Expected: FAIL because `_configure_opponent_pool` is called before random warm-up decision.

- [ ] **Step 3: Move warm-up decision before pool setup**

In `train_deep_cfr`, compute `random_opponents` before `_configure_opponent_pool`:

```python
checkpoint_every = int(cfg_get("checkpoint_save_every", 1000))
random_opponents = agent.num_players - 1 if iteration <= checkpoint_every else 0
if random_opponents > 0:
    agent.clear_opponent_advantage_states()
    opponent_checkpoints = []
else:
    opponent_checkpoints = _configure_opponent_pool(
        agent,
        iteration,
        save_dir,
        traversing_player=0,
        state_cache=opponent_state_cache,
    )
```

Do not change the existing RandomAgent selection below except to reuse `random_opponents`.

- [ ] **Step 4: Run warm-up and existing cutoff tests**

Run:

```bash
pytest tests/test_training_regressions.py::test_training_skips_opponent_pool_setup_during_random_warmup tests/test_training_regressions.py::test_training_stops_passing_random_agent_after_thousandth_checkpoint -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/train.py tests/test_training_regressions.py
git commit -m "perf: skip checkpoint opponent setup during warmup"
```

---

### Task 5: Add timing visibility for opponent pool setup

**Files:**
- Modify: `src/training/train.py`
- Test: `tests/test_training_regressions.py`

**Interfaces:**
- Produces: TensorBoard scalar `Time/OpponentPoolSetup`.
- Produces: console summary field `Time/OpponentPoolSetup=...s`.

- [ ] **Step 1: Write formatting test**

Add to `tests/test_training_regressions.py`:

```python
def test_iteration_summary_includes_opponent_pool_setup_time():
    summary = train_mod._format_iteration_summary(
        iteration_elapsed=10.0,
        traversal_elapsed=4.0,
        advantage_loss=0.1,
        strategy_loss=0.2,
        opponent_setup_elapsed=0.3,
    )

    assert "Time/OpponentPoolSetup=0.3s" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training_regressions.py::test_iteration_summary_includes_opponent_pool_setup_time -v`

Expected: FAIL because `_format_iteration_summary` has no `opponent_setup_elapsed` parameter.

- [ ] **Step 3: Extend summary signature**

Change `_format_iteration_summary` in `src/training/train.py` to:

```python
def _format_iteration_summary(
    iteration_elapsed: float,
    traversal_elapsed: float,
    advantage_loss: float,
    strategy_loss: float,
    opponent_setup_elapsed: float = 0.0,
) -> str:
    return (
        f"Time/Iteration={iteration_elapsed:.1f}s | "
        f"Time/Traversal={traversal_elapsed:.1f}s | "
        f"Time/OpponentPoolSetup={opponent_setup_elapsed:.1f}s | "
        f"Loss/Advantage={advantage_loss:.6f} | "
        f"Loss/Strategy={strategy_loss:.6f}"
    )
```

- [ ] **Step 4: Measure setup in training loop**

Around the warm-up/pool setup block:

```python
opponent_setup_started = time.perf_counter()
# compute random_opponents and maybe call _configure_opponent_pool
opponent_setup_elapsed = time.perf_counter() - opponent_setup_started
```

After existing `writer.add_scalar("Time/Traversal", ...)`, add:

```python
writer.add_scalar("Time/OpponentPoolSetup", opponent_setup_elapsed, iteration)
```

Pass `opponent_setup_elapsed` into `_format_iteration_summary(...)`.

- [ ] **Step 5: Run formatting test**

Run: `pytest tests/test_training_regressions.py::test_iteration_summary_includes_opponent_pool_setup_time -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/training/train.py tests/test_training_regressions.py
git commit -m "chore: log opponent pool setup time"
```

---

### Task 6: Final regression and smoke verification

**Files:**
- Modify: none unless tests reveal a real defect.
- Test: full targeted suite.

**Interfaces:**
- Consumes: all tasks above.
- Produces: verified transition to strategy-net opponents.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
pytest tests/test_checkpoint_lifecycle.py tests/test_training_regressions.py tests/test_action_space.py -q
```

Expected: PASS.

- [ ] **Step 2: Compile source**

Run:

```bash
python -m compileall -q src
```

Expected: PASS with no output.

- [ ] **Step 3: Run a tiny smoke training**

Run:

```bash
python -m src.training.train --iterations 2 --traversals 0 --save-dir .tmp_strategy_opponent_smoke --log-dir .tmp_strategy_opponent_smoke/logs
```

Expected: command exits successfully. Console summary includes `Time/OpponentPoolSetup`.

- [ ] **Step 4: Inspect diff**

Run:

```bash
git diff -- src/core/deep_cfr.py src/training/train.py tests/test_checkpoint_lifecycle.py tests/test_training_regressions.py
```

Expected: diff only changes opponent checkpoint policy source, warm-up setup ordering, and timing visibility.

- [ ] **Step 5: Commit**

```bash
git add src/core/deep_cfr.py src/training/train.py tests/test_checkpoint_lifecycle.py tests/test_training_regressions.py
git commit -m "test: verify strategy opponent pool migration"
```

---

## Self-Review

- Spec coverage: план переводит checkpoint opponents на `strategy_net`, сохраняет `advantage_net` для traverser, не меняет full checkpoint формат, добавляет проверку warm-up и метрику стоимости setup.
- Placeholder scan: нет `TBD`, нет незаполненных шагов, каждый task имеет тест и ожидаемый результат.
- Type consistency: loader возвращает `dict[str, torch.Tensor]`; `_configure_opponent_pool` продолжает принимать `state_cache`; `DeepCFRAgent.set_opponent_strategy_states` принимает тот же shape контракта, что старый advantage API.
