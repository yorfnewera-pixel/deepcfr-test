# Card Context V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить opt-in factorized card/context архитектуру с одинаковым карточным блоком в HU и six-max.

**Architecture:** В `card_context_v1` первые 109 state features проходят через `card_encoder`, остальные — через `context_encoder`; embeddings объединяются перед action head. Legacy `monolithic_v1` остаётся default.

**Tech Stack:** Python, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-card-context-v1-design.md`

## Global Constraints

- Комментарии и ошибки на русском языке.
- Не менять state encoding, action space, D2CFR или transfer весов.
- Legacy monolithic checkpoint остаются совместимы.
- Для `card_context_v1` metadata architecture обязательна.

---

### Task 1: Контракт и factorized PokerNetwork

**Files:**
- Modify: `src/core/model.py`
- Test: `tests/test_card_context_architecture.py`

**Interfaces:**
- Produces: `CARD_FEATURE_SIZE = 109`, architecture constants и `PokerNetwork(..., architecture=...)`.

- [ ] **Step 1: Write the failing test**

```python
def test_card_encoder_ignores_context_features():
    network = PokerNetwork(181, hidden_size=16, architecture="card_context_v1")
    first = torch.zeros(1, 181)
    second = first.clone()
    second[:, 109:] = 1.0
    assert torch.equal(network.encode_cards(first), network.encode_cards(second))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_card_context_architecture.py -q`

Expected: FAIL because the architecture argument or `encode_cards` is absent.

- [ ] **Step 3: Write minimal implementation**

Add constants, validate size is at least 109, add card/context encoders, fuse their embeddings before action head, preserve monolithic state-dict keys.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_card_context_architecture.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit `Добавить card context архитектуру сети` with only Task 1 files.

### Task 2: Агент и checkpoint metadata

**Files:**
- Modify: `config.yaml`
- Modify: `src/utils/config.py`
- Modify: `src/core/deep_cfr.py`
- Modify: `src/training/train.py`
- Test: `tests/test_card_context_architecture.py`
- Test: `tests/test_checkpoint_lifecycle.py`

**Interfaces:**
- Consumes: architecture constants from Task 1.
- Produces: `network_architecture` config and architecture metadata in normal and HU checkpoints.

- [ ] **Step 1: Write the failing test**

```python
def test_card_context_checkpoint_rejects_monolithic_runtime(tmp_path):
    source = DeepCFRAgent(num_players=2, network_architecture="card_context_v1")
    path = tmp_path / "model.pt"
    source.save_model(path)
    target = DeepCFRAgent(num_players=2, network_architecture="monolithic_v1")
    with pytest.raises(ValueError, match="архитектур"):
        target.load_model(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_card_context_architecture.py -q`

Expected: FAIL because checkpoint lacks architecture contract.

- [ ] **Step 3: Write minimal implementation**

Read the defaulted config into every network; write and validate metadata. Metadata absence remains valid only for monolithic legacy checkpoints.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_card_context_architecture.py tests/test_checkpoint_lifecycle.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit `Версионировать card context checkpoint` with Task 2 files.

### Task 3: Runtime construction and regressions

**Files:**
- Modify: `src/evaluation/blueprint_policy.py`
- Modify: `tools/checkpoint_tools.py` only if it constructs `PokerNetwork`
- Test: `tests/test_frozen_blueprint_policy.py`
- Test: `tests/test_card_context_architecture.py`

**Interfaces:**
- Consumes: checkpoint architecture metadata from Task 2.
- Produces: runtime policy constructing the declared architecture.

- [ ] **Step 1: Write the failing test**

```python
def test_blueprint_loads_card_context_checkpoint(tmp_path):
    agent = DeepCFRAgent(num_players=2, network_architecture="card_context_v1")
    path = tmp_path / "card_context.pt"
    agent.save_model(path)
    assert FrozenBlueprintPolicy(checkpoint_path=path, num_players=2).strategy_net.architecture == "card_context_v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_frozen_blueprint_policy.py -q`

Expected: FAIL because runtime always constructs monolithic `PokerNetwork`.

- [ ] **Step 3: Write minimal implementation**

Read architecture metadata, construct matching network and keep strict input/action validation.

- [ ] **Step 4: Run focused regressions**

Run: `python -m pytest tests/test_card_context_architecture.py tests/test_frozen_blueprint_policy.py tests/test_checkpoint_lifecycle.py tests/test_hu_checkpoint_resume.py tests/test_hu_self_play_core.py tests/test_hu_self_play_training.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit `Поддержать card context в runtime` with Task 3 files.
