# D2CFR Dueling Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить воспроизводимый opt-in вариант `d2cfr_dueling_v1` для HU current-policy self-play и six-max без изменения обычного Deep CFR пути.

**Architecture:** `DuelingRegretNetwork` имеет общий encoder и головы counterfactual `V(I)` и `Q(I,a)`; его runtime-выход `R=Q-V` остаётся входом существующего regret matching. Отдельный D2 reservoir хранит все три нормализованные цели, а D2 training branch обучается на трёх masked/weighted MSE без target-network bootstrap. HU содержит две полностью независимые D2 ноги; checkpoint маркирует algorithm variant и запрещает перекрёстный resume.

**Tech Stack:** Python 3, NumPy, PyTorch, pytest, YAML, текущие `pokers` traversal API.

**Spec:** `docs/superpowers/specs/2026-09-10-d2cfr-dueling-core-design.md`

## Global Constraints

- `d2cfr_enabled` по умолчанию равен `false`; обычный action-only путь не меняется семантически.
- Использовать только существующий `six_fixed_v2` контракт из шести action slots и существующий encoder.
- D2CFR первой поставки не включает MC rectification, outcome sampling, history-Q, KL policy loss, Stage B distillation, перенос D2-голов или HU→six-max проекцию.
- `d2cfr_mc_correction_enabled: true` и ненулевой `advantage_regret_clip` в D2 режиме обязаны отклоняться до начала обучения.
- Для HU P0/P1 не разделяют parameters, optimizer, replay или snapshot; strict resume валидирует payload до мутации runtime.
- Не добавлять в Git `planning/буфер_планов/порядок выполнения.txt` и не затрагивать несвязанные изменения рабочего дерева.

---

## File structure

| Файл | Ответственность |
| --- | --- |
| `src/core/model.py` | Dueling-сеть и тип её компонентного результата. |
| `src/core/buffers.py` | Типизированный reservoir для `Q`, `V`, `R`, masks и iteration. |
| `src/utils/config.py`, `config.yaml` | Default и fail-fast валидация D2 config. |
| `src/core/deep_cfr.py` | D2 target scaling, запись traversal-целей, обучение и single-agent checkpoint. |
| `src/core/hu_self_play.py` | Опциональная D2 запись в транзакции и coordinator без фиктивного target network. |
| `src/training/train.py` | HU D2 wiring, full checkpoint/resume и периодические light checkpoints. |
| `tests/test_d2cfr_*.py` | Изолированные unit, training, HU и checkpoint контракты. |

## Task 1: Dueling network, reservoir и конфигурация

**Files:**

- Modify: `src/core/model.py: после PokerNetwork`
- Modify: `src/core/buffers.py: после AdvantageBuffer`
- Modify: `src/utils/config.py: _DEFAULTS и load_config`
- Modify: `config.yaml: рядом с advantage-настройками`
- Create: `tests/test_d2cfr_model_and_buffer.py`

**Interfaces:**

- Produces `DuelingNetworkOutput(state_values, action_values, regrets)`.
- Produces `DuelingRegretNetwork(input_size, hidden_size, num_actions=NUM_ACTIONS, architecture=...)` с `forward_components(x)` и `forward(x)`.
- Produces `DuelingAdvantageBuffer(capacity, state_dim, num_actions=NUM_ACTIONS)` с `add(state, action_values, state_value, regrets, mask, iteration)` и `sample(num_samples=-1)`.
- Produces `_validate_d2cfr_configuration(config: Mapping[str, object]) -> None`, вызываемую из `load_config` до присваивания `_config`.

- [ ] **Step 1: Write failing network and buffer tests**

```python
@pytest.mark.parametrize("architecture", ["monolithic_v1", "card_context_v1"])
def test_dueling_network_returns_exact_regret_difference(architecture):
    network = DuelingRegretNetwork(181, hidden_size=16, architecture=architecture)
    state = torch.randn(3, 181)
    result = network.forward_components(state)
    assert result.state_values.shape == (3, 1)
    assert result.action_values.shape == result.regrets.shape == (3, 6)
    assert torch.equal(result.regrets, result.action_values - result.state_values)
    assert torch.equal(network(state), result.regrets)

def test_dueling_buffer_rejects_invalid_sample_before_mutation():
    buffer = DuelingAdvantageBuffer(2, state_dim=3)
    with pytest.raises(ValueError, match="конеч"):
        buffer.add(np.zeros(3), np.full(6, np.nan), 0.0, np.zeros(6), np.ones(6), 1)
    assert len(buffer) == 0
```

Добавить тесты reservoir replacement, tuple shapes и запрет `num_actions != 6`.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest -q tests/test_d2cfr_model_and_buffer.py`

Expected: FAIL import errors for `DuelingRegretNetwork` and `DuelingAdvantageBuffer`.

- [ ] **Step 3: Add the smallest complete data contracts**

```python
@dataclass(frozen=True)
class DuelingNetworkOutput:
    state_values: torch.Tensor
    action_values: torch.Tensor
    regrets: torch.Tensor

class DuelingRegretNetwork(nn.Module):
    def forward_components(self, x: torch.Tensor) -> DuelingNetworkOutput:
        embedding = self._encode(x)
        state_values = self.state_value_head(embedding)
        action_values = self.action_value_head(embedding)
        return DuelingNetworkOutput(state_values, action_values, action_values - state_values)

    def forward(self, x: torch.Tensor, opponent_features=None) -> torch.Tensor:
        del opponent_features
        return self.forward_components(x).regrets
```

Сделать `_encode` эквивалентным соответствующей ветке `PokerNetwork`: для
`card_context_v1` те же 109 card features, `card_encoder`, `context_encoder`
и конкатенация. В `DuelingAdvantageBuffer.add` сначала преобразовать все шесть
полей в `float32`, проверить точные shapes, binary mask, finite values и
`iteration >= 1`, затем единственной веткой менять массивы и счётчики.

Добавить D2 defaults из спецификации. В `load_config` проверить типы, диапазоны,
сумму весов, MC-флаг и clip до возврата `_config`.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `python -m pytest -q tests/test_d2cfr_model_and_buffer.py tests/test_card_context_architecture.py`

Expected: PASS.

- [ ] **Step 5: Commit task**

```powershell
git add -- src/core/model.py src/core/buffers.py src/utils/config.py config.yaml tests/test_d2cfr_model_and_buffer.py
git commit -m "feat: добавить dueling сеть и D2 replay"
```

## Task 2: D2 targets и обучение одной advantage-ноги

**Files:**

- Modify: `src/core/deep_cfr.py: конструктор, traversal collector, _normalise_regrets, _cfr_traverse_multi, train_advantage_network_multi`
- Create: `tests/test_d2cfr_training.py`

**Interfaces:**

- Consumes `DuelingRegretNetwork`, `DuelingAdvantageBuffer` и validated config из Task 1.
- Produces `DeepCFRAgent._advantage_target_scale(regrets, state, legal_slots) -> float`.
- Produces `DeepCFRAgent._normalise_d2cfr_targets(action_values, state_value, state, legal_slots) -> tuple[np.ndarray, np.float32, np.ndarray]`.
- Produces `DeepCFRAgent._record_d2cfr_advantage_sample(state, action_values, state_value, regrets, mask, iteration) -> None`.
- Produces `DeepCFRAgent.train_d2cfr_advantage_network_multi(batch_size=None, epochs=None, player_id=0) -> float`.

- [ ] **Step 1: Write failing target and loss tests**

```python
def test_d2_targets_share_one_scale_and_keep_r_equal_q_minus_v(agent, state):
    scale = agent._advantage_target_scale(np.array([4, -2, 0, 0, 0, 0], np.float32), state, [0, 1])
    q, v, r = agent._normalise_d2cfr_targets(
        np.array([10, 4, 0, 0, 0, 0], np.float32), 6.0, state, [0, 1]
    )
    assert np.allclose(q[:2] - v, r[:2])
    assert scale == pytest.approx(200.0)

def test_d2_loss_ignores_illegal_q_and_r_slots(agent):
    loss_before = agent.train_d2cfr_advantage_network_multi()
    agent.d2cfr_buffer._action_values[0, 5] = 1e9
    agent.d2cfr_buffer._regrets[0, 5] = -1e9
    loss_after = agent.train_d2cfr_advantage_network_multi()
    assert loss_after == pytest.approx(loss_before)
```

В тесте использовать две одинаково инициализированные копии агента или
зафиксированный seed, чтобы второй assertion не зависел от optimizer step.
Добавить тесты: exact weighted MSE на synthetic batch, `p=0`, replacement сети
и AdamW при `reinitialize=true`, сохранение buffer и отсутствие обращения к
`advantage_target_net` в D2-ветке.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest -q tests/test_d2cfr_training.py`

Expected: FAIL because D2 methods and `d2cfr_buffer` отсутствуют.

- [ ] **Step 3: Route only enabled agents through D2 implementation**

Создать фабрики `_new_advantage_network()` и `_new_advantage_optimizer(network)`;
при D2 они строят `DuelingRegretNetwork`/`AdamW`, иначе прежние объекты.
`advantage_target_net` в D2 равен `None` и к нему нет обращения из D2 кода.

Вынести линейный знаменатель в `_advantage_target_scale`. Обычная
`_normalise_regrets` продолжает давать прежний результат. D2 traversal при
узле traverser-а сохраняет только допустимые `Q_raw`, вычисляет `V_raw` текущей
policy, применяет один scale ко всем трём targets и добавляет sample. Обычный
traversal продолжает писать `AdvantageBuffer` по существующему коду.

Реализовать weighted MSE так, чтобы numerator суммировал только legal slots,
а denominator был `sum(weights) * legal_slot_count` для Q/R и `sum(weights)`
для V. После каждого шага проверять finite loss/gradients и применять прежний
`clip_grad_norm_`. В профиль записывать `regret_loss`, `state_value_loss`,
`action_value_loss`, `total_loss`.

- [ ] **Step 4: Run focused regression tests to verify GREEN**

Run: `python -m pytest -q tests/test_d2cfr_training.py tests/test_training_regressions.py tests/test_card_context_architecture.py`

Expected: PASS.

- [ ] **Step 5: Commit task**

```powershell
git add -- src/core/deep_cfr.py tests/test_d2cfr_training.py
git commit -m "feat: обучать D2CFR counterfactual heads"
```

## Task 3: HU coordinator с двумя D2-ногами

**Files:**

- Modify: `src/core/hu_self_play.py: HuTraversalAdapter, HuCurrentPolicySelfPlayCoordinator`
- Modify: `src/training/train.py: _create_hu_current_policy_coordinator`
- Create: `tests/test_d2cfr_hu.py`
- Modify: `tests/test_hu_self_play_core.py`

**Interfaces:**

- Consumes `DeepCFRAgent._normalise_d2cfr_targets` и `train_d2cfr_advantage_network_multi`.
- Extends `HuTraversalAdapter` with optional `normalise_d2cfr_targets(state, action_values, state_value, mask) -> tuple[np.ndarray, np.float32, np.ndarray]`.
- Extends coordinator with `d2cfr_enabled: bool`; `advantage_target_nets` is `None` only when it is true.
- Produces independent `agent.hu_advantage_nets`, `agent.hu_advantage_optimizers`, `agent.hu_advantage_buffers` for P0/P1 in both variants.

- [ ] **Step 1: Write failing HU isolation tests**

```python
def test_hu_d2_writes_q_v_r_only_to_traversing_player_buffer():
    coordinator = _d2_coordinator()
    coordinator.begin_iteration()
    coordinator.traverse((0, 0), traversing_player=0, iteration=3)
    assert len(coordinator.advantage_buffers[0]) == 1
    assert len(coordinator.advantage_buffers[1]) == 0
    _, q, v, r, mask, _ = coordinator.advantage_buffers[0].sample()
    assert np.allclose(q[0, mask[0] > 0] - v[0], r[0, mask[0] > 0])

def test_hu_d2_does_not_construct_or_sync_target_networks(monkeypatch, agent):
    coordinator = _create_hu_current_policy_coordinator(agent)
    assert coordinator.advantage_target_nets is None
```

Также перенести текущие обычные HU tests без изменения assertions: coordinator с
target networks по-прежнему валидирует их независимость и синхронизирует после
обучения.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest -q tests/test_d2cfr_hu.py tests/test_hu_self_play_core.py`

Expected: FAIL because adapter/coordinator не знают D2 sample и optional targets.

- [ ] **Step 3: Make D2 transaction explicit**

В `_traverse` HU при `actor_id == traverser` сохранить raw `action_values` и
`expected_value`; D2 callback формирует и валидирует `(Q, V, R)`, после чего
координатор записывает один D2 sample в pending list нужного игрока. При
ошибке callback traversal rollback удаляет этот sample вместе со strategy
sample.

Сделать `advantage_target_nets: tuple[nn.Module, nn.Module] | None`. Когда
они отсутствуют, constructor требует `d2cfr_enabled=True`, `run_iteration`
не синхронизирует targets, а training callback получает `target_network=None`.
Когда они есть, не менять прежний validation и copy semantics.

В train factory для D2 построить второй Dueling network, второй AdamW и второй
`DuelingAdvantageBuffer`; strategy путь не менять. Training callback временно
подменяет только текущие `advantage_net`, `optimizer`, `advantage_buffer` и
вызывает dispatch-метод агента.

- [ ] **Step 4: Run HU tests to verify GREEN**

Run: `python -m pytest -q tests/test_d2cfr_hu.py tests/test_hu_self_play_core.py tests/test_hu_self_play_training.py`

Expected: PASS.

- [ ] **Step 5: Commit task**

```powershell
git add -- src/core/hu_self_play.py src/training/train.py tests/test_d2cfr_hu.py tests/test_hu_self_play_core.py
git commit -m "feat: подключить D2CFR к HU self-play"
```

## Task 4: Variant-safe six-max и HU checkpoints

**Files:**

- Modify: `src/core/deep_cfr.py: _buffer_payload, _restore_buffer, _build_checkpoint, _load_checkpoint`
- Modify: `src/training/train.py: HU trajectory config, HU build/validate/load helpers`
- Modify: `tests/test_hu_checkpoint_resume.py`
- Create: `tests/test_d2cfr_checkpoint.py`

**Interfaces:**

- Full D2 payload has `algorithm_variant == "d2cfr_dueling_v1"`.
- Full normal payload has explicit `algorithm_variant == "deep_cfr_action_only_v1"`.
- `DeepCFRAgent._load_checkpoint(path)` and `_load_hu_checkpoint(agent, path)` validate variant, network metadata, optimizer state, buffer payload and RNG before mutating runtime.

- [ ] **Step 1: Write failing checkpoint tests**

```python
def test_hu_d2_checkpoint_round_trip_restores_both_legs_buffer_and_rng(tmp_path, d2_hu_agent):
    path = _save_hu_checkpoint(d2_hu_agent, tmp_path / "hu_d2.pt", seed=17)
    restored = _new_d2_hu_agent()
    _load_hu_checkpoint(restored, path)
    assert restored.iteration_count == d2_hu_agent.iteration_count
    assert len(restored.hu_advantage_buffers[0]) == len(d2_hu_agent.hu_advantage_buffers[0])
    assert len(restored.hu_advantage_buffers[1]) == len(d2_hu_agent.hu_advantage_buffers[1])

def test_d2_resume_rejects_action_only_checkpoint_before_mutation(tmp_path, d2_agent, action_only_path):
    before = {name: tensor.clone() for name, tensor in d2_agent.advantage_net.state_dict().items()}
    with pytest.raises(ValueError, match="algorithm variant"):
        d2_agent.load_model(action_only_path)
    assert_state_dict_equal(d2_agent.advantage_net.state_dict(), before)
```

Добавить обратное направление, повреждённый/неполный D2 optimizer, неверные
shapes D2 buffer и запрет Stage A loader для D2 HU checkpoint.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest -q tests/test_d2cfr_checkpoint.py tests/test_hu_checkpoint_resume.py`

Expected: FAIL because payload ещё не содержит algorithm variant и D2 legs.

- [ ] **Step 3: Serialize and validate before every mutation**

Для single-agent D2 serialизовать `dueling_advantage_net`, `advantage_optimizer`
и D2 buffer; не записывать несуществующий target network. Для HU D2 записывать
две legs `{network, optimizer, buffer}` без `target_network`; architecture
явно описывает `dueling_regret_v1`, а config содержит все D2 поля и target
normalization. Обычный checkpoint получает собственный явный variant и прежний
schema с target network.

Разделить validation и application: функции validation строят проверенные
локальные структуры, load weights/optimizers/buffers/RNG выполняется только
после успешной проверки каждого ключа, shape и metadata. `weights_only=True`
сохраняется для HU safe load. `build_light_checkpoint` остаётся strategy-only
и не объявляется resume-артефактом.

- [ ] **Step 4: Run checkpoint tests to verify GREEN**

Run: `python -m pytest -q tests/test_d2cfr_checkpoint.py tests/test_hu_checkpoint_resume.py tests/test_teacher_transfer.py`

Expected: PASS.

- [ ] **Step 5: Commit task**

```powershell
git add -- src/core/deep_cfr.py src/training/train.py tests/test_d2cfr_checkpoint.py tests/test_hu_checkpoint_resume.py
git commit -m "feat: сохранять D2CFR checkpoint строго"
```

## Task 5: End-to-end smoke, diagnostics и документация запуска

**Files:**

- Modify: `src/training/train.py: logging D2 component losses и resume diagnostics`
- Create: `configs/hu_d2cfr_smoke.yaml`
- Create: `tests/test_d2cfr_smoke.py`
- Modify: `docs/superpowers/specs/2026-09-10-d2cfr-dueling-core-design.md: раздел критериев готовности`

**Interfaces:**

- Consumes complete D2 runtime from Tasks 1–4.
- Produces minimal HU config with `d2cfr_enabled: true`, two players, strict mode, small buffer/batches and `d2cfr_mc_correction_enabled: false`.
- Produces console/TensorBoard diagnostic fields `d2cfr_regret_loss`, `d2cfr_state_value_loss`, `d2cfr_action_value_loss`.

- [ ] **Step 1: Write failing process-level smoke tests**

```python
def test_hu_d2_smoke_creates_full_and_light_checkpoint_and_resumes(tmp_path, monkeypatch):
    agent = train_self_play_multi(
        num_iterations=2, traversals_per_iteration=1, evaluate_every=0,
        save_dir=tmp_path, num_players=2, trainable_players=2,
        hu_current_policy_self_play=True,
    )
    assert (tmp_path / "hu_checkpoint_final.pt").is_file()
    assert (tmp_path / "light_checkpoint_final.pt").is_file()
    assert all(np.isfinite(network(torch.zeros(1, agent.input_size)).detach().numpy()).all()
               for network in agent.hu_advantage_nets)
```

В test fixture загрузить `configs/hu_d2cfr_smoke.yaml`, чтобы глобальный
`config.yaml` не менялся. Добавить six-max one-iteration smoke с D2 flag,
который проверяет шесть выходных slots и отсутствие HU coordinator.

- [ ] **Step 2: Run smoke tests to verify RED**

Run: `python -m pytest -q tests/test_d2cfr_smoke.py`

Expected: FAIL до появления итоговой D2 диагностики или smoke config.

- [ ] **Step 3: Add only observable diagnostics and fixed smoke config**

В логировании выводить три D2 component losses и total loss раздельно для P0
и P1, не меняя формат обычного loss. Конфиг smoke использует `hidden_size: 16`,
`advantage_memory_size: 32`, `strategy_memory_size: 32`, batch size `4`, одну
epoch и `hu_checkpoint_save_every: 1`; он предназначен только для теста, не
для оценки силы. В spec добавить точные команды прогона и напоминание о
раздельном A/B качестве.

- [ ] **Step 4: Run all required verification**

Run: `python -m pytest -q tests/test_d2cfr_model_and_buffer.py tests/test_d2cfr_training.py tests/test_d2cfr_hu.py tests/test_d2cfr_checkpoint.py tests/test_d2cfr_smoke.py tests/test_teacher_transfer.py tests/test_hu_checkpoint_resume.py tests/test_card_context_architecture.py tests/test_training_regressions.py`

Expected: PASS.

Run: `python -m pytest -q --ignore=tests/test_legacy_checkpoint_detection.py --ignore=tests/test_legacy_starting_opponent.py`

Expected: PASS; записать число passed/skipped и длительность в итоговом отчёте.

- [ ] **Step 5: Review and commit task**

Попросить независимый code review после результатов Task 5. Исправить только
подтверждённые findings, снова выполнить обе команды проверки и затем:

```powershell
git add -- src/training/train.py configs/hu_d2cfr_smoke.yaml tests/test_d2cfr_smoke.py docs/superpowers/specs/2026-09-10-d2cfr-dueling-core-design.md
git commit -m "test: добавить smoke проверку D2CFR"
```

## Final acceptance checklist

- [ ] Каждый task имеет отдельный RED→GREEN commit и review gate.
- [ ] `d2cfr_enabled: false` проходит существующие HU, six-max и Stage A tests без смены поведения.
- [ ] `d2cfr_enabled: true` создаёт конечные P0/P1 policies, full checkpoint/resume и strategy-only light checkpoint.
- [ ] Нормальный и D2 full checkpoints перекрёстно отвергаются до мутации runtime.
- [ ] Финальный suite прошёл с указанными двумя legacy exclusions.
- [ ] Итоговый отчёт отделяет техническую валидность от метрики игрового качества и не выбирает checkpoint по training loss.
