#!/usr/bin/env python3
"""П2: разделяющий тест sizing-таргетов — четыре руки на одних раздачах.

ЗАЧЕМ. Замеры #107 §2.8 показали, что sizing-нога не лучше равномерного выбора анкеров
(`TV(net)/TV(uniform)` = 1.033 для advantage, 1.068 для strategy, без улучшения за 20 итераций).
Но все наши метрики мерили СОГЛАСИЕ С ТАРГЕТАМИ, а сами таргеты на 12-14 анкерах из 15
производны от Q-сети с `corr(Q, MC) = 0.156`. Поэтому неизвестно главное: таргеты вообще
правильные? Сеть может добросовестно расходиться с мусором.

Единственный способ узнать — пересчитать значения анкеров MC-обходом на живых раздачах,
без Q-инъекции, и сравнить в игре. Всё, что не требует пересчёта MC, снова померит согласие
с таргетами.

ЧЕТЫРЕ РУКИ (одни раздачи, common random numbers, парные разницы):
  1. oracle_mc     — политика из свежих MC-регретов по доступным анкерам, БЕЗ Q
  2. oracle_mc_q   — то же, но некредитуемые анкеры берут значение от sizing_q_net
  3. uniform       — равномерно по доступным анкерам
  4. blueprint     — strategy_sizing_net, то есть ровно то, что играет choose_action:3422

Action-нога у всех четырёх одна и та же (strategy_net блюпринта): различается ТОЛЬКО источник
сайзинга на hero raise-узлах. Оракул платит обходом только там.

ИСХОДЫ:
  oracle уверенно бьёт uniform -> таргеты хороши, чинить Q-инъекцию и механику обучения
  паритет                      -> регреты бесполезны, sizing_cf_regrets в переделку
  oracle проигрывает uniform   -> таргеты вредны (ср. raise_freq 0.365 -> 0.142 в v8)
  (1) против (2)               -> прямая цена Q-усадки

Использование:
    python tools/sizing_oracle_ab.py -c models/test107v9/multi_checkpoint_iter_5.pt --hands 300

УТЕЧКА БОРДА (починено). Роллаут оракула, продолжающий реальную раздачу, видел
ИСТИННЫЙ будущий борд: колода целиком фиксируется `State.from_seed`, и выбор анкера
получался с подглядыванием в ранаут. Теперь каждый роллаут идёт из пересобранного
через `from_mid_hand` узла с независимо перемешанной остаточной колодой: те же карты,
ставки и улица, но борд дораздаётся из независимого шафла. Внутри узла все анкеры
видят ОДИН И ТО ЖЕ независимый борд (общий по индексу роллаута) — сравнение анкеров
остаётся низкодисперсным, а корреляция с истинным ранаутом исчезает. Старый режим
доступен флагом `--keep-deal-board` (только для воспроизведения старых замеров).

КОНТРОЛЬНЫЕ РУКИ (положительный контроль гейта). После починки утечки оракул
оценивает анкер одним роллаутом по случайному борду и перестал быть верхней границей,
поэтому ноль всех рук против uniform сам по себе ничего не доказывает. Руки
`min_available` / `max_available` льют рейз всегда в первый/последний доступный
анкер (диаметральны на своей строке, без фолбэков). Если заведомо плохая рука не
отличается от uniform — инструмент не видит сайзинг вообще, и вопрос «учится ли
сайзинг» этим гейтом неразрешим. Отдельно печатаются парные разницы по подвыборкам:
(1) раздачи с hero-рейзом; (2) раздачи, где на hero raise-узле доступно >= 2
анкеров (K_avail >= 2) — на остальных узлах обе руки выбирают один и тот же анкер
и разбавляют диапазон нулём.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_INVOCATION_CWD = Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pokers as pkrs  # noqa: E402

from src.core.deep_cfr import DeepCFRAgent  # noqa: E402
from src.core.sizing import regret_matching_anchors  # noqa: E402

ARMS = ("oracle_mc", "oracle_mc_q", "uniform", "blueprint", "min_available", "max_available")


def redeal_remaining_deck(state: Any, shuffle_rng: random.Random) -> Any:
    """Пересобирает узел с независимо перемешанной остаточной колодой.

    Движок тянет борд из `state.deck` по порядку, а колода целиком фиксируется
    `State.from_seed`. Поэтому роллаут, продолжающий реальную раздачу, видит
    истинный будущий борд. `from_mid_hand` принимает колоду отдельно: передаём
    те же карты/ставки/улицу, но перетасованный хвост. `min_bet` конструктор
    восстанавливает как max(bet_chips) — в движке эти величины совпадают во
    всех достижимых состояниях (рейз всегда поднимает min_bet до своей ставки,
    смена улицы обнуляет обе).
    """
    if state.final_state or not state.deck:
        return state
    deck = list(state.deck)
    shuffle_rng.shuffle(deck)
    players = state.players_state
    stake = max(p.stake + p.bet_chips + p.pot_chips for p in players)
    return pkrs.State.from_mid_hand(
        n_players=len(players),
        button=int(state.button),
        sb=float(state.sb),
        bb=float(state.bb),
        stake=float(stake),
        deck=deck,
        hole_cards=[tuple(p.hand) for p in players],
        public_cards=list(state.public_cards),
        stage=state.stage,
        pot=float(state.pot),
        bet_chips=[float(p.bet_chips) for p in players],
        pot_chips=[float(p.pot_chips) for p in players],
        active=[bool(p.active) for p in players],
        last_stage_action=[p.last_stage_action for p in players],
        current_player=int(state.current_player),
        last_raise_increment=float(state.last_raise_increment),
    )


def _resolve_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    from_invocation = (_INVOCATION_CWD / candidate).resolve()
    if from_invocation.is_file():
        return from_invocation
    return (ROOT / candidate).resolve()


class OracleSizing:
    """Политика сайзинга из свежих MC-значений анкеров.

    На hero raise-узле обходит доступные анкеры, для каждого играет раздачу до конца
    блюпринтом и берёт reward. Регрет анкера = value(anchor) - EV, политика = regret matching.
    Это те же величины, что попадают в sizing_advantage_buffer, но БЕЗ Q-инъекции:
    каждый анкер получает собственное MC-значение, а не предсказание сети.
    """

    def __init__(
        self,
        agent: DeepCFRAgent,
        rollouts: int,
        max_anchors: int,
        use_q_for_uncredited: bool,
        min_prob: float,
        depth_cap: int,
        ev_mode: str = "weighted",
        redealt_board: bool = True,
    ) -> None:
        self.agent = agent
        self.rollouts = rollouts
        self.max_anchors = max_anchors
        self.use_q = use_q_for_uncredited
        self.min_prob = min_prob
        self.depth_cap = depth_cap
        self.ev_mode = ev_mode
        self.redealt_board = redealt_board
        self.nodes_evaluated = 0
        self.rollouts_done = 0
        self.redeals = 0
        self.redeal_failures = 0

    def _slot_weights(self, state: Any, hero: int) -> np.ndarray:
        """Стратегия анкеров (regret-matching по advantage_sizing_net), как в конвейере."""
        encoded = self.agent._encode_state_for_sizing(state, hero)
        tensor = torch.from_numpy(np.asarray(encoded, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            logits = self.agent.advantage_sizing_net(tensor)[0].cpu().numpy()
        return regret_matching_anchors(logits.astype(np.float32), min_prob=self.min_prob)

    def _rollout(self, state: Any, hero: int, rng: random.Random, board_seed: int | None = None) -> float:
        """Догоняет раздачу до конца блюпринтом, возвращает reward героя.

        Роллауты идут через choose_action, которая сэмплит из ГЛОБАЛЬНОГО np.random
        (`deep_cfr.py:3461-3528`). Без изоляции состояния оракул сдвигал бы глобальный поток
        и ломал CRN: остальные руки играли бы уже другими случайными числами.

        Каждый роллаут получает свой seed из локального `rng`, поэтому при `rollouts > 1`
        траектории независимы (иначе восстановление np.random-state делало все роллауты
        идентичными и денойзинг через усреднение не работал).

        `board_seed` — независимый шафл остаточной колоды (антимтия с истинным бордом):
        все анкеры одного узла получают одинаковый борд на одинаковом индексе роллаута.
        """
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        np.random.seed(rng.getrandbits(32))
        try:
            current = state
            if board_seed is not None:
                try:
                    current = redeal_remaining_deck(state, random.Random(board_seed))
                    self.redeals += 1
                except Exception:
                    self.redeal_failures += 1
            for _ in range(self.depth_cap):
                if current.final_state:
                    break
                action = self.agent.choose_action(current, player_id=int(current.current_player))
                nxt = current.apply_action(action)
                if nxt.status != pkrs.StateStatus.Ok:
                    return float(current.players_state[hero].reward)
                current = nxt
            self.rollouts_done += 1
            return float(current.players_state[hero].reward)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    def policy(self, state: Any, hero: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает (probs[K], available_mask[K]) по фиксированной сетке анкеров."""
        available, _ = self.agent._anchor_availability(state)
        indices = np.flatnonzero(available)
        anchors = self.agent.anchors_arr
        values = np.full(len(anchors), np.nan, dtype=np.float64)

        if len(indices) == 0:
            return np.zeros(len(anchors), dtype=np.float64), available

        chosen = indices
        if len(indices) > self.max_anchors:
            chosen = np.asarray(
                sorted(rng.sample(list(indices), self.max_anchors)), dtype=np.int64
            )

        # Один набор бордов на узел: анкер A и анкер B на индексе роллаута r видят
        # ОДИН и тот же независимый борд. Сравнение анкеров остаётся парным, а
        # корреляция с истинным ранаутом раздачи исчезает. Оба экземпляра оракула
        # (pure/with_q) тянут те же seed'ы из одинаковых локальных потоков, поэтому
        # их борды совпадают и рук-to-рук сравнение остаётся CRN-парным.
        if self.redealt_board:
            board_seeds = [rng.getrandbits(32) for _ in range(self.rollouts)]
        else:
            board_seeds = [None] * self.rollouts

        self.nodes_evaluated += 1
        for index in chosen:
            action = self.agent.action_type_to_pokers_action(3, state, float(anchors[index]))
            nxt = state.apply_action(action)
            if nxt.status != pkrs.StateStatus.Ok:
                continue
            total = 0.0
            for rollout_index in range(self.rollouts):
                total += self._rollout(nxt, hero, rng, board_seed=board_seeds[rollout_index])
            values[index] = total / self.rollouts

        evaluated = ~np.isnan(values)
        if not evaluated.any():
            probs = np.zeros(len(anchors), dtype=np.float64)
            probs[indices] = 1.0 / len(indices)
            return probs, available

        if self.use_q:
            # Воспроизводит текущий конвейер: некредитуемые анкеры берут значение от Q-сети.
            encoded = self.agent._encode_state_for_sizing(state, hero)
            tensor = torch.from_numpy(np.asarray(encoded, dtype=np.float32)).unsqueeze(0)
            q_values = np.asarray(
                self.agent._evaluate_sizing_q_for_sizes(tensor, anchors), dtype=np.float64
            )
            filled = np.where(evaluated, values, q_values)
        else:
            # Строгая версия: только реально обойдённые анкеры.
            filled = values.copy()

        usable = available & (evaluated if not self.use_q else np.ones_like(available))
        if not usable.any():
            usable = available
        subset = np.flatnonzero(usable)
        if self.ev_mode == "weighted":
            slot_weights = self._slot_weights(state, hero)
            w = slot_weights[subset]
            ev = float((w * filled[subset]).sum() / max(float(w.sum()), 1e-12))
        else:
            ev = float(np.nanmean(filled[subset]))
        regrets = np.nan_to_num(filled[subset] - ev, nan=0.0)
        local = regret_matching_anchors(regrets.astype(np.float32), min_prob=self.min_prob)

        probs = np.zeros(len(anchors), dtype=np.float64)
        probs[subset] = local
        return probs, available


def _uniform_policy(agent: DeepCFRAgent, state: Any) -> np.ndarray:
    available, _ = agent._anchor_availability(state)
    indices = np.flatnonzero(available)
    probs = np.zeros(len(agent.anchors_arr), dtype=np.float64)
    if len(indices):
        probs[indices] = 1.0 / len(indices)
    return probs


def _extreme_anchor_policy(
    agent: DeepCFRAgent, state: Any, mode: str, anchor_tracker: dict, differ_tracker: dict
) -> np.ndarray:
    """Всегда первый или последний доступный анкер: положительный контроль гейта.

    min_available = первый доступный анкер (indices[0]),
    max_available = последний доступный анкер (indices[-1]).
    Фолбэков нет по построению — руки всегда диаметральны на своей строке.
    differ_tracker[mode] поднимается, когда доступных анкеров >= 2: только на
    таких узлах руки реально различаются (K_avail >= 2, фильтр разбавления).
    """
    available, _ = agent._anchor_availability(state)
    indices = np.flatnonzero(available)
    probs = np.zeros(len(agent.anchors_arr), dtype=np.float64)
    if len(indices):
        if mode == "min":
            idx = int(indices[0])
        else:
            idx = int(indices[-1])
        probs[idx] = 1.0
        anchor_tracker[mode].append(float(agent.anchors_arr[idx]))
        if len(indices) >= 2:
            differ_tracker[mode] = True
    return probs


def play_hand(
    agent: DeepCFRAgent,
    hero: int,
    seed: int,
    button: int,
    num_players: int,
    sizing_source: Callable[[Any, random.Random], np.ndarray] | None,
    rng: random.Random,
    depth_cap: int,
) -> tuple[float, int]:
    """Одна раздача. sizing_source=None -> блюпринт (choose_action без вмешательства).

    Действие (рейз/чек/фолд) всегда выбирает `choose_action`; рука (`sizing_source`) влияет
    только на РАЗМЕР рейза, когда выпал рейз. Это делает четыре руки сравнимыми по частоте
    рейза и изолирует вклад сайзинга (не решения «рейзить или нет»).
    """
    state = pkrs.State.from_seed(
        n_players=num_players, button=button, sb=1, bb=2, stake=200.0, seed=seed
    )
    raises = 0
    for _ in range(depth_cap):
        if state.final_state:
            break
        player = int(state.current_player)
        action = agent.choose_action(state, player_id=player)
        if player == hero and sizing_source is not None and action.action == pkrs.ActionEnum.Raise:
            probs = sizing_source(state, rng)
            total = probs.sum()
            if total > 1e-9:
                probs = probs / total
                index = int(rng.choices(range(len(probs)), weights=probs, k=1)[0])
                action = agent.action_type_to_pokers_action(
                    3, state, float(agent.anchors_arr[index])
                )
        if player == hero and action.action == pkrs.ActionEnum.Raise:
            raises += 1
        nxt = state.apply_action(action)
        if nxt.status != pkrs.StateStatus.Ok:
            break
        state = nxt
    return float(state.players_state[hero].reward), raises


def paired_stats(values: list[float]) -> tuple[float, float, float]:
    """Среднее, стандартная ошибка и t-статистика парных разниц."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    stderr = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, stderr, mean / stderr if stderr > 1e-12 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("--hands", type=int, default=2000)
    parser.add_argument("--rollouts", type=int, default=1,
                        help="MC-роллаутов на анкер (CRN уже гасит дисперсию)")
    parser.add_argument("--max-anchors", type=int, default=6,
                        help="сколько доступных анкеров обходить на узле")
    parser.add_argument("--num-players", type=int, default=6)
    parser.add_argument("--hero", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--depth-cap", type=int, default=200)
    parser.add_argument("--ev-mode", choices=("mean", "weighted"), default="weighted",
                        help="как оракул считает ev: mean (простое среднее) или weighted (по стратегии, как конвейер)")
    parser.add_argument("--keep-deal-board", action="store_true",
                        help="СТАРЫЙ РЕЖИМ: роллауты оракула видят истинный борд раздачи (утечка). "
                             "Только для воспроизведения старых замеров.")
    parser.add_argument("-o", "--output", help="путь к текстовому отчёту (по умолчанию рядом с чекпоинтом)")
    args = parser.parse_args()

    torch.set_num_threads(1)
    path = _resolve_path(args.checkpoint)
    out_path = Path(args.output) if args.output else path.with_name(path.stem + "_oracle_ab.txt")

    lines: list[str] = []
    started = time.time()

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"чекпоинт: {path}")
    emit(f"отчёт: {out_path}")

    agent = DeepCFRAgent(player_id=args.hero, num_players=args.num_players, device="cpu")
    agent._load_checkpoint(str(path))
    iteration = int(getattr(agent, "iteration_count", 0) or 0)
    min_prob = agent._current_sizing_min_prob(iteration)
    emit(f"iteration={iteration}  min_prob={min_prob:.4f}  "
         f"strategy_sizing_net_loaded={getattr(agent, 'strategy_sizing_net_loaded', False)}")
    emit(f"раздач={args.hands}  роллаутов/анкер={args.rollouts}  "
         f"анкеров/узел<={args.max_anchors}  ev_mode={args.ev_mode}  seed={args.seed}")
    emit(f"борд в роллаутах: "
         + ("хвост from_seed (СТАРЫЙ РЕЖИМ, утечка)" if args.keep_deal_board
            else "независимый шафл остаточной колоды (утечка починена)"))
    emit()

    oracle_pure = OracleSizing(agent, args.rollouts, args.max_anchors, False, min_prob,
                               args.depth_cap, args.ev_mode, not args.keep_deal_board)
    oracle_with_q = OracleSizing(agent, args.rollouts, args.max_anchors, True, min_prob,
                                 args.depth_cap, args.ev_mode, not args.keep_deal_board)

    sources: dict[str, Callable[[Any, random.Random], np.ndarray] | None] = {
        "oracle_mc": lambda state, rng: oracle_pure.policy(state, args.hero, rng)[0],
        "oracle_mc_q": lambda state, rng: oracle_with_q.policy(state, args.hero, rng)[0],
        "uniform": lambda state, rng: _uniform_policy(agent, state),
        "blueprint": None,
    }
    anchor_tracker: dict[str, list[float]] = {"min": [], "max": []}
    differ_current: dict[str, bool] = {"min": False, "max": False}
    sources["min_available"] = lambda state, rng: _extreme_anchor_policy(
        agent, state, "min", anchor_tracker, differ_current)
    sources["max_available"] = lambda state, rng: _extreme_anchor_policy(
        agent, state, "max", anchor_tracker, differ_current)

    rewards: dict[str, list[float]] = {arm: [] for arm in ARMS}
    raise_counts: dict[str, int] = {arm: 0 for arm in ARMS}
    raise_flags: dict[str, list[bool]] = {arm: [] for arm in ARMS}
    differ_flags: dict[str, list[bool]] = {"min_available": [], "max_available": []}

    for hand in range(args.hands):
        seed = args.seed * 1_000_003 + hand
        button = hand % args.num_players
        for arm in ARMS:
            # CRN: одинаковые карты (seed раздачи), одинаковый локальный поток И одинаковое
            # состояние глобального np.random — choose_action:3461-3528 сэмплит через него,
            # без этого руки расходятся после первого же решения оппонента.
            differ_current["min"] = False
            differ_current["max"] = False
            rng = random.Random(seed * 31 + 7)
            np.random.seed(seed % (2 ** 31 - 1))
            random.seed(seed * 17 + 3)
            reward, raises = play_hand(
                agent, args.hero, seed, button, args.num_players,
                sources[arm], rng, args.depth_cap,
            )
            rewards[arm].append(reward)
            raise_counts[arm] += raises
            raise_flags[arm].append(raises > 0)
            if arm == "min_available":
                differ_flags["min_available"].append(differ_current["min"])
            elif arm == "max_available":
                differ_flags["max_available"].append(differ_current["max"])
        if (hand + 1) % max(args.hands // 10, 1) == 0:
            emit(f"  ...{hand + 1}/{args.hands} раздач")

    emit()
    header = f"{'рука':>12} {'ср.профит':>11} {'stderr':>9} {'raises':>8}"
    emit(header)
    emit("-" * len(header))
    for arm in ARMS:
        mean, stderr, _ = paired_stats(rewards[arm])
        emit(f"{arm:>12} {mean:>11.3f} {stderr:>9.3f} {raise_counts[arm]:>8}")

    emit()
    emit("ПАРНЫЕ РАЗНИЦЫ (одни раздачи, CRN). |t| >= 2 = значимо")
    comparisons = (
        ("oracle_mc", "uniform", "качество таргетов БЕЗ Q"),
        ("oracle_mc_q", "uniform", "качество таргетов С Q-инъекцией"),
        ("oracle_mc", "oracle_mc_q", "ЦЕНА Q-УСАДКИ"),
        ("blueprint", "uniform", "сила бота: сеть против равномерной"),
        ("oracle_mc", "blueprint", "потолок: сколько теряет сеть"),
        ("min_available", "uniform", "КОНТРОЛЬ: мин-доступный vs uniform"),
        ("max_available", "uniform", "КОНТРОЛЬ: макс-доступный vs uniform"),
        ("min_available", "max_available", "ДИАПАЗОН: min vs max доступный"),
    )
    width = max(len(f"{a} - {b}") for a, b, _ in comparisons)
    for left, right, label in comparisons:
        diff = [x - y for x, y in zip(rewards[left], rewards[right])]
        mean, stderr, tstat = paired_stats(diff)
        verdict = "значимо" if abs(tstat) >= 2 else "шум"
        emit(f"  {f'{left} - {right}':<{width}}  {mean:>+8.3f} +/- {stderr:>6.3f}  "
             f"t={tstat:>+6.2f}  {verdict:<8} {label}")

    emit()
    emit("ПОДВЫБОРКА: только раздачи с hero-рейзом (сайзинг влияет лишь на них)")
    for left, right, label in comparisons:
        diff = [x - y for i, (x, y) in enumerate(zip(rewards[left], rewards[right]))
                if raise_flags[left][i] or raise_flags[right][i]]
        n_sub = len(diff)
        mean, stderr, tstat = paired_stats(diff)
        verdict = "значимо" if abs(tstat) >= 2 else "шум"
        emit(f"  {f'{left} - {right}':<{width}}  {mean:>+8.3f} +/- {stderr:>6.3f}  "
             f"t={tstat:>+6.2f}  {verdict:<8} n={n_sub}  {label}")
    emit()
    emit("ПОДВЫБОРКА: раздачи с hero raise-узлом K_avail >= 2 (руки реально различались)")
    k2_flags = [differ_flags["min_available"][i] or differ_flags["max_available"][i]
                for i in range(args.hands)]
    emit(f"  раздач с K_avail >= 2: {sum(k2_flags)} / {args.hands} "
         f"(на остальных min и max доступный анкер совпадают, вклад в разницу нулевой)")
    for left, right, label in comparisons:
        diff = [x - y for i, (x, y) in enumerate(zip(rewards[left], rewards[right]))
                if k2_flags[i]]
        n_sub = len(diff)
        mean, stderr, tstat = paired_stats(diff)
        verdict = "значимо" if abs(tstat) >= 2 else "шум"
        emit(f"  {f'{left} - {right}':<{width}}  {mean:>+8.3f} +/- {stderr:>6.3f}  "
             f"t={tstat:>+6.2f}  {verdict:<8} n={n_sub}  {label}")
    emit()
    avg_min = float(np.mean(anchor_tracker["min"])) if anchor_tracker["min"] else float("nan")
    avg_max = float(np.mean(anchor_tracker["max"])) if anchor_tracker["max"] else float("nan")
    emit(f"контрольные руки: фактический средний anchor_mult: "
         f"min_available={avg_min:.3f} (n={len(anchor_tracker['min'])}), "
         f"max_available={avg_max:.3f} (n={len(anchor_tracker['max'])})")

    emit()
    emit(f"время прогона: {time.time() - started:.0f} с")
    emit(f"oracle узлов обойдено: pure={oracle_pure.nodes_evaluated} "
         f"with_q={oracle_with_q.nodes_evaluated}, "
         f"роллаутов={oracle_pure.rollouts_done + oracle_with_q.rollouts_done}, "
         f"пересборок борда={oracle_pure.redeals + oracle_with_q.redeals} "
         f"(сбоев={oracle_pure.redeal_failures + oracle_with_q.redeal_failures})")
    emit()
    emit("ЧТЕНИЕ: oracle_mc >> uniform -> таргеты хороши, чинить Q и механику обучения")
    emit("        паритет              -> регреты бесполезны, sizing_cf_regrets в переделку")
    emit("        oracle_mc << uniform -> таргеты вредны")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Отчёт сохранён: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
