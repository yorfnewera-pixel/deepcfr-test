#!/usr/bin/env python3
"""S6-декомпозиция: выучила ли sizing-голова что-то кроме маргиналов по анкерам.

ВОПРОС. Гейт 32k показал: сигнал сайзинга в игре есть (always_min − always_max
≈ +0.65 фишки/рука, реплицировано), но блюпринт ≈ uniform. Два несовместимых диагноза:
  (а) голова выучила только константные маргиналы по анкерам — состояние не используется;
  (б) голова не выучила ничего, даже маргиналов.
Разделаются одним офлайн-замером на holdout-строках буфера — без прогонов обучения.

МЕТОД (тот же сплит и таргет, что в _log_sizing_holdout_tv):
  маргинал    = средний логит advantage_sizing_net по НЕ-holdout строкам (credit>=2),
                политика = regret_matching_anchors(маргинал[avail], min_prob итерации)
  на каждой holdout-строке (credit>=2) считаются TV к одному таргету
                pt = regret_matching_anchors(regrets[i][avail], min_prob):
                TV(net), TV(marginal), TV(uniform),
  статистика — на ПАРНЫХ разницах по строкам, не на средних.

ЧТЕНИЕ:
  TV(net) − TV(marginal) в пределах шума  -> вклад состояния ноль; чинить вход/представление
  TV(net) заметно лучше TV(marginal)      -> состояние-зависимость есть, но мелкая; вопрос мощности
  TV(marginal) лучше TV(net)              -> голова активно вредит, искать рассогласование слотов
КОНТРОЛЬ: TV(marginal) обязан быть не хуже TV(uniform) (парная разница <= 0),
иначе маргиналы посчитаны неверно (порядок анкеров / маска).
Дополнительно: target-marginal (regret matching от средних regrets train-строк) —
нижняя граница для любой константной политики: «выучила ли голова хотя бы правильные
маргиналы». Отдельно печатаются n строк и TV(uniform).

Использование:
    python tools/diagnose_sizing_marginal.py -c models/test107v17/multi_checkpoint_iter_30.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_INVOCATION_CWD = Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.core.deep_cfr import DeepCFRAgent  # noqa: E402
from src.core.sizing import regret_matching_anchors  # noqa: E402


def _resolve_path(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    from_invocation = (_INVOCATION_CWD / candidate).resolve()
    if from_invocation.is_file():
        return from_invocation
    return (ROOT / candidate).resolve()


def paired_stats(values):
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, float("nan"), float("nan")
    stderr = float(arr.std(ddof=1) / math.sqrt(arr.size))
    t = mean / stderr if stderr > 1e-12 else float("nan")
    return mean, stderr, t


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    torch.set_num_threads(1)
    path = _resolve_path(args.checkpoint)
    out_path = Path(args.output) if args.output else path.with_name(path.stem + "_marginal_tv.txt")

    lines = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"чекпоинт: {path}")
    agent = DeepCFRAgent(player_id=0, num_players=6, device="cpu")
    agent._load_checkpoint(str(path))
    iteration = int(getattr(agent, "iteration_count", 0) or 0)
    min_prob = agent._current_sizing_min_prob(iteration)
    buf = agent.sizing_advantage_buffer
    emit(f"iteration={iteration}  min_prob={min_prob:.4f}  строк в буфере={len(buf)}")

    n_buf = len(buf)
    masks = buf._masks[:n_buf].astype(bool)
    credits = buf._credit_masks[:n_buf]
    holdout = buf._holdout[:n_buf].astype(bool)
    mismatch = int((masks != (credits > 0.5)).any(axis=1).sum())
    emit(f"строк, где _masks != credit_mask: {mismatch} из {n_buf} "
         f"(0 = включённый sizing_credit_mask_enabled, как ожидалось)")

    credit_ge2 = credits.sum(axis=1) >= 2
    train_sel = (~holdout) & credit_ge2
    hold_sel = holdout & credit_ge2
    emit(f"train-строк для маргинала (не holdout, credit>=2): {int(train_sel.sum())}")
    emit(f"holdout-строк для оценки (holdout, credit>=2):     {int(hold_sel.sum())}")

    net = agent.advantage_sizing_net
    net.eval()
    states_train = buf._states[:n_buf][train_sel].astype(np.float32)
    with torch.no_grad():
        chunks = []
        for start in range(0, len(states_train), 4096):
            chunk = states_train[start:start + 4096]
            chunks.append(net(torch.from_numpy(chunk)).numpy())
    train_logits = np.concatenate(chunks, axis=0)
    marginal_logits = train_logits.mean(axis=0)
    emit(f"маргинальные логиты (среднее по {len(states_train)} train-строкам):")
    emit("  " + "  ".join(f"{v:+.3f}" for v in marginal_logits))

    # Таргет-маргинал: regret matching от средних regrets train-строк (нижняя граница
    # константной политики по этому таргету). Индексируется маской строки как вектор.
    mean_train_regrets = buf._regrets[:n_buf][train_sel].astype(np.float64).mean(axis=0)
    target_marginal_logits = mean_train_regrets.astype(np.float32)

    states = buf._states[:n_buf][hold_sel].astype(np.float32)
    regrets = buf._regrets[:n_buf][hold_sel].astype(np.float64)
    avail_all = masks[hold_sel]
    with torch.no_grad():
        logits = []
        for start in range(0, len(states), 4096):
            chunk = states[start:start + 4096]
            logits.append(net(torch.from_numpy(chunk)).numpy())
    logits = np.concatenate(logits, axis=0)

    tv_net, tv_marginal, tv_uniform, tv_tmarginal = [], [], [], []
    target_prob_sum = np.zeros(logits.shape[1], dtype=np.float64)
    uniform_prob_sum = np.zeros(logits.shape[1], dtype=np.float64)
    target_prob_cnt = np.zeros(logits.shape[1], dtype=np.int64)
    k_counts = []
    target_is_uniform_rows = 0
    for i in range(len(states)):
        avail = avail_all[i]
        pt = regret_matching_anchors(regrets[i][avail].astype(np.float32), min_prob=min_prob)
        pn = regret_matching_anchors(logits[i][avail].astype(np.float32), min_prob=min_prob)
        pm = regret_matching_anchors(marginal_logits[avail].astype(np.float32), min_prob=min_prob)
        ptm = regret_matching_anchors(target_marginal_logits[avail].astype(np.float32), min_prob=min_prob)
        u = np.full(avail.sum(), 1.0 / avail.sum())
        tv_net.append(0.5 * float(np.abs(pn - pt).sum()))
        tv_marginal.append(0.5 * float(np.abs(pm - pt).sum()))
        tv_uniform.append(0.5 * float(np.abs(u - pt).sum()))
        tv_tmarginal.append(0.5 * float(np.abs(ptm - pt).sum()))
        idxs = np.flatnonzero(avail)
        target_prob_sum[idxs] += pt
        uniform_prob_sum[idxs] += u
        target_prob_cnt[idxs] += 1
        k_counts.append(int(avail.sum()))
        if float(np.abs(pt - u).sum()) < 1e-6:
            target_is_uniform_rows += 1

    emit()
    header = f"{'политика':>16} {'TV ср.':>9} {'stderr':>8}"
    emit(header)
    emit("-" * len(header))
    for name, vals in (("net", tv_net), ("marginal", tv_marginal),
                       ("target-marg", tv_tmarginal), ("uniform", tv_uniform)):
        mean, stderr, _ = paired_stats(vals)
        emit(f"{name:>16} {mean:>9.4f} {stderr:>8.4f}")

    emit()
    emit(f"ПАРНЫЕ РАЗНИЦЫ ТВ по {len(states)} holdout-строкам. Ключевая — первая.")
    diffs = (
        ("net − marginal", [a - b for a, b in zip(tv_net, tv_marginal)],
         "вклад состояния: 0 = голова выдала только маргиналы"),
        ("net − uniform", [a - b for a, b in zip(tv_net, tv_uniform)],
         "сравнимость с RATIO из консоли"),
        ("marginal − uniform", [a - b for a, b in zip(tv_marginal, tv_uniform)],
         "КОНТРОЛЬ корректности: обязан быть <= 0"),
        ("net − target-marg", [a - b for a, b in zip(tv_net, tv_tmarginal)],
         "выучила ли голова даже правильные маргиналы"),
    )
    for name, values, label in diffs:
        mean, stderr, t = paired_stats(values)
        verdict = "значимо" if abs(t) >= 2 else "шум"
        emit(f"  {name:>20}  {mean:>+8.4f} +/- {stderr:>7.4f}  t={t:>+6.2f}  {verdict:<8} {label}")

    emit()
    emit("ДИАГНОЗ КОНТРОЛЯ: есть ли у таргетов глобальный наклон вообще.")
    covered = target_prob_cnt > 0
    mean_target_prob = np.divide(target_prob_sum, np.maximum(target_prob_cnt, 1))
    mean_uniform_prob = np.divide(uniform_prob_sum, np.maximum(target_prob_cnt, 1))
    emit(f"  среднее число доступных анкеров на строку: K = {float(np.mean(k_counts)):.2f}")
    emit("  ср. вероятность по анкеру (только строки, где анкер доступен):")
    emit("    anchor:   " + "  ".join(f"{i+1:>5}" for i in range(len(mean_target_prob))))
    emit("    target:   " + "  ".join(f"{v*100:>4.1f}%" for v in mean_target_prob))
    emit("    uniform:  " + "  ".join(f"{v*100:>4.1f}%" for v in mean_uniform_prob))
    emit(f"  строк с таргетом == uniform (все регреты <= 0): {target_is_uniform_rows} из {len(states)}")
    tilt = 0.5 * float(np.abs(
        mean_target_prob[covered] - mean_uniform_prob[covered]).sum())
    emit(f"  TV(средний таргет, средний uniform) = {tilt:.4f} — глобальный наклон таргетов;"
         f" ~0 = наклона нет, uniform лучшая константа, контроль плана неприменим")

    emit()
    emit("РАССОГЛАСОВАНИЕ СЛОТОВ (ветка 3): корреляция logit[a] с regret[b] по строкам,")
    emit("где оба анкера доступны. Диагональ обязана доминировать; систематический")
    emit("сдвиг = перепутанные индексы.")
    n_anchors = logits.shape[1]
    corr = np.full((n_anchors, n_anchors), np.nan)
    for a in range(n_anchors):
        for b in range(n_anchors):
            rows = avail_all[:, a] & avail_all[:, b]
            if rows.sum() < 100:
                continue
            x = logits[rows, a].astype(np.float64)
            y = regrets[rows, b].astype(np.float64)
            if x.std() < 1e-9 or y.std() < 1e-9:
                continue
            corr[a, b] = float(np.corrcoef(x, y)[0, 1])
    diag = np.diag(corr)
    emit("  диагональ (logit[a] ~ regret[a]): "
         + "  ".join(".." if np.isnan(v) else f"{v:+.2f}" for v in diag))
    valid = ~np.isnan(corr)
    diag_mean = float(np.nanmean(diag))
    offdiag = corr.copy()
    np.fill_diagonal(offdiag, np.nan)
    off_valid = ~np.isnan(offdiag)
    emit(f"  ср. диагональная корреляция = {diag_mean:+.3f}")
    emit(f"  ср. внецентровая             = {float(np.nanmean(offdiag)):+.3f}")
    best_off = np.unravel_index(int(np.argmax(np.where(off_valid, offdiag, -np.inf))), offdiag.shape)
    emit(f"  лучший внецентровый элемент: corr[{best_off[0]},{best_off[1]}] = "
         f"{offdiag[best_off]:+.3f}")
    worse = int(np.sum(offdiag[off_valid] > 0))
    emit(f"  внецентровых элементов > 0: {worse} из {int(off_valid.sum())} —"
         f" сколько пар анкеров коррелируют сильнее нуля (ожидаемо ~половина при шуме)")
    del valid

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nОтчёт сохранён: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
