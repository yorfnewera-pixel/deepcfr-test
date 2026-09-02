import numpy as np


def regret_matching_anchors(regrets, min_prob=0.0125):
    """Regret matching по sizing-анкорам с floor для исследования."""
    regrets_arr = np.asarray(regrets, dtype=np.float32)
    if regrets_arr.ndim != 1 or regrets_arr.size == 0:
        raise ValueError("regrets должен быть непустым одномерным массивом")

    positive = np.maximum(regrets_arr, 0.0)
    total = float(positive.sum())
    if total <= 1e-8:
        return np.full(regrets_arr.size, 1.0 / regrets_arr.size, dtype=np.float32)

    floor = float(min_prob)
    if floor < 0.0:
        raise ValueError("min_prob не может быть отрицательным")
    floor = min(floor, 1.0 / regrets_arr.size)
    probs = positive / total
    probs = probs * (1.0 - floor * regrets_arr.size) + floor
    probs = probs / max(float(probs.sum()), 1e-8)
    return probs.astype(np.float32)


def sample_waugh_sizing(probs, anchors):
    """Waugh-интерполяция: один random U выбирает интервал и точку внутри него."""
    probs_arr = np.asarray(probs, dtype=np.float32)
    anchors_arr = np.asarray(anchors, dtype=np.float32)
    if probs_arr.ndim != 1 or anchors_arr.ndim != 1:
        raise ValueError("probs и anchors должны быть одномерными")
    if probs_arr.size != anchors_arr.size or anchors_arr.size < 2:
        raise ValueError("количество probs должно совпадать с anchors и быть >= 2")

    interval_weights = 0.5 * (probs_arr[:-1] + probs_arr[1:])
    total = float(interval_weights.sum())
    if total <= 1e-8:
        return float(np.sum(probs_arr * anchors_arr) / max(float(probs_arr.sum()), 1e-8))

    u = float(np.random.random()) * total
    cumulative = 0.0
    for idx, weight in enumerate(interval_weights):
        if weight <= 0.0:
            continue
        next_cumulative = cumulative + float(weight)
        if u <= next_cumulative:
            local = (u - cumulative) / max(float(weight), 1e-8)
            return float(anchors_arr[idx] + local * (anchors_arr[idx + 1] - anchors_arr[idx]))
        cumulative = next_cumulative
    return float(anchors_arr[-1])


def credit_assignment(size, anchors, regret):
    """Распределяет regret sampled sizing по двум соседним анкорам."""
    anchors_arr = np.asarray(anchors, dtype=np.float32)
    if anchors_arr.ndim != 1 or anchors_arr.size < 2:
        raise ValueError("anchors должен содержать минимум две точки")
    size = float(size)
    regret = float(regret)

    if size <= float(anchors_arr[0]):
        return regret, 0.0, 0, 1
    if size >= float(anchors_arr[-1]):
        last = anchors_arr.size - 1
        return 0.0, regret, last - 1, last

    right_idx = int(np.searchsorted(anchors_arr, size, side="right"))
    left_idx = right_idx - 1
    left_anchor = float(anchors_arr[left_idx])
    right_anchor = float(anchors_arr[right_idx])
    leverage = (size - left_anchor) / max(right_anchor - left_anchor, 1e-8)
    return regret * (1.0 - leverage), regret * leverage, left_idx, right_idx


def compute_sizing_heat_weights(q_values, temperature=0.35, min_advantage=0.03, top_p=0.90):
    """Q-heat weighting: advantage → weight.

    Bug #50: используется для фиксированной сетки сайзингов.

    Args:
        q_values: np.array [K] — Q(state, size_i) для каждого фиксированного бакета
        temperature: softmax temperature
        min_advantage: порог — ниже него heat = 0
        top_p: если задан, отсекает хвосты распределения
    Returns:
        weights: np.array [K] — нормализованные веса
    """
    q_arr = np.asarray(q_values, dtype=np.float32)
    baseline = np.mean(q_arr)
    adv = q_arr - baseline
    heat = np.maximum(adv - min_advantage, 0.0)
    if heat.sum() < 1e-8:
        exp_q = np.exp(q_arr / max(temperature, 1e-8))
        weights = exp_q / (exp_q.sum() + 1e-8)
    else:
        weights = heat / (heat.sum() + 1e-8)

    if top_p < 1.0:
        sorted_idx = np.argsort(weights)[::-1]
        cumsum = np.cumsum(weights[sorted_idx])
        cutoff = np.searchsorted(cumsum, top_p * cumsum[-1]) + 1
        keep = sorted_idx[:cutoff]
        mask = np.zeros_like(weights)
        mask[keep] = 1.0
        weights = weights * mask
        weights = weights / (weights.sum() + 1e-8)

    return weights.astype(np.float32)
