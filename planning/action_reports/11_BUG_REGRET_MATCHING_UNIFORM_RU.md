# БАГ-РЕПОРТ #11: REGRET MATCHING ВЫРОЖДАЕТСЯ В «ФОЛДИ ВСЁ» ПРИ ОТРИЦАТЕЛЬНЫХ REGRETS
# Дата: 04.05.2026
# Критичность: КРИТИЧЕСКАЯ (корневая причина провала обучения против чекпоинтов)
# Статус: ИСПРАВЛЕН

================================================================================
ПРОБЛЕМА
================================================================================

Когда все advantage-значения (регреты) для legal actions отрицательные,
текущий код назначает **uniform** стратегию:

    if sum(advantages_masked) > 0:
        strategy = advantages_masked / sum(advantages_masked)
    else:
        strategy[a] = 1.0 / len(legal_action_types)   # UNIFORM

Оригинальная статья Deep CFR (Brown et al., 2018, Algorithm 3, стр. 164)
делает **иначе**:

    «If sum = 0 then each action is assigned equal probability, but in this
     paper we choose the action with highest regret with probability 1,
     which empirically helps RM better cope with approximation error»

Т.е. при все ≤ 0: **argmax с вероятностью 1**, а НЕ uniform.

================================================================================
ПОЧЕМУ ЭТО КРИТИЧЕСКИ
================================================================================

При игре против сильного оппонента (checkpoint) большинство действий дают
отрицательный EV. Типичная картина:

    Fold:  advantage = -1.0   (потерял блайнд, но минимум потерь)
    Call:  advantage = -8.0   (уравнял, проиграл на флопе)
    Raise: advantage = -12.0  (рейзнул, ре-рейзнули, упал)

Uniform стратегия: σ = [1/3, 1/3, 1/3]
→ Fold «выигрывает» 1/3 лотерею — но Fold — наименее отрицательный
→ НО: 2/3 раз агент выбирает Call/Raise, которые дают БОЛЬШЕ потерь
→ action_values[Raise] сильно отрицательный → regret[Raise] = Raise_ev - EV
   где EV учитывает все три действия → regret[Raise] может быть ещё хуже
→ На следующей итерации: ещё больше отрицательных advantage → порочный круг

Argmax стратегия: σ = [1.0, 0, 0]
→ Агент ВСЕГДА выбирает Fold (наименее отрицательный)
→ action_values[Fold] = -1.0, но Call и Raise НЕ обходятся
   → их action_values остаются 0 (по умолчанию)
→ regret[Call] = 0 - (-1.0) = +1.0  ← ПОЛОЖИТЕЛЬНЫЙ!
→ regret[Raise] = 0 - (-1.0) = +1.0 ← ПОЛОЖИТЕЛЬНЫЙ!
→ На следующей итерации: сеть видит положительный regret для Call/Raise
   → regret matching начинает их использовать → выход из «фолди всё»

Ключевая разница:
- Uniform: все действия обходятся → все отрицательные → нет положительного сигнала
- Argmax: непосещённые действия получают action_value=0 → нулевой minus отрицательный = ПОЛОЖИТЕЛЬНЫЙ regret → сеть учится пробовать другие действия

================================================================================
ВЛИЯНИЕ НА ТРЕНИРОВКУ
================================================================================

Uniform (БАГ):
  Итерация 1: все regret отрицательные → uniform → все обходятся → все отрицательные
  Итерация 2: сеть предсказывает все отрицательные → uniform → ...
  Итерация N: сеть сошлась к «всё отрицательно, фолди всё» → ПРОБУЖДЕНИЯ НЕТ

Argmax (ФИКС):
  Итерация 1: все regret отрицательные → argmax(Fold) → Fold=−1, Call=0, Raise=0
  Итерация 2: regret(Call)=+1, regret(Raise)=+1 → regret matching пробует их
  Итерация 3+: сеть начинает различать состояния → стратегия улучшается

================================================================================
ФАЙЛЫ
================================================================================

src/core/deep_cfr.py (строки 294-299) — cfr_traverse
src/training/train.py (строки 131-135) — _cfr_traverse_with_opponents

================================================================================
ИСПРАВЛЕНИЕ
================================================================================

--- src/core/deep_cfr.py ---

  if sum(advantages_masked) > 0:
      strategy = advantages_masked / sum(advantages_masked)
  else:
-     strategy = np.zeros(self.num_actions)
-     for a in legal_action_types:
-         strategy[a] = 1.0 / len(legal_action_types)
+     strategy = np.zeros(self.num_actions)
+     best_a = max(legal_action_types, key=lambda a: advantages[a])
+     strategy[best_a] = 1.0

--- src/training/train.py ---

  if advantages_masked.sum() > 0:
      strategy = advantages_masked / advantages_masked.sum()
  else:
-     strategy = np.zeros(agent.num_actions)
-     for action_type in legal_action_types:
-         strategy[action_type] = 1.0 / len(legal_action_types)
+     strategy = np.zeros(agent.num_actions)
+     best_a = max(legal_action_types, key=lambda a: advantages[a])
+     strategy[best_a] = 1.0

================================================================================
ССЫЛКА НА ОРИГИНАЛ
================================================================================

Brown, Lerer, Gross, Sandholm. "Deep Counterfactual Regret Minimization",
NeurIPS 2018. Algorithm 3 (Infoset Strategy Computation), строка 164:

  «If sum = 0 then each action is assigned equal probability, but in this
   paper we choose the action with highest regret with probability 1,
   which empirically helps RM better cope with approximation error»

Файл: C:\Users\Cassmall\Desktop\DeepBrown\main.tex, строки 292-310
