# drm_module.py
"""
DRM 3.0 module
Zawiera:
- Rule (cechy wektorowe, historia)
- DRM3System (FRZ adaptive, emergence_pressure, curiosity, anti-stagnation)
- bezpieczne emergency_revival z checkpoint + rollback
- enforce_diversity oparty na wektorach cech (cosine similarity)
- metryki, logi, eksport CSV/JSON, proste ploty
- API do podawania external_reward (opcjonalne)
"""

import math
import random
import time
import json
import csv
import statistics
import math
from copy import deepcopy
from typing import Dict, Any, Callable, List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ---------------- Rule ----------------
class Rule:
    """Reguła z wektorem cech, wagą i historią siły."""
    def __init__(self, name: str, features: Optional[List[float]] = None, weight: float = 1.0):
        self.name = name
        self.features = features or [weight]  # feature vector; at least contains weight
        self.weight = float(weight)
        self.strength = 0.0
        self.usage_count = 0
        self.last_used_cycle = 0
        self.created_cycle = 0
        self.is_new = True
        self.history_strength: List[float] = []
        self.metadata: Dict[str, Any] = {}

    def experience_level(self, current_cycle: int, max_exp: int = 100) -> float:
        age = current_cycle - self.created_cycle
        exp = min(max_exp, age + self.usage_count)
        return exp / max_exp  # 0..1


# ---------------- DRM3System ----------------
class DRM3System:
    """Główny system DRM 3.0 z mechanizmami anty-stagnacji."""
    def __init__(
        self,
        FRZ: float = 45.0,
        stagnation_window: int = 5,
        death_window: int = 10,
        stagnation_change_thresh: float = 0.01,
        stable_threshold: float = 0.5,
        stable_cycles: int = 5,
        mutation_magnitude: float = 0.3,
        emergency_mutation_count: int = 3,
        seed: Optional[int] = None
    ):
        self.FRZ = float(FRZ)
        self.cycle = 0
        self.rules: Dict[str, Rule] = {}
        self.activity_history: List[float] = []
        self.cycles_since_new_rule = 9999
        self.stagnation_window = stagnation_window
        self.death_window = death_window
        self.stagnation_change_thresh = stagnation_change_thresh
        self.event_log: List[Tuple[int, str]] = []
        self.archive: List[Any] = []
        self.stable_threshold = stable_threshold
        self.stable_cycles = stable_cycles
        self.mutation_magnitude = mutation_magnitude
        self.emergency_mutation_count = emergency_mutation_count
        self._last_vitals: Dict[str, Any] = {}
        # checkpoint for rollback
        self._last_checkpoint: Optional[Dict[str, Any]] = None
        self._post_revival_eval_cycles = 5
        self._pending_rollback_check: Optional[Dict[str, Any]] = None
        if seed is not None:
            random.seed(seed)

    # ---------- rule management ----------
    def add_rule(self, name: str, features: Optional[List[float]] = None, weight: float = 1.0) -> bool:
        if name in self.rules:
            return False
        r = Rule(name=name, features=features or [weight], weight=weight)
        r.created_cycle = self.cycle
        self.rules[name] = r
        self.cycles_since_new_rule = 0
        self.event_log.append((self.cycle, f"add_rule:{name}"))
        return True

    def remove_rule(self, name: str) -> bool:
        if name in self.rules:
            del self.rules[name]
            self.event_log.append((self.cycle, f"remove_rule:{name}"))
            return True
        return False

    # ---------- utility similarity (feature vectors) ----------
    @staticmethod
    def _cosine_sim_vec(v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.hypot(*v1) if len(v1) > 1 else abs(v1[0])
        n2 = math.hypot(*v2) if len(v2) > 1 else abs(v2[0])
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    # ---------- core cycle ----------
    def run_cycle(self, external_reward_fn: Optional[Callable[[], float]] = None):
        """Uruchom jedną iterację cyklu systemu.
        optional external_reward_fn: funkcja zwracająca reward scalar (0..1) dla aktualnej puli reguł.
        """
        self.cycle += 1

        # cache vitals
        vitals = {}
        vitals['is_stagnation'] = self._detect_stagnation()
        vitals['is_death_spiral'] = self._detect_death_spiral()
        self._last_vitals = vitals

        # przed ewentualnym revival zapis checkpointu
        if vitals['is_stagnation'] or vitals['is_death_spiral']:
            self._create_checkpoint("pre_revival")

        # revival / diversity enforcement
        if vitals['is_death_spiral']:
            self._emergency_revival()
            # schedule rollback check after few cycles
            self._pending_rollback_check = {
                "checkpoint": deepcopy(self._last_checkpoint),
                "evaluate_after_cycle": self.cycle + self._post_revival_eval_cycles,
                "pre_avg": self.activity_history[-1] if self.activity_history else 0.0
            }
        elif vitals['is_stagnation']:
            self._enforce_diversity()

        # compute strengths and probabilistic uses
        total_strength = 0.0
        rules_list = list(self.rules.values())
        if not rules_list:
            self.activity_history.append(0.0)
            self.cycles_since_new_rule += 1
            return

        for r in rules_list:
            s, terms = self._calculate_strength(r, external_reward_fn)
            r.strength = s
            r.history_strength.append(s)
            total_strength += s
            p_use = self._usage_probability(r, s)
            if random.random() < p_use:
                r.usage_count += 1
                r.last_used_cycle = self.cycle
                r.is_new = False

        avg_strength = total_strength / len(rules_list)
        self.activity_history.append(avg_strength)
        self.cycles_since_new_rule += 1

        # archive stable branches
        self._archive_stable_rules()

        # rollback check
        if self._pending_rollback_check and self.cycle >= self._pending_rollback_check["evaluate_after_cycle"]:
            self._maybe_rollback_after_revival()

    def _usage_probability(self, rule: Rule, strength: float) -> float:
        # normalize strength to probability in stable, controlled way
        if strength <= 0:
            return 0.0
        # soft cap
        p = 1.0 - math.exp(-strength / (1.0 + rule.usage_count * 0.01))
        return min(1.0, max(0.0, p))

    # ---------- strength / components ----------
    def _calculate_strength(self, rule: Rule, external_reward_fn: Optional[Callable[[], float]] = None) -> Tuple[float, Tuple]:
        # term1: base
        Wi = rule.weight
        Ci = rule.usage_count
        Ui = max(0, self.cycle - rule.last_used_cycle)
        T = max(1, self.cycle)
        Ri = 1.0  # can be extended with rule-specific responsiveness

        term1 = Wi * math.log(Ci + 1.0) * (1.0 + Ui / T) * Ri

        # term2: interactions - mean neighbor strength
        neighbor_avg = self._neighbor_average(rule)
        Ii = 0.2
        term2 = 1.0 + Ii * neighbor_avg

        # term3: context sensitivity (placeholder)
        Mi = 1.02
        d_context = 1.0
        max_d = 5.0
        term3 = Mi ** (d_context / max(1e-6, max_d))

        # term4: FRZ adaptive
        term4 = self._calculate_adaptive_frz_cached()

        # term5: emergence pressure
        emergence_pressure = self._calculate_emergence_pressure()
        term5 = 1.0 + emergence_pressure

        # term6: curiosity
        term6 = 1.0 + self._curiosity_bonus(rule)

        # term7: anti-stagnation
        term7 = self._get_anti_stagnation_factor_cached()

        # optional external reward multiplier
        reward = 1.0
        if external_reward_fn is not None:
            try:
                rcv = float(external_reward_fn())  # expected 0..1
                reward = max(0.0, 1.0 + (rcv - 0.5))  # map roughly
            except Exception:
                reward = 1.0

        Si = term1 * term2 * term3 * term4 * term5 * term6 * term7 * reward
        return Si, (term1, term2, term3, term4, term5, term6, term7, reward)

    def _neighbor_average(self, rule: Rule) -> float:
        others = [r.strength for r in self.rules.values() if r is not rule]
        return sum(others) / len(others) if others else 0.0

    # ---------- FRZ / emergence / curiosity / anti-stagnation ----------
    def _calculate_adaptive_frz_cached(self) -> float:
        is_stag = self._last_vitals.get('is_stagnation', self._detect_stagnation())
        is_death = self._last_vitals.get('is_death_spiral', self._detect_death_spiral())
        frz = max(0.0, min(self.FRZ, 100.0))
        if is_death:
            return max(0.01, frz / 100.0)
        if is_stag:
            return max(0.1, 1.5 - (frz / 100.0))
        if len(self.activity_history) < 10:
            return 0.8 + 0.2
        return max(0.01, frz / 100.0)

    def _calculate_emergence_pressure(self) -> float:
        stagnation_pressure = 0.0
        diversity_pressure = 0.0
        novelty_pressure = 0.0
        success_pressure = 0.0
        time_pressure = 0.0

        if self._detect_stagnation():
            stagnation_pressure = 2.0

        unique_frac = self._unique_rule_fraction()
        if unique_frac < 0.8:
            diversity_pressure = 1.5 * (1.0 - unique_frac)

        if self.cycles_since_new_rule > 10:
            novelty_pressure = 1.0

        recent_avg = self.activity_history[-1] if self.activity_history else 0.0
        if recent_avg < 0.5:
            success_pressure = 0.8 * (0.5 - recent_avg)

        if self.cycles_since_new_rule > 20:
            time_pressure = 0.5

        total = stagnation_pressure + diversity_pressure + novelty_pressure + success_pressure + time_pressure
        return total

    def _curiosity_bonus(self, rule: Rule) -> float:
        exploration_bonus = 0.0
        if rule.is_new:
            exploration_bonus += 0.8
        elif rule.usage_count < 5:
            exploration_bonus += 0.6
        novelty_seeking = max(0.2, 1.0 - rule.experience_level(self.cycle, max_exp=100))
        return exploration_bonus * novelty_seeking

    def _get_anti_stagnation_factor_cached(self) -> float:
        if self._last_vitals.get('is_death_spiral', False):
            return 2.0 + random.uniform(0.0, 1.0)
        if self._last_vitals.get('is_stagnation', False):
            return 1.5 + random.uniform(0.0, 0.5)
        return 1.0

    # ---------- detection ----------
    def _detect_stagnation(self) -> bool:
        if len(self.activity_history) < self.stagnation_window:
            return False
        recent = self.activity_history[-self.stagnation_window:]
        avg_changes = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        avg_change = sum(avg_changes) / len(avg_changes) if avg_changes else 0.0
        no_new = (self.cycles_since_new_rule > 0)
        return (avg_change < self.stagnation_change_thresh) and no_new

    def _detect_death_spiral(self) -> bool:
        if len(self.activity_history) < self.death_window:
            return False
        avg_activity = sum(self.activity_history[-self.death_window:]) / self.death_window
        return (avg_activity < 0.05) and (self.cycles_since_new_rule > 10)

    # ---------- diversity and mutation ----------
    def _unique_rule_fraction(self) -> float:
        if not self.rules:
            return 1.0
        names = [r.name for r in self.rules.values()]
        return len(set(names)) / len(names)

    def _enforce_diversity(self):
        rules_list = list(self.rules.values())
        if len(rules_list) < 2:
            return
        best_pair = None
        best_sim = -1.0
        for i in range(len(rules_list)):
            for j in range(i + 1, len(rules_list)):
                a = rules_list[i]
                b = rules_list[j]
                sim = self._cosine_sim_vec(a.features, b.features)
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (a, b)
        if best_pair and best_sim > 0.9:
            a, b = best_pair
            target = a if a.weight < b.weight else b
            old_w = target.weight
            factor = 1.0 + random.uniform(self.mutation_magnitude * 0.1, self.mutation_magnitude)
            target.weight *= factor
            # adjust feature vector first element as proxy
            if target.features:
                target.features[0] *= factor
            target.name = f"{target.name}_mut{int(self.cycle)}"
            target.is_new = True
            self.event_log.append((self.cycle, f"enforce_diversity:mutate {target.name} {old_w:.4f}->{target.weight:.4f}"))

    # ---------- emergency revival with checkpoint and rollback ----------
    def _create_checkpoint(self, label: str = "checkpoint"):
        snap = {
            'cycle': self.cycle,
            'label': label,
            'rules': {name: {'weight': r.weight, 'features': list(r.features), 'strength': r.strength,
                             'usage': r.usage_count, 'last_used': r.last_used_cycle}
                      for name, r in self.rules.items()},
            'activity_history': list(self.activity_history),
            'cycles_since_new_rule': self.cycles_since_new_rule
        }
        self._last_checkpoint = snap
        self.archive.append(('checkpoint', deepcopy(snap)))
        self.event_log.append((self.cycle, f"checkpoint_created:{label}"))

    def _emergency_revival(self):
        self.event_log.append((self.cycle, "emergency_revival:start"))
        # checkpoint already created by caller
        weak = [r for r in self.rules.values() if r.strength < 0.2]
        for _ in range(self.emergency_mutation_count):
            if weak:
                r = random.choice(weak)
                old_w = r.weight
                factor = 1.0 + random.uniform(self.mutation_magnitude * 0.5, self.mutation_magnitude * 1.5)
                r.weight *= factor
                if r.features:
                    r.features[0] *= factor
                r.name = f"{r.name}_rev{int(self.cycle)}"
                r.is_new = True
                self.event_log.append((self.cycle, f"revive_mutate:{r.name} {old_w:.4f}->{r.weight:.4f}"))
        # add a controlled new rule
        new_name = f"auto_rule_{int(time.time() * 1000) % 100000}_{self.cycle}"
        self.add_rule(new_name, features=[random.uniform(0.5, 1.5)], weight=random.uniform(0.5, 1.5))
        self.event_log.append((self.cycle, "emergency_revival:end"))

    def _maybe_rollback_after_revival(self):
        if not self._pending_rollback_check:
            return
        chk = self._pending_rollback_check
        pre_avg = chk.get("pre_avg", 0.0)
        post_avg = self.activity_history[-1] if self.activity_history else 0.0
        # if performance dropped significantly -> rollback
        if post_avg < pre_avg * 0.7:
            # rollback
            cp = chk.get("checkpoint")
            if cp:
                self._rollback_checkpoint(cp)
                self.event_log.append((self.cycle, f"rollback_performed pre_avg={pre_avg:.4f} post_avg={post_avg:.4f}"))
        else:
            self.event_log.append((self.cycle, f"rollback_skipped pre_avg={pre_avg:.4f} post_avg={post_avg:.4f}"))
        self._pending_rollback_check = None

    def _rollback_checkpoint(self, checkpoint: Dict[str, Any]):
        # restore rules and state
        rules_snap = checkpoint.get('rules', {})
        self.rules = {}
        for name, s in rules_snap.items():
            r = Rule(name, features=list(s.get('features', [s.get('weight', 1.0)])), weight=float(s.get('weight', 1.0)))
            r.strength = float(s.get('strength', 0.0))
            r.usage_count = int(s.get('usage', 0))
            r.last_used_cycle = int(s.get('last_used', 0))
            r.created_cycle = checkpoint.get('cycle', 0)
            self.rules[name] = r
        self.activity_history = list(checkpoint.get('activity_history', []))
        self.cycles_since_new_rule = int(checkpoint.get('cycles_since_new_rule', 9999))
        self.event_log.append((self.cycle, "rollback_restored"))

    # ---------- archiving stable branches ----------
    def _archive_stable_rules(self):
        for r in list(self.rules.values()):
            recent = r.history_strength[-self.stable_cycles:]
            if len(recent) == self.stable_cycles and min(recent) > self.stable_threshold:
                snap = {'cycle': self.cycle, 'rule': r.name, 'weight': r.weight, 'strength': r.strength}
                self.archive.append(('stable', snap))
                self.event_log.append((self.cycle, f"archive_stable:{r.name}"))
                # reset history to prevent repeated archives for same streak
                r.history_strength = []

    # ---------- stats & export ----------
    def get_stats(self) -> Dict[str, Any]:
        return {
            'cycle': self.cycle,
            'n_rules': len(self.rules),
            'avg_activity': self.activity_history[-1] if self.activity_history else 0.0,
            'cycles_since_new_rule': self.cycles_since_new_rule,
            'events_last_50': list(self.event_log[-50:]),
            'archive_count': len(self.archive)
        }

    def export_csv(self, path: str):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['cycle', 'n_rules', 'avg_activity', 'cycles_since_new_rule', 'archive_count'])
            w.writerow([self.cycle, len(self.rules), self.activity_history[-1] if self.activity_history else 0.0,
                        self.cycles_since_new_rule, len(self.archive)])

    def export_json(self, path: str):
        out = {
            'cycle': self.cycle,
            'rules': {name: {'weight': r.weight, 'features': r.features, 'strength': r.strength,
                             'usage_count': r.usage_count} for name, r in self.rules.items()},
            'activity_history': self.activity_history,
            'event_log': self.event_log,
            'archive': self.archive
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    def plot_activity(self, out_png: str = "drm_activity.png"):
        if plt is None:
            return
        plt.figure()
        epochs = list(range(1, len(self.activity_history) + 1))
        plt.plot(epochs, self.activity_history, label='avg_activity')
        plt.xlabel('epoch')
        plt.ylabel('avg_activity')
        plt.title('DRM activity over time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()

    # ---------- placeholders for advanced improvements ----------
    def auto_calibrate(self):
        """Placeholder: auto-kalibracja parametrów (grid search / bayes)"""
        # implement external experiment runner to tune FRZ, mutation_magnitude, etc.
        raise NotImplementedError("auto_calibrate not implemented; integrate experiment runner.")

    # ---------- convenience runs ----------
    def run_epochs(self, epochs: int = 100, external_reward_fn: Optional[Callable[[], float]] = None, verbose: bool = False):
        for e in range(epochs):
            self.run_cycle(external_reward_fn=external_reward_fn)
            if verbose and (e + 1) % max(1, epochs // 10) == 0:
                print(f"[cycle {self.cycle}] avg_activity={self.activity_history[-1]:.4f} n_rules={len(self.rules)} archive={len(self.archive)}")

    # optional: quick summary text
    def summary_text(self) -> str:
        s = self.get_stats()
        return f"cycle={s['cycle']} n_rules={s['n_rules']} avg_activity={s['avg_activity']:.4f} cycles_since_new_rule={s['cycles_since_new_rule']} archive_count={s['archive_count']}"

# ---------------- example usage ----------------
if __name__ == "__main__":
    # szybki test modułu
    drm = DRM3System(FRZ=40.0, seed=42, stable_cycles=4, stable_threshold=0.4)
    # dodaj początkowe reguły
    for i in range(12):
        drm.add_rule(f"rule_{i}", features=[random.uniform(0.5, 1.5) for _ in range(4)], weight=random.uniform(0.5, 1.5))

    # opcjonalna zewnętrzna funkcja nagrody (symulowana)
    def ext_reward():
        # prosta symulacja: reward rośnie gdy avg_activity jest niska (promote recovery)
        avg = drm.activity_history[-1] if drm.activity_history else 0.0
        return max(0.0, min(1.0, 0.5 + (0.5 - avg)))

    drm.run_epochs(epochs=200, external_reward_fn=ext_reward, verbose=True)

    # eksport i wykres
    drm.export_csv("drm_stats.csv")
    drm.export_json("drm_state.json")
    if plt is not None:
        drm.plot_activity("drm_activity.png")

    print("Done. Summary:", drm.summary_text())
