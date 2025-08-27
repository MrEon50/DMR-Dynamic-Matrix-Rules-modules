# drm_module_v2.py
import math
import random
import time
from copy import deepcopy

class Rule:
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = float(weight)
        self.strength = 0.0
        self.usage_count = 0
        self.last_used_cycle = 0
        self.created_cycle = 0
        self.is_new = True
        self.history_strength = []

    def experience_level(self, current_cycle, max_exp=100):
        age = current_cycle - self.created_cycle
        exp = min(max_exp, age + self.usage_count)
        return exp / max_exp  # 0..1

class DRM3System:
    def __init__(self,
                 FRZ=45.0,
                 stagnation_window=5,
                 death_window=10,
                 stagnation_change_thresh=0.01,
                 stable_threshold=0.5,
                 stable_cycles=5,
                 mutation_magnitude=0.3,
                 emergency_mutation_count=3,
                 seed=None):
        self.FRZ = float(FRZ)
        self.cycle = 0
        self._rules = {}
        self.activity_history = []
        self.cycles_since_new_rule = 9999
        self.stagnation_window = stagnation_window
        self.death_window = death_window
        self.stagnation_change_thresh = stagnation_change_thresh
        self.event_log = []  # (cycle, event_str)
        self.archive = []  # snapshots of stable branches
        self.stable_threshold = stable_threshold
        self.stable_cycles = stable_cycles
        self.mutation_magnitude = mutation_magnitude
        self.emergency_mutation_count = emergency_mutation_count
        self._last_vitals = {}
        if seed is not None:
            random.seed(seed)

    # ---------- rule management ----------
    def add_rule(self, rule_name, weight=1.0):
        if rule_name in self._rules:
            return False
        r = Rule(rule_name, weight)
        r.created_cycle = self.cycle
        self._rules[rule_name] = r
        self.cycles_since_new_rule = 0
        self.event_log.append((self.cycle, f"add_rule:{rule_name}"))
        return True

    def remove_rule(self, rule_name):
        if rule_name in self._rules:
            del self._rules[rule_name]
            self.event_log.append((self.cycle, f"remove_rule:{rule_name}"))
            return True
        return False

    # ---------- core cycle ----------
    def run_cycle(self):
        self.cycle += 1
        # cache vitals for this cycle to avoid repeated recomputation
        vitals = {}
        vitals['is_stagnation'] = self._detect_stagnation()
        vitals['is_death_spiral'] = self._detect_death_spiral()
        self._last_vitals = vitals

        if vitals['is_death_spiral']:
            self._emergency_revival()
        elif vitals['is_stagnation']:
            self._enforce_diversity()

        total_strength = 0.0
        active_rules = list(self._rules.values())
        if not active_rules:
            self.activity_history.append(0.0)
            self.cycles_since_new_rule += 1
            return

        for r in active_rules:
            s, terms = self._calculate_strength(r)
            r.strength = s
            r.history_strength.append(s)
            total_strength += s
            # probabilistic use scaled by normalized strength
            p_use = min(1.0, max(0.0, s))
            if random.random() < p_use:
                r.usage_count += 1
                r.last_used_cycle = self.cycle
                r.is_new = False

        avg_strength = total_strength / len(active_rules)
        self.activity_history.append(avg_strength)
        self.cycles_since_new_rule += 1

        # archive stable branches if any
        self._archive_stable_rules()

    # ---------- strength / components ----------
    def _calculate_strength(self, rule):
        # Base
        Wi = rule.weight
        Ci = rule.usage_count
        Ui = self.cycle - rule.last_used_cycle if self.cycle > 0 else 0
        T = max(1, self.cycle)
        Ri = 1.0  # placeholder; could be rule-specific

        term1 = Wi * math.log(Ci + 1.0) * (1.0 + Ui / T) * Ri

        # Interaction proxy: average neighbor strength (simple)
        neighbor_avg = self._neighbor_average(rule)
        Ii = 0.2
        term2 = 1.0 + Ii * neighbor_avg

        # Context sensitivity
        Mi = 1.02
        d_context = 1.0
        max_d = 5.0
        term3 = Mi ** (d_context / max(1e-6, max_d))

        # FRZ adaptive
        term4 = self._calculate_adaptive_frz_cached()

        # Emergence pressure (composed)
        emergence_pressure = self._calculate_emergence_pressure()
        term5 = 1.0 + emergence_pressure

        # Curiosity drive using normalized experience_level
        novelty_bonus = self._curiosity_bonus(rule)
        term6 = 1.0 + novelty_bonus

        # Anti-stagnation factor
        term7 = self._get_anti_stagnation_factor_cached()

        Si = term1 * term2 * term3 * term4 * term5 * term6 * term7
        return Si, (term1, term2, term3, term4, term5, term6, term7)

    def _neighbor_average(self, rule):
        # simple proxy: mean strength of other rules
        others = [r.strength for r in self._rules.values() if r is not rule]
        return sum(others) / len(others) if others else 0.0

    # ---------- FRZ / emergence / curiosity / anti-stagnation ----------
    def _calculate_adaptive_frz_cached(self):
        # Use cached vitals if present
        if 'is_stagnation' in self._last_vitals:
            is_stag = self._last_vitals['is_stagnation']
            is_death = self._last_vitals['is_death_spiral']
        else:
            is_stag = self._detect_stagnation()
            is_death = self._detect_death_spiral()
        # FRZ normalized 0..100 expected; map to 0..1
        frz = max(0.0, min(self.FRZ, 100.0))
        if is_death:
            return max(0.01, frz / 100.0)
        if is_stag:
            return max(0.1, 1.5 - (frz / 100.0))
        # learning phase heuristic
        if len(self.activity_history) < 10:
            return 0.8 + 0.2
        return max(0.01, frz / 100.0)

    def _calculate_emergence_pressure(self):
        # compose pressure factors explicitly
        stagnation_pressure = 0.0
        diversity_pressure = 0.0
        novelty_pressure = 0.0
        success_pressure = 0.0
        time_pressure = 0.0

        if self._detect_stagnation():
            stagnation_pressure = 2.0
        # diversity: fraction unique names; lower -> more pressure
        unique_frac = self._unique_rule_fraction()
        if unique_frac < 0.8:
            diversity_pressure = 1.5 * (1.0 - unique_frac)
        # novelty: if cycles_since_new_rule large
        if self.cycles_since_new_rule > 10:
            novelty_pressure = 1.0
        # success proxy: if avg strength low
        recent_avg = self.activity_history[-1] if self.activity_history else 0.0
        if recent_avg < 0.5:
            success_pressure = 0.8 * (0.5 - recent_avg)
        # time pressure
        if self.cycles_since_new_rule > 20:
            time_pressure = 0.5

        base = 1.0
        total = stagnation_pressure + diversity_pressure + novelty_pressure + success_pressure + time_pressure
        return base * total

    def _curiosity_bonus(self, rule):
        # exploration_bonus based on rule novelty and experience
        exploration_bonus = 0.0
        if rule.is_new:
            exploration_bonus += 0.8
        elif rule.usage_count < 5:
            exploration_bonus += 0.6
        # novelty_seeking decreases with experience
        novelty_seeking = max(0.2, 1.0 - rule.experience_level(self.cycle, max_exp=100))
        return exploration_bonus * novelty_seeking

    def _get_anti_stagnation_factor_cached(self):
        if 'is_death_spiral' in self._last_vitals and self._last_vitals['is_death_spiral']:
            return 2.0 + random.uniform(0.0, 1.0)
        if 'is_stagnation' in self._last_vitals and self._last_vitals['is_stagnation']:
            return 1.5 + random.uniform(0.0, 0.5)
        return 1.0

    # ---------- detection ----------
    def _detect_stagnation(self):
        if len(self.activity_history) < self.stagnation_window:
            return False
        recent = self.activity_history[-self.stagnation_window:]
        avg_changes = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
        avg_change = sum(avg_changes) / len(avg_changes) if avg_changes else 0.0
        no_new = (self.cycles_since_new_rule > 0)
        return (avg_change < self.stagnation_change_thresh) and no_new

    def _detect_death_spiral(self):
        if len(self.activity_history) < self.death_window:
            return False
        avg_activity = sum(self.activity_history[-self.death_window:]) / self.death_window
        return (avg_activity < 0.05) and (self.cycles_since_new_rule > 10)

    # ---------- diversity and mutation ----------
    def _unique_rule_fraction(self):
        names = [r.name for r in self._rules.values()]
        if not names:
            return 1.0
        return len(set(names)) / len(names)

    def _enforce_diversity(self):
        # compute simple vector similarity on (weight, strength)
        rules = list(self._rules.values())
        if len(rules) < 2:
            return
        # find pair with highest similarity
        best_pair = None
        best_sim = -1.0
        for i in range(len(rules)):
            for j in range(i+1, len(rules)):
                a, b = rules[i], rules[j]
                sim = self._cosine_sim((a.weight, a.strength), (b.weight, b.strength))
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (a, b)
        if best_sim > 0.9:
            a, b = best_pair
            # mutate smaller weight rule slightly
            target = a if a.weight < b.weight else b
            old_w = target.weight
            target.weight *= (1.0 + random.uniform(self.mutation_magnitude*0.1, self.mutation_magnitude))
            target.name = f"{target.name}_mut{int(self.cycle)}"
            target.is_new = True
            self.event_log.append((self.cycle, f"enforce_diversity:mutate {target.name} from {old_w:.3f} to {target.weight:.3f}"))

    def _cosine_sim(self, v1, v2):
        import math
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        norm1 = math.hypot(v1[0], v1[1])
        norm2 = math.hypot(v2[0], v2[1])
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    # ---------- emergency revival ----------
    def _emergency_revival(self):
        # checkpoint top-K rules
        snapshot = {name: {'weight': r.weight, 'strength': r.strength, 'usage': r.usage_count} for name, r in self._rules.items()}
        self.event_log.append((self.cycle, "emergency_revival:start"))
        self.archive.append(('checkpoint', self.cycle, snapshot))

        # find weak rules and apply controlled mutations
        weak = [r for r in self._rules.values() if r.strength < 0.2]
        for _ in range(self.emergency_mutation_count):
            if weak:
                r = random.choice(weak)
                old_w = r.weight
                r.weight *= (1.0 + random.uniform(self.mutation_magnitude*0.5, self.mutation_magnitude*1.5))
                r.name = f"{r.name}_rev{int(self.cycle)}"
                r.is_new = True
                self.event_log.append((self.cycle, f"revive_mutate:{r.name} {old_w:.3f}->{r.weight:.3f}"))

        # optionally add a small number of new random rules
        for i in range(1):
            new_name = f"auto_rule_{int(time.time()*1000)%100000}_{self.cycle}"
            self.add_rule(new_name, weight=random.uniform(0.5, 1.5))

        self.event_log.append((self.cycle, "emergency_revival:end"))

    # ---------- archiving stable branches ----------
    def _archive_stable_rules(self):
        for r in self._rules.values():
            # keep last N strength values
            recent = r.history_strength[-self.stable_cycles:]
            if len(recent) == self.stable_cycles and min(recent) > self.stable_threshold:
                # snapshot rule
                snap = {'cycle': self.cycle, 'rule': r.name, 'weight': r.weight, 'strength': r.strength}
                self.archive.append(('stable', snap))
                self.event_log.append((self.cycle, f"archive_stable:{r.name}"))
                # mark to avoid duplicate archives for same streak
                r.history_strength = []  # reset history to avoid repeated archives

    # ---------- utilities ----------
    def get_stats(self):
        return {
            'cycle': self.cycle,
            'n_rules': len(self._rules),
            'avg_activity': self.activity_history[-1] if self.activity_history else 0.0,
            'cycles_since_new_rule': self.cycles_since_new_rule,
            'events': list(self.event_log[-50:]),
            'archive_count': len(self.archive)
        }
