import math
import random
import time

class Rule:
    """Reprezentuje pojedynczą regułę w systemie DRM 3.0."""
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight
        self.strength = 0.0
        self.usage_count = 0
        self.last_used_cycle = 0
        self.is_new = True

class DRM3System:
    """
    Główny silnik DRM 3.0.
    Zarządza regułami, wykrywa stagnację i stosuje mechanizmy antystagnacyjne.
    """
    def __init__(self):
        self._rules = {}
        self.cycle = 0
        self._activity_history = []
        self._new_rules_count = 0

    def add_rule(self, rule_name, weight=1.0):
        """Dodaje nową regułę do systemu."""
        if rule_name not in self._rules:
            new_rule = Rule(rule_name, weight)
            self._rules[rule_name] = new_rule
            self._new_rules_count += 1
            return True
        return False

    def get_rule_strength(self, rule_name):
        """Pobiera siłę danej reguły."""
        rule = self._rules.get(rule_name)
        if rule:
            return rule.strength
        return 0.0

    def run_cycle(self):
        """Wykonuje jeden cykl obliczeniowy dla wszystkich reguł."""
        self.cycle += 1
        
        is_stagnation = self._detect_stagnation()
        is_death_spiral = self._detect_death_spiral()
        
        if is_death_spiral:
            self._emergency_revival()
        elif is_stagnation:
            self._enforce_diversity()
        
        total_strength = 0
        active_rules = list(self._rules.values())

        if not active_rules:
            return

        for rule in active_rules:
            strength = self._calculate_strength(rule)
            rule.strength = strength
            total_strength += strength
            
            # Symulacja użycia reguły na podstawie jej siły
            if random.random() < strength:
                rule.usage_count += 1
                rule.last_used_cycle = self.cycle
                rule.is_new = False

        avg_strength = total_strength / len(active_rules)
        self._activity_history.append(avg_strength)
        self._new_rules_count = 0
        
    def _calculate_strength(self, rule):
        """
        Główna formuła DRM 3.0 do obliczania siły reguły.
        """
        Wi = rule.weight
        Ci = rule.usage_count
        Ui = self.cycle - rule.last_used_cycle
        T = self.cycle
        Ri = 1.0  # Uproszczona responsywność
        
        # FRZ_Adaptive
        frz_adaptive = self._calculate_adaptive_frz()
        
        # Emergence_Pressure
        emergence_pressure = self._calculate_emergence_pressure()
        
        # Curiosity_Drive
        curiosity_drive = self._calculate_curiosity_drive(rule)
        
        # Anti_Stagnation_Factor
        anti_stagnation_factor = self._get_anti_stagnation_factor()

        base = Wi * math.log(Ci + 1) * (1 + Ui / T) * Ri if T > 0 else 0
        
        return (
            base * frz_adaptive * emergence_pressure * curiosity_drive * anti_stagnation_factor
        )

    def _calculate_adaptive_frz(self):
        """
        FRZ_Adaptive: Inteligentny modulator.
        Zwiększa wartość, gdy wykryje stagnację.
        """
        frz_value = 0.5  # Uproszczona stała wartość
        if self._detect_stagnation():
            return 1.5 - (frz_value / 100)
        return frz_value / 100

    def _calculate_emergence_pressure(self):
        """
        Emergence_Pressure: Presja środowiskowa.
        """
        pressure = 0
        if self._detect_stagnation():
            pressure += 2.0  # Stagnation pressure
        if self._check_diversity():
            pressure += 1.5  # Diversity pressure
        return 1.0 + pressure
    
    def _calculate_curiosity_drive(self, rule):
        """
        Curiosity_Drive: Bonus dla nowości i rzadkości.
        """
        exploration_bonus = 0.0
        if rule.is_new:
            exploration_bonus += 0.8
        elif rule.usage_count < 5:
            exploration_bonus += 0.6
        
        novelty_seeking = max(0.2, 1.0 - (self.cycle / 100))
        return 1.0 + (exploration_bonus * novelty_seeking)

    def _get_anti_stagnation_factor(self):
        """
        Anti_Stagnation_Factor: Aktywny przeciwnik nudy.
        """
        if self._detect_stagnation():
            return 1.5 + random.uniform(0, 0.5)
        if self._detect_death_spiral():
            return 2.0 + random.uniform(0, 1.0)
        return 1.0

    def _detect_stagnation(self):
        """
        Wykrywa, czy system zaczyna stagnować.
        """
        if len(self._activity_history) < 5:
            return False
        
        recent_changes = [abs(self._activity_history[i] - self._activity_history[i-1]) for i in range(len(self._activity_history) - 4, len(self._activity_history))]
        avg_change = sum(recent_changes) / len(recent_changes)
        
        no_new_rules = self._new_rules_count == 0
        
        return avg_change < 0.01 and no_new_rules

    def _detect_death_spiral(self):
        """
        Wykrywa "spiralę śmierci" - krytyczny stan stagnacji.
        """
        if len(self._activity_history) < 10:
            return False
        
        avg_activity = sum(self._activity_history[-10:]) / 10
        return avg_activity < 0.05 and self._new_rules_count == 0

    def _emergency_revival(self):
        """
        Reanimuje system w przypadku spirali śmierci.
        """
        print("🚨 Awaryjna reanimacja systemu: Wstrzykuję chaos!")
        weak_rules = [r for r in self._rules.values() if r.strength < 0.1]
        for rule in weak_rules:
            rule.strength *= 2.0  # Wzmocnienie słabych reguł
        
        new_rule_name = f"Rule_Revived_{time.time()}"
        self.add_rule(new_rule_name)
    
    def _enforce_diversity(self):
        """
        Wymusza różnorodność, mutując podobne reguły.
        """
        # Uproszczone wykrywanie podobieństwa
        rule_names = [r.name for r in self._rules.values()]
        if len(set(rule_names)) / len(rule_names) < 0.5:
            print("⚠️ Wykryto zbyt dużą jednorodność: Wymuszam mutacje!")
            rule_to_mutate = random.choice(list(self._rules.values()))
            rule_to_mutate.name = f"{rule_to_mutate.name}_mutated"
            rule_to_mutate.is_new = True
    
    def _check_diversity(self):
        """
        Sprawdza poziom różnorodności reguł.
        """
        rule_names = [r.name for r in self._rules.values()]
        if len(rule_names) == 0:
            return True
        return len(set(rule_names)) / len(rule_names) < 0.8