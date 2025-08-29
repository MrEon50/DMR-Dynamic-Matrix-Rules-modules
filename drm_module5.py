
"""drm_module5.py
Unified Dynamic Rule Matrix (DRM) module.
Features:
 - Rule types: LOGICAL, HEURISTIC, HYBRID with semantic fields (pre/post/params/tests/provenance)
 - Bayesian posterior (post_mean, post_var) for rule performance
 - Strength function S_i combining weight, usage, novelty, curiosity, emergence pressure
 - ReplayBuffer, RuleGenerator (latent), and generator training hook
 - Multiplicative update with adaptive eta and KL trust-region
 - StagnationDetector, EmergencyRevival, DiversityEnforcement
 - Semantic validator: validate_rule(context) with strict logic for LOGICAL rules
 - Serialization (to_dict/from_dict), explainability, mutate, compose
 - Lightweight APIs: add_rule, generate_rules, run_cycle, save/load JSON, export
Dependencies: numpy optional for acceleration
"""
from __future__ import annotations
import math, random, json, time
from typing import Dict, Any, List, Optional, Callable, Tuple
from collections import deque, defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    np = None
    HAS_NUMPY = False

EPS = 1e-9

# Rule types
LOGICAL = "LOGICAL"
HEURISTIC = "HEURISTIC"
HYBRID = "HYBRID"

# ------------------ Rule ------------------
class Rule:
    def __init__(self,
                 id: str,
                 name: Optional[str] = None,
                 rtype: str = HEURISTIC,
                 init_weight: float = 1.0,
                 init_mean: float = 0.5,
                 init_var: float = 0.25,
                 latent_z: Optional[List[float]] = None,
                 pre_conditions: Optional[List[str]] = None,
                 post_conditions: Optional[List[str]] = None,
                 params: Optional[Dict[str, Dict[str, Any]]] = None,
                 tests: Optional[List[Callable[['Rule', Dict[str, Any]], bool]]] = None,
                 provenance: Optional[Dict[str, Any]] = None):
        self.id = id
        self.name = name or id
        self.type = rtype
        # probabilistic
        self.weight = float(max(EPS, init_weight))
        self.post_mean = float(init_mean)
        self.post_var = float(init_var)
        self.observations = 0
        self.usage_count = 0
        self.is_new = True
        self.quarantined = False
        self.latent_z = list(latent_z) if latent_z is not None else None
        # semantic
        self.pre_conditions = list(pre_conditions) if pre_conditions else []
        self.post_conditions = list(post_conditions) if post_conditions else []
        self.params = dict(params) if params else {}
        self.tests = list(tests) if tests else []
        self.provenance = dict(provenance) if provenance else {}
        self.created_at = time.time()
        # diagnostics history (small)
        self.history = deque(maxlen=200)

    # Bayesian posterior update (Gaussian conjugate)
    def update_posterior(self, reward: Optional[float], obs_var: float = 0.05):
        if reward is None:
            return
        self.observations += 1
        self.usage_count += 1
        self.is_new = False
        prior_prec = 1.0 / max(EPS, self.post_var)
        like_prec = 1.0 / max(EPS, obs_var)
        post_var = 1.0 / (prior_prec + like_prec)
        post_mean = post_var * (self.post_mean * prior_prec + reward * like_prec)
        self.post_mean = float(post_mean)
        self.post_var = float(post_var)
        self.history.append(("up", reward, self.post_mean, self.post_var))

    def sample_posterior(self) -> float:
        sigma = math.sqrt(max(EPS, self.post_var))
        return random.gauss(self.post_mean, sigma)

    def post_prob_below(self, thr: float) -> float:
        if self.post_var <= 0:
            return 1.0 if self.post_mean < thr else 0.0
        z = (thr - self.post_mean) / math.sqrt(self.post_var)
        cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        return float(min(1.0, max(0.0, cdf)))

    # Semantics evaluation
    def evaluate_semantics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        res = {"syntactic": True, "pre_ok": True, "tests_pass": True, "param_ok": True, "statistical_ok": True, "score": 0.0}
        # syntactic: required context keys
        for t in self.pre_conditions:
            if t not in context:
                res["syntactic"] = False
                res["pre_ok"] = False
        # run tests
        for fn in self.tests:
            try:
                ok = bool(fn(self, context))
            except Exception:
                ok = False
            if not ok:
                res["tests_pass"] = False
        # params range check
        for k, meta in self.params.items():
            if "value" not in meta:
                res["param_ok"] = False
                continue
            v = meta["value"]
            if "min" in meta and v < meta["min"] - EPS:
                res["param_ok"] = False
            if "max" in meta and v > meta["max"] + EPS:
                res["param_ok"] = False
        # statistical quick check
        res["statistical_ok"] = (self.post_mean >= 0.05)
        # score aggregate
        score = 0.0
        score += 1.0 if res["syntactic"] else 0.0
        score += 1.0 if res["pre_ok"] else 0.0
        score += 1.0 if res["tests_pass"] else 0.0
        score += 1.0 if res["param_ok"] else 0.0
        score += self.post_mean
        res["score"] = score / 5.0
        return res

    def explain(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "weight": self.weight,
            "post_mean": self.post_mean,
            "post_var": self.post_var,
            "observations": self.observations,
            "usage_count": self.usage_count,
            "quarantined": self.quarantined,
            "pre_conditions": list(self.pre_conditions),
            "post_conditions": list(self.post_conditions),
            "params": dict(self.params),
            "provenance": dict(self.provenance),
        }

    def mutate(self, op: str = "tweak_param", magnitude: float = 0.1) -> bool:
        if self.type == LOGICAL:
            return False
        if op == "tweak_param" and self.params:
            for k, meta in self.params.items():
                if "value" in meta and "min" in meta and "max" in meta:
                    v = float(meta["value"])
                    span = float(max(EPS, meta["max"] - meta["min"]))
                    delta = (random.uniform(-1, 1) * magnitude * span)
                    meta["value"] = max(meta["min"], min(meta["max"], v + delta))
            return True
        if op == "mutate_weight":
            factor = 1.0 + random.uniform(-magnitude, magnitude)
            self.weight *= factor
            return True
        if op == "perturb_latent" and self.latent_z is not None:
            self.latent_z = [z + random.gauss(0, magnitude) for z in self.latent_z]
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "type": self.type,
            "weight": self.weight, "post_mean": self.post_mean, "post_var": self.post_var,
            "observations": self.observations, "usage_count": self.usage_count,
            "quarantined": self.quarantined, "latent_z": list(self.latent_z) if self.latent_z is not None else None,
            "pre_conditions": list(self.pre_conditions), "post_conditions": list(self.post_conditions),
            "params": dict(self.params), "provenance": dict(self.provenance)
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'Rule':
        return Rule(id=d["id"], name=d.get("name"), rtype=d.get("type", HEURISTIC),
                    init_weight=d.get("weight",1.0), init_mean=d.get("post_mean",0.5), init_var=d.get("post_var",0.25),
                    latent_z=d.get("latent_z"), pre_conditions=d.get("pre_conditions"),
                    post_conditions=d.get("post_conditions"), params=d.get("params"),
                    provenance=d.get("provenance"))

# ------------------ ReplayBuffer ------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.buf = deque(maxlen=capacity)
    def add(self, entry: Tuple[str, float]):
        self.buf.append(entry)
    def sample(self, k:int) -> List[Tuple[str,float]]:
        if not self.buf: return []
        k = min(k, len(self.buf))
        return random.sample(list(self.buf), k)
    def __len__(self):
        return len(self.buf)

# ------------------ RuleGenerator (latent) ------------------
class RuleGenerator:
    def __init__(self, latent_dim:int=8, seed:Optional[int]=None):
        self.latent_dim = int(latent_dim)
        if seed is not None: random.seed(seed)
        self.registry: Dict[str, List[float]] = {}
    def sample_latent(self):
        if HAS_NUMPY:
            return np.random.normal(size=(self.latent_dim,)).tolist()
        else:
            return [random.gauss(0,1) for _ in range(self.latent_dim)]
    def generate(self, prefix="gen", idx:Optional[int]=None) -> Rule:
        z = self.sample_latent()
        raw = sum(random.gauss(0,0.5)*zi for zi in z) + random.gauss(0,0.1)
        mean = 1.0/(1.0+math.exp(-raw))
        mean = max(0.01,min(0.99, mean))
        rid = f"{prefix}_{idx}" if idx is not None else f"{prefix}_{random.randint(0,10**9)}"
        r = Rule(id=rid, name=rid, rtype=HEURISTIC, init_weight=1.0, init_mean=mean, init_var=0.05, latent_z=z,
                 params={"alpha":{"type":"float","min":0.0,"max":1.0,"value":0.1}}, provenance={"created_by":"generator"})
        self.registry[r.id] = z
        return r
    def train_on_replay(self, replay: ReplayBuffer, rules: Dict[str,Rule], lr:float=0.01, epochs:int=1):
        # placeholder: optional training if numpy available
        return

# ------------------ Utilities ------------------
def entropy_from_dist(dist: Dict[str,float]) -> float:
    return -sum(p*math.log(p+EPS) for p in dist.values())

def kl_divergence(p: Dict[str,float], q: Dict[str,float]) -> float:
    keys = set(p.keys())|set(q.keys())
    kl = 0.0
    for k in keys:
        pk = p.get(k, EPS); qk = q.get(k, EPS)
        kl += pk*math.log((pk+EPS)/(qk+EPS))
    return kl

# ------------------ Stagnation Detector & Diversity ------------------
class StagnationDetector:
    def __init__(self, window:int=20, entropy_drop_thresh:float=0.3, no_improve_steps:int=50):
        self.window = window
        self.entropy_history = deque(maxlen=window)
        self.no_improve_steps = no_improve_steps
        self.best_score = -1e9
        self.steps_since_improve = 0
        self.entropy_drop_thresh = entropy_drop_thresh
    def observe(self, entropy:float, global_score:float):
        self.entropy_history.append(entropy)
        if global_score > self.best_score + 1e-9:
            self.best_score = global_score
            self.steps_since_improve = 0
        else:
            self.steps_since_improve += 1
    def is_stagnant(self) -> bool:
        if len(self.entropy_history) < self.window: return False
        e0 = self.entropy_history[0]; e1 = self.entropy_history[-1]
        if e0 - e1 > self.entropy_drop_thresh: return True
        if self.steps_since_improve >= self.no_improve_steps: return True
        return False

class DiversityEnforcer:
    def __init__(self, sim_threshold: float = 0.95):
        self.sim_threshold = sim_threshold
    def rule_similarity(self, a:Rule, b:Rule) -> float:
        if a.latent_z is None or b.latent_z is None:
            return 0.0
        if HAS_NUMPY:
            va = np.array(a.latent_z); vb = np.array(b.latent_z)
            ca = np.linalg.norm(va); cb = np.linalg.norm(vb)
            if ca < EPS or cb < EPS: return 0.0
            return float((va.dot(vb))/(ca*cb))
        else:
            # cosine approx
            dot = sum(x*y for x,y in zip(a.latent_z, b.latent_z))
            na = math.sqrt(sum(x*x for x in a.latent_z))
            nb = math.sqrt(sum(x*x for x in b.latent_z))
            if na < EPS or nb < EPS: return 0.0
            return dot/(na*nb)
    def enforce(self, rules: Dict[str,Rule]) -> List[Tuple[str,str,float]]:
        pairs = []
        ids = list(rules.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a = rules[ids[i]]; b = rules[ids[j]]
                sim = self.rule_similarity(a,b)
                if sim > self.sim_threshold:
                    pairs.append((a.id,b.id,sim))
        return pairs

# ------------------ DRMSystem ------------------
class DRMSystem:
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.replay = ReplayBuffer()
        self.generators: List[RuleGenerator] = []
        self.stagnation = StagnationDetector()
        self.diversity = DiversityEnforcer()
        self.audit_log: List[Dict[str,Any]] = []
        self.archived: Dict[str, Rule] = {}
    # basic CRUD
    def add_rule(self, rule: Rule):
        self.rules[rule.id] = rule
        rule.weight = max(EPS, rule.weight)
        self.audit("add", rule.id, {"type": rule.type})
    def remove_rule(self, rule_id: str, archive: bool = True):
        if rule_id in self.rules:
            r = self.rules.pop(rule_id)
            if archive:
                self.archived[rule_id] = r
            self.audit("remove", rule_id, {})
    def register_generator(self, gen: RuleGenerator):
        self.generators.append(gen)
    def generate_rules(self, gen: RuleGenerator, count:int=1, prefix:str="g"):
        for i in range(count):
            r = gen.generate(prefix, i)
            self.add_rule(r)
    def get_distribution(self) -> Dict[str,float]:
        names = list(self.rules.keys())
        weights = [max(EPS, self.rules[n].weight) for n in names]
        s = sum(weights) or EPS
        return {n: w/s for n,w in zip(names,weights)}

    # core strength function (Si) per concept: weight * log(usage+1) * (1 + novelty/T) * post_mean scaled
    def compute_strengths(self, curiosity:float=1.0, frz:float=1.0) -> Dict[str,float]:
        strengths = {}
        for rid, r in self.rules.items():
            w = r.weight
            usage = r.usage_count
            novelty = 1.0/(1.0 + usage)
            curiosity_term = curiosity * novelty if r.is_new else 0.0
            base = w * math.log(1.0 + usage + EPS)
            score = base * (1.0 + curiosity_term) * (r.post_mean + 1e-3) * frz
            strengths[rid] = float(score)
        return strengths

    # multiplicative update with adaptive eta and KL trust region
    def multiplicative_update(self, eta:float=0.05, beta:float=0.5, lam:float=0.5, kl_max:float=0.5):
        old = self.get_distribution()
        scores = self.compute_scores(beta=beta, lam=lam)
        names = list(self.rules.keys())
        weights = [max(EPS, self.rules[n].weight) for n in names]
        tentative = [w * math.exp(eta * scores.get(n,0.0)) for w,n in zip(weights,names)]
        s = sum(tentative) or EPS
        tentative = [v/s for v in tentative]
        kl = kl_divergence({n:tentative[i] for i,n in enumerate(names)}, old)
        if kl > kl_max and kl > 0:
            eta_scaled = max(1e-6, eta * (kl_max/kl))
            tentative = [w * math.exp(eta_scaled * scores.get(n,0.0)) for w,n in zip(weights,names)]
            s2 = sum(tentative) or EPS
            tentative = [v/s2 for v in tentative]
        avg_scale = len(names) or 1
        for i,n in enumerate(names):
            self.rules[n].weight = max(EPS, tentative[i]*avg_scale)
        return kl

    def compute_scores(self, beta:float=0.5, lam:float=0.5, exploration_bonus:float=0.02) -> Dict[str,float]:
        scores = {}
        for name,r in self.rules.items():
            if r.quarantined:
                scores[name] = -1e9; continue
            novelty = 1.0/(1.0 + r.usage_count)
            risk = r.post_var
            exploration = exploration_bonus if r.is_new else 0.0
            scores[name] = float(r.post_mean + beta*novelty - lam*risk + exploration)
        return scores

    # semantic validate
    def validate_rule(self, rule_id: str, context: Dict[str,Any]) -> Dict[str,Any]:
        if rule_id not in self.rules:
            return {"error":"not_found"}
        r = self.rules[rule_id]
        res = r.evaluate_semantics(context)
        decision = {"activate": False, "quarantine": False, "reason": None, "score": res["score"], "details": res}
        if r.type == LOGICAL:
            if res["tests_pass"] and res["syntactic"] and res["param_ok"]:
                decision["activate"] = True
            else:
                decision["quarantine"] = True; decision["reason"] = "logical_failed"
        elif r.type == HYBRID:
            if (res["tests_pass"] and res["param_ok"]) or r.post_mean > 0.5:
                decision["activate"] = True
            else:
                decision["quarantine"] = True; decision["reason"] = "hybrid_low_conf"
        else:
            if not res["tests_pass"] and r.post_mean < 0.2:
                decision["quarantine"] = True; decision["reason"] = "heuristic_failing"
            else:
                decision["activate"] = True
        if decision["quarantine"]:
            r.quarantined = True
            self.audit("quarantine", r.id, {"reason": decision["reason"]})
        return decision

    def explain_rule(self, rule_id: str) -> Dict[str,Any]:
        if rule_id not in self.rules:
            return {"error":"not_found"}
        return self.rules[rule_id].explain()

    def mutate_rule(self, rule_id:str, op:str="tweak_param", magnitude:float=0.1) -> Dict[str,Any]:
        if rule_id not in self.rules:
            return {"error":"not_found"}
        r = self.rules[rule_id]
        if r.type == LOGICAL:
            return {"mutated":False, "reason":"logical_protected"}
        ok = r.mutate(op=op, magnitude=magnitude)
        self.audit("mutate", r.id, {"op":op, "magnitude":magnitude, "success":ok})
        return {"mutated": ok, "new_params": r.params, "new_weight": r.weight}

    # detect dezinformation (quarantine low posterior)
    def detect_dezinformation(self, mu_min:float=0.1, tau:float=0.95) -> List[str]:
        removed = []
        for rid,r in list(self.rules.items()):
            if r.quarantined: continue
            if r.post_prob_below(mu_min) > tau:
                r.quarantined = True
                removed.append(rid)
                self.audit("quarantine_low_posterior", rid, {})
        return removed

    # Diversity enforcement and emergency revival
    def enforce_diversity(self):
        pairs = self.diversity.enforce(self.rules)
        for a,b,sim in pairs:
            # slightly perturb one rule
            if b in self.rules:
                self.rules[b].mutate(op="perturb_latent", magnitude=0.2)
                self.audit("diversity_perturb", b, {"sim":sim})

    def emergency_revival(self, num_new:int=3):
        # generate new rules from generators and boost low-weight rules
        for gen in self.generators:
            for i in range(num_new):
                r = gen.generate(prefix="rev", idx=None)
                self.add_rule(r)
                self.audit("revival_generate", r.id, {})
        # boost weakest heuristics slightly
        heuristics = [r for r in self.rules.values() if r.type == HEURISTIC and not r.quarantined]
        heuristics.sort(key=lambda x: x.weight)
        for r in heuristics[:max(1, len(heuristics)//5)]:
            r.weight *= 1.5
            self.audit("revival_boost", r.id, {"new_weight": r.weight})

    # lifecycle run_cycle
    def run_cycle(self, evaluator: Callable[[Rule], Optional[float]],
                  eta:float=0.05, beta:float=0.5, lam:float=0.5,
                  kl_max:float=0.5, mu_min:float=0.1, tau:float=0.95,
                  replay_batch:int=0, generator_train:bool=False):
        names = list(self.rules.keys())
        if not names:
            return {"status":"no_rules"}
        sampled=[]
        for nid in names:
            r = self.rules[nid]
            reward = evaluator(r)
            if reward is not None:
                self.replay.add((nid, float(reward)))
            r.update_posterior(reward)
            sampled.append((nid, reward))
        # optional replay stabilization
        if replay_batch and len(self.replay) > 0:
            for rn, rew in self.replay.sample(replay_batch):
                if rn in self.rules:
                    self.rules[rn].update_posterior(rew)
        # train generators optionally
        if generator_train and self.generators and len(self.replay) > 0:
            for gen in self.generators:
                gen.train_on_replay(self.replay, self.rules)
        kl = self.multiplicative_update(eta=eta, beta=beta, lam=lam, kl_max=kl_max)
        quarantined = self.detect_dezinformation(mu_min=mu_min, tau=tau)
        dist = self.get_distribution()
        ent = entropy_from_dist(dist)
        # compute global score (avg post_mean)
        global_score = sum(r.post_mean for r in self.rules.values())/max(1,len(self.rules))
        self.stagnation.observe(ent, global_score)
        if self.stagnation.is_stagnant():
            self.audit("stagnation_detected", None, {"entropy": ent, "global_score": global_score})
            self.enforce_diversity()
            self.emergency_revival(num_new=3)
        return {"evaluations": sampled, "quarantined": quarantined, "entropy": ent, "distribution": dist, "kl": kl}

    # explainability & serialization
    def audit(self, action:str, rule_id:Optional[str], info:Dict[str,Any]):
        self.audit_log.append({"time": time.time(), "action": action, "rule": rule_id, "info": info})

    def to_dict(self) -> Dict[str,Any]:
        return {"rules":[r.to_dict() for r in self.rules.values()], "audit": list(self.audit_log)}

    def save_json(self, path:str):
        with open(path,"w",encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_json(self, path:str):
        with open(path,"r",encoding="utf-8") as f:
            d = json.load(f)
        for rd in d.get("rules", []):
            r = Rule.from_dict(rd)
            self.rules[r.id] = r
        self.audit_log = d.get("audit", [])

# ------------------ minimal sanity demonstration when run as script ------------------
if __name__ == "__main__":
    ds = DRMSystem()
    gen = RuleGenerator(latent_dim=6, seed=1)
    ds.register_generator(gen)
    # logical rule
    r1 = Rule(id="mass_cons", name="mass_cons", rtype=LOGICAL,
              pre_conditions=["domain_advec1d"],
              post_conditions=["mass_change<1e-3"],
              params={"alpha":{"type":"float","min":0.0,"max":1.0,"value":0.0}},
              tests=[lambda rule,ctx: ("mass" in ctx and abs(ctx.get("mass_change",1.0)) < 1e-3)],
              provenance={"created_by":"user"})
    ds.add_rule(r1)
    # heuristic rule
    r2 = gen.generate(prefix="g", idx=0)
    ds.add_rule(r2)
    # fake evaluator: heuristic reward depends on random, logical reward depends on context mass change small
    def evaluator(rule:Rule):
        if rule.type == LOGICAL:
            # if small mass change, reward near 1 else 0
            return 1.0 if ("mass_change" in demo_ctx and abs(demo_ctx["mass_change"]) < 1e-3) else 0.0
        else:
            # heuristic: reward correlated with post_mean + noise
            return max(0.0, min(1.0, rule.post_mean + random.gauss(0, 0.1)))
    # context where mass change small
    demo_ctx = {"domain_advec1d":True, "mass":1.0, "mass_change":5e-4}
    print("Validate logical:", ds.validate_rule("mass_cons", demo_ctx))
    # run a cycle
    res = ds.run_cycle(evaluator, eta=0.1, beta=0.6, lam=0.4, kl_max=0.8, replay_batch=2, generator_train=False)
    print("Cycle result keys:", list(res.keys()))
    print("Entropy:", res["entropy"])
    print("Rules:", {k: v.post_mean for k,v in ds.rules.items()})
