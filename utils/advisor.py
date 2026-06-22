"""
====================================================
  Smart Water Usage Advisor — AI Core Logic
====================================================
This module contains the main AI/ML brain of the application.

WHAT IT DOES:
  - Uses a deterministic rule-based classifier to categorize water usage
    as Low / Moderate / High / Critical
  - Optionally trains a Random Forest model and uses it as the primary
    classifier (with rule-based fallback for transparency)
  - Generates personalized conservation recommendations
  - Calculates savings estimates
  - Produces historical trend data for charts

LIBRARIES USED:
  - scikit-learn : RandomForestClassifier (ensemble of decision trees)
  - numpy        : Fast numerical computations
  - pandas       : Data manipulation and analysis
  - joblib       : Caches the trained model to disk

------------------------------------------------------------------
FIX LOG — all bugs identified in the previous review are addressed here
------------------------------------------------------------------
FIX #1  _build_features / form-key mismatch
        Added FORM_KEY_MAP so HTML form field names (e.g. "bathroom")
        are explicitly translated to internal benchmark keys (e.g.
        "bathing").  User input is no longer silently dropped when the
        form field name differs from the benchmark key.

FIX #2  analyze() — score formula
        Replaced the ambiguous / sign-broken formula with two clearly
        named metrics:
          • conservation_score  (100 = excellent, 0 = at/above critical)
          • usage_pct           (0 = zero use, 100 = at critical ceiling)
        Both are clamped to [0, 100] so extreme values never go negative.

FIX #3  analyze() — ML model actually used for classification
        self.model.predict() is now the primary classification path.
        _classify_per_capita() is kept as a fallback and for labelling
        training data.  A diagnostic warning is logged whenever the two
        methods disagree so divergence is visible in the server log.

FIX #4 & #5  _activity_breakdown() — scaling for ALL per-person activities
        car_wash and gardening are now included in SCALES_WITH_PEOPLE so
        their benchmarks are multiplied by household size before comparison,
        matching the treatment of drinking / cooking / bathing / etc.
        The set is the single source of truth; adding a new activity only
        requires updating that set.

FIX #6  _generate_training_data() — realistic household amounts
        Per-activity amounts now scale with people (± 20 % noise) so the
        model learns that larger households produce higher absolute totals,
        not just higher per-capita values.

FIX #7  _estimate_savings() — bottle size made explicit
        BOTTLE_SIZE_LITERS constant replaces the silent ÷1 assumption so
        the unit conversion is visible and easy to change.

FIX #8  Key coupling between self.benchmarks and FEATURE_COLUMNS
        FEATURE_COLUMNS is now the single authoritative ordering for the
        feature vector.  _build_features() and _generate_training_data()
        both build arrays by iterating FEATURE_COLUMNS, not by relying on
        dict insertion order.  BENCHMARK_DEFAULTS is keyed to exactly the
        same names as FEATURE_COLUMNS.

FIX #9  Cache version guard
        The joblib payload now includes a CACHE_VERSION integer and a copy
        of FEATURE_COLUMNS.  _load_cached_model() rejects the cache and
        retrains whenever either value does not match the current code,
        preventing stale models from silently producing wrong predictions.
------------------------------------------------------------------
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_PATH      = BASE_DIR / "data" / "household_water_consumption.csv"
MODEL_CACHE_PATH = BASE_DIR / "model_cache" / "water_advisor_model.joblib"

# ---------------------------------------------------------------------------
# FIX #9 — increment this whenever FEATURE_COLUMNS or thresholds change so
# the old cache is automatically invalidated on the next server start.
# ---------------------------------------------------------------------------
CACHE_VERSION = 2

# ---------------------------------------------------------------------------
# FIX #8 — FEATURE_COLUMNS is the single source of truth for feature order.
# Every place that builds a feature array iterates this list, not the
# benchmark dict, so ordering can never silently drift.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    "People",
    "Drinking_Liters",
    "Cooking_Liters",
    "Bathroom_Liters",
    "Toilet_Liters",
    "Laundry_Liters",
    "Dishwashing_Liters",
    "Gardening_Liters",
    "CarWash_Liters",
]

# ---------------------------------------------------------------------------
# FIX #1 — explicit mapping from HTML form field names to internal keys.
# Add entries here whenever the HTML form uses a different name to what the
# CSV / FEATURE_COLUMNS uses.  Nothing else needs to change.
# ---------------------------------------------------------------------------
FORM_KEY_MAP: Dict[str, str] = {
    # form field  →  FEATURE_COLUMNS name
    "people":       "People",
    "drinking":     "Drinking_Liters",
    "cooking":      "Cooking_Liters",
    "bathroom":     "Bathroom_Liters",   # HTML sends "bathroom"; column is "Bathroom_Liters"
    "toilet":       "Toilet_Liters",
    "laundry":      "Laundry_Liters",
    "dishwashing":  "Dishwashing_Liters",
    "gardening":    "Gardening_Liters",
    "car_wash":     "CarWash_Liters",
}

# ---------------------------------------------------------------------------
# FIX #8 — benchmark defaults keyed to FEATURE_COLUMNS names (not to a
# separate benchmark dict with different keys).
# Values are WHO/BIS per-person/day litres; People is a count, not litres.
# ---------------------------------------------------------------------------
BENCHMARK_DEFAULTS: Dict[str, float] = {
    "People":            3.0,   # default household size if omitted
    "Drinking_Liters":   3.0,
    "Cooking_Liters":    6.0,
    "Bathroom_Liters":  80.0,
    "Toilet_Liters":    30.0,
    "Laundry_Liters":   40.0,
    "Dishwashing_Liters": 15.0,
    "Gardening_Liters": 20.0,
    "CarWash_Liters":    5.0,
}

# Activity columns only (everything except "People")
ACTIVITY_COLUMNS: List[str] = [c for c in FEATURE_COLUMNS if c != "People"]

# ---------------------------------------------------------------------------
# FIX #4 & #5 — ALL activities that scale linearly with household size.
# car_wash and gardening are now included.  Benchmark comparison in
# _activity_breakdown() uses this set as the single source of truth.
# ---------------------------------------------------------------------------
SCALES_WITH_PEOPLE: set = {
    "Drinking_Liters",
    "Cooking_Liters",
    "Bathroom_Liters",
    "Toilet_Liters",
    "Laundry_Liters",
    "Dishwashing_Liters",
    "Gardening_Liters",
    "CarWash_Liters",
}

# ---------------------------------------------------------------------------
# FIX #7 — explicit bottle size constant so the unit conversion is visible.
# ---------------------------------------------------------------------------
BOTTLE_SIZE_LITERS: float = 1.0   # 1-litre PET bottle (change to 0.5 if preferred)


class WaterUsageAdvisor:
    """
    The AI-powered advisor class.

    Primary classification uses the trained Random Forest model.
    A rule-based fallback (_classify_per_capita) is always available and
    is used to label training data and as a sanity check against the model.

    RANDOM FOREST explained simply:
      Imagine asking 100 different experts the same question and taking the
      majority vote.  That is essentially what Random Forest does — it builds
      100 decision trees and combines their predictions.
    """

    def __init__(self) -> None:
        self.model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=8,
        )
        self.scaler: StandardScaler = StandardScaler()
        self.is_trained: bool = False
        self._train_lock: threading.Lock = threading.Lock()

        self.categories: List[str] = ["Low", "Moderate", "High", "Critical"]

        # Per-capita liters/day thresholds separating the four categories.
        # Centralised here and used by _classify_per_capita(), the synthetic
        # data generator, and the score calculation so they can never diverge.
        self.thresholds: Dict[str, float] = {
            "low_max":      120.0,
            "moderate_max": 200.0,
            "high_max":     300.0,
        }

        # Recommended daily per-person usage (sum of per-person benchmarks).
        self.daily_recommended: float = sum(
            v for k, v in BENCHMARK_DEFAULTS.items() if k != "People"
        )  # ≈ 199 L/person/day

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(value: Any, default: float, minimum: float = 0.0) -> float:
        """Converts *value* to a non-negative float, returning *default* on
        any failure (None, blank string, non-numeric text, negative number)."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, v)

    @staticmethod
    def _safe_int(value: Any, default: int, minimum: int = 1) -> int:
        """Converts *value* to an int no less than *minimum*."""
        try:
            v = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, v)

    def _classify_per_capita(self, per_capita: float) -> int:
        """
        Rule-based fallback classifier.

        Maps a per-capita litres/day figure to a category index:
          0 = Low, 1 = Moderate, 2 = High, 3 = Critical

        Used to:
          • label rows when building training data
          • provide a sanity-check against the ML model's prediction
          • serve as the classification path when the model is not trained
        """
        if per_capita < self.thresholds["low_max"]:
            return 0
        if per_capita < self.thresholds["moderate_max"]:
            return 1
        if per_capita < self.thresholds["high_max"]:
            return 2
        return 3

    def _resolve_input(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        FIX #1 — translates raw form data (HTML field names) to the internal
        FEATURE_COLUMNS namespace and applies safe type conversion.

        Returns a dict keyed by FEATURE_COLUMNS names with validated floats.
        """
        resolved: Dict[str, float] = {}
        for form_key, col_name in FORM_KEY_MAP.items():
            raw = data.get(form_key)
            default = BENCHMARK_DEFAULTS[col_name]
            if col_name == "People":
                resolved[col_name] = float(self._safe_int(raw, default=int(default), minimum=1))
            else:
                resolved[col_name] = self._safe_float(raw, default=default)
        return resolved

    # ── Training ────────────────────────────────────────────────────────────

    def _generate_training_data(self, n_samples: int = 2000):
        """
        Generates synthetic training data used as a fallback when the real
        CSV is unavailable.

        FIX #6 — per-activity amounts now scale with household size (± 20 %
        noise) so the model learns that larger households produce higher
        absolute totals.  Previously all households used the same activity
        distributions regardless of size, making the People feature almost
        useless to the model.

        Returns
        -------
        X : np.ndarray, shape (n_samples, len(FEATURE_COLUMNS))
        y : np.ndarray, shape (n_samples,)  — integer category labels
        """
        rng = np.random.default_rng(42)

        records: List[List[float]] = []
        labels:  List[int]        = []

        # Per-person mean and std for each activity column (same order as
        # ACTIVITY_COLUMNS so we can zip cleanly).
        per_person_params = {
            "Drinking_Liters":    (3,  1),
            "Cooking_Liters":     (6,  2),
            "Bathroom_Liters":    (80, 30),
            "Toilet_Liters":      (30, 10),
            "Laundry_Liters":     (40, 20),
            "Dishwashing_Liters": (15,  8),
            "Gardening_Liters":   (20, 20),
            "CarWash_Liters":     (5,   8),
        }

        for _ in range(n_samples):
            people = int(rng.integers(1, 9))  # 1-8

            # FIX #6: scale by people with ±20 % household-level variation so
            # the model sees realistic absolute volumes for each household size.
            household_scale = people * rng.uniform(0.8, 1.2)

            activity_values: List[float] = []
            for col in ACTIVITY_COLUMNS:
                mu, sigma = per_person_params[col]
                val = max(0.0, rng.normal(mu * household_scale, sigma * people))
                activity_values.append(val)

            total     = sum(activity_values)
            per_cap   = total / people

            # Feature vector order must match FEATURE_COLUMNS exactly.
            records.append([float(people)] + activity_values)
            labels.append(self._classify_per_capita(per_cap))

        return np.array(records), np.array(labels)

    def _load_cached_model(self) -> bool:
        """
        Loads a previously trained model from disk.

        FIX #9 — validates CACHE_VERSION and FEATURE_COLUMNS before accepting
        the cache.  A mismatch triggers a full retrain so stale models can
        never produce silent wrong predictions after a code change.
        """
        if not MODEL_CACHE_PATH.exists():
            return False
        try:
            cached = joblib.load(MODEL_CACHE_PATH)

            # Version guard
            if cached.get("version") != CACHE_VERSION:
                logger.warning(
                    "Cache version mismatch (cached=%s, current=%s); retraining.",
                    cached.get("version"), CACHE_VERSION,
                )
                return False

            # Feature-columns guard
            if cached.get("feature_columns") != FEATURE_COLUMNS:
                logger.warning("Cache feature columns differ from current code; retraining.")
                return False

            self.model    = cached["model"]
            self.scaler   = cached["scaler"]
            self.is_trained = True
            logger.info("Loaded cached model (v%s) from %s", CACHE_VERSION, MODEL_CACHE_PATH)
            return True

        except Exception as exc:
            logger.warning("Could not load cached model (%s); retraining.", exc)
            return False

    def _save_cached_model(self) -> None:
        """Persists the trained model to disk with version metadata."""
        try:
            MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "version":         CACHE_VERSION,
                    "feature_columns": FEATURE_COLUMNS,   # FIX #9
                    "model":           self.model,
                    "scaler":          self.scaler,
                },
                MODEL_CACHE_PATH,
            )
            logger.info("Model cached to %s", MODEL_CACHE_PATH)
        except Exception as exc:
            # Caching is an optimisation, not a requirement.
            logger.warning("Could not cache trained model: %s", exc)

    def train(self, force_retrain: bool = False) -> None:
        """
        Trains the Random Forest classifier.

        Call order:
          1. Try the on-disk cache (fast path).
          2. Try the real CSV dataset.
          3. Fall back to synthetic data if neither is available.

        Thread-safe: protected by _train_lock with double-checked locking.
        """
        if not force_retrain and self._load_cached_model():
            return

        X: Optional[np.ndarray] = None
        y: Optional[np.ndarray] = None

        try:
            logger.info("Loading dataset from %s", DATA_PATH)
            df = pd.read_csv(DATA_PATH)

            # Validate that all required columns are present before proceeding.
            missing = set(FEATURE_COLUMNS) - set(df.columns)
            if missing:
                raise KeyError(f"CSV is missing columns: {missing}")

            X = df[FEATURE_COLUMNS].to_numpy()
            y = df["PerCapita_Liters"].apply(self._classify_per_capita).to_numpy()

        except (FileNotFoundError, KeyError, pd.errors.EmptyDataError) as exc:
            logger.warning(
                "Real dataset unavailable or malformed (%s); training on synthetic data.", exc
            )
            X, y = self._generate_training_data()

        logger.info("Scaling %d training samples across %d features…", len(X), X.shape[1])
        X_scaled = self.scaler.fit_transform(X)

        logger.info("Training Random Forest…")
        self.model.fit(X_scaled, y)

        self.is_trained = True
        logger.info("Model trained successfully.")
        self._save_cached_model()

    # ── Feature construction ─────────────────────────────────────────────────

    def _build_features(self, resolved: Dict[str, float]) -> np.ndarray:
        """
        Converts the resolved (already-translated and type-safe) input dict
        into the feature array the model expects.

        FIX #8 — iterates FEATURE_COLUMNS (not self.benchmarks) so the
        column order is guaranteed to match what the model was trained on,
        regardless of dict insertion order.

        Parameters
        ----------
        resolved : output of _resolve_input()

        Returns
        -------
        np.ndarray of shape (1, len(FEATURE_COLUMNS))
        """
        row = [resolved.get(col, BENCHMARK_DEFAULTS[col]) for col in FEATURE_COLUMNS]
        return np.array([row])

    # ── Analysis ────────────────────────────────────────────────────────────

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis entry point.

        Parameters
        ----------
        data : raw form data from the HTTP request

        Returns
        -------
        dict with keys:
          category, category_index, confidence, total_liters, per_capita,
          benchmark, conservation_score, usage_pct, breakdown,
          recommendations, savings, probabilities
        """
        # Lazy training — thread-safe double-checked locking.
        if not self.is_trained:
            with self._train_lock:
                if not self.is_trained:
                    self.train()

        # FIX #1 — translate & validate form input once, reuse everywhere.
        resolved = self._resolve_input(data)

        people  = int(resolved["People"])
        total   = sum(resolved[col] for col in ACTIVITY_COLUMNS)
        per_cap = total / people

        # Build the scaled feature array.
        features        = self._build_features(resolved)
        features_scaled = self.scaler.transform(features)

        # ── FIX #3: use the trained model as the primary classifier ──────
        model_class_idx = int(self.model.predict(features_scaled)[0])

        # Rule-based classification kept as a sanity check.
        rule_class_idx = self._classify_per_capita(per_cap)

        if model_class_idx != rule_class_idx:
            logger.warning(
                "Model prediction (%s) differs from rule-based (%s) for per_cap=%.1f L.",
                self.categories[model_class_idx],
                self.categories[rule_class_idx],
                per_cap,
            )

        # The model's prediction is the authoritative result.
        category_index = model_class_idx

        # Probability distribution across all four classes.
        pred_proba = self.model.predict_proba(features_scaled)[0]
        logger.debug("Model classes: %s  probabilities: %s", self.model.classes_, pred_proba)

        # Map probabilities through actual class indices (guards against a
        # training set missing an entire category).
        proba_by_class = dict(zip(self.model.classes_, pred_proba))
        probabilities  = {
            cat: round(float(proba_by_class.get(idx, 0.0)) * 100, 1)
            for idx, cat in enumerate(self.categories)
        }
        confidence = round(float(max(pred_proba)) * 100, 1)

        # ── FIX #2: two unambiguous score metrics ─────────────────────────
        usage_ratio = per_cap / self.thresholds["high_max"]

        # conservation_score: 100 = excellent efficiency, 0 = at/above Critical ceiling
        conservation_score = max(0, min(100, round((1.0 - usage_ratio) * 100)))

        # usage_pct: 0 = zero use, 100 = at Critical ceiling (higher = worse)
        usage_pct = max(0, min(100, round(usage_ratio * 100)))

        breakdown       = self._activity_breakdown(resolved, people)
        recommendations = self._generate_recommendations(resolved, category_index, breakdown)
        savings         = self._estimate_savings(total, category_index, people)

        return {
            "category":           self.categories[category_index],
            "category_index":     category_index,
            "confidence":         confidence,
            "total_liters":       round(total, 1),
            "per_capita":         round(per_cap, 1),
            "benchmark":          self.daily_recommended,
            "conservation_score": conservation_score,   # FIX #2 — replaces ambiguous "score"
            "usage_pct":          usage_pct,            # FIX #2 — second metric for charts
            "breakdown":          breakdown,
            "recommendations":    recommendations,
            "savings":            savings,
            "probabilities":      probabilities,
            # Included for transparency / debugging — remove in production if preferred.
            "rule_category":      self.categories[rule_class_idx],
        }

    # ── Activity breakdown ───────────────────────────────────────────────────

    def _activity_breakdown(
        self, resolved: Dict[str, float], people: int
    ) -> List[Dict[str, Any]]:
        """
        Compares each activity's actual household usage against the
        household-adjusted benchmark and assigns a status label.

        FIX #4 & #5 — uses SCALES_WITH_PEOPLE to scale every per-person
        activity (including car_wash and gardening) before comparison.

        Parameters
        ----------
        resolved : output of _resolve_input()
        people   : validated household size

        Returns
        -------
        List of dicts sorted by ratio descending (worst offenders first).
        """
        display: Dict[str, tuple] = {
            "Drinking_Liters":    ("Drinking water",   "💧"),
            "Cooking_Liters":     ("Cooking",           "🍳"),
            "Bathroom_Liters":    ("Bathing/shower",    "🚿"),
            "Toilet_Liters":      ("Toilet flushing",   "🚽"),
            "Laundry_Liters":     ("Laundry",           "👕"),
            "Dishwashing_Liters": ("Dishwashing",       "🍽️"),
            "Gardening_Liters":   ("Gardening",         "🌱"),
            "CarWash_Liters":     ("Car washing",       "🚗"),
        }

        result: List[Dict[str, Any]] = []

        for col in ACTIVITY_COLUMNS:
            label, icon         = display[col]
            per_person_bench    = BENCHMARK_DEFAULTS[col]
            actual_value        = resolved.get(col, per_person_bench)

            # FIX #4 & #5 — all entries in SCALES_WITH_PEOPLE get scaled.
            household_benchmark = (
                per_person_bench * people
                if col in SCALES_WITH_PEOPLE
                else per_person_bench
            )

            ratio   = actual_value / household_benchmark if household_benchmark > 0 else 1.0
            per_head = round(actual_value / people, 1)

            if ratio <= 0.8:
                status = "excellent"
            elif ratio <= 1.2:
                status = "good"
            elif ratio <= 2.0:
                status = "high"
            else:
                status = "critical"

            result.append(
                {
                    "key":       col,
                    "label":     label,
                    "icon":      icon,
                    "value":     round(actual_value, 1),
                    "per_head":  per_head,
                    "benchmark": per_person_bench,
                    "household_benchmark": round(household_benchmark, 1),
                    "ratio":     round(ratio, 2),
                    "status":    status,
                }
            )

        result.sort(key=lambda x: x["ratio"], reverse=True)
        return result

    # ── Recommendations ──────────────────────────────────────────────────────

    def _generate_recommendations(
        self,
        resolved: Dict[str, float],
        category_index: int,
        breakdown: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Returns up to 6 prioritised, actionable recommendations.

        General tips are always included.  Activity-specific tips are added
        for any activity whose household usage exceeds 150 % of its benchmark.
        An encouragement tip is appended for Low / Moderate usage.
        """
        tip_bank: Dict[str, List[Dict[str, Any]]] = {
            "Bathroom_Liters": [
                {
                    "title":    "Switch to shorter showers",
                    "detail":   "Reducing shower time from 10 to 5 minutes saves ~35 litres per person daily.",
                    "saving":   35,
                    "icon":     "🚿",
                    "priority": "high",
                },
                {
                    "title":    "Install a low-flow showerhead",
                    "detail":   "A low-flow head uses 6-8 L/min instead of 15 L/min — a 50 % reduction.",
                    "saving":   50,
                    "icon":     "🔧",
                    "priority": "medium",
                },
            ],
            "Toilet_Liters": [
                {
                    "title":    "Use a dual-flush toilet",
                    "detail":   "Dual-flush models use 3 L (liquid) and 6 L (solid) vs. 13 L for old single-flush.",
                    "saving":   25,
                    "icon":     "🚽",
                    "priority": "medium",
                },
                {
                    "title":    "Check for toilet leaks",
                    "detail":   "A leaking toilet can waste 200–400 litres per day. Add food colouring to the tank to detect leaks.",
                    "saving":   200,
                    "icon":     "🔍",
                    "priority": "high",
                },
            ],
            "Laundry_Liters": [
                {
                    "title":    "Only run full loads",
                    "detail":   "A full washing machine load uses ~40 L. Half-loads use nearly the same — always fill it up.",
                    "saving":   30,
                    "icon":     "👕",
                    "priority": "medium",
                },
                {
                    "title":    "Use cold-water cycles",
                    "detail":   "Cold washes are equally effective for most laundry and save energy too.",
                    "saving":   10,
                    "icon":     "❄️",
                    "priority": "low",
                },
            ],
            "Gardening_Liters": [
                {
                    "title":    "Water plants at dawn or dusk",
                    "detail":   "Watering in cooler hours reduces evaporation by up to 30 %.",
                    "saving":   15,
                    "icon":     "🌅",
                    "priority": "medium",
                },
                {
                    "title":    "Use drip irrigation",
                    "detail":   "Drip systems deliver water directly to roots, cutting garden use by 30–50 %.",
                    "saving":   40,
                    "icon":     "🌱",
                    "priority": "high",
                },
                {
                    "title":    "Collect rainwater",
                    "detail":   "A 200 L rainwater barrel can offset garden watering for days after rainfall.",
                    "saving":   20,
                    "icon":     "🌧️",
                    "priority": "medium",
                },
            ],
            "Dishwashing_Liters": [
                {
                    "title":    "Use a dishwasher (full load)",
                    "detail":   "A full dishwasher uses ~12 L vs. ~40 L for handwashing the same amount.",
                    "saving":   12,
                    "icon":     "🍽️",
                    "priority": "low",
                },
                {
                    "title":    "Don't leave the tap running",
                    "detail":   "A running tap uses ~6 L/min. Fill a basin instead of rinsing under flowing water.",
                    "saving":   18,
                    "icon":     "🚰",
                    "priority": "medium",
                },
            ],
            "CarWash_Liters": [
                {
                    "title":    "Use a bucket instead of a hose",
                    "detail":   "A hose uses 150–400 L per wash. A bucket wash uses only ~30 L.",
                    "saving":   120,
                    "icon":     "🚗",
                    "priority": "high",
                },
            ],
            "Cooking_Liters": [
                {
                    "title":    "Reuse vegetable rinse water",
                    "detail":   "Water used to rinse vegetables can water plants directly.",
                    "saving":   3,
                    "icon":     "🥦",
                    "priority": "low",
                },
            ],
            "general": [
                {
                    "title":    "Fix all dripping taps",
                    "detail":   "A tap dripping once per second wastes ~3 000 litres per month. Fix leaks promptly.",
                    "saving":   100,
                    "icon":     "🔧",
                    "priority": "high",
                },
                {
                    "title":    "Install water-saving aerators",
                    "detail":   "Tap aerators add air to the water flow, reducing usage by 30–50 % with no change in experience.",
                    "saving":   30,
                    "icon":     "💡",
                    "priority": "medium",
                },
            ],
        }

        recs: List[Dict[str, Any]] = list(tip_bank["general"])

        for activity in breakdown:
            if activity["ratio"] > 1.5 and activity["key"] in tip_bank:
                recs.extend(tip_bank[activity["key"]])

        if category_index <= 1:
            recs.append(
                {
                    "title":    "Great job! Keep monitoring",
                    "detail":   "Your usage is within healthy limits. Keep tracking to maintain this habit.",
                    "saving":   0,
                    "icon":     "🌟",
                    "priority": "low",
                }
            )

        priority_order = {"high": 0, "medium": 1, "low": 2}
        recs.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return recs[:6]

    # ── Savings estimate ─────────────────────────────────────────────────────

    def _estimate_savings(
        self, total_liters: float, category_index: int, people: int
    ) -> Dict[str, Any]:
        """
        Estimates water and cost savings from adopting the recommendations.

        FIX #7 — BOTTLE_SIZE_LITERS makes the unit conversion explicit so
        it is easy to change and impossible to miss.

        Assumes ₹5 per 1 000 litres (India average municipal tariff).
        """
        reduction_pct_map = {0: 5, 1: 10, 2: 25, 3: 40}
        pct = reduction_pct_map.get(category_index, 20)

        daily_saving_L   = round(total_liters * pct / 100, 1)
        monthly_saving_L = round(daily_saving_L * 30, 1)
        yearly_saving_L  = round(daily_saving_L * 365, 1)

        cost_per_liter   = 0.005   # ₹5 per 1 000 L → ₹0.005 per L
        monthly_saving_inr = round(monthly_saving_L * cost_per_liter, 2)
        yearly_saving_inr  = round(yearly_saving_L  * cost_per_liter, 2)

        # FIX #7 — explicit bottle size so the conversion is auditable.
        bottles_per_day = round(daily_saving_L / BOTTLE_SIZE_LITERS, 0)

        return {
            "reduction_pct":    pct,
            "daily_liters":     daily_saving_L,
            "monthly_liters":   monthly_saving_L,
            "yearly_liters":    yearly_saving_L,
            "monthly_inr":      monthly_saving_inr,
            "yearly_inr":       yearly_saving_inr,
            "bottles_saved_daily": bottles_per_day,        # FIX #7
            "bottle_size_liters":  BOTTLE_SIZE_LITERS,     # FIX #7 — expose to frontend
        }

    # ── Historical trend ─────────────────────────────────────────────────────

    def get_historical_trend(self) -> Dict[str, Any]:
        """
        Returns simulated 30-day historical water usage data for chart display.

        NOTE: This is placeholder/demo data.  Wiring it up to real per-user
        history (e.g. from a database) is a planned feature addition and is
        flagged as a known limitation in the project report.
        """
        rng    = np.random.default_rng(0)
        days   = list(range(1, 31))
        trend  = np.linspace(220, 160, 30)
        noise  = rng.normal(0, 20, 30)
        values = np.clip(trend + noise, 80, 350)

        return {
            "labels":    [f"Day {d}" for d in days],
            "values":    [round(float(v), 1) for v in values],
            "benchmark": self.daily_recommended,
        }