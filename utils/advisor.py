"""
====================================================
  Smart Water Usage Advisor — AI Core Logic
====================================================
This module contains the main AI/ML brain of the application.

WHAT IT DOES:
  - Trains a machine learning model to classify water usage
    as Low / Moderate / High / Critical
  - Generates personalized conservation recommendations
  - Calculates savings estimates
  - Produces historical trend data for charts

LIBRARIES USED:
  - scikit-learn: RandomForestClassifier (ensemble of decision trees)
  - numpy: Fast numerical computations
  - pandas: Data manipulation and analysis
  - joblib: Caches the trained model to disk (ships with scikit-learn,
    no new dependency to install)

------------------------------------------------------------------
CHANGE LOG — reliability / performance fixes applied in this revision
------------------------------------------------------------------
1. Removed an unused `import random`.
2. `train()` no longer crashes if the CSV is missing or malformed —
   it now falls back to the synthetic data generator
   (`_generate_training_data`, which used to be dead code that was
   defined but never actually called anywhere).
3. Fixed a real bug in `analyze()`: usage-category probabilities were
   zipped against `self.categories` assuming the model's classes were
   always `[0, 1, 2, 3]` in that exact order. If the training data was
   ever missing one category (e.g. no "Critical" rows happened to be
   sampled), this silently produced mislabeled probabilities.
   Probabilities are now mapped through `self.model.classes_` explicitly.
4. Removed a data-leakage problem: `Total_Liters` and `PerCapita_Liters`
   used to be BOTH input features AND the exact value the label was
   computed from. That let the model degenerate into a single threshold
   check instead of learning anything from behaviour (showering,
   laundry, etc.). The model now trains only on the raw per-activity
   numbers, so predictions reflect real usage patterns.
5. Removed duplicated "magic number" benchmark values that were
   hardcoded in three different places (`_build_features`, `analyze`,
   `_activity_breakdown`) — they now all read from one
   `self.benchmarks` dictionary, so they can't drift out of sync.
6. Centralized the Low/Moderate/High/Critical thresholds into
   `self.thresholds` and a single `_classify_per_capita()` helper
   (previously duplicated almost identically in two places).
7. Added input sanitization (`_safe_float` / `_safe_int`) so malformed
   or missing form input (blank field, stray text, negative number)
   can't crash a request.
8. Added model caching with `joblib` — the model trains once and is
   reloaded from disk afterward, instead of retraining from scratch
   on every server restart.
9. Added a threading lock around lazy training so two concurrent
   requests can't trigger duplicate simultaneous training.
10. Replaced leftover debug `print()` statements with proper
    `logging` calls (silent by default, visible if the app enables
    DEBUG logging).
11. Resolved the CSV/cache paths relative to this file's own location
    instead of the process's current working directory, so behavior
    doesn't change depending on how/where the app is launched.
------------------------------------------------------------------
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "household_water_consumption.csv"

MODEL_CACHE_PATH = BASE_DIR / "model_cache" / "water_advisor_model.joblib"

# Column order must match the order features are assembled in
# _build_features() and _generate_training_data() below.
FEATURE_COLUMNS = [
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


class WaterUsageAdvisor:
    """
    The AI-powered advisor class.

    Trains a Random Forest classifier on household water-usage data,
    then uses that model to classify new input and generate
    recommendations.

    RANDOM FOREST explained simply:
      Imagine asking 100 different experts the same question and
      taking the majority vote. That's essentially what Random Forest
      does — it builds 100 decision trees and combines their
      predictions.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=8,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self._train_lock = threading.Lock()

        self.categories = ["Low", "Moderate", "High", "Critical"]

        # Per-capita liters/day thresholds that separate the categories.
        # Centralized here so they can't drift out of sync between the
        # synthetic generator, the CSV-based trainer, and the score calc.
        self.thresholds = {"low_max": 120, "moderate_max": 200, "high_max": 300}

        # Average daily water-use benchmarks (liters/person/day)
        # Source: WHO / UN Water recommendations
        self.benchmarks = {
            "drinking": 3,
            "cooking": 6,
            "bathing": 80,
            "toilet": 30,
            "laundry": 40,
            "dishes": 15,
            "gardening": 20,
            "car_wash": 5,
        }

        # Total recommended daily usage (WHO minimum is 50L, comfortable is 100-150L)
        self.daily_recommended = sum(self.benchmarks.values())  # ~199 liters

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(value: Any, default: float, minimum: float = 0.0) -> float:
        """Safely converts user input to a float, falling back to a
        sane default instead of raising on bad input (blank strings,
        None, non-numeric text, negative numbers, etc.)."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, v)

    @staticmethod
    def _safe_int(value: Any, default: int, minimum: int = 1) -> int:
        try:
            v = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, v)

    def _classify_per_capita(self, per_capita: float) -> int:
        """Maps a per-capita liters/day figure to a category index.
        Single source of truth for the thresholds — used by both the
        CSV trainer and the synthetic data fallback."""
        if per_capita < self.thresholds["low_max"]:
            return 0  # Low
        if per_capita < self.thresholds["moderate_max"]:
            return 1  # Moderate
        if per_capita < self.thresholds["high_max"]:
            return 2  # High
        return 3  # Critical

    # ── Training ────────────────────────────────────────────────────────────

    def _generate_training_data(self, n_samples: int = 2000):
        """
        Generates synthetic training data. Used automatically as a
        fallback if the real CSV dataset can't be loaded, so the app
        keeps working (e.g. on a fresh checkout missing the data file)
        instead of crashing.

        Returns:
          X: Feature matrix (input variables)
          y: Target labels (Low/Moderate/High/Critical)
        """
        rng = np.random.default_rng(42)  # Reproducible random numbers

        records: List[List[float]] = []
        labels: List[int] = []

        for _ in range(n_samples):
            drinking = max(0.0, rng.normal(3, 1))
            cooking = max(0.0, rng.normal(6, 2))
            bathing = max(0.0, rng.normal(80, 30))
            toilet = max(0.0, rng.normal(30, 10))
            laundry = max(0.0, rng.normal(40, 20))
            dishes = max(0.0, rng.normal(15, 8))
            gardening = max(0.0, rng.normal(20, 20))
            car_wash = max(0.0, rng.normal(5, 8))

            people = int(rng.integers(1, 9))  # 1-8 people
            total = (
                drinking + cooking + bathing + toilet + laundry + dishes + gardening + car_wash
            )
            per_capita = total / people

            # NOTE: total/per_capita are deliberately NOT included as
            # features below — only the raw activity numbers are. This
            # avoids handing the model the literal value its label is
            # derived from (see change #4 above).
            records.append(
                [people, drinking, cooking, bathing, toilet, laundry, dishes, gardening, car_wash]
            )
            labels.append(self._classify_per_capita(per_capita))

        return np.array(records), np.array(labels)

    def _load_cached_model(self) -> bool:
        """Loads a previously trained model from disk, if present."""
        if not MODEL_CACHE_PATH.exists():
            return False
        try:
            cached = joblib.load(MODEL_CACHE_PATH)
            self.model = cached["model"]
            self.scaler = cached["scaler"]
            self.is_trained = True
            logger.info("Loaded cached water-usage model from %s", MODEL_CACHE_PATH)
            return True
        except Exception as exc:  # corrupted cache, version mismatch, etc.
            logger.warning("Could not load cached model (%s); retraining.", exc)
            return False

    def _save_cached_model(self) -> None:
        try:
            MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": self.model, "scaler": self.scaler}, MODEL_CACHE_PATH)
        except Exception as exc:
            # Caching is an optimization, not a requirement — never let a
            # failed write (e.g. read-only filesystem) break the app.
            logger.warning("Could not cache trained model: %s", exc)

    def train(self, force_retrain: bool = False) -> None:
        """
        Trains the AI model. Called once when the app starts (or
        lazily on first request). Tries the cache first, then the real
        CSV dataset, and falls back to synthetic data if neither is
        available — so the app stays reliable even with a missing or
        corrupted dataset.
        """
        if not force_retrain and self._load_cached_model():
            return

        try:
            logger.info("Loading dataset from %s", DATA_PATH)
            df = pd.read_csv(DATA_PATH)
            # .to_numpy() strips pandas column names before fitting, so the
            # scaler/model are trained on a plain array — matching the plain
            # array used later at prediction time and avoiding a spurious
            # "feature names" mismatch warning from scikit-learn.
            X = df[FEATURE_COLUMNS].to_numpy()
            y = df["PerCapita_Liters"].apply(self._classify_per_capita).to_numpy()
        except (FileNotFoundError, KeyError, pd.errors.EmptyDataError) as exc:
            logger.warning(
                "Real dataset unavailable or malformed (%s); training on synthetic data instead.",
                exc,
            )
            X, y = self._generate_training_data()

        logger.info("Scaling features...")
        # StandardScaler: transforms each feature to have mean=0, std=1
        # so large-scale features don't dominate the model.
        X_scaled = self.scaler.fit_transform(X)

        logger.info("Training Random Forest model...")
        self.model.fit(X_scaled, y)

        self.is_trained = True
        logger.info("Model trained successfully.")
        self._save_cached_model()

    # ── Analysis ────────────────────────────────────────────────────────────

    def _build_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Converts raw user input into the feature array the model expects."""
        people = self._safe_int(data.get("people"), default=1, minimum=1)
        values = [
            self._safe_float(data.get(key), default=float(default))
            for key, default in self.benchmarks.items()
        ]
        return np.array([[people, *values]])

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis function. Takes user input, runs the AI
        prediction, and builds the full response payload.

        Args:
          data (dict): Form data from the frontend

        Returns:
          dict with: category, score, breakdown, recommendations, savings
        """
        if not self.is_trained:
            with self._train_lock:
                if not self.is_trained:  # double-checked after acquiring the lock
                    self.train()

        features = self._build_features(data)
        people = self._safe_int(data.get("people"), default=1, minimum=1)
        total = sum(
            self._safe_float(data.get(key), default=float(default))
            for key, default in self.benchmarks.items()
        )
        per_cap = total / people

        features_scaled = self.scaler.transform(features)
        pred_label = int(self.model.predict(features_scaled)[0])
        if pred_label >= len(self.categories):
            pred_label = len(self.categories) - 1

        pred_proba = self.model.predict_proba(features_scaled)[0]
        confidence = round(float(max(pred_proba)) * 100, 1)
        logger.debug("Model classes: %s", self.model.classes_)
        logger.debug("Prediction probabilities: %s", pred_proba)

        # Map probabilities by actual class label rather than assuming
        # the model saw every category during training (fix for bug #3).
        proba_by_class = dict(zip(self.model.classes_, pred_proba))
        probabilities = {
            cat: round(float(proba_by_class.get(idx, 0.0)) * 100, 1)
            for idx, cat in enumerate(self.categories)
        }

        # Usage score vs benchmark (lower is better, shown as % of the
        # "Critical" threshold reached, capped at 100).
        score = min(100, round((per_cap / self.thresholds["high_max"]) * 100))

        breakdown = self._activity_breakdown(data, people)
        recommendations = self._generate_recommendations(data, pred_label, breakdown)
        savings = self._estimate_savings(total, pred_label, people)

        return {
            "category": self.categories[pred_label],
            "category_index": pred_label,
            "confidence": confidence,
            "total_liters": round(total, 1),
            "per_capita": round(per_cap, 1),
            "benchmark": self.daily_recommended,
            "score": score,
            "breakdown": breakdown,
            "recommendations": recommendations,
            "savings": savings,
            "probabilities": probabilities,
        }

    def _activity_breakdown(self, data: Dict[str, Any], people: int) -> List[Dict[str, Any]]:
        """
        Compares each activity's usage against the recommended benchmark.
        Flags activities that are significantly over the benchmark.

        Returns list of dicts with status for each activity.
        """
        display = {
            "drinking": ("Drinking Water", "💧"),
            "cooking": ("Cooking", "🍳"),
            "bathing": ("Bathing/Shower", "🚿"),
            "toilet": ("Toilet Flushing", "🚽"),
            "laundry": ("Laundry", "👕"),
            "dishes": ("Dishwashing", "🍽️"),
            "gardening": ("Gardening", "🌱"),
            "car_wash": ("Car Washing", "🚗"),
        }

        result = []
        for key, benchmark in self.benchmarks.items():
            label, icon = display[key]
            value = self._safe_float(data.get(key), default=float(benchmark))
            ratio = value / benchmark if benchmark > 0 else 1.0
            per_head = round(value / people, 1)

            # Status based on how much above/below benchmark
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
                    "key": key,
                    "label": label,
                    "icon": icon,
                    "value": round(value, 1),
                    "per_head": per_head,
                    "benchmark": benchmark,
                    "ratio": round(ratio, 2),
                    "status": status,
                }
            )

        # Sort by ratio descending so worst offenders appear first
        result.sort(key=lambda x: x["ratio"], reverse=True)
        return result

    def _generate_recommendations(
        self, data: Dict[str, Any], category_index: int, breakdown: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generates personalized, actionable recommendations based on
        which activities are consuming the most water.

        Returns list of recommendation objects.
        """
        tip_bank = {
            "bathing": [
                {
                    "title": "Switch to Shorter Showers",
                    "detail": "Reducing shower time from 10 to 5 minutes saves ~35 liters per person daily.",
                    "saving": 35,
                    "icon": "🚿",
                    "priority": "high",
                },
                {
                    "title": "Install a Low-Flow Showerhead",
                    "detail": "A low-flow showerhead uses 6-8 L/min instead of 15 L/min - a 50% reduction.",
                    "saving": 50,
                    "icon": "🔧",
                    "priority": "medium",
                },
            ],
            "toilet": [
                {
                    "title": "Use a Dual-Flush Toilet",
                    "detail": "Dual-flush models use 3L for liquid and 6L for solid waste vs. 13L for old single-flush.",
                    "saving": 25,
                    "icon": "🚽",
                    "priority": "medium",
                },
                {
                    "title": "Check for Toilet Leaks",
                    "detail": "A leaking toilet can waste 200-400 liters per day. Put food colouring in the tank to detect leaks.",
                    "saving": 200,
                    "icon": "🔍",
                    "priority": "high",
                },
            ],
            "laundry": [
                {
                    "title": "Only Run Full Loads",
                    "detail": "A full washing machine load uses ~40L. Half-loads use nearly the same water - always fill it up.",
                    "saving": 30,
                    "icon": "👕",
                    "priority": "medium",
                },
                {
                    "title": "Use Cold Water Cycles",
                    "detail": "Cold water washes are equally effective for most laundry and save energy too.",
                    "saving": 10,
                    "icon": "❄️",
                    "priority": "low",
                },
            ],
            "gardening": [
                {
                    "title": "Water Plants at Dawn or Dusk",
                    "detail": "Watering in the cooler parts of the day reduces evaporation by up to 30%.",
                    "saving": 15,
                    "icon": "🌅",
                    "priority": "medium",
                },
                {
                    "title": "Use Drip Irrigation",
                    "detail": "Drip systems deliver water directly to roots, cutting garden water use by 30-50%.",
                    "saving": 40,
                    "icon": "🌱",
                    "priority": "high",
                },
                {
                    "title": "Collect Rainwater",
                    "detail": "A 200L rainwater barrel can offset garden watering for days after rainfall.",
                    "saving": 20,
                    "icon": "🌧️",
                    "priority": "medium",
                },
            ],
            "dishes": [
                {
                    "title": "Use a Dishwasher (Full Load)",
                    "detail": "A full dishwasher uses ~12L vs. ~40L for handwashing the same amount.",
                    "saving": 12,
                    "icon": "🍽️",
                    "priority": "low",
                },
                {
                    "title": "Don't Leave the Tap Running",
                    "detail": "A running tap uses ~6L per minute. Fill a basin instead of rinsing under a running tap.",
                    "saving": 18,
                    "icon": "🚰",
                    "priority": "medium",
                },
            ],
            "car_wash": [
                {
                    "title": "Use a Bucket Instead of a Hose",
                    "detail": "A hose uses 150-400L per wash. A bucket wash uses only ~30L.",
                    "saving": 120,
                    "icon": "🚗",
                    "priority": "high",
                },
            ],
            "cooking": [
                {
                    "title": "Reuse Vegetable Rinse Water",
                    "detail": "Water used to rinse vegetables can be reused to water plants.",
                    "saving": 3,
                    "icon": "🥦",
                    "priority": "low",
                },
            ],
            "general": [
                {
                    "title": "Fix All Dripping Taps",
                    "detail": "A tap dripping once per second wastes ~3,000 liters per month. Fix leaks promptly.",
                    "saving": 100,
                    "icon": "🔧",
                    "priority": "high",
                },
                {
                    "title": "Install Water-Saving Aerators",
                    "detail": "Tap aerators add air to water flow, reducing usage by 30-50% with no change in experience.",
                    "saving": 30,
                    "icon": "💡",
                    "priority": "medium",
                },
            ],
        }

        recs = list(tip_bank["general"])

        # Add activity-specific tips for any activity over 150% of benchmark
        for activity in breakdown:
            if activity["ratio"] > 1.5 and activity["key"] in tip_bank:
                recs.extend(tip_bank[activity["key"]])

        # For moderate/low usage, still show an encouraging tip
        if category_index <= 1:
            recs.append(
                {
                    "title": "Great Job! Keep Monitoring",
                    "detail": "Your water usage is within healthy limits. Continue tracking daily to maintain this habit.",
                    "saving": 0,
                    "icon": "🌟",
                    "priority": "low",
                }
            )

        priority_order = {"high": 0, "medium": 1, "low": 2}
        recs.sort(key=lambda x: priority_order.get(x["priority"], 3))

        # Return top 6 recommendations to avoid overwhelming the user
        return recs[:6]

    def _estimate_savings(self, total_liters: float, category_index: int, people: int) -> Dict[str, Any]:
        """
        Estimates how much water and money could be saved if the user
        follows the recommendations.

        Assumes water costs approximately ₹5 per 1000 liters (India average).
        """
        reduction_pct = {0: 5, 1: 10, 2: 25, 3: 40}
        pct = reduction_pct.get(category_index, 20)

        daily_saving_liters = round(total_liters * pct / 100, 1)
        monthly_saving_liters = round(daily_saving_liters * 30, 1)
        yearly_saving_liters = round(daily_saving_liters * 365, 1)

        cost_per_liter = 0.005  # ₹5 per 1000 liters
        monthly_saving_inr = round(monthly_saving_liters * cost_per_liter, 2)
        yearly_saving_inr = round(yearly_saving_liters * cost_per_liter, 2)

        return {
            "reduction_pct": pct,
            "daily_liters": daily_saving_liters,
            "monthly_liters": monthly_saving_liters,
            "yearly_liters": yearly_saving_liters,
            "monthly_inr": monthly_saving_inr,
            "yearly_inr": yearly_saving_inr,
            "bottles_saved_daily": round(daily_saving_liters, 0),  # 1L bottles
        }

    # ── Historical Trend ────────────────────────────────────────────────────

    def get_historical_trend(self) -> Dict[str, Any]:
        """
        Returns simulated 30-day historical water usage data for chart
        display. NOTE: this is still placeholder/demo data — wiring it
        up to real per-user history (e.g. from a database) would be a
        feature addition rather than a bug fix, but worth flagging for
        your report as a known limitation.
        """
        rng = np.random.default_rng(0)
        days = list(range(1, 31))
        trend = np.linspace(220, 160, 30)
        noise = rng.normal(0, 20, 30)
        values = np.clip(trend + noise, 80, 350)

        return {
            "labels": [f"Day {d}" for d in days],
            "values": [round(float(v), 1) for v in values],
            "benchmark": self.daily_recommended,
        }