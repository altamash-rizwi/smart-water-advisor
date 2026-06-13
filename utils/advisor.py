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
  - scikit-learn: Popular ML library for Python
    (RandomForestClassifier = ensemble of decision trees)
  - numpy: Fast numerical computations
  - pandas: Data manipulation and analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import random


class WaterUsageAdvisor:
    """
    The AI-powered advisor class.

    It trains a Random Forest classifier on synthetic water usage data,
    then uses that model to classify new inputs and generate recommendations.

    RANDOM FOREST explained simply:
      Imagine asking 100 different experts the same question and taking
      the majority vote. That's essentially what Random Forest does —
      it builds 100 decision trees and combines their predictions.
    """

    def __init__(self):
        """
        Constructor — sets up all the components.
        Called automatically when you do: advisor = WaterUsageAdvisor()
        """
        # The ML model — RandomForest is great for beginners:
        # it's robust, handles small datasets well, and rarely overfits
        self.model = RandomForestClassifier(
            n_estimators=100,   # 100 decision trees
            random_state=42,    # Fixed seed → reproducible results
            max_depth=8         # Limit tree depth to avoid overfitting
        )

        # Scaler normalizes input features so all values are on the same scale
        # (e.g., "bathing minutes" and "liters" are very different scales)
        self.scaler = StandardScaler()

        # Track whether the model has been trained
        self.is_trained = False

        # Usage category labels (what the model predicts)
        self.categories = ["Low", "Moderate", "High", "Critical"]

        # Average daily water use benchmarks (in liters)
        # Source: WHO / UN Water recommendations
        self.benchmarks = {
            "drinking":   3,    # 2–4 liters per person per day
            "cooking":    6,    # ~5–7 liters
            "bathing":    80,   # Shower: ~60–100 liters
            "toilet":     30,   # ~6 liters per flush × 5 flushes
            "laundry":    40,   # Washing machine ~40–80 liters/load
            "dishes":     15,   # Handwashing dishes
            "gardening":  20,   # Depends heavily on garden size
            "car_wash":   5,    # Amortized daily (~150L per wash)
        }

        # Total recommended daily usage (WHO minimum is 50L, comfortable is 100–150L)
        self.daily_recommended = sum(self.benchmarks.values())  # ~199 liters

    # ── Training ────────────────────────────────────────────────────────────────

    def _generate_training_data(self, n_samples=2000):
        """
        Generates synthetic training data since we don't have real sensor data.

        In a production system, you'd replace this with real data from:
        - Water meters / IoT sensors
        - Government water utility datasets
        - User-contributed survey data

        Returns:
          X: Feature matrix (input variables)
          y: Target labels (Low/Moderate/High/Critical)
        """
        np.random.seed(42)  # Reproducible random numbers

        records = []
        labels  = []

        for _ in range(n_samples):
            # Randomly generate daily usage for each activity (in liters)
            # np.random.normal(mean, std) draws from a bell curve
            drinking  = max(0, np.random.normal(3,   1))
            cooking   = max(0, np.random.normal(6,   2))
            bathing   = max(0, np.random.normal(80,  30))
            toilet    = max(0, np.random.normal(30,  10))
            laundry   = max(0, np.random.normal(40,  20))
            dishes    = max(0, np.random.normal(15,   8))
            gardening = max(0, np.random.normal(20,  20))
            car_wash  = max(0, np.random.normal(5,    8))

            total = (drinking + cooking + bathing + toilet +
                     laundry + dishes + gardening + car_wash)

            # Number of people in the household (1–8)
            people = np.random.randint(1, 9)

            # Per-capita usage
            per_capita = total / people

            # Features we feed into the model
            features = [drinking, cooking, bathing, toilet,
                        laundry, dishes, gardening, car_wash,
                        total, people, per_capita]

            # Label based on per-capita usage thresholds
            # These thresholds are based on WHO water-use guidelines
            if per_capita < 80:
                label = 0   # Low — very conservative
            elif per_capita < 150:
                label = 1   # Moderate — within healthy range
            elif per_capita < 250:
                label = 2   # High — above average
            else:
                label = 3   # Critical — wasteful

            records.append(features)
            labels.append(label)

        return np.array(records), np.array(labels)

    def train(self):
        """
        Trains the AI model.
        Called once when the app starts.
        """
        print("  [AI] Loading real dataset...")

        df = pd.read_csv("data/household_water_consumption.csv")

        # Features
        X = df[[
            "Bathroom_Liters",
            "Kitchen_Liters",
            "Laundry_Liters",
            "Gardening_Liters"
        ]]

        # Create categories
        def classify_usage(total):
            if total < 300:
                return 0
            elif total < 500:
                return 1
            elif total < 700:
                return 2
            return 3

        y = df["Total_Liters"].apply(classify_usage)

        print("  [AI] Scaling features...")
        # StandardScaler: transforms each feature to have mean=0, std=1
        # This is important so large-scale features don't dominate the model
        X_scaled = self.scaler.fit_transform(X)

        print("  [AI] Training Random Forest model...")
        self.model.fit(X_scaled, y)

        self.is_trained = True
        print("  [AI] Model trained successfully! ✓")

    # ── Analysis ────────────────────────────────────────────────────────────────

    def _build_features(self, data):
        """
        Converts raw user input dict into the feature array the model expects.

        Args:
          data (dict): User's reported usage per activity

        Returns:
          numpy array of shape (1, 11)
        """
        drinking  = float(data.get("drinking",  3))
        cooking   = float(data.get("cooking",   6))
        bathing   = float(data.get("bathing",  80))
        toilet    = float(data.get("toilet",   30))
        laundry   = float(data.get("laundry",  40))
        dishes    = float(data.get("dishes",   15))
        gardening = float(data.get("gardening",20))
        car_wash  = float(data.get("car_wash",  5))
        people    = max(1, int(data.get("people", 1)))

        total      = (drinking + cooking + bathing + toilet +
                      laundry + dishes + gardening + car_wash)
        per_capita = total / people

        return np.array([[
            bathing,
            cooking,
            laundry,
            gardening
        ]])

    def analyze(self, data):
        """
        Main analysis function.
        Takes user input, runs AI prediction, and builds a full response.

        Args:
          data (dict): Form data from the frontend

        Returns:
          dict with: category, score, breakdown, recommendations, savings
        """
        if not self.is_trained:
            self.train()

        # Build feature array
        features = self._build_features(data)
        total = (
             float(data.get("bathing", 80)) +
             float(data.get("cooking", 6)) +
             float(data.get("laundry", 40)) +
             float(data.get("gardening", 20))
         )

        people = max(1, int(data.get("people", 1)))

        per_cap = total / people

        # Scale features and predict
        features_scaled = self.scaler.transform(features)
        pred_label      = int(self.model.predict(features_scaled)[0])
        if pred_label >= len(self.categories):
             pred_label = len(self.categories) - 1
        pred_proba      = self.model.predict_proba(features_scaled)[0]

        # Confidence score (0–100)
        confidence = round(float(max(pred_proba)) * 100, 1)
        print("Model classes:", self.model.classes_)
        print("Prediction probabilities:", pred_proba)

        # Usage score vs benchmark (lower is better but we show how over/under they are)
        score = min(100, round((per_cap / self.daily_recommended) * 100))

        # Generate per-activity breakdown
        breakdown = self._activity_breakdown(data, people)

        # Generate personalized recommendations
        recommendations = self._generate_recommendations(data, pred_label, breakdown)

        # Estimate potential water and cost savings
        savings = self._estimate_savings(total, pred_label, people)

        return {
            "category": self.categories[min(pred_label, len(self.categories)-1)],
            "category_index":  pred_label,
            "confidence":      confidence,
            "total_liters":    round(total, 1),
            "per_capita":      round(per_cap, 1),
            "benchmark":       self.daily_recommended,
            "score":           score,
            "breakdown":       breakdown,
            "recommendations": recommendations,
            "savings":         savings,
            "probabilities": {
                cat: round(float(p) * 100, 1)
                for cat, p in zip(self.categories, pred_proba)
            }
        }

    def _activity_breakdown(self, data, people):
        """
        Compares each activity's usage against the recommended benchmark.
        Flags activities that are significantly over the benchmark.

        Returns list of dicts with status for each activity.
        """
        activities = {
            "drinking":  ("Drinking Water",   "💧", 3),
            "cooking":   ("Cooking",           "🍳", 6),
            "bathing":   ("Bathing/Shower",    "🚿", 80),
            "toilet":    ("Toilet Flushing",   "🚽", 30),
            "laundry":   ("Laundry",           "👕", 40),
            "dishes":    ("Dishwashing",       "🍽️", 15),
            "gardening": ("Gardening",         "🌱", 20),
            "car_wash":  ("Car Washing",       "🚗", 5),
        }

        result = []
        for key, (label, icon, benchmark) in activities.items():
            value     = float(data.get(key, benchmark))
            ratio     = value / benchmark if benchmark > 0 else 1
            per_head  = round(value / people, 1)

            # Status based on how much above/below benchmark
            if ratio <= 0.8:
                status = "excellent"
            elif ratio <= 1.2:
                status = "good"
            elif ratio <= 2.0:
                status = "high"
            else:
                status = "critical"

            result.append({
                "key":       key,
                "label":     label,
                "icon":      icon,
                "value":     round(value, 1),
                "per_head":  per_head,
                "benchmark": benchmark,
                "ratio":     round(ratio, 2),
                "status":    status,
            })

        # Sort by ratio descending so worst offenders appear first
        result.sort(key=lambda x: x["ratio"], reverse=True)
        return result

    def _generate_recommendations(self, data, category_index, breakdown):
        """
        Generates personalized, actionable recommendations based on
        which activities are consuming the most water.

        Returns list of recommendation objects.
        """
        # Recommendation database — keyed by activity
        tip_bank = {
            "bathing": [
                {
                    "title":   "Switch to Shorter Showers",
                    "detail":  "Reducing shower time from 10 to 5 minutes saves ~35 liters per person daily.",
                    "saving":  35,
                    "icon":    "🚿",
                    "priority":"high"
                },
                {
                    "title":   "Install a Low-Flow Showerhead",
                    "detail":  "A low-flow showerhead uses 6–8 L/min instead of 15 L/min — a 50% reduction.",
                    "saving":  50,
                    "icon":    "🔧",
                    "priority":"medium"
                }
            ],
            "toilet": [
                {
                    "title":   "Use a Dual-Flush Toilet",
                    "detail":  "Dual-flush models use 3L for liquid and 6L for solid waste vs. 13L for old single-flush.",
                    "saving":  25,
                    "icon":    "🚽",
                    "priority":"medium"
                },
                {
                    "title":   "Check for Toilet Leaks",
                    "detail":  "A leaking toilet can waste 200–400 liters per day. Put food colouring in the tank to detect leaks.",
                    "saving":  200,
                    "icon":    "🔍",
                    "priority":"high"
                }
            ],
            "laundry": [
                {
                    "title":   "Only Run Full Loads",
                    "detail":  "A full washing machine load uses ~40L. Half-loads use nearly the same water — always fill it up.",
                    "saving":  30,
                    "icon":    "👕",
                    "priority":"medium"
                },
                {
                    "title":   "Use Cold Water Cycles",
                    "detail":  "Cold water washes are equally effective for most laundry and save energy too.",
                    "saving":  10,
                    "icon":    "❄️",
                    "priority":"low"
                }
            ],
            "gardening": [
                {
                    "title":   "Water Plants at Dawn or Dusk",
                    "detail":  "Watering in the cooler parts of the day reduces evaporation by up to 30%.",
                    "saving":  15,
                    "icon":    "🌅",
                    "priority":"medium"
                },
                {
                    "title":   "Use Drip Irrigation",
                    "detail":  "Drip systems deliver water directly to roots, cutting garden water use by 30–50%.",
                    "saving":  40,
                    "icon":    "🌱",
                    "priority":"high"
                },
                {
                    "title":   "Collect Rainwater",
                    "detail":  "A 200L rainwater barrel can offset garden watering for days after rainfall.",
                    "saving":  20,
                    "icon":    "🌧️",
                    "priority":"medium"
                }
            ],
            "dishes": [
                {
                    "title":   "Use a Dishwasher (Full Load)",
                    "detail":  "A full dishwasher uses ~12L vs. ~40L for handwashing the same amount.",
                    "saving":  12,
                    "icon":    "🍽️",
                    "priority":"low"
                },
                {
                    "title":   "Don't Leave the Tap Running",
                    "detail":  "A running tap uses ~6L per minute. Fill a basin instead of rinsing under a running tap.",
                    "saving":  18,
                    "icon":    "🚰",
                    "priority":"medium"
                }
            ],
            "car_wash": [
                {
                    "title":   "Use a Bucket Instead of a Hose",
                    "detail":  "A hose uses 150–400L per wash. A bucket wash uses only ~30L.",
                    "saving":  120,
                    "icon":    "🚗",
                    "priority":"high"
                }
            ],
            "cooking": [
                {
                    "title":   "Reuse Vegetable Rinse Water",
                    "detail":  "Water used to rinse vegetables can be reused to water plants.",
                    "saving":  3,
                    "icon":    "🥦",
                    "priority":"low"
                }
            ],
            "general": [
                {
                    "title":   "Fix All Dripping Taps",
                    "detail":  "A tap dripping once per second wastes ~3,000 liters per month. Fix leaks promptly.",
                    "saving":  100,
                    "icon":    "🔧",
                    "priority":"high"
                },
                {
                    "title":   "Install Water-Saving Aerators",
                    "detail":  "Tap aerators add air to water flow, reducing usage by 30–50% with no change in experience.",
                    "saving":  30,
                    "icon":    "💡",
                    "priority":"medium"
                }
            ]
        }

        recs = []

        # Always include general tips
        recs.extend(tip_bank["general"])

        # Add activity-specific tips for any activity over 150% of benchmark
        for activity in breakdown:
            if activity["ratio"] > 1.5 and activity["key"] in tip_bank:
                recs.extend(tip_bank[activity["key"]])

        # For moderate/low usage, still show a couple of good tips
        if category_index <= 1:
            recs.append({
                "title":   "Great Job! Keep Monitoring",
                "detail":  "Your water usage is within healthy limits. Continue tracking daily to maintain this habit.",
                "saving":  0,
                "icon":    "🌟",
                "priority":"low"
            })

        # Sort by priority: high → medium → low
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recs.sort(key=lambda x: priority_order.get(x["priority"], 3))

        # Return top 6 recommendations to avoid overwhelming the user
        return recs[:6]

    def _estimate_savings(self, total_liters, category_index, people):
        """
        Estimates how much water and money could be saved if the user
        follows the recommendations.

        Assumes water costs approximately ₹5 per 1000 liters (India average).
        """
        # Potential reduction percentages per category
        reduction_pct = {0: 5, 1: 10, 2: 25, 3: 40}
        pct = reduction_pct.get(category_index, 20)

        daily_saving_liters  = round(total_liters * pct / 100, 1)
        monthly_saving_liters= round(daily_saving_liters * 30, 1)
        yearly_saving_liters = round(daily_saving_liters * 365, 1)

        # Cost estimates (₹5 per 1000 liters = ₹0.005 per liter)
        cost_per_liter = 0.005
        monthly_saving_inr = round(monthly_saving_liters * cost_per_liter, 2)
        yearly_saving_inr  = round(yearly_saving_liters  * cost_per_liter, 2)

        return {
            "reduction_pct":       pct,
            "daily_liters":        daily_saving_liters,
            "monthly_liters":      monthly_saving_liters,
            "yearly_liters":       yearly_saving_liters,
            "monthly_inr":         monthly_saving_inr,
            "yearly_inr":          yearly_saving_inr,
            "bottles_saved_daily": round(daily_saving_liters / 1, 0),  # 1L bottles
        }

    # ── Historical Trend ────────────────────────────────────────────────────────

    def get_historical_trend(self):
        """
        Returns simulated 30-day historical water usage data for chart display.
        In a real app, this data would come from a database.
        """
        np.random.seed(0)
        days    = list(range(1, 31))
        # Simulate a slight downward trend (as user becomes more aware)
        trend   = np.linspace(220, 160, 30)
        noise   = np.random.normal(0, 20, 30)
        values  = np.clip(trend + noise, 80, 350)

        return {
            "labels": [f"Day {d}" for d in days],
            "values": [round(float(v), 1) for v in values],
            "benchmark": self.daily_recommended
        }
