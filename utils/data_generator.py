"""
====================================================
  Smart Water Usage Advisor — Sample Data Generator
====================================================
Generates realistic sample data for demonstration purposes.
Users can click "Load Sample Data" in the UI to auto-fill
the form without entering their own numbers.
"""

import random


def generate_sample_data():
    """
    Returns a dictionary of realistic daily water usage values in liters.
    
    These values represent a typical family of 4 in an urban Indian household.
    The numbers intentionally include some above-benchmark activities so the
    AI has something interesting to flag and recommend.
    """
    # Use a random scenario each time so the demo feels dynamic
    scenarios = [
        {
            # Scenario 1: High bathing + gardening usage
            "name":      "Urban Family (High Bathing)",
            "drinking":  12,    # 4 people × 3L
            "cooking":   25,    # Cooking for 4
            "bathing":   160,   # Long showers for 4 people
            "toilet":    50,    # 4 people × ~12L/day
            "laundry":   80,    # 2 loads
            "dishes":    30,    # After 3 meals
            "gardening": 60,    # Medium garden
            "car_wash":  20,    # Washed car today
            "people":    4,
        },
        {
            # Scenario 2: Eco-conscious household
            "name":      "Eco-Conscious Household",
            "drinking":  8,
            "cooking":   12,
            "bathing":   60,    # Short showers
            "toilet":    20,    # Dual-flush toilet
            "laundry":   40,    # 1 full load
            "dishes":    10,
            "gardening": 10,    # Drip irrigation
            "car_wash":  0,
            "people":    2,
        },
        {
            # Scenario 3: College hostel student
            "name":      "Single Student",
            "drinking":  3,
            "cooking":   4,
            "bathing":   70,
            "toilet":    18,
            "laundry":   15,
            "dishes":    5,
            "gardening": 0,
            "car_wash":  0,
            "people":    1,
        },
    ]

    # Pick a random scenario
    sample = random.choice(scenarios)
    # Remove the name field before returning — it's just for labelling
    sample.pop("name", None)
    return sample
