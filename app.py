"""
====================================================
  Smart Water Usage Advisor — Main Application
====================================================
This is the entry point of the web application.
It uses Flask — a lightweight Python web framework
that lets us serve web pages and handle API requests.

HOW TO RUN:
  1. Install dependencies:  pip install flask pandas numpy scikit-learn joblib
  2. Run this file:          python app.py
  3. Open browser at:        http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import json
import os

# Import our custom modules
from utils.data_generator import generate_sample_data
from utils.advisor import WaterUsageAdvisor

# ── Create the Flask app ──────────────────────────────────────────────────────
# __name__ tells Flask where to find templates and static files
app = Flask(__name__)

# ── Initialize the AI advisor ─────────────────────────────────────────────────
# This creates an instance of our AI class and trains the model on startup
advisor = WaterUsageAdvisor()
advisor.train()   # Train the AI model when the server starts

# ── Routes (URL endpoints) ────────────────────────────────────────────────────

@app.route("/")
def index():
    """
    Home page route.
    When someone visits http://localhost:5000 — this function runs
    and returns the main HTML page.
    """
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    API endpoint that receives water usage data from the user,
    runs it through the AI advisor, and returns recommendations.

    The frontend sends a JSON POST request here, and we respond
    with analysis results + conservation tips.
    """
    try:
        # Get the JSON data sent from the browser
        data = request.get_json()

        # Validate that we actually received data
        if not data:
            return jsonify({"error": "No data received"}), 400

        # Pass the data to our AI advisor for analysis
        result = advisor.analyze(data)

        # Return the analysis as JSON back to the frontend
        return jsonify(result)

    except Exception as e:
        # If something goes wrong, return a helpful error message
        return jsonify({"error": str(e)}), 500


@app.route("/sample-data")
def sample_data():
    """
    Returns sample water usage data so users can see how the app works
    without needing to enter their own data first.
    """
    data = generate_sample_data()
    return jsonify(data)


@app.route("/history")
def history():
    """
    Returns simulated historical water usage data for the chart display.
    In a real app, this would fetch from a database.
    """
    historical = advisor.get_historical_trend()
    return jsonify(historical)


# ── Run the app ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  Smart Water Usage Advisor — Starting Server")
    print("  Open your browser at: http://localhost:5000")
    print("="*55)
    # debug=True automatically reloads the server when you change code
    port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
