# 💧 Smart Water Usage Advisor
### AI for Sustainability · 1M1B × IBM SkillsBuild × AICTE
**SDG 6: Clean Water & Sanitation · SDG 11: Sustainable Cities**

---

## 📌 Project Overview

The Smart Water Usage Advisor is an AI-powered web application that:
- Accepts daily water usage data per household activity
- Uses a **Random Forest ML model** to classify usage as Low / Moderate / High / Critical
- Generates **personalized conservation recommendations**
- Estimates **water & cost savings** if recommendations are followed
- Displays a **30-day usage trend chart**

---

## 🗂️ Project Structure

```
smart_water_advisor/
│
├── app.py                  ← Main Flask server (run this)
├── requirements.txt        ← Python package dependencies
│
├── utils/
│   ├── __init__.py
│   ├── advisor.py          ← AI model + recommendation engine
│   └── data_generator.py   ← Sample data generator
│
├── templates/
│   └── index.html          ← Main web page (HTML)
│
└── static/
    ├── css/
    │   └── style.css       ← All styling
    └── js/
        └── app.js          ← Frontend JavaScript logic
```

---

## ⚙️ Setup & Installation

### Step 1 — Make sure Python is installed
```bash
python --version   # Should be 3.8 or higher
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the application
```bash
python app.py
```

### Step 4 — Open in browser
```
http://localhost:5000
```

---

## 🧠 How the AI Works

1. **Training Data** — 2,000 synthetic water usage records are generated at startup
2. **Features** — 11 input features: 8 activity usages + total + people + per-capita
3. **Model** — Random Forest Classifier with 100 decision trees
4. **Labels** — Low (<80L/person) · Moderate (80–150L) · High (150–250L) · Critical (>250L)
5. **Scaling** — StandardScaler normalizes all features before training/prediction

---

## 🌱 Recommended Enhancements

| Enhancement | Difficulty | Impact |
|---|---|---|
| Connect to a real database (SQLite / PostgreSQL) | Medium | High |
| User login and history tracking | Medium | High |
| IoT sensor integration via MQTT | Hard | Very High |
| Export report as PDF | Easy | Medium |
| Email/SMS alerts when usage is critical | Medium | High |
| Mobile app (React Native / Flutter) | Hard | High |
| Multilingual support (Hindi, Gujarati) | Easy | Medium |
| Predictive forecasting (next 7 days) | Medium | High |

---

## 📚 Technologies Used

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask |
| Machine Learning | scikit-learn (RandomForestClassifier) |
| Data Processing | NumPy, Pandas |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js |

---

## 👤 Author
**Md Altamash Rizwi** · altushaikh076@gmail.com  
B.Tech Computer Engineering · Aditya Silver Oak Institute Of Technology
