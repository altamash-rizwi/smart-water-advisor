/**
 * ====================================================
 *   Smart Water Usage Advisor — Frontend JavaScript
 * ====================================================
 * This file handles all browser-side logic:
 *   - Reading form values
 *   - Sending data to the Flask API
 *   - Rendering the results on the page
 *   - Drawing the Chart.js trend chart
 *
 * No external JS libraries needed except Chart.js (loaded in HTML).
 */

"use strict";

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Reads the value of a form input by its HTML id.
 * parseFloat converts the string to a decimal number.
 * The || 0 part means "use 0 if the field is empty".
 */
function getVal(id) {
  return parseFloat(document.getElementById(id).value) || 0;
}

/** Show an element by removing the 'hidden' class */
function show(id) { document.getElementById(id).classList.remove("hidden"); }

/** Hide an element by adding the 'hidden' class */
function hide(id) { document.getElementById(id).classList.add("hidden"); }

/** Set inner HTML of an element */
function setHTML(id, html) { document.getElementById(id).innerHTML = html; }

/** Set text content of an element */
function setText(id, text) { document.getElementById(id).textContent = text; }


// ── Main Analysis Function ────────────────────────────────────────────────────

/**
 * Called when the user clicks "Analyze My Usage".
 * 1. Collects form values
 * 2. Sends them to /analyze via fetch (HTTP POST)
 * 3. Renders the returned results
 */
async function analyzeUsage() {
  // Gather all input values into an object
  const formData = {
    people:    parseInt(document.getElementById("people").value) || 1,
    drinking:  getVal("drinking"),
    cooking:   getVal("cooking"),
    bathing:   getVal("bathing"),
    toilet:    getVal("toilet"),
    laundry:   getVal("laundry"),
    dishes:    getVal("dishes"),
    gardening: getVal("gardening"),
    car_wash:  getVal("car_wash"),
  };

  // Show loading spinner, hide old results
  show("loading");
  hide("results");

  try {
    // fetch() sends an HTTP request — this is how the browser talks to Flask
    // We use POST because we're sending data (not just requesting a page)
    const response = await fetch("/analyze", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(formData),   // Convert JS object → JSON string
    });

    // Parse the JSON response from Flask back into a JS object
    const data = await response.json();

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    // Render all the result sections
    renderCategoryBanner(data);
    renderScoreBar(data);
    renderBreakdown(data.breakdown);
    renderRecommendations(data.recommendations);
    renderSavings(data.savings);

    // Show the results panel with a smooth animation
    const resultsEl = document.getElementById("results");
    resultsEl.classList.remove("hidden");
    resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    alert("Something went wrong. Make sure the Flask server is running.");
    console.error(err);
  } finally {
    // Always hide loading spinner when done (success or error)
    hide("loading");
  }
}


// ── Render Functions ──────────────────────────────────────────────────────────

/**
 * Renders the big category banner at the top of results.
 * Shows: category name (Low/Moderate/High/Critical), icon, totals.
 */
function renderCategoryBanner(data) {
  const icons = { Low: "💧", Moderate: "🌊", High: "⚠️", Critical: "🔴" };
  const cat   = data.category;

  const banner = document.getElementById("category-banner");
  // Remove old category class and add new one
  ["cat-Low","cat-Moderate","cat-High","cat-Critical"].forEach(c => banner.classList.remove(c));
  banner.classList.add("cat-" + cat);

  setText("category-icon",  icons[cat] || "💧");
  setText("category-value", cat + " Usage");
  setText("total-liters",   data.total_liters + "L");
  setText("per-capita",     data.per_capita + "L");
  setText("benchmark-val",  data.benchmark + "L");
}

/**
 * Animates the horizontal usage score bar.
 * Score is capped at 150 for display purposes.
 */
function renderScoreBar(data) {
  const bar     = document.getElementById("score-bar");
  const scorePct = Math.min((data.per_capita / data.benchmark) * 100, 150);

  // Set colour based on category
  const colours = {
    Low: "#02C39A", Moderate: "#028090",
    High: "#f0a500", Critical: "#e53935"
  };
  bar.style.background = colours[data.category] || "#028090";
  // Use setTimeout so the CSS transition actually plays
  setTimeout(() => { bar.style.width = Math.min(scorePct, 100) + "%"; }, 50);
}

/**
 * Renders the per-activity breakdown list.
 * Each row shows icon, activity name, a mini bar, usage in liters, and status chip.
 */
function renderBreakdown(breakdown) {
  const container = document.getElementById("breakdown-list");
  container.innerHTML = "";

  breakdown.forEach(item => {
    // Mini bar width = ratio vs benchmark, capped at 100%
    const barW = Math.min((item.ratio * 100), 100);

    const div = document.createElement("div");
    div.className = "breakdown-item";
    div.innerHTML = `
      <span class="breakdown-icon">${item.icon}</span>
      <div class="breakdown-bar-wrap">
        <span class="breakdown-name">${item.label}</span>
        <div class="mini-track">
          <div class="mini-fill fill-${item.status}" style="width:${barW}%"></div>
        </div>
      </div>
      <div style="text-align:right">
        <div class="breakdown-val">${item.value}L</div>
        <span class="status-chip chip-${item.status}">${item.status}</span>
      </div>
    `;
    container.appendChild(div);
  });
}

/**
 * Renders AI recommendation cards.
 * Each card shows priority colour, icon, title, detail, and potential saving.
 */
function renderRecommendations(recs) {
  const container = document.getElementById("recommendations-list");
  container.innerHTML = "";

  if (!recs || recs.length === 0) {
    container.innerHTML = "<p style='color:var(--muted);font-size:.85rem'>No recommendations at this time.</p>";
    return;
  }

  recs.forEach(rec => {
    const savingHTML = rec.saving > 0
      ? `<span class="rec-saving">💧 Save ~${rec.saving}L/day</span>`
      : "";

    const div = document.createElement("div");
    div.className = `rec-card priority-${rec.priority}`;
    div.innerHTML = `
      <span class="rec-icon">${rec.icon}</span>
      <div class="rec-body">
        <div class="rec-title">${rec.title}</div>
        <div class="rec-detail">${rec.detail}</div>
        ${savingHTML}
      </div>
    `;
    container.appendChild(div);
  });
}

/**
 * Renders the savings estimate cards.
 * Shows daily / monthly / yearly potential savings.
 */
function renderSavings(savings) {
  const container = document.getElementById("savings-cards");

  const cards = [
    { num: savings.daily_liters,   unit: "Liters/Day",   label: "🌊 Daily Saving" },
    { num: savings.monthly_liters, unit: "Liters/Month",  label: "📅 Monthly Saving" },
    { num: savings.yearly_liters,  unit: "Liters/Year",   label: "🌍 Yearly Saving" },
    { num: savings.reduction_pct + "%", unit: "Reduction", label: "📉 Usage Cut" },
    { num: "₹" + savings.monthly_inr, unit: "/Month",    label: "💰 Cost Saving" },
    { num: "₹" + savings.yearly_inr,  unit: "/Year",     label: "💰 Annual Saving" },
  ];

  container.innerHTML = cards.map(c => `
    <div class="saving-card">
      <span class="saving-num">${c.num}</span>
      <span class="saving-unit">${c.unit}</span>
      <div class="saving-label">${c.label}</div>
    </div>
  `).join("");
}


// ── Sample Data & Reset ───────────────────────────────────────────────────────

/**
 * Fetches sample data from Flask and fills in the form.
 * Great for first-time users who want to see the app in action.
 */
async function loadSampleData() {
  try {
    const response = await fetch("/sample-data");
    const data     = await response.json();

    // Fill each input field with the sample value
    Object.keys(data).forEach(key => {
      const el = document.getElementById(key);
      if (el) el.value = data[key];
    });

    // Auto-run analysis so the user sees results immediately
    await analyzeUsage();

  } catch (err) {
    alert("Could not load sample data. Is the server running?");
  }
}

/**
 * Resets all form fields to their default values.
 */
function resetForm() {
  const defaults = {
    people: 1, drinking: 3, cooking: 6, bathing: 80,
    toilet: 30, laundry: 40, dishes: 15, gardening: 20, car_wash: 5
  };
  Object.keys(defaults).forEach(key => {
    const el = document.getElementById(key);
    if (el) el.value = defaults[key];
  });
  hide("results");
}


// ── Trend Chart ───────────────────────────────────────────────────────────────

/**
 * Draws the 30-day water usage trend chart using Chart.js.
 *
 * Chart.js documentation: https://www.chartjs.org/docs/
 *
 * @param {Object} data - { labels: [...], values: [...], benchmark: 199 }
 */
function drawTrendChart(data) {
  const ctx = document.getElementById("trendChart").getContext("2d");

  new Chart(ctx, {
    type: "line",
    data: {
      labels: data.labels,
      datasets: [
        {
          // Actual usage line
          label:           "Daily Usage (L)",
          data:            data.values,
          borderColor:     "#028090",
          backgroundColor: "rgba(2,128,144,0.08)",
          borderWidth:     2.5,
          pointRadius:     3,
          pointBackgroundColor: "#028090",
          fill:            true,
          tension:         0.4,    // Smooth curve
        },
        {
          // Benchmark reference line — straight horizontal
          label:           "Benchmark (199L)",
          data:            new Array(data.labels.length).fill(data.benchmark),
          borderColor:     "#02C39A",
          borderWidth:     1.5,
          borderDash:      [6, 4],  // Dashed line
          pointRadius:     0,
          fill:            false,
        }
      ]
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { font: { size: 12 }, color: "#0A2342" }
        },
        tooltip: {
          callbacks: {
            // Add unit to tooltip values
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y}L`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: "#5a7a92", maxTicksLimit: 10 },
          grid:  { color: "#e0ecef" }
        },
        y: {
          ticks: {
            color: "#5a7a92",
            callback: val => val + "L"    // Append "L" to y-axis labels
          },
          grid: { color: "#e0ecef" },
          suggestedMin: 50,
          suggestedMax: 350
        }
      }
    }
  });
}
