// ===== Slider live readouts =====
const sliders = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"];
sliders.forEach((id) => {
  const el = document.getElementById(id);
  const out = document.getElementById(id + "_val");
  el.addEventListener("input", () => {
    out.textContent = el.value;
  });
});

// ===== Gauge tick marks (drawn once) =====
function drawTicks() {
  const g = document.getElementById("ticks");
  const cx = 120, cy = 130, r1 = 95, r2 = 108;
  for (let i = 0; i <= 10; i++) {
    const angle = Math.PI - (i / 10) * Math.PI; // 180deg sweep
    const x1 = cx + r1 * Math.cos(angle);
    const y1 = cy - r1 * Math.sin(angle);
    const x2 = cx + r2 * Math.cos(angle);
    const y2 = cy - r2 * Math.sin(angle);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    g.appendChild(line);
  }
}
drawTicks();

// ===== Gauge update =====
const ARC_LENGTH = 315; // approx path length for the 180deg arc used

function updateGauge(prob) {
  const arc = document.getElementById("gauge_arc");
  const needle = document.getElementById("needle");
  const numberEl = document.getElementById("prob_number");
  const statusEl = document.getElementById("status_badge");

  const clamped = Math.max(0, Math.min(100, prob));
  const dash = (clamped / 100) * ARC_LENGTH;

  let color = "#45D8B8"; // teal
  if (clamped >= 66) color = "#E4483A"; // red
  else if (clamped >= 33) color = "#F2A93B"; // amber

  arc.setAttribute("stroke-dasharray", `${dash} ${ARC_LENGTH}`);
  arc.setAttribute("stroke", color);

  const angleDeg = (clamped / 100) * 180; // 0 = left(180deg), 100 = right(0deg)
  const needleRotation = -180 + angleDeg;
  needle.setAttribute("transform", `rotate(${needleRotation})`);

  numberEl.innerHTML = `${clamped.toFixed(1)}<span class="pct">%</span>`;

  if (clamped >= 50) {
    statusEl.textContent = "FAILURE RISK — INSPECT MACHINE";
    statusEl.className = "gauge-status danger";
  } else {
    statusEl.textContent = "NORMAL OPERATION";
    statusEl.className = "gauge-status ok";
  }
}

function updateRiskBreakdown(breakdown) {
  document.querySelectorAll(".risk-row").forEach((row) => {
    const key = row.dataset.key;
    const val = breakdown[key] || 0;
    row.querySelector(".risk-fill").style.width = val + "%";
    row.querySelector(".risk-pct").textContent = val.toFixed(0) + "%";
  });
}

// ===== Predict button =====
document.getElementById("predict_btn").addEventListener("click", async () => {
  const btn = document.getElementById("predict_btn");
  btn.textContent = "Running...";
  btn.disabled = true;

  const payload = {
    type: document.getElementById("type").value,
    air_temp: document.getElementById("air_temp").value,
    process_temp: document.getElementById("process_temp").value,
    rpm: document.getElementById("rpm").value,
    torque: document.getElementById("torque").value,
    tool_wear: document.getElementById("tool_wear").value,
  };

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (data.error) {
      alert(data.error);
    } else {
      updateGauge(data.probability);
      updateRiskBreakdown(data.risk_breakdown);
    }
  } catch (err) {
    alert("Could not reach the prediction server. Is app.py running?");
  } finally {
    btn.textContent = "Run Diagnostic";
    btn.disabled = false;
  }
});
