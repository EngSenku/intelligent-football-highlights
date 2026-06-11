const API_BASE_URL = "http://127.0.0.1:8000";

// ── DOM refs ───────────────────────────────────────────────────
const videoFileInput = document.getElementById("videoFile");
const uploadBtn = document.getElementById("uploadBtn");
const dropzone = document.getElementById("dropzone");
const fileSelected = document.getElementById("fileSelected");
const fileName = document.getElementById("fileName");

const statusCard = document.getElementById("statusCard");
const jobStatusBadge = document.getElementById("jobStatusBadge");
const displayJobId = document.getElementById("displayJobId");
const displayStage = document.getElementById("displayStage");
const stageDot = document.getElementById("stageDot");
const progressText = document.getElementById("progressText");
const scanFill = document.getElementById("scanFill");
const scanHead = document.getElementById("scanHead");
const errorContainer = document.getElementById("errorContainer");
const errorText = document.getElementById("errorText");

const resultsCard = document.getElementById("resultsCard");
const downloadBtn = document.getElementById("downloadBtn");
const resultsJson = document.getElementById("resultsJson");
const eventList = document.getElementById("eventList");
const statEvents = document.getElementById("statEvents");
const statDuration = document.getElementById("statDuration");
const statRatio = document.getElementById("statRatio");

let pollingInterval = null;

// ── File selection ─────────────────────────────────────────────
videoFileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    const f = e.target.files[0];
    fileName.textContent = f.name;
    fileSelected.classList.remove("hidden");
    dropzone.querySelector(".dropzone-inner").classList.add("hidden");
    uploadBtn.disabled = false;
  }
});

// Drag-over visual cue
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});
dropzone.addEventListener("dragleave", () =>
  dropzone.classList.remove("drag-over"),
);
dropzone.addEventListener("drop", () => dropzone.classList.remove("drag-over"));

// ── Upload ────────────────────────────────────────────────────
uploadBtn.addEventListener("click", async () => {
  if (videoFileInput.files.length === 0) return;

  const file = videoFileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  uploadBtn.disabled = true;
  uploadBtn.querySelector(".btn-label").textContent = "Uploading…";
  errorContainer.classList.add("hidden");

  try {
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.detail || "Upload failed.");
    }
    const data = await response.json();
    uploadBtn.querySelector(".btn-label").textContent = "Analyse Match";
    uploadBtn.disabled = false;
    startJobTracking(data.job_id);
  } catch (err) {
    uploadBtn.disabled = false;
    uploadBtn.querySelector(".btn-label").textContent = "Analyse Match";
    showJobError(err.message);
    statusCard.classList.remove("hidden");
  }
});

// ── Progress helpers ──────────────────────────────────────────
function setProgress(pct) {
  const p = Math.min(100, Math.max(0, pct));
  scanFill.style.width = `${p}%`;
  scanHead.style.left = `calc(${p}% - 1.5px)`;
  progressText.textContent = `${p}%`;
}

// ── Job tracking ──────────────────────────────────────────────
function startJobTracking(jobId) {
  displayJobId.textContent = jobId;
  statusCard.classList.remove("hidden");
  resultsCard.classList.add("hidden");
  errorContainer.classList.add("hidden");
  setProgress(0);
  stageDot.classList.remove("active");

  if (pollingInterval) clearInterval(pollingInterval);
  checkJobStatus(jobId);
  pollingInterval = setInterval(() => checkJobStatus(jobId), 2500);
}

async function checkJobStatus(jobId) {
  try {
    const response = await fetch(`${API_BASE_URL}/status/${jobId}`);
    if (!response.ok) throw new Error("Could not retrieve status.");

    const s = await response.json();
    const status = s.status;
    const stage = s.stage || "Processing…";
    const progress = s.progress || 0;

    displayStage.textContent = stage;
    setProgress(progress);

    // badge
    jobStatusBadge.textContent = status;
    jobStatusBadge.className = "status-badge";

    if (status === "completed") {
      jobStatusBadge.classList.add("completed");
      stageDot.classList.remove("active");
      clearInterval(pollingInterval);
      fetchAndDisplayResults(jobId);
    } else if (status === "failed") {
      jobStatusBadge.classList.add("failed");
      stageDot.classList.remove("active");
      clearInterval(pollingInterval);
      showJobError(s.error || "Processing failed.");
    } else {
      jobStatusBadge.classList.add("running");
      stageDot.classList.add("active");
    }
  } catch (err) {
    clearInterval(pollingInterval);
    showJobError(err.message);
  }
}

// ── Results ───────────────────────────────────────────────────
const TYPE_LABELS = {
  goal_candidate: "⚽  Goal",
  penalty_candidate: "🎯  Penalty",
  big_chance_candidate: "🔥  Big Chance",
  chance_candidate: "💥  Chance",
  foul_or_card_candidate: "🟨  Foul / Card",
  corner_or_set_piece_candidate: "⚑  Corner",
  highlight_candidate: "✦  Highlight",
};

function fmtSeconds(sec) {
  if (!sec && sec !== 0) return "—";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0)
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

async function fetchAndDisplayResults(jobId) {
  try {
    const response = await fetch(`${API_BASE_URL}/results/${jobId}`);
    if (!response.ok) throw new Error("Could not fetch results.");
    const data = await response.json();

    // Stats
    const evCount = data.events_detected ?? data.events?.length ?? "—";
    statEvents.querySelector(".stat-num").textContent = evCount;
    statDuration.querySelector(".stat-num").textContent =
      data.highlights_duration ? fmtSeconds(data.highlights_duration) : "—";
    statRatio.querySelector(".stat-num").textContent = data.compression_ratio
      ? `${data.compression_ratio.toFixed(1)}×`
      : "—";

    // Event list
    eventList.innerHTML = "";
    if (data.events && data.events.length > 0) {
      data.events.forEach((ev) => {
        const row = document.createElement("div");
        row.className = "event-item";
        row.dataset.type = ev.event_type;

        const label = TYPE_LABELS[ev.event_type] ?? ev.event_type;
        const conf = ev.confidence ? `${Math.round(ev.confidence * 100)}%` : "";
        const ts = ev.timestamp_hhmmss ?? fmtSeconds(ev.timestamp);

        row.innerHTML = `
          <span class="event-time">${ts}</span>
          <span class="event-type">${label}</span>
          <span class="type-pill">${ev.event_type.replace("_candidate", "")}</span>
          <span class="event-conf">${conf}</span>
        `;
        eventList.appendChild(row);
      });
    }

    // Download
    downloadBtn.href = `${API_BASE_URL}/download/${jobId}`;

    // Raw JSON
    resultsJson.textContent = JSON.stringify(data, null, 2);

    resultsCard.classList.remove("hidden");
  } catch (err) {
    showJobError(`Failed to load results: ${err.message}`);
  }
}

function showJobError(msg) {
  errorText.textContent = msg;
  errorContainer.classList.remove("hidden");
}
