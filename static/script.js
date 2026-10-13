const reviewInput = document.getElementById("reviewInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const charCount = document.getElementById("charCount");
const errorBox = document.getElementById("errorBox");
const resultCard = document.getElementById("resultCard");

reviewInput.addEventListener("input", () => {
  charCount.textContent = `${reviewInput.value.length} characters`;
});

function showError(element, message) {
  element.textContent = message;
  element.classList.remove("hidden");
}
function hideError(element) {
  element.textContent = "";
  element.classList.add("hidden");
}

analyzeBtn.addEventListener("click", async () => {
  hideError(errorBox);
  resultCard.classList.add("hidden");
  const review = reviewInput.value.trim();
  if (!review) {
    showError(errorBox, "Please enter a review first.");
    return;
  }
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = "Analyzing…";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({review})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    const positive = data.sentiment === "Positive";
    document.getElementById("sentimentText").textContent = data.sentiment;
    document.getElementById("sentimentBadge").textContent = data.sentiment;
    document.getElementById("sentimentBadge").classList.toggle("negative", !positive);
    document.getElementById("confidenceText").textContent = `${data.confidence}% confidence`;
    document.getElementById("confidenceMeter").style.width = `${data.confidence}%`;
    document.getElementById("confidenceMeter").style.background = positive ? "var(--green)" : "var(--red)";
    document.getElementById("positiveProb").textContent = `${data.positive_probability}%`;
    document.getElementById("negativeProb").textContent = `${data.negative_probability}%`;
    resultCard.classList.remove("hidden");
  } catch (error) {
    showError(errorBox, error.message);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = 'Analyze sentiment <span>↗</span>';
  }
});

const csvInput = document.getElementById("csvFile");
const csvForm = document.getElementById("csvForm");
const csvError = document.getElementById("csvError");
const csvResults = document.getElementById("csvResults");
const csvSubmit = csvForm.querySelector('button[type="submit"]');

csvInput.addEventListener("change", () => {
  document.getElementById("fileName").textContent =
    csvInput.files.length ? csvInput.files[0].name : "No file selected";
});

csvForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError(csvError);
  csvResults.classList.add("hidden");
  if (!csvInput.files.length) {
    showError(csvError, "Please choose a CSV file.");
    return;
  }
  const formData = new FormData();
  formData.append("file", csvInput.files[0]);
  csvSubmit.disabled = true;
  csvSubmit.textContent = "Analyzing reviews…";
  try {
    const response = await fetch("/analyze_csv", {method: "POST", body: formData});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "CSV analysis failed.");
    document.getElementById("totalStat").textContent = data.total;
    document.getElementById("positiveStat").textContent = data.positive;
    document.getElementById("negativeStat").textContent = data.negative;
    document.getElementById("positiveBar").style.width = `${data.positive / data.total * 100}%`;
    document.getElementById("negativeBar").style.width = `${data.negative / data.total * 100}%`;
    const tbody = document.getElementById("csvTable");
    tbody.replaceChildren();
    data.results.forEach((item) => {
      const tr = document.createElement("tr");
      const reviewCell = document.createElement("td");
      reviewCell.textContent = item.review;
      const sentimentCell = document.createElement("td");
      const tag = document.createElement("span");
      tag.className = `tag ${item.sentiment === "Negative" ? "negative" : ""}`;
      tag.textContent = item.sentiment;
      sentimentCell.appendChild(tag);
      const confidenceCell = document.createElement("td");
      confidenceCell.textContent = `${item.confidence}%`;
      tr.append(reviewCell, sentimentCell, confidenceCell);
      tbody.appendChild(tr);
    });
    csvResults.classList.remove("hidden");
  } catch (error) {
    showError(csvError, error.message);
  } finally {
    csvSubmit.disabled = false;
    csvSubmit.innerHTML = 'Analyze CSV <span>↗</span>';
  }
});
