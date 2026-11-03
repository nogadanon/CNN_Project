// Locally (Live Server) the API runs separately on port 8000; when hosted, the same server serves both
const API = ["localhost", "127.0.0.1"].includes(location.hostname) && location.port !== "8000" ? "http://127.0.0.1:8000" : "";
const MAX_MB = 10;
const ALLOWED_TYPES = ["image/jpeg", "image/png"]; // the model handles WEBP poorly

const $ = (id) => document.getElementById(id);
const state = { file: null, predicted: null, probs: [], selected: null, confirmed: false };

const fmtPct = (p) => (p * 100).toFixed(p >= 0.1 ? 0 : 1) + "%";
const fmtPrice = (n) => "₪" + Math.round(n).toLocaleString("he-IL");

// ---------- upload ----------
const dz = $("dropzone");
["fileInput", "cameraInput"].forEach((id) =>
  $(id).addEventListener("change", (e) => e.target.files[0] && handleFile(e.target.files[0])));
["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]));

$("newBtn").addEventListener("click", () => {
  $("result").hidden = $("similar").hidden = true;
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (matchMedia("(hover: hover)").matches) $("fileInput").click(); // on phones the user picks gallery/camera
});

async function handleFile(file) {
  if (!ALLOWED_TYPES.includes(file.type)) return showError("ניתן להעלות רק קבצי JPG או PNG");
  $("fileInput").value = $("cameraInput").value = ""; // allow picking the same photo again
  if (file.size > MAX_MB * 1024 * 1024) return showError(`התמונה גדולה מדי (מקסימום ${MAX_MB}MB)`);
  showError(null);
  state.file = file;
  state.uploadId = newId(); // one id per photo, so re-saving with another class replaces it on the server
  $("preview").src = URL.createObjectURL(file);
  $("result").hidden = false;
  $("similar").hidden = true;
  $("loader").classList.add("on");
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${API}/api/predict`, { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    state.predicted = data.predicted;
    state.probs = data.probabilities;
    select(data.predicted);
  } catch (err) {
    showError(serverError(err));
    $("result").hidden = true;
  } finally {
    $("loader").classList.remove("on");
  }
}

// ---------- class selection ----------
function select(name) {
  state.selected = name;
  state.confirmed = false;
  render();
}

function render() {
  const { predicted, selected, probs, confirmed } = state;
  const manual = selected !== predicted;
  const p = probs.find((c) => c.name === selected)?.prob ?? 0;

  $("verdictLabel").textContent = manual ? "נבחר ידנית" : "המודל זיהה";
  $("chosenName").textContent = selected;
  $("chosenProb").innerHTML = manual
    ? `המודל העריך <b>${fmtPct(p)}</b> · הצעת המודל: ${predicted}`
    : `ודאות המודל: <b>${fmtPct(p)}</b>`;

  const btn = $("confirmBtn");
  btn.textContent = !confirmed ? "שמור תמונה ואשר קטגוריה"
    : state.saved ? "✓ התמונה נשמרה והקטגוריה אושרה" : "✓ הקטגוריה אושרה";
  btn.classList.toggle("done", confirmed);
  $("hint").textContent = confirmed ? "מעולה! עכשיו אפשר לראות מוצרים דומים." : "לא מדויק? בחרו קטגוריה אחרת מהרשימה.";
  $("hint").classList.toggle("ok", confirmed);
  $("similarBtn").hidden = !confirmed;

  $("classList").innerHTML = probs.map((c) => `
    <li class="${c.name === selected ? "selected" : ""}" data-name="${c.name}">
      <span class="dot"></span>
      <span class="name">${c.name}${c.name === predicted ? '<span class="tag">הצעת המודל</span>' : ""}</span>
      <span class="pct">${fmtPct(c.prob)}</span>
      <span class="bar"><i style="width:${Math.max(c.prob * 100, 0.5)}%"></i></span>
    </li>`).join("");
}

$("classList").addEventListener("click", (e) => {
  const li = e.target.closest("li");
  if (li) { select(li.dataset.name); $("similar").hidden = true; }
});

$("confirmBtn").addEventListener("click", async () => {
  if (state.confirmed) return;
  const btn = $("confirmBtn");
  btn.disabled = true;
  btn.textContent = "שומר…";
  state.saved = false;
  try {
    const fd = new FormData();
    fd.append("file", state.file);
    fd.append("label", state.selected);
    fd.append("upload_id", state.uploadId);
    fd.append("predicted", state.predicted);
    fd.append("confidence", state.probs[0].prob); // probs are sorted, [0] is the model's choice
    const res = await fetch(`${API}/api/save`, { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    state.saved = true;
  } catch (err) {
    showError("הקטגוריה אושרה, אך שמירת התמונה נכשלה: " + serverError(err));
  } finally {
    state.confirmed = true; // similar products don't depend on the save succeeding
    btn.disabled = false;
    render();
  }
});

function newId() {
  if (crypto.randomUUID) return crypto.randomUUID(); // only on https/localhost
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}

// ---------- similar products ----------
$("similarBtn").addEventListener("click", async () => {
  const btn = $("similarBtn");
  btn.disabled = true;
  btn.textContent = "מחפש מוצרים דומים…";
  try {
    const fd = new FormData();
    fd.append("file", state.file);
    fd.append("label", state.selected);
    const res = await fetch(`${API}/api/similar`, { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    renderSimilar(await res.json());
  } catch (err) {
    showError(serverError(err));
  } finally {
    btn.disabled = false;
    btn.textContent = "הצג מוצרים דומים";
  }
});

function renderSimilar(items) {
  $("similarSub").textContent = `${items.length} מוצרים מהקטגוריה "${state.selected}", מסודרים לפי מידת הדמיון`;
  $("grid").innerHTML = items.map((it) => {
    const price = it.price_from ? fmtPrice(it.price_from) + (it.price_to ? " – " + fmtPrice(it.price_to) : "") : "";
    return `
      <article class="card">
        <div class="img">
          <img src="${API}${it.image}" alt="${it.name || ""}" loading="lazy">
          <span class="match">דמיון ${Math.round(it.score * 100)}%</span>
        </div>
        <div class="body">
          ${it.company ? `<span class="company">${it.company}</span>` : ""}
          <span class="title">${it.name || it.id}</span>
          ${it.style ? `<span class="meta">סגנון: ${it.style}</span>` : ""}
          <div class="bottom">
            <span class="price">${price}</span>
            ${it.url ? `<a href="${it.url}" target="_blank" rel="noopener">לעמוד המוצר</a>` : ""}
          </div>
        </div>
      </article>`;
  }).join("");
  $("similar").hidden = false;
  $("similar").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------- helpers ----------
function showError(msg) {
  $("error").hidden = !msg;
  $("error").textContent = msg || "";
  if (msg && !$("result").hidden) $("result").scrollIntoView({ behavior: "smooth" });
}

function serverError(err) {
  if (!(err instanceof TypeError)) return err.message;
  return API ? "השרת לא זמין. ודאו ש-run_server.bat רץ (פורט 8000)." : "השרת לא זמין כרגע, נסו שוב בעוד דקה.";
}

// Similarity index is built once on first server start - show progress until ready
async function pollStatus() {
  const el = $("status");
  try {
    const s = await (await fetch(`${API}/api/status`)).json();
    if (s.error) { el.hidden = false; el.textContent = "שגיאה בבניית מאגר הדמיון"; return; }
    if (s.ready) { el.hidden = true; return; }
    el.hidden = false;
    el.textContent = `מכין מאגר דמיון… ${s.total ? Math.round((s.done / s.total) * 100) : 0}%`;
  } catch {
    el.hidden = false;
    el.textContent = "השרת לא מחובר";
  }
  setTimeout(pollStatus, 3000);
}
pollStatus();
