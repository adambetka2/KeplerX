"use strict";

const $ = (id) => document.getElementById(id);
const labels = {
  transit: "TRANSIT",
  shallow: "SHALLOW",
  event: "SINGLE EVENT",
  legacy: "SURVIVOR REVIEW",
};
const views = { psf_flat: "PSF FLAT", psf_qdiag: "PSF", sap_qdiag: "SAP" };
const state = {
  items: [],
  decisions: {},
  category: "transit",
  id: null,
  proposal: 0,
  view: "psf_flat",
  fullView: "sap_qdiag",
  serial: 0,
  windowSerial: 0,
  fullSerial: 0,
  undo: [],
  busy: false,
  fullRange: false,
  range: null,
};
const number = (v, digits = 2) =>
  v == null || !Number.isFinite(v)
    ? "—"
    : v.toLocaleString("en", { maximumFractionDigits: digits });
const choice = (id) => state.decisions[id]?.choice || "UNREVIEWED";
const median = (values) => {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function request(path, data) {
  const options = data
    ? {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Review-Token": state.token,
        },
        body: JSON.stringify(data),
      }
    : {};
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok)
    throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}

function notice(message = "") {
  $("notice").textContent = message;
}

function list() {
  const filter = $("filter").value;
  const search = $("search").value.trim().toLowerCase();
  return state.items.filter(
    (i) =>
      i.category === state.category &&
      (filter === "ALL" || choice(i.id) === filter) &&
      (!search || `${i.id} ${i.source_id}`.toLowerCase().includes(search)),
  );
}

function sidebar() {
  const counts = { YES: 0, NO: 0, UNREVIEWED: 0 };
  state.items.forEach((i) => counts[choice(i.id)]++);
  $("kept").textContent = number(counts.YES, 0);
  $("denied").textContent = number(counts.NO, 0);
  $("pending").textContent = number(counts.UNREVIEWED, 0);
  $("progress").style.width =
    `${(100 * (counts.YES + counts.NO)) / state.items.length}%`;
  $("categories").replaceChildren();
  for (const [category, name] of Object.entries(labels)) {
    const total = state.items.filter((i) => i.category === category);
    const pending = total.filter((i) => choice(i.id) === "UNREVIEWED").length;
    const button = element(
      "button",
      name,
      category === state.category ? "active" : "",
    );
    button.append(
      element("span", `${number(pending, 0)} / ${number(total.length, 0)}`),
    );
    button.onclick = () => {
      state.category = category;
      selectFirst();
    };
    $("categories").append(button);
  }
  const queue = list();
  const position = queue.findIndex((i) => i.id === state.id);
  $("queue").replaceChildren();
  const start = Math.max(0, position - 30);
  if (start)
    $("queue").append(
      element("div", `${start} earlier signals · use ←`, "queue-offset"),
    );
  queue.slice(start, start + 100).forEach((item) => {
    const row = element(
      "button",
      undefined,
      `queue-row ${item.id === state.id ? "selected" : ""}`,
    );
    const status = choice(item.id);
    if (status !== "UNREVIEWED")
      row.append(
        element("span", status, `choice ${status === "NO" ? "no" : ""}`),
      );
    row.append(element("strong", item.id.replace("BOI-", "")));
    row.append(
      element(
        "small",
        `${item.period == null ? "Sparse event" : `${number(item.period, 3)} d`} · ${item.views.map((v) => views[v] || v).join(" / ")}`,
      ),
    );
    row.onclick = () => load(item.id);
    $("queue").append(row);
  });
  const selected = $("queue").querySelector(".selected");
  if (selected)
    $("queue").scrollLeft =
      selected.offsetLeft -
      $("queue").offsetLeft -
      ($("queue").clientWidth - selected.offsetWidth) / 2;
  $("queue-count").textContent =
    `${number(queue.length, 0)} signals in this queue`;
  $("export-json").href = `/api/export/${state.dataset}.json`;
  $("export-csv").href = `/api/export/${state.dataset}.csv`;
  $("undo").disabled = state.busy || !state.undo.length;
  localStorage.setItem(
    "boi-review-location",
    JSON.stringify({
      dataset: state.dataset,
      category: state.category,
      id: state.id,
    }),
  );
}

function selectFirst() {
  const queue = list();
  load(queue[0]?.id || null);
}

function available() {
  const ready = !!state.evidence && !!state.plot && !!state.full && !state.busy;
  document.querySelectorAll("button,select,input").forEach((node) => {
    node.disabled = state.busy;
  });
  $("yes").disabled = !ready;
  $("no").disabled = !ready;
  $("previous").disabled = state.busy;
  $("skip").disabled = state.busy;
  $("undo").disabled = state.busy || !state.undo.length;
}

function status() {
  const d = state.decisions[state.id];
  $("current-decision").textContent = d?.choice || "UNREVIEWED";
  $("current-decision").className = d?.choice === "YES" ? "yes" : "";
  $("reason").value = d?.reason || "";
  $("note").value = d?.note || "";
}

async function load(id, proposalIndex = 0) {
  const serial = ++state.serial;
  state.id = id;
  state.proposal = proposalIndex;
  state.evidence = state.plot = state.full = null;
  state.range = null;
  state.fullRange = false;
  $("full-mode").value = "time";
  sidebar();
  available();
  $("empty").hidden = !!id;
  $("review-area").hidden = !id;
  if (!id) return;
  notice();
  $("boi-name").textContent = id;
  $("signal-lane").textContent = `${labels[state.category]} / BOI`;
  const queue = list();
  $("review-position").textContent =
    `${queue.findIndex((i) => i.id === id) + 1} / ${number(queue.length, 0)}`;
  status();
  $("plot-message").hidden = $("full-message").hidden = false;
  $("plot-message").textContent = "Loading original event windows…";
  $("full-message").textContent = "Loading full light curve…";
  try {
    const evidence = await request(`/api/evidence/${id}/${proposalIndex}`);
    if (serial !== state.serial) return;
    state.evidence = evidence;
    state.view = evidence.proposal.discovery_view;
    renderEvidence();
    await Promise.all([loadWindows(), loadFull()]);
  } catch (error) {
    if (serial === state.serial) notice(error.message);
  }
}

function readouts(id, rows) {
  $(id).replaceChildren();
  rows.forEach(([label, value]) => {
    const row = element("div", undefined, "readout");
    row.append(element("span", label), element("b", value));
    $(id).append(row);
  });
}

function metrics() {
  const p = state.evidence.proposal;
  const measured = state.evidence.measurements[state.view] || {};
  const depth = measured.primary_depth;
  const rows = [
    ["PERIOD", number(p.period_days, 4), "days"],
    [
      "DURATION",
      number(p.duration_days == null ? null : p.duration_days * 24),
      "hours",
    ],
    [
      "DEPTH",
      number(
        depth?.depth_fraction == null ? null : depth.depth_fraction * 1e6,
        0,
      ),
      "ppm",
    ],
    ["WINDOWS", number(p.event_units.length, 0), ""],
  ];
  $("metrics").replaceChildren();
  rows.forEach(([name, value, unit]) => {
    const metric = element("div", undefined, "metric");
    metric.append(
      element("label", name),
      element("strong", value),
      element("small", unit),
    );
    $("metrics").append(metric);
  });
  const secondary = measured.secondary_phase_half;
  readouts("orbital-checks", [
    ["Odd / even", `${number(measured.odd_even?.sigma)} σ`],
    [
      "Secondary at phase 0.5",
      `${number(secondary?.depth_fraction == null ? null : secondary.depth_fraction * 1e6, 0)} ppm`,
    ],
    [
      "Depth uncertainty",
      `${number(depth?.depth_error == null ? null : depth.depth_error * 1e6, 0)} ppm`,
    ],
  ]);
}

function renderEvidence() {
  const e = state.evidence;
  const c = e.coordinates;
  $("signal-lane").textContent = `${labels[detectorLane(e.proposal)]} / BOI`;
  $("source-line").textContent =
    `GAIA ${e.proposal.source_id}${c.kic ? ` · KIC ${c.kic}` : ""} · ${number(c.ra_deg, 5)}°, ${number(c.dec_deg, 5)}°`;
  $("source-line").title = $("source-line").textContent;
  $("proposal").replaceChildren();
  e.proposals.forEach((p, i) => {
    const option = element(
      "option",
      `${i + 1} · ${labels[detectorLane(p)]} / ${views[p.discovery_view]}`,
    );
    option.value = i;
    $("proposal").append(option);
  });
  $("proposal").value = state.proposal;
  readouts("host", [
    ["Gaia G", number(e.stellar.phot_g_mean_mag)],
    ["Gaia temperature", `${number(e.stellar.teff_gspphot, 0)} K`],
    ["Gaia radius", `${number(e.stellar.radius_flame, 3)} R☉`],
    [
      "Catalogue matches",
      number(
        (e.catalogues.matches || []).length +
          (e.catalogues.eb_ctoi_matches || []).length,
        0,
      ),
    ],
  ]);
  $("gates").replaceChildren();
  e.gates.forEach((g) => {
    const badge = element("span", `${g.stage} ${g.decision}`);
    badge.title = g.reason;
    $("gates").append(badge);
  });
  $("original").href = `/api/original/${state.id}/${state.proposal}`;
  $("related").replaceChildren();
  const related = state.items.filter(
    (i) => i.source_id === e.proposal.source_id && i.id !== state.id,
  );
  if (related.length) $("related").append(element("span", "SAME SOURCE"));
  related.forEach((i) => {
    const button = element(
      "button",
      `${labels[i.category]} · ${i.period == null ? "event" : `${number(i.period, 3)} d`}`,
    );
    button.onclick = () => {
      state.category = i.category;
      $("filter").value = "ALL";
      $("search").value = "";
      load(i.id);
    };
    $("related").append(button);
  });
  metrics();
  $("full-mode").querySelector('[value="phase"]').disabled = !(
    e.proposal.period_days > 0
  );
}

function detectorLane(p) {
  if (p.detector_classifications.includes("TRANSIT_LIKE")) return "transit";
  if (p.signal_id.startsWith("shallow-v1-")) return "shallow";
  if (p.signal_id.startsWith("signal-")) return "event";
  return "legacy";
}

async function loadWindows() {
  const serial = state.serial,
    sequence = ++state.windowSerial;
  state.plot = null;
  available();
  $("plot-message").hidden = false;
  $("plot-message").textContent = "Loading original event windows…";
  [...$("views").children].forEach((b) =>
    b.classList.toggle("active", b.dataset.view === state.view),
  );
  metrics();
  try {
    const data = await request(
      `/api/plot/${state.id}/${state.proposal}/${state.view}`,
    );
    if (serial !== state.serial || sequence !== state.windowSerial) return;
    state.plot = data;
    $("event").replaceChildren();
    data.windows.forEach((w, i) => {
      const option = element(
        "option",
        `${i + 1} · Q${w.quarter} · ${number(w.centre_bkjd, 3)} BKJD${w.status !== "MEASURED" ? " · no baseline" : ""}`,
      );
      option.value = i;
      $("event").append(option);
    });
    $("mode").querySelector('[value="odd-even"]').disabled = !(
      state.evidence.proposal.period_days > 0
    );
    if (
      $("mode").value === "odd-even" &&
      !(state.evidence.proposal.period_days > 0)
    )
      $("mode").value = "event";
    drawWindows();
    drawTimeline();
    drawFull();
  } catch (error) {
    if (serial === state.serial && sequence === state.windowSerial)
      $("plot-message").textContent = error.message;
  }
  available();
}

async function loadFull() {
  const serial = state.serial,
    sequence = ++state.fullSerial;
  state.full = null;
  available();
  $("full-message").hidden = false;
  $("full-message").textContent = "Loading full light curve…";
  [...$("full-views").children].forEach((b) =>
    b.classList.toggle("active", b.dataset.view === state.fullView),
  );
  try {
    const data = await request(
      `/api/curve/${state.id}/${state.proposal}/${state.fullView}`,
    );
    if (serial !== state.serial || sequence !== state.fullSerial) return;
    state.full = data;
    state.range = null;
    $("quarter").replaceChildren(element("option", "All quarters"));
    $("quarter").firstChild.value = "all";
    Object.keys(data.quarter_baselines)
      .sort((a, b) => Number(a) - Number(b))
      .forEach((q) => {
        const option = element("option", `Quarter ${q}`);
        option.value = q;
        $("quarter").append(option);
      });
    drawFull();
  } catch (error) {
    if (serial === state.serial && sequence === state.fullSerial)
      $("full-message").textContent = error.message;
  }
  available();
}

function canvas(id) {
  const target = $(id),
    rect = target.getBoundingClientRect(),
    ratio = window.devicePixelRatio || 1;
  target.width = Math.round(rect.width * ratio);
  target.height = Math.round(rect.height * ratio);
  const ctx = target.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.font = "10px Menlo, monospace";
  return { ctx, target, width: rect.width, height: rect.height };
}

function extent(values) {
  let low = Infinity,
    high = -Infinity;
  values.forEach((v) => {
    low = Math.min(low, v);
    high = Math.max(high, v);
  });
  if (high === low) {
    low -= 1;
    high += 1;
  }
  return [low, high];
}

function ticks(range, count) {
  const rough = (range[1] - range[0]) / count;
  const power = 10 ** Math.floor(Math.log10(rough));
  const fraction = rough / power;
  const step =
    (fraction >= Math.sqrt(50)
      ? 10
      : fraction >= Math.sqrt(10)
        ? 5
        : fraction >= Math.sqrt(2)
          ? 2
          : 1) * power;
  const values = [];
  for (
    let i = Math.ceil(range[0] / step);
    i <= Math.floor(range[1] / step);
    i++
  )
    values.push(Math.abs(i) < 1e-9 ? 0 : Number((i * step).toPrecision(12)));
  return { values, digits: Math.max(0, -Math.floor(Math.log10(step))) };
}

function axes(surface, xr, yr, unit, xUnit) {
  const { ctx, target, width, height } = surface;
  const yt = ticks(
      yr,
      Math.max(2, Math.min(6, Math.floor((height - 56) / 55))),
    ),
    xt = ticks(xr, Math.max(2, Math.floor(width / 100)));
  const yLabels = yt.values.map((v) => number(v, yt.digits));
  const left = Math.max(
      48,
      ...yLabels.map((v) => ctx.measureText(v).width + 13),
    ),
    right = width - 16,
    top = 20,
    bottom = height - 36;
  const x = (v) => left + ((v - xr[0]) / (xr[1] - xr[0])) * (right - left);
  const y = (v) => bottom - ((v - yr[0]) / (yr[1] - yr[0])) * (bottom - top);
  ctx.fillStyle = "#d4ad6c";
  ctx.fillText(unit, 4, 12);
  yt.values.forEach((value, i) => {
    const yy = y(value);
    ctx.strokeStyle = value === 0 ? "#c18f4355" : "#c18f4322";
    ctx.beginPath();
    ctx.moveTo(left, yy);
    ctx.lineTo(right, yy);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(yLabels[i], left - 9, yy + 3);
  });
  xt.values.forEach((value) => {
    const label = number(value, xt.digits),
      xx = x(value);
    const half = ctx.measureText(label).width / 2;
    ctx.textAlign =
      xx - half < left ? "left" : xx + half > width - 4 ? "right" : "center";
    ctx.fillText(label, xx, bottom + 17);
  });
  ctx.textAlign = "right";
  ctx.fillText(xUnit, right, height - 3);
  ctx.textAlign = "left";
  Object.assign(target.dataset, {
    xRange: JSON.stringify(xr),
    yRange: JSON.stringify(yr),
    xTicks: JSON.stringify(xt.values),
    yTicks: JSON.stringify(yt.values),
    yUnit: unit,
    xUnit,
  });
  return { x, y, left, right, top, bottom, xr, yr, ...surface };
}

function bins(points, xr, count = 70) {
  const groups = Array.from({ length: count }, () => []);
  points.forEach((p) => {
    const i = Math.min(
      count - 1,
      Math.floor(((p.x - xr[0]) / (xr[1] - xr[0])) * count),
    );
    if (p.x >= xr[0] && p.x <= xr[1]) groups[i].push(p.y);
  });
  return groups.flatMap((values, i) =>
    values.length
      ? [
          {
            x: xr[0] + ((i + 0.5) / count) * (xr[1] - xr[0]),
            y: median(values),
          },
        ]
      : [],
  );
}

function scatter(chart, points, color, limit = 50000, radius = 1) {
  const { ctx, x, y, left, top, right, bottom } = chart;
  const step = Math.max(1, Math.ceil(points.length / limit));
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, right - left, bottom - top);
  ctx.clip();
  ctx.fillStyle = color;
  for (let i = 0; i < points.length; i += step)
    ctx.fillRect(
      x(points[i].x) - radius / 2,
      y(points[i].y) - radius / 2,
      radius,
      radius,
    );
  ctx.restore();
  return step;
}

function line(chart, points, color) {
  const { ctx, x, y, left, top, right, bottom } = chart;
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, right - left, bottom - top);
  ctx.clip();
  ctx.strokeStyle = ctx.fillStyle = color;
  ctx.lineWidth = 1.4;
  ctx.shadowColor = color;
  ctx.shadowBlur = 3;
  points.forEach((p, i) => {
    if (i && p.x - points[i - 1].x < (chart.xr[1] - chart.xr[0]) / 15) {
      ctx.beginPath();
      ctx.moveTo(x(points[i - 1].x), y(points[i - 1].y));
      ctx.lineTo(x(p.x), y(p.y));
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(x(p.x), y(p.y), 1.7, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function drawWindows() {
  if (!state.plot) return;
  const mode = $("mode").value,
    p = state.evidence.proposal;
  const selected = Number($("event").value || 0);
  const windows =
    mode === "event"
      ? state.plot.windows.slice(selected, selected + 1)
      : state.plot.windows;
  const groups = [[], []];
  for (const w of windows) {
    if (w.status !== "MEASURED") continue;
    const cycle =
      p.period_days > 0
        ? Math.round((w.centre_bkjd - p.epoch_bkjd) / p.period_days)
        : 0;
    const index = mode === "odd-even" ? ((cycle % 2) + 2) % 2 : 0;
    w.time_hours.forEach((t, i) =>
      groups[index].push({ x: t, y: w.flux_ppm[i] }),
    );
  }
  const points = groups.flat();
  $("plot-message").hidden = !!points.length;
  if (!points.length) {
    $("plot-message").textContent =
      "No measured window with a usable baseline in this view.";
    return;
  }
  const xr = extent(points.map((p) => p.x));
  const medians = groups.map((g) => bins(g, xr));
  let yr = extent(points.map((p) => p.y));
  if (!state.fullRange) {
    const sorted = points.map((p) => p.y).sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)],
      q3 = sorted[Math.floor(sorted.length * 0.75)];
    const br = extent(medians.flat().map((p) => p.y));
    yr = [
      Math.min(br[0], q1 - 3 * (q3 - q1)),
      Math.max(br[1], q3 + 3 * (q3 - q1)),
    ];
  }
  const pad = Math.max(1, (yr[1] - yr[0]) * 0.08);
  yr = [yr[0] - pad, yr[1] + pad];
  const chart = axes(canvas("curve"), xr, yr, "Δflux [ppm]", "Δt [h]");
  if (xr[0] <= 0 && xr[1] >= 0) {
    chart.ctx.strokeStyle = "#a9996155";
    chart.ctx.setLineDash([3, 4]);
    chart.ctx.beginPath();
    chart.ctx.moveTo(chart.x(0), chart.top);
    chart.ctx.lineTo(chart.x(0), chart.bottom);
    chart.ctx.stroke();
    chart.ctx.setLineDash([]);
  }
  const colors = ["#ffcd72", "#8be7ee"];
  groups.forEach((g, i) => {
    scatter(chart, g, i ? "#70dae64d" : "#efbd6742");
    line(chart, medians[i], colors[i]);
  });
  const clipped = points.filter((p) => p.y < yr[0] || p.y > yr[1]).length;
  const measured = windows.filter((w) => w.status === "MEASURED").length;
  const sampled = groups.some((g) => g.length > 50000) ? " · dots thinned" : "";
  $("plot-caption").textContent =
    `${measured}/${windows.length} windows · ${number(points.length, 0)} points${mode === "odd-even" ? " · EVEN amber / ODD cyan" : ""}${clipped ? ` · ${number(clipped, 0)} off-scale` : ""}${sampled}`;
  $("scale").textContent = state.fullRange ? "SIGNAL RANGE" : "FULL RANGE";
}

function drawTimeline() {
  if (!state.plot?.windows.length) return;
  const surface = canvas("timeline"),
    { ctx, width } = surface;
  const times = state.plot.windows.map((w) => w.centre_bkjd),
    xr = extent(times);
  const x = (t) => 15 + ((t - xr[0]) / (xr[1] - xr[0])) * (width - 30);
  ctx.strokeStyle = "#a0784766";
  ctx.beginPath();
  ctx.moveTo(15, 15);
  ctx.lineTo(width - 15, 15);
  ctx.stroke();
  state.plot.windows.forEach((w, i) => {
    ctx.fillStyle = i === Number($("event").value) ? "#ffcd72" : "#8cccd1";
    ctx.fillRect(x(w.centre_bkjd) - 1, 8, 2, 13);
  });
  ctx.fillStyle = "#bda275";
  ctx.fillText(number(xr[0], 1), 15, 28);
  ctx.textAlign = "right";
  ctx.fillText(number(xr[1], 1), width - 15, 28);
  state.timeline = { xr, left: 15, right: width - 15 };
}

function fullPoints() {
  const a = state.full.arrays,
    points = [];
  const quarter = $("quarter").value,
    normalized = $("flux-mode").value === "normalized";
  for (let i = 0; i < a.time.length; i++) {
    if (
      a.time[i] == null ||
      a.flux[i] == null ||
      (quarter !== "all" && String(a.quarter[i]) !== quarter)
    )
      continue;
    const p = state.evidence.proposal;
    const phase = $("full-mode").value === "phase";
    const x = phase
      ? (((((a.time[i] - p.epoch_bkjd) / p.period_days + 0.5) % 1) + 1) % 1) -
        0.5
      : a.time[i];
    if (state.range && (x < state.range[0] || x > state.range[1])) continue;
    const baseline = state.full.quarter_baselines[a.quarter[i]];
    if (normalized && baseline == null) continue;
    const y = normalized ? (a.flux[i] / baseline - 1) * 1e6 : a.flux[i];
    points.push({ x, y, valid: a.valid[i] });
  }
  return points;
}

function drawFull() {
  if (!state.full) return;
  const points = fullPoints();
  $("full-message").hidden = !!points.length;
  if (!points.length) {
    $("full-message").textContent =
      "No finite data in this selection. Choose native flux if this quarter has no valid baseline.";
    return;
  }
  const phase = $("full-mode").value === "phase";
  const xr =
    state.range || (phase ? [-0.5, 0.5] : extent(points.map((p) => p.x)));
  let yr = extent(points.map((p) => p.y));
  const pad = Math.max(1, (yr[1] - yr[0]) * 0.06);
  yr = [yr[0] - pad, yr[1] + pad];
  const chart = axes(
    canvas("full-curve"),
    xr,
    yr,
    $("flux-mode").value === "native" ? "Flux [e−/s]" : "Δflux [ppm]",
    phase ? "Phase" : "BKJD",
  );
  const good = points.filter((p) => p.valid),
    masked = points.filter((p) => !p.valid);
  const step = scatter(chart, good, "#63d6e2cc", 60000, 1.1);
  scatter(chart, masked, "#cc935d66", 20000, 1.1);
  const events = phase ? [{ centre_bkjd: 0 }] : state.evidence.events;
  chart.ctx.strokeStyle = "#b9985260";
  chart.ctx.lineWidth = 1;
  events.forEach((e, i) => {
    if (e.centre_bkjd < xr[0] || e.centre_bkjd > xr[1]) return;
    const x = chart.x(e.centre_bkjd);
    chart.ctx.strokeStyle =
      i === Number($("event").value) ? "#f1b576" : "#b9985260";
    chart.ctx.beginPath();
    chart.ctx.moveTo(x, chart.top);
    chart.ctx.lineTo(x, chart.bottom);
    chart.ctx.stroke();
  });
  if (phase) line(chart, bins(good, xr, 120), "#a3f1f5");
  else
    state.evidence.source_bois
      .filter((r) => r.boi_id !== state.id)
      .forEach((r) => {
        chart.ctx.strokeStyle = "#73a8c766";
        r.events.forEach((e) => {
          if (e.centre_bkjd < xr[0] || e.centre_bkjd > xr[1]) return;
          const x = chart.x(e.centre_bkjd);
          chart.ctx.beginPath();
          chart.ctx.moveTo(x, chart.top);
          chart.ctx.lineTo(x, chart.top + 12);
          chart.ctx.stroke();
        });
      });
  state.fullChart = chart;
  const absent =
    state.full.cadences -
    state.full.arrays.time.filter(
      (v, i) => v != null && state.full.arrays.flux[i] != null,
    ).length;
  $("full-caption").textContent =
    `${number(points.length, 0)} points · ${number(masked.length, 0)} masked (amber)${absent ? ` · ${absent} nonfinite` : ""}${step > 1 ? ` · dots 1:${step}` : ""}`;
}

function nearestEvent(time) {
  let nearest = 0,
    distance = Infinity;
  state.evidence.events.forEach((e, i) => {
    if (Math.abs(e.centre_bkjd - time) < distance) {
      distance = Math.abs(e.centre_bkjd - time);
      nearest = i;
    }
  });
  $("event").value = nearest;
  $("mode").value = "event";
  drawWindows();
  drawTimeline();
  drawFull();
}

function move(direction) {
  if (state.busy) return;
  const queue = list(),
    current = queue.findIndex((i) => i.id === state.id);
  const next = queue[current + direction];
  if (next) load(next.id);
}

async function decide(value) {
  if ($(value === "YES" ? "yes" : "no").disabled) return;
  const id = state.id,
    dataset = state.dataset;
  const previous = state.decisions[id] ? { ...state.decisions[id] } : null;
  const position = list().findIndex((i) => i.id === id);
  state.busy = true;
  available();
  notice();
  try {
    const row = await request("/api/decision", {
      dataset,
      boi: id,
      choice: value,
      reason: $("reason").value,
      note: $("note").value,
      revision: previous?.revision || 0,
    });
    state.decisions[id] = row;
    state.undo.push({ dataset, id, previous });
    $("save-status").textContent =
      `Saved ${new Date(row.updated).toLocaleTimeString()}`;
    $("save-status").title = `${value} · ${id}`;
    const queue = list();
    const next =
      $("filter").value === "UNREVIEWED"
        ? queue[Math.min(position, queue.length - 1)]
        : queue[position + 1];
    if (next) load(next.id);
    else if (!queue.length) load(null);
    else {
      status();
      sidebar();
    }
  } catch (error) {
    notice(error.message);
    state.decisions = await request(`/api/dataset/${dataset}`);
    status();
    sidebar();
  } finally {
    state.busy = false;
    available();
  }
}

async function undo() {
  if (state.busy || !state.undo.length) return;
  const last = state.undo.at(-1);
  if (last.dataset !== state.dataset) {
    notice("Switch back to the previous dataset to undo that decision.");
    return;
  }
  state.busy = true;
  available();
  try {
    const p = last.previous;
    const row = await request("/api/decision", {
      dataset: state.dataset,
      boi: last.id,
      choice: p?.choice || "UNREVIEWED",
      reason: p?.reason || "",
      note: p?.note || "",
      revision: state.decisions[last.id].revision,
    });
    state.decisions[last.id] = row;
    state.undo.pop();
    state.category = state.items.find((i) => i.id === last.id).category;
    $("filter").value = "ALL";
    $("search").value = "";
    load(last.id);
    $("save-status").textContent = "Restored";
    $("save-status").title = last.id;
  } catch (error) {
    notice(error.message);
  } finally {
    state.busy = false;
    available();
  }
}

function datasetOptions() {
  $("dataset").replaceChildren();
  state.datasets.forEach((d) => {
    const option = element("option", d.name);
    option.value = d.id;
    $("dataset").append(option);
  });
  $("dataset").value = state.dataset;
}

async function changeDataset(dataset) {
  state.busy = true;
  available();
  try {
    const decisions = await request(`/api/dataset/${dataset}`);
    state.dataset = dataset;
    state.decisions = decisions;
    state.undo = [];
    selectFirst();
  } finally {
    state.busy = false;
    datasetOptions();
    available();
  }
}

function interactions() {
  $("dataset").onchange = () =>
    changeDataset($("dataset").value).catch((e) => notice(e.message));
  $("new-dataset").onclick = () => {
    $("create-form").hidden = false;
    $("dataset-name").focus();
  };
  $("cancel-create").onclick = () => {
    $("create-form").hidden = true;
  };
  $("create-form").onsubmit = async (event) => {
    event.preventDefault();
    state.busy = true;
    available();
    try {
      const row = await request("/api/datasets", {
        name: $("dataset-name").value,
      });
      state.datasets.push(row);
      $("create-form").hidden = true;
      $("dataset-name").value = "";
      await changeDataset(row.id);
    } catch (error) {
      notice(error.message);
    } finally {
      state.busy = false;
      available();
    }
  };
  $("filter").onchange = selectFirst;
  $("search").oninput = selectFirst;
  $("proposal").onchange = () => load(state.id, Number($("proposal").value));
  $("views").onclick = (event) => {
    if (event.target.dataset.view && state.evidence) {
      state.view = event.target.dataset.view;
      loadWindows();
    }
  };
  $("full-views").onclick = (event) => {
    if (event.target.dataset.view && state.evidence) {
      state.fullView = event.target.dataset.view;
      loadFull();
    }
  };
  $("mode").onchange = drawWindows;
  $("event").onchange = () => {
    $("mode").value = "event";
    drawWindows();
    drawTimeline();
    drawFull();
  };
  $("scale").onclick = () => {
    state.fullRange = !state.fullRange;
    drawWindows();
  };
  $("quarter").onchange = () => {
    state.range = null;
    drawFull();
  };
  $("flux-mode").onchange = drawFull;
  $("full-mode").onchange = () => {
    state.range = null;
    drawFull();
  };
  $("reset-range").onclick = () => {
    state.range = null;
    $("quarter").value = "all";
    drawFull();
  };
  $("yes").onclick = () => decide("YES");
  $("no").onclick = () => decide("NO");
  $("undo").onclick = undo;
  $("previous").onclick = () => move(-1);
  $("skip").onclick = () => move(1);
  $("timeline").onclick = (event) => {
    if (!state.timeline || !state.evidence) return;
    const c = state.timeline,
      x = event.offsetX;
    nearestEvent(
      c.xr[0] + ((x - c.left) / (c.right - c.left)) * (c.xr[1] - c.xr[0]),
    );
  };
  let drag = null;
  $("full-curve").onpointerdown = (event) => {
    if (state.fullChart) {
      drag = event.offsetX;
      event.target.setPointerCapture(event.pointerId);
    }
  };
  $("full-curve").onpointerup = (event) => {
    if (drag === null || !state.fullChart) return;
    const c = state.fullChart;
    const time = (x) =>
      c.xr[0] +
      Math.max(0, Math.min(1, (x - c.left) / (c.right - c.left))) *
        (c.xr[1] - c.xr[0]);
    if (Math.abs(event.offsetX - drag) > 8) {
      state.range = [
        time(Math.min(drag, event.offsetX)),
        time(Math.max(drag, event.offsetX)),
      ];
      drawFull();
    } else if ($("full-mode").value === "time")
      nearestEvent(time(event.offsetX));
    drag = null;
  };
  $("full-curve").onpointercancel = () => {
    drag = null;
  };
  document.addEventListener("keydown", (event) => {
    if (
      event.target.closest("input,select,textarea") ||
      event.metaKey ||
      event.ctrlKey ||
      event.altKey ||
      event.repeat
    )
      return;
    const actions = {
      y: () => decide("YES"),
      n: () => decide("NO"),
      z: undo,
      ArrowRight: () => move(1),
      ArrowLeft: () => move(-1),
    };
    const action = actions[event.key] || actions[event.key.toLowerCase()];
    if (action) {
      event.preventDefault();
      action();
    }
  });
  const observer = new ResizeObserver(() => {
    drawWindows();
    drawTimeline();
    drawFull();
  });
  ["curve", "full-curve", "timeline"].forEach((id) => observer.observe($(id)));
}

async function start() {
  try {
    Object.assign(state, await request("/api/boot"));
    let saved = {};
    try {
      saved = JSON.parse(localStorage.getItem("boi-review-location") || "{}");
    } catch {
      /* Ignore an invalid browser preference; reviews live in SQLite. */
    }
    state.dataset = state.datasets.some((d) => d.id === saved.dataset)
      ? saved.dataset
      : state.datasets[0].id;
    state.category = labels[saved.category] ? saved.category : "transit";
    state.decisions = await request(`/api/dataset/${state.dataset}`);
    datasetOptions();
    interactions();
    const queue = list();
    load(queue.some((i) => i.id === saved.id) ? saved.id : queue[0]?.id);
  } catch (error) {
    notice(error.message);
  }
}

start();
