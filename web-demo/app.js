const RE_VALUES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60];
const POINT_COUNT = 200;

const i18n = {
  zh: {
    productEyebrow: "微流道三角截面设计",
    productName: "TriFocus ForceLab",
    demoModel: "演示代理模型",
    stepOne: "01 输入",
    inputTitle: "只输入三条边",
    predictButton: "生成并预测",
    stepTwo: "02 雷诺数扫描",
    reTitle: "寻找可聚焦区间",
    currentRe: "当前 Re",
    stepThree: "03 图层",
    layerTitle: "力场视图",
    layerTotal: "合力",
    layerFLX: "FL-x",
    layerFLY: "FL-y",
    layerFDX: "Dean-x",
    layerFDY: "Dean-y",
    layerFocus: "聚焦",
    downloadCsv: "下载 CSV",
    downloadPng: "下载 PNG",
    demoNote: "Demo surrogate · TabPFN API ready",
    visualEyebrow: "实时截面力场",
    visualTitle: "合力分布与聚焦位置",
    legendLow: "低",
    legendHigh: "高",
    focusEyebrow: "聚焦判断",
    focusRange: "可聚焦 Re",
    focusPoint: "聚焦位置",
    pointCount: "采样点",
    modelVersion: "模型版本",
    metricName: "指标",
    metricMin: "最小",
    metricAvg: "平均",
    metricMax: "最大",
    areaLabel: "面积",
    perimeterLabel: "周长",
    heightLabel: "高度",
    angleLabel: "角度",
    invalidNumber: "请输入三条正数边长。",
    invalidTriangle: "输入边长无法构成三角形。",
    focused: "聚焦良好",
    weak_focus: "弱聚焦",
    not_focused: "不聚焦",
    unknown: "暂无法判断",
    reasonFocused: "合力低值区稳定，Re 区间内力场向同一位置收敛。",
    reasonWeak: "存在可见低合力区域，但部分 Re 下收敛性仍需模型确认。",
    reasonNone: "Re 扫描内没有形成稳定低合力区。",
    noRange: "未形成",
    total: "合力",
    fl: "惯性升力",
    fd: "Dean 力",
    focusScore: "聚焦分数",
    x: "x",
    y: "y",
  },
  en: {
    productEyebrow: "Microchannel triangular-section design",
    productName: "TriFocus ForceLab",
    demoModel: "Demo surrogate model",
    stepOne: "01 Input",
    inputTitle: "Only three side lengths",
    predictButton: "Generate and predict",
    stepTwo: "02 Reynolds sweep",
    reTitle: "Find focus window",
    currentRe: "Current Re",
    stepThree: "03 Layer",
    layerTitle: "Force-field view",
    layerTotal: "Total",
    layerFLX: "FL-x",
    layerFLY: "FL-y",
    layerFDX: "Dean-x",
    layerFDY: "Dean-y",
    layerFocus: "Focus",
    downloadCsv: "Download CSV",
    downloadPng: "Download PNG",
    demoNote: "Demo surrogate · TabPFN API ready",
    visualEyebrow: "Live cross-section force field",
    visualTitle: "Total force and focus location",
    legendLow: "Low",
    legendHigh: "High",
    focusEyebrow: "Focus assessment",
    focusRange: "Focus Re",
    focusPoint: "Focus point",
    pointCount: "Points",
    modelVersion: "Model version",
    metricName: "Metric",
    metricMin: "Min",
    metricAvg: "Mean",
    metricMax: "Max",
    areaLabel: "Area",
    perimeterLabel: "Perimeter",
    heightLabel: "Height",
    angleLabel: "Angles",
    invalidNumber: "Enter three positive side lengths.",
    invalidTriangle: "The side lengths cannot form a triangle.",
    focused: "Focused",
    weak_focus: "Weak focus",
    not_focused: "Not focused",
    unknown: "Unknown",
    reasonFocused: "The low-force region is stable and the field converges to one location across the Re window.",
    reasonWeak: "A low-force region is visible, but convergence still needs model confirmation for part of the Re sweep.",
    reasonNone: "No stable low-force region formed in the scanned Re range.",
    noRange: "None",
    total: "Total",
    fl: "Inertial lift",
    fd: "Dean force",
    focusScore: "Focus score",
    x: "x",
    y: "y",
  },
};

const state = {
  lang: "zh",
  layer: "total",
  selectedRe: 36,
  result: null,
  canvasTransform: null,
};

const el = {
  form: document.getElementById("triangleForm"),
  sideAB: document.getElementById("sideAB"),
  sideAC: document.getElementById("sideAC"),
  sideBC: document.getElementById("sideBC"),
  formError: document.getElementById("formError"),
  canvas: document.getElementById("forceCanvas"),
  tooltip: document.getElementById("tooltip"),
  reStrip: document.getElementById("reStrip"),
  currentReLabel: document.getElementById("currentReLabel"),
  languageToggle: document.getElementById("languageToggle"),
  focusStatus: document.getElementById("focusStatus"),
  confidenceFill: document.getElementById("confidenceFill"),
  focusReason: document.getElementById("focusReason"),
  focusRange: document.getElementById("focusRange"),
  focusPoint: document.getElementById("focusPoint"),
  pointCount: document.getElementById("pointCount"),
  statsBody: document.getElementById("statsBody"),
  areaValue: document.getElementById("areaValue"),
  perimeterValue: document.getElementById("perimeterValue"),
  heightValue: document.getElementById("heightValue"),
  anglesValue: document.getElementById("anglesValue"),
  downloadCsv: document.getElementById("downloadCsv"),
  downloadPng: document.getElementById("downloadPng"),
  visualTitle: document.getElementById("visualTitle"),
};

const ctx = el.canvas.getContext("2d");

function t(key) {
  return i18n[state.lang][key] || i18n.zh[key] || key;
}

function setLanguage(lang) {
  state.lang = lang;
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  el.languageToggle.textContent = lang === "zh" ? "EN" : "中文";
  if (state.result) {
    updateAll();
  }
}

function readSides() {
  return {
    ab: Number(el.sideAB.value),
    ac: Number(el.sideAC.value),
    bc: Number(el.sideBC.value),
  };
}

function validateSides({ ab, ac, bc }) {
  if (![ab, ac, bc].every((v) => Number.isFinite(v) && v > 0)) {
    return t("invalidNumber");
  }
  if (ab + ac <= bc || ab + bc <= ac || ac + bc <= ab) {
    return t("invalidTriangle");
  }
  return "";
}

function angleFromSides(a, b, c) {
  const cos = clamp((b * b + c * c - a * a) / (2 * b * c), -1, 1);
  return (Math.acos(cos) * 180) / Math.PI;
}

function buildGeometry({ ab, ac, bc }) {
  const xA = (ab * ab + bc * bc - ac * ac) / (2 * bc);
  const yA = Math.sqrt(Math.max(0, ab * ab - xA * xA));
  const vertices = {
    A: { x: xA, y: yA },
    B: { x: 0, y: 0 },
    C: { x: bc, y: 0 },
  };
  return {
    sides: { ab, ac, bc },
    vertices,
    area: (bc * yA) / 2,
    perimeter: ab + ac + bc,
    height: yA,
    centroid: { x: (xA + bc) / 3, y: yA / 3 },
    angles: {
      A: angleFromSides(bc, ab, ac),
      B: angleFromSides(ac, ab, bc),
      C: angleFromSides(ab, ac, bc),
    },
  };
}

function halton(index, base) {
  let f = 1;
  let r = 0;
  let i = index;
  while (i > 0) {
    f /= base;
    r += f * (i % base);
    i = Math.floor(i / base);
  }
  return r;
}

function pointLineDistance(p, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return Math.abs(dy * p.x - dx * p.y + b.x * a.y - b.y * a.x) / Math.hypot(dx, dy);
}

function generatePoints(geometry) {
  const { A, B, C } = geometry.vertices;
  const points = [];
  for (let i = 1; i <= POINT_COUNT; i += 1) {
    let u = halton(i, 2);
    let v = halton(i, 3);
    if (u + v > 1) {
      u = 1 - u;
      v = 1 - v;
    }
    const x = B.x + u * (C.x - B.x) + v * (A.x - B.x);
    const y = B.y + u * (C.y - B.y) + v * (A.y - B.y);
    const p = { point_id: `p${String(i).padStart(4, "0")}`, x, y };
    p.distBase = pointLineDistance(p, B, C);
    p.distLeft = pointLineDistance(p, A, B);
    p.distRight = pointLineDistance(p, A, C);
    p.distCentroid = Math.hypot(p.x - geometry.centroid.x, p.y - geometry.centroid.y);
    p.xNorm = safeDiv(p.x, geometry.sides.bc);
    p.yNorm = safeDiv(p.y, geometry.height);
    points.push(p);
  }
  return points;
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function safeDiv(a, b) {
  return b === 0 ? 0 : a / b;
}

function focusModel(geometry, re) {
  const { ab, ac, bc } = geometry.sides;
  const aspect = bc / geometry.height;
  const symmetry = 1 - Math.min(Math.abs(ab - ac) / Math.max(ab, ac), 0.95);
  const aspectScore = Math.exp(-Math.pow((aspect - 1.45) / 0.68, 2));
  const geometryScore = clamp(0.22 + 0.52 * symmetry + 0.32 * aspectScore, 0.16, 0.98);
  const optRe = clamp(31 + (aspect - 1.25) * 12 + (1 - symmetry) * 10, 18, 52);
  const reScore = Math.exp(-Math.pow((re - optRe) / 16, 2));
  const score = clamp(geometryScore * (0.18 + 0.88 * reScore), 0.03, 0.98);
  const skew = (ab - ac) / (ab + ac);
  const fx = clamp(
    geometry.centroid.x + (re - 36) * 0.0035 * bc - skew * 0.09 * bc,
    Math.min(geometry.vertices.A.x, 0, bc) + bc * 0.12,
    Math.max(geometry.vertices.A.x, 0, bc) - bc * 0.1
  );
  const fy = clamp(
    geometry.centroid.y + Math.sin(re / 9) * 0.045 * geometry.height + (1 - aspectScore) * 0.04 * geometry.height,
    geometry.height * 0.16,
    geometry.height * 0.78
  );
  return { score, x: fx, y: fy, optRe, geometryScore };
}

function predictForRe(geometry, basePoints, re) {
  const focus = focusModel(geometry, re);
  const scale = Math.max(geometry.sides.bc, geometry.height);
  const reFactor = re / 60;
  const skew = (geometry.sides.ab - geometry.sides.ac) / (geometry.sides.ab + geometry.sides.ac);
  const points = basePoints.map((p) => {
    const dx = focus.x - p.x;
    const dy = focus.y - p.y;
    const nx = dx / scale;
    const ny = dy / scale;
    const dist = Math.hypot(nx, ny);
    const wallBias = safeDiv(p.distRight - p.distLeft, geometry.sides.bc);
    const verticalBias = p.yNorm - 0.5;
    const convergence = 0.038 * focus.score * dist;
    const swirl = 0.016 * (1 - focus.score) * (0.45 + reFactor);
    const totalBaseX = convergence * nx + swirl * -ny + 0.0025 * wallBias;
    const totalBaseY = convergence * ny + swirl * nx - 0.0018 * verticalBias;
    const flMix = 0.57 + 0.07 * Math.sin(re / 12);
    const flX = totalBaseX * flMix + 0.0036 * wallBias + 0.001 * skew;
    const flY = totalBaseY * flMix - 0.0028 * verticalBias + 0.0012 * Math.sin(Math.PI * p.xNorm);
    const fdX = totalBaseX - flX;
    const fdY = totalBaseY - flY;
    const totalX = flX + fdX;
    const totalY = flY + fdY;
    return {
      ...p,
      re,
      fl_x_pred: flX,
      fl_y_pred: flY,
      fl_magnitude_pred: Math.hypot(flX, flY),
      fd_x_pred: fdX,
      fd_y_pred: fdY,
      fd_magnitude_pred: Math.hypot(fdX, fdY),
      total_force_x: totalX,
      total_force_y: totalY,
      total_force_magnitude: Math.hypot(totalX, totalY),
      focus_score: focus.score,
      focus_distance_um: Math.hypot(p.x - focus.x, p.y - focus.y),
      focus_region_flag: Math.hypot(p.x - focus.x, p.y - focus.y) < scale * 0.08 ? 1 : 0,
    };
  });
  return { re, focus, points };
}

function runPrediction() {
  const sides = readSides();
  const error = validateSides(sides);
  el.formError.textContent = error;
  if (error) return;

  const geometry = buildGeometry(sides);
  const basePoints = generatePoints(geometry);
  const sweep = RE_VALUES.map((re) => predictForRe(geometry, basePoints, re));
  const focusScores = sweep.map((s) => ({ re: s.re, score: s.focus.score }));
  const best = sweep.reduce((a, b) => (b.focus.score > a.focus.score ? b : a));
  const focusedRes = focusScores.filter((s) => s.score >= 0.52);
  const status = best.focus.score >= 0.72 ? "focused" : best.focus.score >= 0.48 ? "weak_focus" : "not_focused";

  state.result = {
    input: sides,
    geometry,
    basePoints,
    sweep,
    bestRe: best.re,
    focus: {
      status,
      confidence: best.focus.score,
      x: best.focus.x,
      y: best.focus.y,
      reRange: focusedRes.length ? [focusedRes[0].re, focusedRes[focusedRes.length - 1].re] : null,
      method: "force_rule",
    },
  };
  state.selectedRe = best.re;
  updateAll();
}

function activePrediction() {
  return state.result.sweep.find((item) => item.re === state.selectedRe) || state.result.sweep[0];
}

function updateAll() {
  renderReStrip();
  updateSummary();
  resizeCanvasToDisplay();
  renderCanvas();
}

function renderReStrip() {
  el.reStrip.innerHTML = "";
  state.result.sweep.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `re-button${item.re === state.selectedRe ? " active" : ""}${item.focus.score >= 0.52 ? " focused" : ""}`;
    button.textContent = String(item.re);
    button.style.setProperty("--score-color", scoreColor(item.focus.score));
    button.addEventListener("click", () => {
      state.selectedRe = item.re;
      updateAll();
    });
    el.reStrip.appendChild(button);
  });
  el.currentReLabel.textContent = String(state.selectedRe);
}

function scoreColor(score) {
  if (score > 0.72) return "#2e9d63";
  if (score > 0.52) return "#f3b340";
  return "#e45b4f";
}

function updateSummary() {
  const result = state.result;
  const active = activePrediction();
  const g = result.geometry;
  const focus = result.focus;
  const locale = state.lang === "zh" ? "zh-CN" : "en-US";

  el.focusStatus.textContent = t(focus.status);
  el.confidenceFill.style.width = `${Math.round(focus.confidence * 100)}%`;
  el.focusReason.textContent =
    focus.status === "focused" ? t("reasonFocused") : focus.status === "weak_focus" ? t("reasonWeak") : t("reasonNone");
  el.focusRange.textContent = focus.reRange ? `${focus.reRange[0]} - ${focus.reRange[1]}` : t("noRange");
  el.focusPoint.textContent = `${format(focus.x, 1)} um, ${format(focus.y, 1)} um`;
  el.pointCount.textContent = String(POINT_COUNT);

  el.areaValue.textContent = `${format(g.area, 0, locale)} um²`;
  el.perimeterValue.textContent = `${format(g.perimeter, 0, locale)} um`;
  el.heightValue.textContent = `${format(g.height, 0, locale)} um`;
  el.anglesValue.textContent = `${format(g.angles.A, 1)}° / ${format(g.angles.B, 1)}° / ${format(g.angles.C, 1)}°`;

  const stats = [
    [t("fl"), active.points.map((p) => p.fl_magnitude_pred)],
    [t("fd"), active.points.map((p) => p.fd_magnitude_pred)],
    [t("total"), active.points.map((p) => p.total_force_magnitude)],
    [t("focusScore"), active.points.map((p) => p.focus_score)],
  ];
  el.statsBody.innerHTML = stats
    .map(([label, values]) => {
      const s = summarize(values);
      return `<tr><td>${label}</td><td>${format(s.min, 4)}</td><td>${format(s.avg, 4)}</td><td>${format(s.max, 4)}</td></tr>`;
    })
    .join("");

  const titleByLayer = {
    total: state.lang === "zh" ? "合力分布与聚焦位置" : "Total force and focus location",
    flx: state.lang === "zh" ? "惯性升力 x 方向投影" : "Inertial lift projection on x",
    fly: state.lang === "zh" ? "惯性升力 y 方向投影" : "Inertial lift projection on y",
    fdx: state.lang === "zh" ? "Dean force x 分量" : "Dean force component on x",
    fdy: state.lang === "zh" ? "Dean force y 分量" : "Dean force component on y",
    focus: state.lang === "zh" ? "聚焦区域与 Re 稳定性" : "Focus region and Re stability",
  };
  el.visualTitle.textContent = titleByLayer[state.layer];
}

function summarize(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const avg = values.reduce((sum, v) => sum + v, 0) / values.length;
  return { min, max, avg };
}

function format(value, digits = 2, locale = "en-US") {
  return Number(value).toLocaleString(locale, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits > 2 ? digits : 0,
  });
}

function resizeCanvasToDisplay() {
  const rect = el.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(420, Math.floor(rect.width * dpr));
  const height = Math.max(360, Math.floor(rect.height * dpr));
  if (el.canvas.width !== width || el.canvas.height !== height) {
    el.canvas.width = width;
    el.canvas.height = height;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function getCanvasSize() {
  const rect = el.canvas.getBoundingClientRect();
  return { w: rect.width, h: rect.height };
}

function computeTransform() {
  const { geometry } = state.result;
  const { A, B, C } = geometry.vertices;
  const xs = [A.x, B.x, C.x];
  const ys = [A.y, B.y, C.y];
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const { w, h } = getCanvasSize();
  const pad = Math.min(72, Math.max(42, Math.min(w, h) * 0.1));
  const sx = (w - pad * 2) / Math.max(1, maxX - minX);
  const sy = (h - pad * 2) / Math.max(1, maxY - minY);
  const scale = Math.min(sx, sy);
  const usedW = (maxX - minX) * scale;
  const usedH = (maxY - minY) * scale;
  const offsetX = (w - usedW) / 2 - minX * scale;
  const offsetY = (h - usedH) / 2 + maxY * scale;
  const transform = {
    toScreen: (p) => ({ x: offsetX + p.x * scale, y: offsetY - p.y * scale }),
    toData: (p) => ({ x: (p.x - offsetX) / scale, y: (offsetY - p.y) / scale }),
    scale,
  };
  state.canvasTransform = transform;
  return transform;
}

function renderCanvas() {
  if (!state.result) return;
  const { w, h } = getCanvasSize();
  const transform = computeTransform();
  ctx.clearRect(0, 0, w, h);
  drawGrid(w, h);

  const active = activePrediction();
  const valueInfo = layerValues(active.points);
  const sortedPoints = [...active.points].sort((a, b) => valueInfo.value(a) - valueInfo.value(b));

  drawHeatPoints(sortedPoints, valueInfo, transform);
  if (state.layer === "total" || state.layer === "focus") {
    drawArrows(active.points, transform);
  }
  drawTriangle(transform);
  drawFocus(active.focus, transform);
  drawAxes(transform);
}

function drawGrid(w, h) {
  ctx.save();
  ctx.strokeStyle = "rgba(21,34,37,0.06)";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 42) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 42) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  ctx.restore();
}

function layerValues(points) {
  const map = {
    total: { value: (p) => p.total_force_magnitude, signed: false },
    flx: { value: (p) => p.fl_x_pred, signed: true },
    fly: { value: (p) => p.fl_y_pred, signed: true },
    fdx: { value: (p) => p.fd_x_pred, signed: true },
    fdy: { value: (p) => p.fd_y_pred, signed: true },
    focus: { value: (p) => 1 / (1 + p.focus_distance_um), signed: false },
  };
  const info = map[state.layer] || map.total;
  const values = points.map(info.value);
  return {
    ...info,
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

function drawHeatPoints(points, valueInfo, transform) {
  points.forEach((p) => {
    const screen = transform.toScreen(p);
    const color = colorFor(valueInfo.value(p), valueInfo.min, valueInfo.max, valueInfo.signed);
    const radius = state.layer === "focus" && p.focus_region_flag ? 6 : 4.2;
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = state.layer === "focus" && !p.focus_region_flag ? 0.45 : 0.88;
    ctx.fill();
  });
  ctx.globalAlpha = 1;
}

function colorFor(value, min, max, signed) {
  if (signed) {
    const span = Math.max(Math.abs(min), Math.abs(max), 1e-8);
    const normalized = clamp(value / span, -1, 1);
    if (normalized < 0) {
      return blend("#277da1", "#f7fbfb", normalized + 1);
    }
    return blend("#f7fbfb", "#d94141", normalized);
  }
  const n = clamp((value - min) / Math.max(max - min, 1e-8), 0, 1);
  if (n < 0.25) return blend("#277da1", "#43aa8b", n / 0.25);
  if (n < 0.5) return blend("#43aa8b", "#f9c74f", (n - 0.25) / 0.25);
  if (n < 0.75) return blend("#f9c74f", "#f3722c", (n - 0.5) / 0.25);
  return blend("#f3722c", "#d94141", (n - 0.75) / 0.25);
}

function blend(a, b, tValue) {
  const tClamped = clamp(tValue, 0, 1);
  const ca = hexToRgb(a);
  const cb = hexToRgb(b);
  return `rgb(${Math.round(ca.r + (cb.r - ca.r) * tClamped)}, ${Math.round(ca.g + (cb.g - ca.g) * tClamped)}, ${Math.round(ca.b + (cb.b - ca.b) * tClamped)})`;
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  return {
    r: parseInt(clean.slice(0, 2), 16),
    g: parseInt(clean.slice(2, 4), 16),
    b: parseInt(clean.slice(4, 6), 16),
  };
}

function drawArrows(points, transform) {
  ctx.save();
  ctx.strokeStyle = "rgba(21,34,37,0.58)";
  ctx.fillStyle = "rgba(21,34,37,0.58)";
  ctx.lineWidth = 1.2;
  const every = Math.max(1, Math.floor(points.length / 38));
  points.forEach((p, idx) => {
    if (idx % every !== 0) return;
    const s = transform.toScreen(p);
    const mag = Math.hypot(p.total_force_x, p.total_force_y);
    if (mag < 1e-8) return;
    const len = clamp(mag * 1250, 8, 20);
    const dx = (p.total_force_x / mag) * len;
    const dy = -(p.total_force_y / mag) * len;
    drawArrow(s.x, s.y, s.x + dx, s.y + dy);
  });
  ctx.restore();
}

function drawArrow(x1, y1, x2, y2) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - Math.cos(angle - 0.55) * 5, y2 - Math.sin(angle - 0.55) * 5);
  ctx.lineTo(x2 - Math.cos(angle + 0.55) * 5, y2 - Math.sin(angle + 0.55) * 5);
  ctx.closePath();
  ctx.fill();
}

function drawTriangle(transform) {
  const { A, B, C } = state.result.geometry.vertices;
  const a = transform.toScreen(A);
  const b = transform.toScreen(B);
  const c = transform.toScreen(C);
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.lineTo(c.x, c.y);
  ctx.closePath();
  ctx.fillStyle = "rgba(255,255,255,0.2)";
  ctx.fill();
  ctx.lineWidth = 2.4;
  ctx.strokeStyle = "#152225";
  ctx.stroke();
  drawVertexLabel("A", a.x, a.y - 14);
  drawVertexLabel("B", b.x - 18, b.y + 20);
  drawVertexLabel("C", c.x + 18, c.y + 20);
  ctx.restore();
}

function drawVertexLabel(label, x, y) {
  ctx.save();
  ctx.font = "800 13px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#152225";
  ctx.fillText(label, x, y);
  ctx.restore();
}

function drawFocus(focus, transform) {
  const s = transform.toScreen(focus);
  const pulse = 18 + focus.score * 20;
  ctx.save();
  ctx.beginPath();
  ctx.arc(s.x, s.y, pulse, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(11,143,138,0.12)";
  ctx.fill();
  ctx.strokeStyle = "rgba(11,143,138,0.42)";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(s.x, s.y, 6, 0, Math.PI * 2);
  ctx.fillStyle = "#0b8f8a";
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

function drawAxes(transform) {
  const { geometry } = state.result;
  const origin = transform.toScreen({ x: 0, y: 0 });
  const xEnd = transform.toScreen({ x: geometry.sides.bc * 0.18, y: 0 });
  const yEnd = transform.toScreen({ x: 0, y: geometry.height * 0.18 });
  ctx.save();
  ctx.strokeStyle = "rgba(21,34,37,0.45)";
  ctx.fillStyle = "rgba(21,34,37,0.65)";
  ctx.lineWidth = 1.2;
  drawArrow(origin.x, origin.y, xEnd.x, xEnd.y);
  drawArrow(origin.x, origin.y, yEnd.x, yEnd.y);
  ctx.font = "800 12px system-ui, sans-serif";
  ctx.fillText(t("x"), xEnd.x + 8, xEnd.y + 4);
  ctx.fillText(t("y"), yEnd.x + 6, yEnd.y - 7);
  ctx.restore();
}

function nearestPoint(clientX, clientY) {
  if (!state.result || !state.canvasTransform) return null;
  const rect = el.canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const points = activePrediction().points;
  let best = null;
  let bestDist = Infinity;
  for (const p of points) {
    const s = state.canvasTransform.toScreen(p);
    const d = Math.hypot(s.x - x, s.y - y);
    if (d < bestDist) {
      bestDist = d;
      best = { point: p, screen: s };
    }
  }
  return bestDist < 16 ? best : null;
}

function showTooltip(event) {
  const nearest = nearestPoint(event.clientX, event.clientY);
  if (!nearest) {
    el.tooltip.hidden = true;
    return;
  }
  const p = nearest.point;
  el.tooltip.innerHTML = `
    <strong>${p.point_id} · Re ${p.re}</strong><br>
    x ${format(p.x, 1)} um · y ${format(p.y, 1)} um<br>
    FL-x ${format(p.fl_x_pred, 4)} · FL-y ${format(p.fl_y_pred, 4)}<br>
    Dean-x ${format(p.fd_x_pred, 4)} · Dean-y ${format(p.fd_y_pred, 4)}<br>
    ${t("total")} ${format(p.total_force_magnitude, 4)}
  `;
  const rect = el.canvas.getBoundingClientRect();
  el.tooltip.style.left = `${Math.min(rect.width - 236, nearest.screen.x + 12)}px`;
  el.tooltip.style.top = `${Math.max(8, nearest.screen.y - 18)}px`;
  el.tooltip.hidden = false;
}

function hideTooltip() {
  el.tooltip.hidden = true;
}

function downloadCsv() {
  if (!state.result) return;
  const rows = [
    [
      "point_id",
      "re",
      "x_um",
      "y_um",
      "fl_x_pred",
      "fl_y_pred",
      "fl_magnitude_pred",
      "fd_x_pred",
      "fd_y_pred",
      "fd_magnitude_pred",
      "total_force_x",
      "total_force_y",
      "total_force_magnitude",
      "focus_region_flag",
    ],
  ];
  state.result.sweep.forEach((item) => {
    item.points.forEach((p) => {
      rows.push([
        p.point_id,
        p.re,
        p.x.toFixed(4),
        p.y.toFixed(4),
        p.fl_x_pred.toFixed(8),
        p.fl_y_pred.toFixed(8),
        p.fl_magnitude_pred.toFixed(8),
        p.fd_x_pred.toFixed(8),
        p.fd_y_pred.toFixed(8),
        p.fd_magnitude_pred.toFixed(8),
        p.total_force_x.toFixed(8),
        p.total_force_y.toFixed(8),
        p.total_force_magnitude.toFixed(8),
        p.focus_region_flag,
      ]);
    });
  });
  const csv = rows.map((row) => row.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  triggerDownload(blob, `trifocus_prediction_${Date.now()}.csv`);
}

function downloadPng() {
  if (!state.result) return;
  el.canvas.toBlob((blob) => {
    if (!blob) return;
    triggerDownload(blob, `trifocus_${state.layer}_re${state.selectedRe}.png`);
  });
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

el.form.addEventListener("submit", (event) => {
  event.preventDefault();
  runPrediction();
});

document.querySelectorAll(".layer-tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".layer-tab").forEach((node) => node.classList.remove("active"));
    button.classList.add("active");
    state.layer = button.dataset.layer;
    updateAll();
  });
});

el.languageToggle.addEventListener("click", () => {
  setLanguage(state.lang === "zh" ? "en" : "zh");
});

el.canvas.addEventListener("mousemove", showTooltip);
el.canvas.addEventListener("mouseleave", hideTooltip);
el.downloadCsv.addEventListener("click", downloadCsv);
el.downloadPng.addEventListener("click", downloadPng);
window.addEventListener("resize", () => {
  if (state.result) updateAll();
});

setLanguage("zh");
runPrediction();
