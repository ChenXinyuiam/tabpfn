const RE_VALUES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60];
const POINT_COUNT = 400;
const CURVATURE_RADIUS = {
  min: 300,
  max: 5000,
  reference: 1200,
};

const i18n = {
  zh: {
    productEyebrow: "微流道三角截面设计",
    productName: "TriFocus ForceLab",
    checkingBackend: "检测后端服务",
    backendReady: "后端 API 已连接",
    offlineReady: "离线演示模式",
    stepOne: "01 输入",
    inputTitle: "截面几何与曲率半径",
    radiusLabel: "曲率半径 Rc",
    radiusUnit: "um",
    radiusHelp: "Dean 力修正把 Rc 作为连续设计变量。",
    designCalloutTitle: "设计杠杆",
    designCalloutText: "Rc 越小，二次流和 Dean 力越强；Rc 越大，侧向偏转越弱。",
    predictButton: "生成并预测",
    predicting: "计算中...",
    presetTitle: "演示案例",
    presetBalanced: "平衡型",
    presetTall: "高窄型",
    presetAsym: "偏斜型",
    pipelineEyebrow: "Pipeline",
    pipelineTitle: "实时推理链路",
    pipeGeometry: "几何计算",
    pipePoints: "400 点采样",
    pipeRe: "Re 扫描",
    pipeForce: "曲率修正与力场预测",
    pipeFocus: "聚焦窗口",
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
    arrowDensity: "箭头密度",
    arrowSampled: "抽样",
    arrowAll: "全部点",
    downloadCsv: "下载 CSV",
    downloadPng: "下载 PNG",
    downloadReport: "下载报告",
    engineNoteBackend: "完整演示模式：后端使用 TabPFN 力场，并用 Rc 对 Dean 力做曲率修正。",
    engineNoteOffline: "离线演示模式：浏览器替代模型使用三边、Rc 与 Re 生成结果，未调用 TabPFN。",
    visualEyebrow: "实时截面力场",
    visualTitle: "合力分布与聚焦位置",
    legendLow: "低",
    legendHigh: "高",
    vectorNote: "箭头可切换抽样/全部",
    focusEyebrow: "聚焦判断",
    focusRange: "可聚焦 Re",
    focusPoint: "聚焦位置",
    pointCount: "采样点",
    radiusMetric: "曲率半径 Rc",
    deanGainMetric: "Dean 增益",
    designReadoutEyebrow: "设计解读",
    tightCurve: "急弯",
    gentleCurve: "缓弯",
    regimeTight: "急弯强化",
    regimeNominal: "标称曲率",
    regimeGentle: "缓弯弱化",
    adviceTight: "当前 Rc 会显著放大 Dean 力，适合快速筛选强横向迁移方案；若聚焦点贴近边界，应增大 Rc。",
    adviceNominal: "当前 Rc 接近参考设计，Dean 力和惯性升力处于较均衡状态，适合作为芯片初版参数。",
    adviceGentle: "当前 Rc 较大，Dean 力较弱，适合降低二次流扰动；若聚焦窗口变窄，应减小 Rc 或提高 Re。",
    engineLabel: "引擎",
    generatedRows: "生成行数",
    latencyLabel: "延迟",
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
    invalidRadius: "请输入 300 到 5000 um 之间的曲率半径。",
    backendFailed: "后端不可用，已切换到离线演示。",
    focused: "聚焦良好",
    weak_focus: "弱聚焦",
    not_focused: "不聚焦",
    unknown: "暂无法判断",
    reasonFocused: "低合力区稳定，Re 区间内力场向同一位置收敛。",
    reasonWeak: "存在可见低合力区，但部分 Re 下收敛性仍需真实模型确认。",
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
    checkingBackend: "Checking backend",
    backendReady: "Backend API connected",
    offlineReady: "Offline demo mode",
    stepOne: "01 Input",
    inputTitle: "Geometry and curvature radius",
    radiusLabel: "Curvature radius Rc",
    radiusUnit: "um",
    radiusHelp: "Dean-force correction uses Rc as a continuous design variable.",
    designCalloutTitle: "Design lever",
    designCalloutText: "Smaller Rc strengthens Dean force; larger Rc weakens secondary-flow steering.",
    predictButton: "Generate and predict",
    predicting: "Computing...",
    presetTitle: "Demo presets",
    presetBalanced: "Balanced",
    presetTall: "Tall",
    presetAsym: "Asymmetric",
    pipelineEyebrow: "Pipeline",
    pipelineTitle: "Live inference path",
    pipeGeometry: "Geometry",
    pipePoints: "400 points",
    pipeRe: "Re sweep",
    pipeForce: "Curvature correction + force field",
    pipeFocus: "Focus window",
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
    arrowDensity: "Arrow density",
    arrowSampled: "Sampled",
    arrowAll: "All points",
    downloadCsv: "Download CSV",
    downloadPng: "Download PNG",
    downloadReport: "Download report",
    engineNoteBackend: "Full demo mode: the backend uses TabPFN force fields and applies an Rc correction to Dean force.",
    engineNoteOffline: "Offline demo mode: the browser surrogate uses side lengths, Rc, and Re. TabPFN is not called.",
    visualEyebrow: "Live cross-section force field",
    visualTitle: "Total force and focus location",
    legendLow: "Low",
    legendHigh: "High",
    vectorNote: "Arrows can be sampled or full",
    focusEyebrow: "Focus assessment",
    focusRange: "Focus Re",
    focusPoint: "Focus point",
    pointCount: "Points",
    radiusMetric: "Curvature R",
    deanGainMetric: "Dean gain",
    designReadoutEyebrow: "Design interpretation",
    tightCurve: "Tight curve",
    gentleCurve: "Gentle curve",
    regimeTight: "Tight-curve amplification",
    regimeNominal: "Nominal curvature",
    regimeGentle: "Gentle-curve attenuation",
    adviceTight: "This Rc strongly amplifies Dean force. It is useful for screening aggressive lateral migration; increase Rc if the focus point moves too close to a wall.",
    adviceNominal: "This Rc is near the reference design. Dean force and inertial lift are comparatively balanced, making it a good first-pass chip setting.",
    adviceGentle: "This Rc weakens Dean force and reduces secondary-flow steering. Decrease Rc or increase Re if the focus window becomes too narrow.",
    engineLabel: "Engine",
    generatedRows: "Generated rows",
    latencyLabel: "Latency",
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
    invalidRadius: "Enter a curvature radius between 300 and 5000 um.",
    backendFailed: "Backend unavailable; switched to offline demo.",
    focused: "Focused",
    weak_focus: "Weak focus",
    not_focused: "Not focused",
    unknown: "Unknown",
    reasonFocused: "The low-force region is stable and the field converges to one location across the Re window.",
    reasonWeak: "A low-force region is visible, but convergence still needs real-model confirmation for part of the Re sweep.",
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
  arrowMode: "sampled",
  selectedRe: 36,
  result: null,
  canvasTransform: null,
  backendAvailable: false,
  engine: "offline",
};

const el = {
  form: document.getElementById("triangleForm"),
  visualPanel: document.querySelector(".visual-panel"),
  sideAB: document.getElementById("sideAB"),
  sideAC: document.getElementById("sideAC"),
  sideBC: document.getElementById("sideBC"),
  curvatureRadius: document.getElementById("curvatureRadius"),
  curvatureRadiusSlider: document.getElementById("curvatureRadiusSlider"),
  formError: document.getElementById("formError"),
  predictButton: document.getElementById("predictButton"),
  canvas: document.getElementById("forceCanvas"),
  tooltip: document.getElementById("tooltip"),
  reStrip: document.getElementById("reStrip"),
  currentReLabel: document.getElementById("currentReLabel"),
  languageToggle: document.getElementById("languageToggle"),
  modelStatus: document.getElementById("modelStatus"),
  modelStatusText: document.getElementById("modelStatusText"),
  engineNote: document.getElementById("engineNote"),
  focusStatus: document.getElementById("focusStatus"),
  confidenceFill: document.getElementById("confidenceFill"),
  focusReason: document.getElementById("focusReason"),
  focusRange: document.getElementById("focusRange"),
  focusPoint: document.getElementById("focusPoint"),
  pointCount: document.getElementById("pointCount"),
  radiusValue: document.getElementById("radiusValue"),
  deanGainValue: document.getElementById("deanGainValue"),
  designRegime: document.getElementById("designRegime"),
  designAdvice: document.getElementById("designAdvice"),
  sensitivityMarker: document.getElementById("sensitivityMarker"),
  engineValue: document.getElementById("engineValue"),
  generatedRows: document.getElementById("generatedRows"),
  latencyValue: document.getElementById("latencyValue"),
  statsBody: document.getElementById("statsBody"),
  areaValue: document.getElementById("areaValue"),
  perimeterValue: document.getElementById("perimeterValue"),
  heightValue: document.getElementById("heightValue"),
  anglesValue: document.getElementById("anglesValue"),
  downloadCsv: document.getElementById("downloadCsv"),
  downloadPng: document.getElementById("downloadPng"),
  downloadReport: document.getElementById("downloadReport"),
  visualTitle: document.getElementById("visualTitle"),
};

const ctx = el.canvas.getContext("2d");

function t(key) {
  return i18n[state.lang][key] || i18n.en[key] || key;
}

function setLanguage(lang) {
  state.lang = lang;
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  el.languageToggle.textContent = lang === "zh" ? "EN" : "中文";
  updateEngineStatus();
  if (state.result) updateAll();
}

function setPipelineStatus(activeIndex = -1, doneAll = false) {
  const items = [...document.querySelectorAll("#pipelineList li")];
  items.forEach((item, index) => {
    item.classList.toggle("active", index === activeIndex && !doneAll);
    item.classList.toggle("done", doneAll || index < activeIndex);
  });
}

function updateEngineStatus() {
  el.modelStatus.classList.remove("offline", "error");
  if (state.engine === "backend") {
    el.modelStatusText.textContent = t("backendReady");
    el.engineNote.textContent = t("engineNoteBackend");
  } else {
    el.modelStatus.classList.add("offline");
    el.modelStatusText.textContent = t("offlineReady");
    el.engineNote.textContent = t("engineNoteOffline");
  }
}

function readSides() {
  return {
    ab: Number(el.sideAB.value),
    ac: Number(el.sideAC.value),
    bc: Number(el.sideBC.value),
    curvatureRadius: Number(el.curvatureRadius.value),
  };
}

function validateSides({ ab, ac, bc, curvatureRadius }) {
  if (![ab, ac, bc].every((value) => Number.isFinite(value) && value > 0)) return t("invalidNumber");
  if (ab + ac <= bc || ab + bc <= ac || ac + bc <= ab) return t("invalidTriangle");
  if (!Number.isFinite(curvatureRadius) || curvatureRadius < CURVATURE_RADIUS.min || curvatureRadius > CURVATURE_RADIUS.max) return t("invalidRadius");
  return "";
}

async function checkBackend() {
  if (!["http:", "https:"].includes(window.location.protocol)) {
    state.backendAvailable = false;
    state.engine = "offline";
    updateEngineStatus();
    return;
  }
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("health failed");
    state.backendAvailable = true;
    state.engine = "backend";
  } catch {
    state.backendAvailable = false;
    state.engine = "offline";
  }
  updateEngineStatus();
}

async function runPrediction() {
  const sides = readSides();
  const error = validateSides(sides);
  el.formError.textContent = error;
  if (error) return;

  el.predictButton.disabled = true;
  el.visualPanel.classList.add("is-computing");
  el.predictButton.querySelector("span").textContent = t("predicting");
  setPipelineStatus(0);

  try {
    let result = null;
    if (state.backendAvailable) {
      try {
        result = await predictWithBackend(sides);
        state.engine = "backend";
      } catch {
        state.backendAvailable = false;
        state.engine = "offline";
        el.formError.textContent = t("backendFailed");
      }
    }
    if (!result) {
      result = predictLocally(sides);
      state.engine = "offline";
    }

    state.result = normalizeResult(result);
    state.selectedRe = state.result.bestRe;
    updateEngineStatus();
    setPipelineStatus(-1, true);
    updateAll();
  } finally {
    el.predictButton.disabled = false;
    el.visualPanel.classList.remove("is-computing");
    el.predictButton.querySelector("span").textContent = t("predictButton");
  }
}

async function predictWithBackend(sides) {
  setPipelineStatus(2);
  const response = await fetch("/api/predict-triangle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      side_ab_um: sides.ab,
      side_ac_um: sides.ac,
      side_bc_um: sides.bc,
      curvature_radius_um: sides.curvatureRadius,
    }),
  });
  if (!response.ok) throw new Error("prediction failed");
  setPipelineStatus(4);
  return response.json();
}

function normalizeResult(result) {
  const inputRadius = result.input?.curvature_radius_um ?? result.input?.curvatureRadius ?? readSides().curvatureRadius;
  const defaults = {
    engine: "browser-offline-surrogate",
    model_version: "offline-demo-v1",
    feature_version: "triangle-force-features-v2-radius",
    curvature_radius_um: inputRadius,
    fd_model_route: `dean_rc_${inputRadius}um`,
    point_count_per_re: POINT_COUNT,
    generated_rows: POINT_COUNT * RE_VALUES.length,
    re_values: RE_VALUES,
    inference_latency_ms: null,
    data_source: "generated_from_geometry_and_curvature_radius",
    uses_tabpfn: false,
  };
  return {
    ...result,
    input: {
      ...result.input,
      curvature_radius_um: inputRadius,
    },
    prediction_meta: {
      ...defaults,
      ...(result.prediction_meta || {}),
      curvature_radius_um: result.prediction_meta?.curvature_radius_um ?? inputRadius,
      fd_model_route: result.prediction_meta?.fd_model_route ?? `dean_rc_${inputRadius}um`,
    },
  };
}

function angleFromSides(a, b, c) {
  const cos = clamp((b * b + c * c - a * a) / (2 * b * c), -1, 1);
  return (Math.acos(cos) * 180) / Math.PI;
}

function buildGeometry({ ab, ac, bc }) {
  const xA = (ab * ab + bc * bc - ac * ac) / (2 * bc);
  const yA = Math.sqrt(Math.max(0, ab * ab - xA * xA));
  const vertices = { A: { x: xA, y: yA }, B: { x: 0, y: 0 }, C: { x: bc, y: 0 } };
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
  let factor = 1;
  let result = 0;
  let i = index;
  while (i > 0) {
    factor /= base;
    result += factor * (i % base);
    i = Math.floor(i / base);
  }
  return result;
}

function pointLineDistance(point, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return Math.abs(dy * point.x - dx * point.y + b.x * a.y - b.y * a.x) / Math.hypot(dx, dy);
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
    const point = { point_id: `p${String(i).padStart(4, "0")}`, x, y };
    point.distBase = pointLineDistance(point, B, C);
    point.distLeft = pointLineDistance(point, A, B);
    point.distRight = pointLineDistance(point, A, C);
    point.distCentroid = Math.hypot(point.x - geometry.centroid.x, point.y - geometry.centroid.y);
    point.xNorm = safeDiv(point.x, geometry.sides.bc);
    point.yNorm = safeDiv(point.y, geometry.height);
    points.push(point);
  }
  return points;
}

function curvatureFactor(curvatureRadius) {
  return clamp(CURVATURE_RADIUS.reference / curvatureRadius, 0.35, 2.35);
}

function focusModel(geometry, re, curvatureRadius) {
  const { ab, ac, bc } = geometry.sides;
  const aspect = bc / geometry.height;
  const symmetry = 1 - Math.min(Math.abs(ab - ac) / Math.max(ab, ac), 0.95);
  const aspectScore = Math.exp(-(((aspect - 1.45) / 0.68) ** 2));
  const geometryScore = clamp(0.22 + 0.52 * symmetry + 0.32 * aspectScore, 0.16, 0.98);
  const deanGain = curvatureFactor(curvatureRadius);
  const optRe = clamp(31 + (aspect - 1.25) * 12 + (1 - symmetry) * 10 + (deanGain - 1) * 7.5, 18, 56);
  const reScore = Math.exp(-(((re - optRe) / 16) ** 2));
  const radiusScore = clamp(0.9 + 0.1 * deanGain, 0.78, 1.14);
  const score = clamp(geometryScore * radiusScore * (0.18 + 0.88 * reScore), 0.03, 0.98);
  const skew = (ab - ac) / (ab + ac);
  const fx = clamp(
    geometry.centroid.x + (re - 36) * 0.0035 * bc - skew * 0.09 * bc + (deanGain - 1) * 0.035 * bc,
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

function predictForRe(geometry, basePoints, re, curvatureRadius) {
  const focus = focusModel(geometry, re, curvatureRadius);
  const scale = Math.max(geometry.sides.bc, geometry.height);
  const reFactor = re / 60;
  const skew = (geometry.sides.ab - geometry.sides.ac) / (geometry.sides.ab + geometry.sides.ac);
  const points = basePoints.map((point) => {
    const dx = focus.x - point.x;
    const dy = focus.y - point.y;
    const nx = dx / scale;
    const ny = dy / scale;
    const dist = Math.hypot(nx, ny);
    const wallBias = safeDiv(point.distRight - point.distLeft, geometry.sides.bc);
    const verticalBias = point.yNorm - 0.5;
    const convergence = 0.038 * focus.score * dist;
    const swirl = 0.016 * (1 - focus.score) * (0.45 + reFactor);
    const totalBaseX = convergence * nx + swirl * -ny + 0.0025 * wallBias;
    const totalBaseY = convergence * ny + swirl * nx - 0.0018 * verticalBias;
    const flMix = 0.57 + 0.07 * Math.sin(re / 12);
    const flX = totalBaseX * flMix + 0.0036 * wallBias + 0.001 * skew;
    const flY = totalBaseY * flMix - 0.0028 * verticalBias + 0.0012 * Math.sin(Math.PI * point.xNorm);
    const deanRadiusFactor = curvatureFactor(curvatureRadius);
    const fdX = (totalBaseX - flX) * deanRadiusFactor;
    const fdY = (totalBaseY - flY) * deanRadiusFactor;
    const totalX = flX + fdX;
    const totalY = flY + fdY;
    const focusDistance = Math.hypot(point.x - focus.x, point.y - focus.y);
    return {
      ...point,
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
      focus_distance_um: focusDistance,
      focus_region_flag: focusDistance < scale * 0.08 ? 1 : 0,
    };
  });
  return { re, focus, points };
}

function predictLocally(sides) {
  setPipelineStatus(1);
  const geometry = buildGeometry(sides);
  const basePoints = generatePoints(geometry);
  setPipelineStatus(2);
  const sweep = RE_VALUES.map((re) => predictForRe(geometry, basePoints, re, sides.curvatureRadius));
  const best = sweep.reduce((a, b) => (b.focus.score > a.focus.score ? b : a));
  const focusedRes = sweep.filter((item) => item.focus.score >= 0.52);
  const status = best.focus.score >= 0.72 ? "focused" : best.focus.score >= 0.48 ? "weak_focus" : "not_focused";
  setPipelineStatus(4);
  return {
    status: "ok",
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
      method: "force_rule_demo",
    },
    prediction_meta: {
      engine: "browser-offline-surrogate",
      model_version: "offline-demo-v1",
      feature_version: "triangle-force-features-v2-radius",
      curvature_radius_um: sides.curvatureRadius,
      fd_model_route: `dean_rc_${sides.curvatureRadius}um`,
      point_count_per_re: POINT_COUNT,
      generated_rows: POINT_COUNT * RE_VALUES.length,
      re_values: RE_VALUES,
      inference_latency_ms: null,
      data_source: "generated_from_geometry_and_curvature_radius",
      uses_tabpfn: false,
    },
  };
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

  el.focusStatus.textContent = t(focus.status || "unknown");
  el.confidenceFill.style.width = `${Math.round((focus.confidence || 0) * 100)}%`;
  el.focusReason.textContent =
    focus.status === "focused" ? t("reasonFocused") : focus.status === "weak_focus" ? t("reasonWeak") : t("reasonNone");
  el.focusRange.textContent = focus.reRange ? `${focus.reRange[0]} - ${focus.reRange[1]}` : t("noRange");
  el.focusPoint.textContent = `${format(focus.x, 1)} um, ${format(focus.y, 1)} um`;
  el.pointCount.textContent = String(result.prediction_meta.point_count_per_re || POINT_COUNT);
  const rc = Number(result.prediction_meta.curvature_radius_um ?? result.input.curvature_radius_um ?? result.input.curvatureRadius ?? CURVATURE_RADIUS.reference);
  const deanGain = Number(result.prediction_meta.dean_curvature_factor ?? curvatureFactor(rc));
  el.radiusValue.textContent = `Rc = ${format(rc, 0, locale)} um`;
  el.deanGainValue.textContent = `${format(deanGain, 2)}x`;
  updateDesignInterpretation(rc, deanGain);
  el.engineValue.textContent = result.prediction_meta.engine || state.engine;
  el.generatedRows.textContent = format(result.prediction_meta.generated_rows || active.points.length * result.sweep.length, 0, locale);
  el.latencyValue.textContent =
    result.prediction_meta.inference_latency_ms == null ? (state.engine === "backend" ? "< 1 ms" : "local") : `${format(result.prediction_meta.inference_latency_ms, 1)} ms`;

  el.areaValue.textContent = `${format(g.area, 0, locale)} um²`;
  el.perimeterValue.textContent = `${format(g.perimeter, 0, locale)} um`;
  el.heightValue.textContent = `${format(g.height, 0, locale)} um`;
  el.anglesValue.textContent = `${format(g.angles.A, 1)}° / ${format(g.angles.B, 1)}° / ${format(g.angles.C, 1)}°`;

  const stats = [
    [t("fl"), active.points.map((point) => point.fl_magnitude_pred)],
    [t("fd"), active.points.map((point) => point.fd_magnitude_pred)],
    [t("total"), active.points.map((point) => point.total_force_magnitude)],
    [t("focusScore"), active.points.map((point) => point.focus_score)],
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

function updateDesignInterpretation(rc, deanGain) {
  const regime =
    deanGain >= 1.25 ? { key: "regimeTight", advice: "adviceTight" } :
    deanGain <= 0.75 ? { key: "regimeGentle", advice: "adviceGentle" } :
    { key: "regimeNominal", advice: "adviceNominal" };
  el.designRegime.textContent = t(regime.key);
  el.designAdvice.textContent = t(regime.advice);
  const normalized = clamp((rc - CURVATURE_RADIUS.min) / (CURVATURE_RADIUS.max - CURVATURE_RADIUS.min), 0, 1);
  el.sensitivityMarker.style.left = `${normalized * 100}%`;
  el.sensitivityMarker.textContent = `${format(rc, 0)} um`;
}

function summarize(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
  return { min, max, avg };
}

function format(value, digits = 2, locale = "en-US") {
  return Number(value || 0).toLocaleString(locale, {
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
  const scale = Math.min((w - pad * 2) / Math.max(1, maxX - minX), (h - pad * 2) / Math.max(1, maxY - minY));
  const usedW = (maxX - minX) * scale;
  const usedH = (maxY - minY) * scale;
  const offsetX = (w - usedW) / 2 - minX * scale;
  const upwardBias = h > 520 ? h * 0.13 : h * 0.08;
  const offsetY = (h - usedH) / 2 + maxY * scale - upwardBias;
  const transform = {
    toScreen: (point) => ({ x: offsetX + point.x * scale, y: offsetY - point.y * scale }),
    toData: (point) => ({ x: (point.x - offsetX) / scale, y: (offsetY - point.y) / scale }),
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
  if (state.layer === "total" || state.layer === "focus") drawArrows(active.points, transform);
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
    total: { value: (point) => point.total_force_magnitude, signed: false },
    flx: { value: (point) => point.fl_x_pred, signed: true },
    fly: { value: (point) => point.fl_y_pred, signed: true },
    fdx: { value: (point) => point.fd_x_pred, signed: true },
    fdy: { value: (point) => point.fd_y_pred, signed: true },
    focus: { value: (point) => 1 / (1 + point.focus_distance_um), signed: false },
  };
  const info = map[state.layer] || map.total;
  const values = points.map(info.value);
  return { ...info, min: Math.min(...values), max: Math.max(...values) };
}

function drawHeatPoints(points, valueInfo, transform) {
  points.forEach((point) => {
    const screen = transform.toScreen(point);
    const color = colorFor(valueInfo.value(point), valueInfo.min, valueInfo.max, valueInfo.signed);
    const radius = state.layer === "focus" && point.focus_region_flag ? 5.6 : 3.25;
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, radius + 2.6, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = state.layer === "focus" && !point.focus_region_flag ? 0.1 : 0.16;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = state.layer === "focus" && !point.focus_region_flag ? 0.5 : 0.92;
    ctx.fill();
  });
  ctx.globalAlpha = 1;
}

function colorFor(value, min, max, signed) {
  if (signed) {
    const span = Math.max(Math.abs(min), Math.abs(max), 1e-8);
    const normalized = clamp(value / span, -1, 1);
    return normalized < 0 ? blend("#277da1", "#f7fbfb", normalized + 1) : blend("#f7fbfb", "#d94141", normalized);
  }
  const n = clamp((value - min) / Math.max(max - min, 1e-8), 0, 1);
  if (n < 0.25) return blend("#277da1", "#43aa8b", n / 0.25);
  if (n < 0.5) return blend("#43aa8b", "#f9c74f", (n - 0.25) / 0.25);
  if (n < 0.75) return blend("#f9c74f", "#f3722c", (n - 0.5) / 0.25);
  return blend("#f3722c", "#d94141", (n - 0.75) / 0.25);
}

function blend(a, b, tValue) {
  const v = clamp(tValue, 0, 1);
  const ca = hexToRgb(a);
  const cb = hexToRgb(b);
  return `rgb(${Math.round(ca.r + (cb.r - ca.r) * v)}, ${Math.round(ca.g + (cb.g - ca.g) * v)}, ${Math.round(ca.b + (cb.b - ca.b) * v)})`;
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  return { r: parseInt(clean.slice(0, 2), 16), g: parseInt(clean.slice(2, 4), 16), b: parseInt(clean.slice(4, 6), 16) };
}

function drawArrows(points, transform) {
  ctx.save();
  const isAll = state.arrowMode === "all";
  ctx.strokeStyle = isAll ? "rgba(21,34,37,0.28)" : "rgba(21,34,37,0.5)";
  ctx.fillStyle = isAll ? "rgba(21,34,37,0.28)" : "rgba(21,34,37,0.5)";
  ctx.lineWidth = isAll ? 0.75 : 1.2;
  const every = isAll ? 1 : Math.max(1, Math.ceil(points.length / 34));
  points.forEach((point, index) => {
    if (index % every !== 0) return;
    const screen = transform.toScreen(point);
    const mag = Math.hypot(point.total_force_x, point.total_force_y);
    if (mag < 1e-8) return;
    const len = isAll ? clamp(mag * 900, 4, 10) : clamp(mag * 1250, 8, 20);
    drawArrow(screen.x, screen.y, screen.x + (point.total_force_x / mag) * len, screen.y - (point.total_force_y / mag) * len);
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
  const screen = transform.toScreen(focus);
  const pulse = 18 + focus.score * 20;
  ctx.save();
  ctx.beginPath();
  ctx.arc(screen.x, screen.y, pulse, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(11,143,138,0.12)";
  ctx.fill();
  ctx.strokeStyle = "rgba(11,143,138,0.42)";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(screen.x, screen.y, 6, 0, Math.PI * 2);
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
  let best = null;
  let bestDist = Infinity;
  activePrediction().points.forEach((point) => {
    const screen = state.canvasTransform.toScreen(point);
    const dist = Math.hypot(screen.x - x, screen.y - y);
    if (dist < bestDist) {
      bestDist = dist;
      best = { point, screen };
    }
  });
  return bestDist < 16 ? best : null;
}

function showTooltip(event) {
  const nearest = nearestPoint(event.clientX, event.clientY);
  if (!nearest) {
    el.tooltip.hidden = true;
    return;
  }
  const point = nearest.point;
  el.tooltip.innerHTML = `
    <strong>${point.point_id} · Re ${point.re}</strong><br>
    x ${format(point.x, 1)} um · y ${format(point.y, 1)} um<br>
    FL-x ${format(point.fl_x_pred, 4)} · FL-y ${format(point.fl_y_pred, 4)}<br>
    Dean-x ${format(point.fd_x_pred, 4)} · Dean-y ${format(point.fd_y_pred, 4)}<br>
    ${t("total")} ${format(point.total_force_magnitude, 4)}
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
      "curvature_radius_um",
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
    item.points.forEach((point) => {
      rows.push([
        point.point_id,
        state.result.prediction_meta.curvature_radius_um,
        point.re,
        point.x.toFixed(4),
        point.y.toFixed(4),
        point.fl_x_pred.toFixed(8),
        point.fl_y_pred.toFixed(8),
        point.fl_magnitude_pred.toFixed(8),
        point.fd_x_pred.toFixed(8),
        point.fd_y_pred.toFixed(8),
        point.fd_magnitude_pred.toFixed(8),
        point.total_force_x.toFixed(8),
        point.total_force_y.toFixed(8),
        point.total_force_magnitude.toFixed(8),
        point.focus_region_flag,
      ]);
    });
  });
  triggerDownload(new Blob([rows.map((row) => row.join(",")).join("\n")], { type: "text/csv;charset=utf-8" }), `trifocus_prediction_${Date.now()}.csv`);
}

function downloadPng() {
  if (!state.result) return;
  el.canvas.toBlob((blob) => {
    if (blob) triggerDownload(blob, `trifocus_${state.layer}_re${state.selectedRe}.png`);
  });
}

function downloadReport() {
  if (!state.result) return;
  const result = state.result;
  const g = result.geometry;
  const focus = result.focus;
  const meta = result.prediction_meta;
  const report = [
    "# TriFocus ForceLab Demo Report",
    "",
    "## Input",
    "",
    `- AB: ${result.input.side_ab_um ?? result.input.ab} um`,
    `- AC: ${result.input.side_ac_um ?? result.input.ac} um`,
    `- BC: ${result.input.side_bc_um ?? result.input.bc} um`,
    `- Curvature radius Rc: ${meta.curvature_radius_um ?? result.input.curvature_radius_um ?? result.input.curvatureRadius} um`,
    "",
    "## Geometry",
    "",
    `- Area: ${format(g.area, 2)} um2`,
    `- Perimeter: ${format(g.perimeter, 2)} um`,
    `- Height: ${format(g.height, 2)} um`,
    `- Angles: ${format(g.angles.A, 2)} / ${format(g.angles.B, 2)} / ${format(g.angles.C, 2)} deg`,
    "",
    "## Prediction",
    "",
    `- Engine: ${meta.engine}`,
    `- Uses TabPFN: ${meta.uses_tabpfn ? "yes" : "no"}`,
    `- Dean force correction: ${meta.fd_model_route}`,
    `- Points per Re: ${meta.point_count_per_re}`,
    `- Re values: ${(meta.re_values || RE_VALUES).join(", ")}`,
    `- Generated rows: ${meta.generated_rows || result.sweep.length * activePrediction().points.length}`,
    `- Latency: ${meta.inference_latency_ms == null ? "local/offline" : `${meta.inference_latency_ms} ms`}`,
    "",
    "## Focus",
    "",
    `- Status: ${focus.status}`,
    `- Confidence: ${format(focus.confidence, 4)}`,
    `- Focus point: ${format(focus.x, 2)} um, ${format(focus.y, 2)} um`,
    `- Focus Re range: ${focus.reRange ? `${focus.reRange[0]} - ${focus.reRange[1]}` : "none"}`,
    "",
    "## Note",
    "",
    "This demo backend is API-shaped and deterministic. Replace the force prediction function with TabPFN inference for production-grade predictions.",
    "",
  ].join("\n");
  triggerDownload(new Blob([report], { type: "text/markdown;charset=utf-8" }), `trifocus_report_${Date.now()}.md`);
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

const presets = {
  balanced: { ab: 690, ac: 520, bc: 750, curvatureRadius: 1200 },
  tall: { ab: 720, ac: 710, bc: 520, curvatureRadius: 2200 },
  asym: { ab: 830, ac: 430, bc: 920, curvatureRadius: 650 },
};

function applyPreset(name) {
  const preset = presets[name];
  if (!preset) return;
  el.sideAB.value = preset.ab;
  el.sideAC.value = preset.ac;
  el.sideBC.value = preset.bc;
  setCurvatureRadius(preset.curvatureRadius);
  runPrediction();
}

function setCurvatureRadius(value) {
  const next = clamp(Number(value), CURVATURE_RADIUS.min, CURVATURE_RADIUS.max);
  el.curvatureRadius.value = String(next);
  el.curvatureRadiusSlider.value = String(next);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function safeDiv(a, b) {
  return b === 0 ? 0 : a / b;
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
    if (state.result) updateAll();
  });
});

document.querySelectorAll(".arrow-mode").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".arrow-mode").forEach((node) => node.classList.remove("active"));
    button.classList.add("active");
    state.arrowMode = button.dataset.arrowMode;
    if (state.result) renderCanvas();
  });
});

el.languageToggle.addEventListener("click", () => setLanguage(state.lang === "zh" ? "en" : "zh"));
el.curvatureRadius.addEventListener("input", () => {
  setCurvatureRadius(el.curvatureRadius.value);
});
el.curvatureRadiusSlider.addEventListener("input", () => {
  setCurvatureRadius(el.curvatureRadiusSlider.value);
});
el.canvas.addEventListener("mousemove", showTooltip);
el.canvas.addEventListener("mouseleave", hideTooltip);
el.downloadCsv.addEventListener("click", downloadCsv);
el.downloadPng.addEventListener("click", downloadPng);
el.downloadReport.addEventListener("click", downloadReport);
document.querySelectorAll(".preset-button").forEach((button) => {
  button.addEventListener("click", () => applyPreset(button.dataset.preset));
});
window.addEventListener("resize", () => {
  if (state.result) updateAll();
});

setLanguage("zh");
checkBackend().then(runPrediction);
