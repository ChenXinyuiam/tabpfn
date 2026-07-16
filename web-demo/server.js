"use strict";

const fs = require("fs");
const http = require("http");
const path = require("path");
const { performance } = require("perf_hooks");

const RE_VALUES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60];
const POINT_COUNT = 200;
const RADIUS_GROUPS = [10, 20, 30];
const MODEL_VERSION = "demo-node-backend-v2-radius";
const FEATURE_VERSION = "triangle-force-features-v2-radius";
const ROOT = __dirname;
const HOST = process.env.HOST || "0.0.0.0";
const PORT = Number(process.env.PORT || 8000);

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

function safeDiv(a, b) {
  return b === 0 ? 0 : a / b;
}

function angleFromSides(a, b, c) {
  const cosValue = clamp((b * b + c * c - a * a) / (2 * b * c), -1, 1);
  return (Math.acos(cosValue) * 180) / Math.PI;
}

function validateSides(ab, ac, bc) {
  if (![ab, ac, bc].every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error("side lengths must be positive numbers");
  }
  if (ab + ac <= bc || ab + bc <= ac || ac + bc <= ab) {
    throw new Error("side lengths cannot form a triangle");
  }
}

function validateRadiusGroup(value) {
  const radiusGroup = Number(value);
  if (!Number.isInteger(radiusGroup) || !RADIUS_GROUPS.includes(radiusGroup)) {
    throw new Error("radius_group must be one of 10, 20, or 30");
  }
  return radiusGroup;
}

function buildGeometry(ab, ac, bc) {
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
    point.distCentroid = Math.hypot(x - geometry.centroid.x, y - geometry.centroid.y);
    point.xNorm = safeDiv(x, geometry.sides.bc);
    point.yNorm = safeDiv(y, geometry.height);
    points.push(point);
  }
  return points;
}

function focusModel(geometry, reValue, radiusGroup) {
  const { ab, ac, bc } = geometry.sides;
  const aspect = safeDiv(bc, geometry.height);
  const symmetry = 1 - Math.min(Math.abs(ab - ac) / Math.max(ab, ac), 0.95);
  const aspectScore = Math.exp(-(((aspect - 1.45) / 0.68) ** 2));
  const geometryScore = clamp(0.22 + 0.52 * symmetry + 0.32 * aspectScore, 0.16, 0.98);
  const curvatureFactor = 20 / radiusGroup;
  const optRe = clamp(31 + (aspect - 1.25) * 12 + (1 - symmetry) * 10 + (radiusGroup - 20) * 0.42, 18, 56);
  const reScore = Math.exp(-(((reValue - optRe) / 16) ** 2));
  const radiusScore = clamp(0.88 + 0.12 * curvatureFactor, 0.78, 1.12);
  const score = clamp(geometryScore * radiusScore * (0.18 + 0.88 * reScore), 0.03, 0.98);
  const skew = (ab - ac) / (ab + ac);
  const fx = clamp(
    geometry.centroid.x + (reValue - 36) * 0.0035 * bc - skew * 0.09 * bc + (20 - radiusGroup) * 0.0015 * bc,
    Math.min(geometry.vertices.A.x, 0, bc) + bc * 0.12,
    Math.max(geometry.vertices.A.x, 0, bc) - bc * 0.1
  );
  const fy = clamp(
    geometry.centroid.y + Math.sin(reValue / 9) * 0.045 * geometry.height + (1 - aspectScore) * 0.04 * geometry.height,
    geometry.height * 0.16,
    geometry.height * 0.78
  );
  return { score, x: fx, y: fy, optRe, geometryScore };
}

function predictForRe(geometry, basePoints, reValue, radiusGroup) {
  const focus = focusModel(geometry, reValue, radiusGroup);
  const scale = Math.max(geometry.sides.bc, geometry.height);
  const reFactor = reValue / 60;
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
    const flMix = 0.57 + 0.07 * Math.sin(reValue / 12);
    const flX = totalBaseX * flMix + 0.0036 * wallBias + 0.001 * skew;
    const flY = totalBaseY * flMix - 0.0028 * verticalBias + 0.0012 * Math.sin(Math.PI * point.xNorm);
    const deanRadiusFactor = 20 / radiusGroup;
    const fdX = (totalBaseX - flX) * deanRadiusFactor;
    const fdY = (totalBaseY - flY) * deanRadiusFactor;
    const totalX = flX + fdX;
    const totalY = flY + fdY;
    const focusDistance = Math.hypot(point.x - focus.x, point.y - focus.y);
    return {
      ...point,
      re: reValue,
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
  return { re: reValue, focus, points };
}

function predictTriangle(payload) {
  const startedAt = performance.now();
  const ab = Number(payload.side_ab_um ?? payload.sideAB ?? 0);
  const ac = Number(payload.side_ac_um ?? payload.sideAC ?? 0);
  const bc = Number(payload.side_bc_um ?? payload.sideBC ?? 0);
  const radiusGroup = validateRadiusGroup(payload.radius_group ?? payload.radiusGroup);
  validateSides(ab, ac, bc);

  const geometry = buildGeometry(ab, ac, bc);
  const basePoints = generatePoints(geometry);
  const sweep = RE_VALUES.map((reValue) => predictForRe(geometry, basePoints, reValue, radiusGroup));
  const best = sweep.reduce((currentBest, item) => (item.focus.score > currentBest.focus.score ? item : currentBest));
  const focusedRe = sweep.filter((item) => item.focus.score >= 0.52).map((item) => item.re);
  const score = best.focus.score;
  const status = score >= 0.72 ? "focused" : score >= 0.48 ? "weak_focus" : "not_focused";

  return {
    status: "ok",
    input: { side_ab_um: ab, side_ac_um: ac, side_bc_um: bc, radius_group: radiusGroup },
    geometry,
    basePoints,
    sweep,
    bestRe: best.re,
    focus: {
      status,
      confidence: score,
      x: best.focus.x,
      y: best.focus.y,
      reRange: focusedRe.length ? [focusedRe[0], focusedRe[focusedRe.length - 1]] : null,
      method: "force_rule_demo",
    },
    prediction_meta: {
      engine: "node-demo-backend",
      model_version: MODEL_VERSION,
      feature_version: FEATURE_VERSION,
      radius_group: radiusGroup,
      fd_model_route: `demo_fd_radius_${radiusGroup}`,
      point_count_per_re: POINT_COUNT,
      generated_rows: sweep.length * basePoints.length,
      re_values: RE_VALUES,
      inference_latency_ms: Math.round((performance.now() - startedAt) * 100) / 100,
      data_source: "generated_from_geometry_and_radius_group",
      fl_output_mode: "directional_projection_xy",
      fd_output_mode: "force_vector",
      uses_tabpfn: false,
      notes: [
        "Deterministic Node demo backend for BT panel deployment.",
        "Replace this module with TabPFN inference for production predictions.",
      ],
    },
  };
}

function jsonResponse(res, statusCode, payload) {
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": body.length,
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 1024 * 1024) {
        reject(new Error("request body too large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch {
        reject(new Error("invalid JSON body"));
      }
    });
    req.on("error", reject);
  });
}

function schemaPayload() {
  return {
    service: "TriFocus ForceLab demo API",
    version: MODEL_VERSION,
    endpoints: {
      "GET /api/health": {
        purpose: "Check whether the demo backend is running.",
        returns: ["status", "engine", "model_version", "point_count_per_re", "re_values"],
      },
      "POST /api/predict-triangle": {
        purpose: "Generate force-field predictions from triangle geometry and curvature radius group.",
        input: {
          side_ab_um: "number, triangle side AB in micrometers",
          side_ac_um: "number, triangle side AC in micrometers",
          side_bc_um: "number, triangle side BC in micrometers",
          radius_group: "required integer enum: 10, 20, or 30",
        },
        output: ["geometry", "basePoints", "sweep", "bestRe", "focus", "prediction_meta"],
      },
    },
    prediction_contract: {
      point_count_per_re: POINT_COUNT,
      re_values: RE_VALUES,
      supported_radius_groups: RADIUS_GROUPS,
      fl_output_mode: "directional_projection_xy",
      fd_output_mode: "force_vector",
      generated_rows_per_request: POINT_COUNT * RE_VALUES.length,
      uses_tabpfn: false,
      production_note: "Replace the deterministic predictor with TabPFN inference behind the same endpoint.",
    },
  };
}

function serveStatic(req, res, pathname) {
  const requested = pathname === "/" ? "/index.html" : pathname;
  const normalized = path.normalize(decodeURIComponent(requested)).replace(/^(\.\.[/\\])+/, "");
  const filePath = path.join(ROOT, normalized);
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, (error, data) => {
    if (error) {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Not found");
      return;
    }
    res.writeHead(200, {
      "Content-Type": MIME_TYPES[path.extname(filePath).toLowerCase()] || "application/octet-stream",
      "Cache-Control": "no-store",
    });
    res.end(data);
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const pathname = url.pathname;

  try {
    if (req.method === "GET" && pathname === "/api/health") {
      jsonResponse(res, 200, {
        status: "ok",
        engine: "node-demo-backend",
        model_loaded: true,
        model_version: MODEL_VERSION,
        uses_tabpfn: false,
        point_count_per_re: POINT_COUNT,
        re_values: RE_VALUES,
        supported_radius_groups: RADIUS_GROUPS,
      });
      return;
    }

    if (req.method === "GET" && pathname === "/api/schema") {
      jsonResponse(res, 200, schemaPayload());
      return;
    }

    if (req.method === "POST" && pathname === "/api/predict-triangle") {
      const payload = await readJsonBody(req);
      jsonResponse(res, 200, predictTriangle(payload));
      return;
    }

    if (pathname.startsWith("/api/")) {
      jsonResponse(res, 404, { status: "error", error: "not found" });
      return;
    }

    serveStatic(req, res, pathname);
  } catch (error) {
    const statusCode = /triangle|positive|radius_group|JSON|large/.test(error.message) ? 400 : 500;
    jsonResponse(res, statusCode, { status: "error", error: error.message });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`TriFocus ForceLab Node demo: http://${HOST}:${PORT}`);
});
