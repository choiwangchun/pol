const { app, BrowserWindow, ipcMain } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const ENV_PATH = path.join(PROJECT_ROOT, ".env");
const LOG_DIR = path.join(PROJECT_ROOT, "logs");
const RESULT_PATH = path.join(LOG_DIR, "result.json");
const EXECUTIONS_PATH = path.join(LOG_DIR, "executions.csv");
const LIVE_BALANCE_REFRESH_MS = 10_000;

loadEnvFile(ENV_PATH);

let mainWindow = null;
let botProcess = null;
let metricsTimer = null;
let lastMetrics = null;
let logBuffer = [];
const LOG_BUFFER_LIMIT = 1000;
let liveBalanceRefreshInFlight = false;
let liveBalanceCache = {
  ok: false,
  result: null,
  updatedAt: null,
  error: null,
};

const runtimeState = {
  running: false,
  mode: "dry-run",
  pid: null,
  startedAt: null,
  stoppedAt: null,
  lastExitCode: null,
  lastError: null,
  lastCommand: null,
};

function loadEnvFile(filePath) {
  if (!fs.existsSync(filePath)) {
    return;
  }
  try {
    const text = fs.readFileSync(filePath, "utf-8");
    let loadedCount = 0;
    text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .forEach((line) => {
        if (!line || line.startsWith("#")) {
          return;
        }
        const match = line.match(/^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
        if (!match) {
          return;
        }
        const key = match[1];
        const rawValue = match[2].trim();
        if (process.env[key] !== undefined) {
          return;
        }
        let value = rawValue;
        if (
          (value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'"))
        ) {
          const quote = value[0];
          value = value.slice(1, -1);
          if (quote === '"') {
            value = value
              .replace(/\\n/g, "\n")
              .replace(/\\r/g, "\r")
              .replace(/\\t/g, "\t")
              .replace(/\\"/g, '"')
              .replace(/\\\\/g, "\\");
          }
        }
        process.env[key] = value;
        loadedCount += 1;
      });
    if (loadedCount > 0) {
      console.info(`[pm1h-desktop] loaded ${loadedCount} env vars from ${filePath}`);
    }
  } catch (error) {
    const message = error && error.message ? error.message : String(error);
    console.warn(`[pm1h-desktop] failed to load env file ${filePath}: ${message}`);
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1420,
    height: 920,
    minWidth: 1200,
    minHeight: 760,
    backgroundColor: "#0b1020",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });
  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
}

function sendToRenderer(channel, payload) {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  mainWindow.webContents.send(channel, payload);
}

function pushLog(level, message) {
  const timestamp = new Date().toISOString();
  const line = `${timestamp} | ${level} | ${message}`;
  logBuffer.push(line);
  if (logBuffer.length > LOG_BUFFER_LIMIT) {
    logBuffer = logBuffer.slice(logBuffer.length - LOG_BUFFER_LIMIT);
  }
  sendToRenderer("bot:log", line);
}

function buildBotCommand(config) {
  const mode = config.mode === "live" ? "live" : "dry-run";
  const bankroll = Number(config.bankroll || 1000);
  const tickSeconds = Number(config.tickSeconds || 1);
  const fCap = Number(config.fCap || 0.25);
  const minOrderNotional = Number(config.minOrderNotional || 0.3);
  const args = [
    "run",
    "--python",
    "3.11",
    "python",
    "-m",
    "pm1h_edge_trader.main",
    "--mode",
    mode,
    "--bankroll",
    String(bankroll),
    "--tick-seconds",
    String(tickSeconds),
    "--f-cap",
    String(fCap),
    "--min-order-notional",
    String(minOrderNotional),
  ];
  if (Number(config.maxTicks || 0) > 0) {
    args.push("--max-ticks", String(Number(config.maxTicks)));
  }
  if (mode === "live") {
    args.push("--hard-kill-on-daily-loss");
  }
  if (config.freshStart) {
    args.push("--fresh-start");
  }
  if (config.disableWebsocket) {
    args.push("--disable-websocket");
  }
  if (config.enablePolicyBandit) {
    args.push("--enable-policy-bandit");
    if (config.policyShadowMode) {
      args.push("--policy-shadow-mode");
    }
    args.push("--policy-exploration-epsilon", String(Number(config.policyExplorationEpsilon || 0.05)));
    args.push("--policy-ucb-c", String(Number(config.policyUcbC || 1.0)));
  }
  return { command: "uv", args };
}

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"' && line[i + 1] === '"') {
      current += '"';
      i += 1;
      continue;
    }
    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (char === "," && !inQuotes) {
      values.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  values.push(current);
  return values;
}

function parseExecutionsCsv(filePath) {
  if (!fs.existsSync(filePath)) {
    return {
      totalRows: 0,
      openPositions: [],
      recentEvents: [],
    };
  }
  const text = fs.readFileSync(filePath, "utf-8");
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trimEnd())
    .filter((line) => line.length > 0);
  if (lines.length <= 1) {
    return {
      totalRows: 0,
      openPositions: [],
      recentEvents: [],
    };
  }

  const headers = parseCsvLine(lines[0]);
  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const raw = parseCsvLine(lines[i]);
    const row = {};
    headers.forEach((header, idx) => {
      row[header] = raw[idx] ?? "";
    });
    rows.push(row);
  }

  const unsettled = new Map();
  rows.forEach((row) => {
    const status = String(row.status || "").toLowerCase();
    const orderId = String(row.order_id || "").trim();
    if (!orderId || orderId === "-") {
      return;
    }
    if (status === "paper_fill" || status === "live_fill") {
      unsettled.set(orderId, row);
      return;
    }
    if (status === "paper_settle" || status === "live_settle") {
      unsettled.delete(orderId);
    }
  });

  const openPositions = Array.from(unsettled.values())
    .map((row) => ({
      timestamp: row.timestamp || "",
      marketId: row.market_id || "",
      orderId: row.order_id || "",
      side: row.side || "",
      price: Number(row.price || 0),
      size: Number(row.size || 0),
      notional: Number(row.price || 0) * Number(row.size || 0),
      status: row.status || "",
    }))
    .sort((a, b) => String(b.timestamp).localeCompare(String(a.timestamp)));

  const recentEvents = rows
    .slice(Math.max(0, rows.length - 20))
    .map((row) => ({
      timestamp: row.timestamp || "",
      orderId: row.order_id || "",
      side: row.side || "",
      status: row.status || "",
      pnl: row.pnl || "",
    }))
    .reverse();

  return {
    totalRows: rows.length,
    openPositions,
    recentEvents,
  };
}

function readResultJson(filePath) {
  if (!fs.existsSync(filePath)) {
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch (_error) {
    return null;
  }
}

function currentSnapshot() {
  const result = readResultJson(RESULT_PATH);
  const executions = parseExecutionsCsv(EXECUTIONS_PATH);
  const liveBalance = liveBalanceCache.result
    ? {
        ...liveBalanceCache.result,
        updated_at: liveBalanceCache.updatedAt,
        ok: liveBalanceCache.ok,
        error: liveBalanceCache.error,
      }
    : null;
  return {
    runtime: { ...runtimeState },
    result,
    executions,
    liveBalance,
    logs: logBuffer.slice(Math.max(0, logBuffer.length - 200)),
  };
}

function publishMetrics() {
  scheduleLiveBalanceRefresh();
  const snapshot = currentSnapshot();
  lastMetrics = snapshot;
  sendToRenderer("bot:metrics", snapshot);
}

function runOneShot(command, args, cwd) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", (error) => reject(error));
    child.on("close", (code) => {
      resolve({ code, stdout, stderr });
    });
  });
}

async function fetchLiveBankrollSnapshot() {
  const args = [
    "run",
    "--python",
    "3.11",
    "python",
    "-m",
    "pm1h_edge_trader.live_balance",
  ];
  const result = await runOneShot("uv", args, PROJECT_ROOT);
  const stdoutLines = result.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  const lastLine = stdoutLines.length > 0 ? stdoutLines[stdoutLines.length - 1] : "";
  if (!lastLine.startsWith("{") || !lastLine.endsWith("}")) {
    return { ok: false, error: "Live balance response parsing failed.", raw: result };
  }
  let parsed = null;
  try {
    parsed = JSON.parse(lastLine);
  } catch (_error) {
    parsed = null;
  }
  if (!parsed) {
    return { ok: false, error: "Live balance response JSON is invalid.", raw: result };
  }
  if (!parsed.ok || result.code !== 0) {
    return {
      ok: false,
      error: parsed.error || `Live balance command failed (${result.code}).`,
      raw: result,
    };
  }
  return {
    ok: true,
    result: parsed.result || null,
  };
}

function normalizeLiveBalanceCacheResponse() {
  if (liveBalanceCache.ok && liveBalanceCache.result) {
    return { ok: true, result: liveBalanceCache.result };
  }
  return { ok: false, error: liveBalanceCache.error || "Failed to fetch live bankroll." };
}

async function refreshLiveBalanceCache({ force = false, emit = false } = {}) {
  const nowMs = Date.now();
  if (!force && liveBalanceCache.updatedAt) {
    const updatedMs = Date.parse(liveBalanceCache.updatedAt);
    if (Number.isFinite(updatedMs) && nowMs - updatedMs < LIVE_BALANCE_REFRESH_MS) {
      return normalizeLiveBalanceCacheResponse();
    }
  }
  if (liveBalanceRefreshInFlight) {
    return normalizeLiveBalanceCacheResponse();
  }
  liveBalanceRefreshInFlight = true;
  try {
    const snapshot = await fetchLiveBankrollSnapshot();
    if (snapshot.ok && snapshot.result) {
      liveBalanceCache = {
        ok: true,
        result: snapshot.result,
        updatedAt: new Date().toISOString(),
        error: null,
      };
      if (emit) {
        sendToRenderer("bot:live-bankroll", {
          ...snapshot.result,
          updated_at: liveBalanceCache.updatedAt,
        });
      }
      return { ok: true, result: snapshot.result };
    }
    liveBalanceCache = {
      ...liveBalanceCache,
      ok: false,
      error: snapshot.error || "Failed to fetch live bankroll.",
    };
    return { ok: false, error: liveBalanceCache.error };
  } finally {
    liveBalanceRefreshInFlight = false;
  }
}

function scheduleLiveBalanceRefresh() {
  if (!runtimeState.running || runtimeState.mode !== "live") {
    return;
  }
  void refreshLiveBalanceCache({ force: false, emit: true });
}

ipcMain.handle("bot:start", async (_event, config) => {
  if (botProcess) {
    return { ok: false, error: "Bot process is already running." };
  }
  const startConfig = { ...(config || {}) };
  if (startConfig.mode === "live") {
    const liveBalance = await refreshLiveBalanceCache({ force: true, emit: true });
    if (!liveBalance.ok || !liveBalance.result) {
      return { ok: false, error: liveBalance.error || "Failed to fetch live bankroll." };
    }
    const bankroll = Number(liveBalance.result.bankroll || 0);
    if (!(bankroll > 0)) {
      return { ok: false, error: "Live bankroll is zero or invalid." };
    }
    startConfig.bankroll = bankroll;
    sendToRenderer("bot:live-bankroll", liveBalance.result);
    pushLog(
      "INFO",
      `Live bankroll auto-applied: balance=${liveBalance.result.balance} allowance=${liveBalance.result.allowance} bankroll=${bankroll}`,
    );
  }
  const { command, args } = buildBotCommand(startConfig);
  runtimeState.running = true;
  runtimeState.mode = startConfig.mode === "live" ? "live" : "dry-run";
  runtimeState.lastError = null;
  runtimeState.lastExitCode = null;
  runtimeState.startedAt = new Date().toISOString();
  runtimeState.stoppedAt = null;
  runtimeState.lastCommand = `${command} ${args.join(" ")}`;

  pushLog("INFO", `Starting bot: ${runtimeState.lastCommand}`);
  botProcess = spawn(command, args, {
    cwd: PROJECT_ROOT,
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
  });

  runtimeState.pid = botProcess.pid || null;
  sendToRenderer("bot:status", { ...runtimeState });

  botProcess.stdout.on("data", (chunk) => {
    const text = String(chunk);
    text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .forEach((line) => pushLog("OUT", line));
  });

  botProcess.stderr.on("data", (chunk) => {
    const text = String(chunk);
    text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .forEach((line) => pushLog("ERR", line));
  });

  botProcess.on("error", (error) => {
    runtimeState.lastError = `${error.name}: ${error.message}`;
    pushLog("ERROR", runtimeState.lastError);
  });

  botProcess.on("close", (code) => {
    pushLog("INFO", `Bot process exited with code ${code}`);
    runtimeState.running = false;
    runtimeState.pid = null;
    runtimeState.stoppedAt = new Date().toISOString();
    runtimeState.lastExitCode = code;
    botProcess = null;
    sendToRenderer("bot:status", { ...runtimeState });
    publishMetrics();
  });

  publishMetrics();
  return { ok: true, runtime: { ...runtimeState } };
});

ipcMain.handle("bot:get-live-bankroll", async () => {
  const snapshot = await refreshLiveBalanceCache({ force: true, emit: true });
  if (!snapshot.ok) {
    return snapshot;
  }
  return snapshot;
});

ipcMain.handle("bot:stop", async () => {
  if (!botProcess) {
    return { ok: true, message: "Bot is not running." };
  }
  pushLog("INFO", "Stopping bot process...");
  botProcess.kill("SIGTERM");
  setTimeout(() => {
    if (botProcess) {
      pushLog("WARN", "Bot did not stop in time, sending SIGKILL.");
      botProcess.kill("SIGKILL");
    }
  }, 5000);
  return { ok: true, message: "Stop signal sent." };
});

ipcMain.handle("bot:manual-entry", async (_event, payload) => {
  const mode = payload && payload.mode === "live" ? "live" : "dry-run";
  const direction = payload && payload.direction === "down" ? "down" : "up";
  const usd = Number(payload?.usd || 0);
  if (!(usd > 0)) {
    return { ok: false, error: "USD notional must be greater than 0." };
  }
  const args = [
    "run",
    "--python",
    "3.11",
    "python",
    "-m",
    "pm1h_edge_trader.manual_entry",
    "--mode",
    mode,
    "--direction",
    direction,
    "--usd",
    String(usd),
    "--log-dir",
    "logs",
  ];
  if (payload?.price && Number(payload.price) > 0) {
    args.push("--price", String(Number(payload.price)));
  }
  if (payload?.disableWebsocket) {
    args.push("--disable-websocket");
  }

  pushLog("INFO", `Manual entry requested: mode=${mode} direction=${direction} usd=${usd}`);
  const result = await runOneShot("uv", args, PROJECT_ROOT);
  if (result.stderr && result.stderr.trim().length > 0) {
    result.stderr
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .forEach((line) => pushLog("ERR", line));
  }

  const stdoutLines = result.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  const lastLine = stdoutLines.length > 0 ? stdoutLines[stdoutLines.length - 1] : "";
  let payloadJson = null;
  if (lastLine.startsWith("{") && lastLine.endsWith("}")) {
    try {
      payloadJson = JSON.parse(lastLine);
    } catch (_error) {
      payloadJson = null;
    }
  }

  publishMetrics();
  if (result.code !== 0) {
    return {
      ok: false,
      error: payloadJson?.error || `Manual entry failed with code ${result.code}.`,
      rawStdout: result.stdout,
    };
  }
  return {
    ok: true,
    response: payloadJson || { ok: true, raw: result.stdout },
  };
});

ipcMain.handle("bot:snapshot", async () => {
  if (!lastMetrics) {
    publishMetrics();
  }
  return lastMetrics || currentSnapshot();
});

app.whenReady().then(() => {
  if (!fs.existsSync(LOG_DIR)) {
    fs.mkdirSync(LOG_DIR, { recursive: true });
  }
  createWindow();
  publishMetrics();
  metricsTimer = setInterval(publishMetrics, 1000);
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on("before-quit", () => {
  if (metricsTimer) {
    clearInterval(metricsTimer);
    metricsTimer = null;
  }
  if (botProcess) {
    botProcess.kill("SIGTERM");
  }
});
