const runtimeBadge = document.getElementById("runtimeBadge");
const currentBalance = document.getElementById("currentBalance");
const realizedPnl = document.getElementById("realizedPnl");
const winRate = document.getElementById("winRate");
const unsettledNotional = document.getElementById("unsettledNotional");
const openCount = document.getElementById("openCount");
const openPositionsBody = document.getElementById("openPositionsBody");
const logPane = document.getElementById("logPane");
const entryResult = document.getElementById("entryResult");
const commandPreview = document.getElementById("commandPreview");
const liveBalanceHint = document.getElementById("liveBalanceHint");
const riskBanner = document.getElementById("riskBanner");
const entryValidationHint = document.getElementById("entryValidationHint");

const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const entryButton = document.getElementById("entryButton");
const clearLogButton = document.getElementById("clearLogButton");

const START_LIMITS = {
  bankroll: { min: 1, max: 1_000_000_000, fallback: 1000, decimals: 2 },
  tickSeconds: { min: 0.2, max: 60, fallback: 1, decimals: 2 },
  maxTicks: { min: 0, max: 100_000_000, fallback: 0, integer: true },
  fCap: { min: 0.01, max: 1, fallback: 0.25, decimals: 2 },
  minOrderNotional: { min: 0.1, max: 100_000, fallback: 0.3, decimals: 2 },
  policyExplorationEpsilon: { min: 0, max: 1, fallback: 0.05, decimals: 2 },
  policyUcbC: { min: 0, max: 20, fallback: 1.0, decimals: 2 },
};

function parseFinite(value, fallback) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return fallback;
  }
  return num;
}

function clampNumber(value, limits) {
  let clamped = parseFinite(value, limits.fallback);
  clamped = Math.min(limits.max, Math.max(limits.min, clamped));
  if (limits.integer) {
    clamped = Math.round(clamped);
  }
  return clamped;
}

function syncNumericInput(id, value, limits) {
  const element = document.getElementById(id);
  if (!element) {
    return;
  }
  if (limits.integer) {
    element.value = String(Math.round(value));
    return;
  }
  if (typeof limits.decimals === "number") {
    element.value = Number(value).toFixed(limits.decimals);
    return;
  }
  element.value = String(value);
}

function money(value) {
  const num = Number(value || 0);
  return `$${num.toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 })}`;
}

function pct(value) {
  const num = Number(value || 0);
  return `${(num * 100).toFixed(2)}%`;
}

function formatLiveBalanceHint(payload) {
  if (!payload) {
    return "";
  }
  const base = `balance=${payload.balance} allowance=${payload.allowance} bankroll=${payload.bankroll}`;
  if (payload.updated_at) {
    return `${base} updated_at=${payload.updated_at}`;
  }
  return base;
}

function isLiveModeContext(runtime) {
  const runtimeMode = String(runtime?.mode || "").toLowerCase();
  if (runtimeMode === "live") {
    return true;
  }
  return document.getElementById("mode").value === "live";
}

function setBadge(running) {
  runtimeBadge.textContent = running ? "Running" : "Stopped";
  runtimeBadge.classList.toggle("running", !!running);
  runtimeBadge.classList.toggle("stopped", !running);
}

function renderRows(openPositions) {
  openPositionsBody.innerHTML = "";
  if (!Array.isArray(openPositions) || openPositions.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 7;
    cell.textContent = "열린 포지션이 없습니다.";
    row.appendChild(cell);
    openPositionsBody.appendChild(row);
    return;
  }

  openPositions.forEach((pos) => {
    const row = document.createElement("tr");
    const cols = [
      String(pos.timestamp || "").replace("T", " ").slice(0, 19),
      pos.marketId || "-",
      pos.orderId || "-",
      pos.side || "-",
      Number(pos.price || 0).toFixed(3),
      Number(pos.size || 0).toFixed(4),
      money(pos.notional || 0),
    ];
    cols.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = String(value);
      row.appendChild(td);
    });
    openPositionsBody.appendChild(row);
  });
}

function appendLog(line) {
  const existing = logPane.textContent.split("\n").filter((item) => item.length > 0);
  existing.push(line);
  const trimmed = existing.slice(Math.max(0, existing.length - 250));
  logPane.textContent = trimmed.join("\n");
  logPane.scrollTop = logPane.scrollHeight;
}

function renderSnapshot(snapshot) {
  if (!snapshot) {
    return;
  }
  const runtime = snapshot.runtime || {};
  const result = snapshot.result || {};
  const executions = snapshot.executions || {};
  const liveBalance = snapshot.liveBalance || null;
  const liveMode = isLiveModeContext(runtime);

  setBadge(!!runtime.running);
  const liveBalanceValue = Number(liveBalance?.balance);
  const currentBalanceValue =
    liveMode && Number.isFinite(liveBalanceValue) ? liveBalanceValue : Number(result.current_balance || 0);
  currentBalance.textContent = money(currentBalanceValue);
  realizedPnl.textContent = money(result.realized_pnl || 0);
  realizedPnl.style.color = Number(result.realized_pnl || 0) >= 0 ? "#7ff0be" : "#ff9ba6";
  winRate.textContent = pct(result.win_rate || 0);
  unsettledNotional.textContent = money(result.unsettled_notional || 0);
  if (liveMode && liveBalance) {
    liveBalanceHint.textContent = formatLiveBalanceHint(liveBalance);
  } else if (!liveMode) {
    liveBalanceHint.textContent = "";
  }

  const openPositions = executions.openPositions || [];
  openCount.textContent = String(openPositions.length);
  renderRows(openPositions);

  const logs = Array.isArray(snapshot.logs) ? snapshot.logs : [];
  if (logs.length > 0) {
    logPane.textContent = logs.slice(Math.max(0, logs.length - 250)).join("\n");
    logPane.scrollTop = logPane.scrollHeight;
  }
}

function collectStartConfig() {
  const isLiveMode = document.getElementById("mode").value === "live";
  const bankrollInput = document.getElementById("bankroll");
  bankrollInput.readOnly = isLiveMode;
  bankrollInput.style.opacity = isLiveMode ? "0.84" : "1";

  const bankroll = clampNumber(bankrollInput.value, START_LIMITS.bankroll);
  const tickSeconds = clampNumber(document.getElementById("tickSeconds").value, START_LIMITS.tickSeconds);
  const maxTicks = clampNumber(document.getElementById("maxTicks").value, START_LIMITS.maxTicks);
  const fCap = clampNumber(document.getElementById("fCap").value, START_LIMITS.fCap);
  const minOrderNotional = clampNumber(
    document.getElementById("minOrderNotional").value,
    START_LIMITS.minOrderNotional,
  );
  const policyExplorationEpsilon = clampNumber(
    document.getElementById("policyExplorationEpsilon").value,
    START_LIMITS.policyExplorationEpsilon,
  );
  const policyUcbC = clampNumber(document.getElementById("policyUcbC").value, START_LIMITS.policyUcbC);

  syncNumericInput("bankroll", bankroll, START_LIMITS.bankroll);
  syncNumericInput("tickSeconds", tickSeconds, START_LIMITS.tickSeconds);
  syncNumericInput("maxTicks", maxTicks, START_LIMITS.maxTicks);
  syncNumericInput("fCap", fCap, START_LIMITS.fCap);
  syncNumericInput("minOrderNotional", minOrderNotional, START_LIMITS.minOrderNotional);
  syncNumericInput("policyExplorationEpsilon", policyExplorationEpsilon, START_LIMITS.policyExplorationEpsilon);
  syncNumericInput("policyUcbC", policyUcbC, START_LIMITS.policyUcbC);

  const config = {
    mode: document.getElementById("mode").value,
    bankroll,
    tickSeconds,
    maxTicks,
    fCap,
    minOrderNotional,
    freshStart: document.getElementById("freshStart").checked,
    disableWebsocket: document.getElementById("disableWebsocket").checked,
    enablePolicyBandit: document.getElementById("enablePolicyBandit").checked,
    policyShadowMode: document.getElementById("policyShadowMode").checked,
    policyExplorationEpsilon,
    policyUcbC,
  };
  const parts = [
    "uv run --python 3.11 python -m pm1h_edge_trader.main",
    `--mode ${config.mode}`,
    `--bankroll ${config.bankroll}`,
    `--tick-seconds ${config.tickSeconds}`,
    `--f-cap ${config.fCap}`,
    `--min-order-notional ${config.minOrderNotional}`,
  ];
  if (config.maxTicks > 0) {
    parts.push(`--max-ticks ${config.maxTicks}`);
  }
  if (config.mode === "live") {
    parts.push("--hard-kill-on-daily-loss");
  }
  if (config.freshStart) {
    parts.push("--fresh-start");
  }
  if (config.disableWebsocket) {
    parts.push("--disable-websocket");
  }
  if (config.enablePolicyBandit) {
    parts.push("--enable-policy-bandit");
    if (config.policyShadowMode) {
      parts.push("--policy-shadow-mode");
    }
    parts.push(`--policy-exploration-epsilon ${config.policyExplorationEpsilon}`);
    parts.push(`--policy-ucb-c ${config.policyUcbC}`);
  }
  commandPreview.textContent = parts.join(" ");
  if (config.mode === "live") {
    riskBanner.textContent = "LIVE + HARD KILL ON DAILY LOSS (운영 중 즉시 중단 가능)";
    riskBanner.classList.add("visible");
  } else {
    riskBanner.textContent = "";
    riskBanner.classList.remove("visible");
  }
  return config;
}

async function syncLiveBankrollIfNeeded() {
  const mode = document.getElementById("mode").value;
  if (mode !== "live") {
    liveBalanceHint.textContent = "";
    collectStartConfig();
    return;
  }
  liveBalanceHint.textContent = "Live 잔액 조회 중...";
  const response = await window.traderApi.getLiveBankroll();
  if (!response.ok || !response.result) {
    liveBalanceHint.textContent = `Live 잔액 조회 실패: ${response.error || "unknown error"}`;
    return;
  }
  const bankroll = Number(response.result.bankroll || 0);
  if (bankroll > 0) {
    document.getElementById("bankroll").value = bankroll.toFixed(2);
  }
  liveBalanceHint.textContent = formatLiveBalanceHint(response.result);
  collectStartConfig();
}

async function startBot() {
  startButton.disabled = true;
  try {
    if (document.getElementById("mode").value === "live") {
      await syncLiveBankrollIfNeeded();
    }
    const response = await window.traderApi.startBot(collectStartConfig());
    if (!response.ok) {
      appendLog(`${new Date().toISOString()} | ERROR | ${response.error}`);
    }
  } finally {
    startButton.disabled = false;
  }
}

async function stopBot() {
  stopButton.disabled = true;
  try {
    const response = await window.traderApi.stopBot();
    if (!response.ok) {
      appendLog(`${new Date().toISOString()} | ERROR | ${response.error}`);
    }
  } finally {
    stopButton.disabled = false;
  }
}

async function manualEntry() {
  if (!validateManualEntryForm()) {
    return;
  }
  entryButton.disabled = true;
  const entryPriceValue = document.getElementById("entryPrice").value.trim();
  const payload = {
    mode: document.getElementById("entryMode").value,
    direction: document.getElementById("entryDirection").value,
    usd: Number(document.getElementById("entryUsd").value || 0),
    price: entryPriceValue.length > 0 ? Number(entryPriceValue) : null,
    disableWebsocket: document.getElementById("entryDisableWebsocket").checked,
  };
  try {
    const response = await window.traderApi.manualEntry(payload);
    entryResult.textContent = JSON.stringify(response, null, 2);
  } catch (error) {
    entryResult.textContent = JSON.stringify(
      {
        ok: false,
        error: `${error.name}: ${error.message}`,
      },
      null,
      2,
    );
  } finally {
    entryButton.disabled = false;
  }
}

function validateManualEntryForm() {
  const usdInput = document.getElementById("entryUsd");
  const priceInput = document.getElementById("entryPrice");
  const usd = parseFinite(usdInput.value, NaN);
  const usdValid = Number.isFinite(usd) && usd > 0 && usd <= 1_000_000;
  const priceText = priceInput.value.trim();
  let priceValid = true;
  if (priceText.length > 0) {
    const price = parseFinite(priceText, NaN);
    priceValid = Number.isFinite(price) && price > 0 && price < 1;
  }

  usdInput.classList.toggle("input-invalid", !usdValid);
  priceInput.classList.toggle("input-invalid", !priceValid);
  entryButton.disabled = !(usdValid && priceValid);

  if (!usdValid) {
    entryValidationHint.textContent = "진입 금액은 0보다 크고 1,000,000 이하이어야 합니다.";
    return false;
  }
  if (!priceValid) {
    entryValidationHint.textContent = "지정가는 비우거나 0~1 사이 값이어야 합니다.";
    return false;
  }
  entryValidationHint.textContent = "";
  return true;
}

startButton.addEventListener("click", startBot);
stopButton.addEventListener("click", stopBot);
entryButton.addEventListener("click", manualEntry);
clearLogButton.addEventListener("click", () => {
  logPane.textContent = "";
});

[
  "mode",
  "bankroll",
  "tickSeconds",
  "maxTicks",
  "fCap",
  "minOrderNotional",
  "freshStart",
  "disableWebsocket",
  "enablePolicyBandit",
  "policyShadowMode",
  "policyExplorationEpsilon",
  "policyUcbC",
].forEach(
  (id) => {
    const element = document.getElementById(id);
    element.addEventListener("change", collectStartConfig);
    element.addEventListener("input", collectStartConfig);
  },
);

["entryUsd", "entryPrice", "entryMode", "entryDirection"].forEach((id) => {
  const element = document.getElementById(id);
  element.addEventListener("change", validateManualEntryForm);
  element.addEventListener("input", validateManualEntryForm);
});

document.getElementById("mode").addEventListener("change", () => {
  syncLiveBankrollIfNeeded();
});

window.traderApi.onStatus((status) => {
  setBadge(!!status.running);
});

window.traderApi.onLog((line) => {
  appendLog(line);
});

window.traderApi.onMetrics((snapshot) => {
  renderSnapshot(snapshot);
});

window.traderApi.onLiveBankroll((payload) => {
  if (!payload) {
    return;
  }
  const bankroll = Number(payload.bankroll || 0);
  if (bankroll > 0) {
    document.getElementById("bankroll").value = bankroll.toFixed(2);
  }
  liveBalanceHint.textContent = formatLiveBalanceHint(payload);
  collectStartConfig();
});

async function bootstrap() {
  collectStartConfig();
  validateManualEntryForm();
  const snapshot = await window.traderApi.getSnapshot();
  renderSnapshot(snapshot);
  await syncLiveBankrollIfNeeded();
}

bootstrap();
