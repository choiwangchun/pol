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

const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const entryButton = document.getElementById("entryButton");
const clearLogButton = document.getElementById("clearLogButton");

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

  const config = {
    mode: document.getElementById("mode").value,
    bankroll: Number(bankrollInput.value || 1000),
    tickSeconds: Number(document.getElementById("tickSeconds").value || 1),
    maxTicks: Number(document.getElementById("maxTicks").value || 0),
    fCap: Number(document.getElementById("fCap").value || 0.25),
    minOrderNotional: Number(document.getElementById("minOrderNotional").value || 0.3),
    freshStart: document.getElementById("freshStart").checked,
    disableWebsocket: document.getElementById("disableWebsocket").checked,
    enablePolicyBandit: document.getElementById("enablePolicyBandit").checked,
    policyShadowMode: document.getElementById("policyShadowMode").checked,
    policyExplorationEpsilon: Number(document.getElementById("policyExplorationEpsilon").value || 0.05),
    policyUcbC: Number(document.getElementById("policyUcbC").value || 1.0),
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
  entryButton.disabled = true;
  const payload = {
    mode: document.getElementById("entryMode").value,
    direction: document.getElementById("entryDirection").value,
    usd: Number(document.getElementById("entryUsd").value || 0),
    price: Number(document.getElementById("entryPrice").value || 0),
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
  const snapshot = await window.traderApi.getSnapshot();
  renderSnapshot(snapshot);
  await syncLiveBankrollIfNeeded();
}

bootstrap();
