function parseFiniteNumber(value, fallback) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return fallback;
  }
  return num;
}

function clampNumber(value, { min, max, fallback, integer = false }) {
  let num = parseFiniteNumber(value, fallback);
  num = Math.min(max, Math.max(min, num));
  if (integer) {
    num = Math.round(num);
  }
  return num;
}

function sanitizeStartConfig(config) {
  const raw = config || {};
  return {
    mode: raw.mode === "live" ? "live" : "dry-run",
    bankroll: clampNumber(raw.bankroll, {
      min: 1,
      max: 1_000_000_000,
      fallback: 1000,
    }),
    tickSeconds: clampNumber(raw.tickSeconds, {
      min: 0.2,
      max: 60,
      fallback: 1,
    }),
    maxTicks: clampNumber(raw.maxTicks, {
      min: 0,
      max: 100_000_000,
      fallback: 0,
      integer: true,
    }),
    fCap: clampNumber(raw.fCap, {
      min: 0.01,
      max: 1.0,
      fallback: 0.25,
    }),
    minOrderNotional: clampNumber(raw.minOrderNotional, {
      min: 0.1,
      max: 100_000,
      fallback: 0.3,
    }),
    freshStart: !!raw.freshStart,
    freshStartIgnoreState: !!raw.freshStartIgnoreState,
    disableWebsocket: !!raw.disableWebsocket,
    manualUnlatch: !!raw.manualUnlatch,
    enablePolicyBandit: !!raw.enablePolicyBandit,
    policyShadowMode: !!raw.policyShadowMode,
    policyExplorationEpsilon: clampNumber(raw.policyExplorationEpsilon, {
      min: 0.0,
      max: 1.0,
      fallback: 0.05,
    }),
    policyUcbC: clampNumber(raw.policyUcbC, {
      min: 0.0,
      max: 20.0,
      fallback: 1.0,
    }),
  };
}

function buildBotCommand(config) {
  const safe = sanitizeStartConfig(config);
  const mode = safe.mode;
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
    String(safe.bankroll),
    "--tick-seconds",
    String(safe.tickSeconds),
    "--f-cap",
    String(safe.fCap),
    "--min-order-notional",
    String(safe.minOrderNotional),
  ];
  if (safe.maxTicks > 0) {
    args.push("--max-ticks", String(safe.maxTicks));
  }
  if (mode === "live") {
    args.push("--hard-kill-on-daily-loss");
  }
  if (safe.freshStart) {
    args.push("--fresh-start");
    if (safe.freshStartIgnoreState) {
      args.push("--fresh-start-ignore-state");
    }
  }
  if (safe.disableWebsocket) {
    args.push("--disable-websocket");
  }
  if (safe.manualUnlatch) {
    args.push("--manual-unlatch");
  }
  if (safe.enablePolicyBandit) {
    args.push("--enable-policy-bandit");
    if (safe.policyShadowMode) {
      args.push("--policy-shadow-mode");
    }
    args.push("--policy-exploration-epsilon", String(safe.policyExplorationEpsilon));
    args.push("--policy-ucb-c", String(safe.policyUcbC));
  }
  return { command: "uv", args };
}

module.exports = {
  sanitizeStartConfig,
  buildBotCommand,
};
