const fs = require("node:fs");

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

function parseNumber(value, fallback = 0) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return parsed;
}

function normalizeStatus(value) {
  return String(value || "").trim().toLowerCase();
}

function parseExecutionsCsv(filePath) {
  if (!fs.existsSync(filePath)) {
    return {
      totalRows: 0,
      openPositions: [],
      history: [],
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
      history: [],
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

  const fillStatuses = new Set(["paper_fill", "live_fill"]);
  const settleStatuses = new Set(["paper_settle", "live_settle"]);

  const unsettled = new Map();
  rows.forEach((row) => {
    const status = normalizeStatus(row.status);
    const orderId = String(row.order_id || "").trim();
    if (!orderId || orderId === "-") {
      return;
    }
    if (fillStatuses.has(status)) {
      unsettled.set(orderId, row);
      return;
    }
    if (settleStatuses.has(status)) {
      unsettled.delete(orderId);
    }
  });

  const openPositions = Array.from(unsettled.values())
    .map((row) => {
      const price = parseNumber(row.price, 0);
      const size = parseNumber(row.size, 0);
      return {
        timestamp: row.timestamp || "",
        marketId: row.market_id || "",
        orderId: row.order_id || "",
        side: row.side || "",
        price,
        size,
        notional: price * size,
        status: row.status || "",
      };
    })
    .sort((a, b) => String(b.timestamp).localeCompare(String(a.timestamp)));

  const history = rows
    .filter((row) => {
      const status = normalizeStatus(row.status);
      return fillStatuses.has(status) || settleStatuses.has(status);
    })
    .map((row) => {
      const status = normalizeStatus(row.status);
      const price = parseNumber(row.price, 0);
      const size = parseNumber(row.size, 0);
      return {
        timestamp: row.timestamp || "",
        marketId: row.market_id || "",
        orderId: row.order_id || "",
        side: row.side || "",
        status: row.status || "",
        action: settleStatuses.has(status) ? "sell" : "buy",
        price,
        size,
        notional: price * size,
        pnl: parseNumber(row.pnl, 0),
      };
    })
    .sort((a, b) => String(b.timestamp).localeCompare(String(a.timestamp)))
    .slice(0, 120);

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
    history,
    recentEvents,
  };
}

module.exports = {
  parseCsvLine,
  parseExecutionsCsv,
};
