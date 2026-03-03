const test = require("node:test");
const assert = require("node:assert/strict");

const { buildBotCommand } = require("./lib/bot-command");

test("buildBotCommand adds --manual-unlatch when enabled", () => {
  const command = buildBotCommand({
    mode: "live",
    bankroll: 1000,
    tickSeconds: 1,
    fCap: 0.25,
    minOrderNotional: 0.3,
    manualUnlatch: true,
  });

  assert.equal(command.command, "uv");
  assert.ok(command.args.includes("--manual-unlatch"));
});

test("buildBotCommand does not add --manual-unlatch by default", () => {
  const command = buildBotCommand({
    mode: "live",
    bankroll: 1000,
    tickSeconds: 1,
    fCap: 0.25,
    minOrderNotional: 0.3,
  });

  assert.equal(command.args.includes("--manual-unlatch"), false);
});

test("buildBotCommand adds --fresh-start-ignore-state when both fresh-start options are enabled", () => {
  const command = buildBotCommand({
    mode: "dry-run",
    bankroll: 1000,
    tickSeconds: 1,
    fCap: 0.25,
    minOrderNotional: 0.3,
    freshStart: true,
    freshStartIgnoreState: true,
  });

  assert.ok(command.args.includes("--fresh-start"));
  assert.ok(command.args.includes("--fresh-start-ignore-state"));
});

test("buildBotCommand does not add --fresh-start-ignore-state without --fresh-start", () => {
  const command = buildBotCommand({
    mode: "dry-run",
    bankroll: 1000,
    tickSeconds: 1,
    fCap: 0.25,
    minOrderNotional: 0.3,
    freshStartIgnoreState: true,
  });

  assert.equal(command.args.includes("--fresh-start-ignore-state"), false);
});
