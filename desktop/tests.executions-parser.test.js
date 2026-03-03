const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { parseExecutionsCsv } = require('./lib/executions');

test('parseExecutionsCsv returns open positions and history rows from fills/settles', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pm1h-exec-'));
  const csvPath = path.join(tmpDir, 'executions.csv');
  const header = [
    'timestamp',
    'market_id',
    'order_id',
    'side',
    'price',
    'size',
    'status',
    'pnl',
  ].join(',');
  const rows = [
    '2026-02-28T10:00:00+00:00,mkt-1,ord-1,down,0.90,11.1,paper_fill,',
    '2026-02-28T10:30:00+00:00,mkt-1,ord-1,down,1.00,11.1,paper_settle,1.11',
    '2026-02-28T11:00:00+00:00,mkt-2,ord-2,up,0.55,20,paper_fill,',
  ];
  fs.writeFileSync(csvPath, `${header}\n${rows.join('\n')}\n`, 'utf-8');

  const parsed = parseExecutionsCsv(csvPath);

  assert.equal(parsed.totalRows, 3);
  assert.equal(parsed.openPositions.length, 1);
  assert.equal(parsed.openPositions[0].orderId, 'ord-2');

  assert.equal(parsed.history.length, 3);
  assert.equal(parsed.history[0].orderId, 'ord-2');
  assert.equal(parsed.history[0].action, 'buy');
  assert.equal(parsed.history[1].orderId, 'ord-1');
  assert.equal(parsed.history[1].action, 'sell');
  assert.equal(parsed.history[1].pnl, 1.11);
  assert.equal(parsed.history[2].action, 'buy');
});

test('parseExecutionsCsv returns empty defaults when file is missing', () => {
  const parsed = parseExecutionsCsv('/tmp/definitely-missing-pm1h-file.csv');
  assert.equal(parsed.totalRows, 0);
  assert.deepEqual(parsed.openPositions, []);
  assert.deepEqual(parsed.history, []);
});
