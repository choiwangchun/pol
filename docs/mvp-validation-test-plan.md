# MVP Validation Test Plan

Purpose: validate core PRD success metrics in a credential-free stub mode before wiring full integration.

## Scope
- Integration-style validation through the real execution engine (`LimitOrderExecutionEngine`) using `DryRunExecutionAdapter`
- Automated coverage in `tests/integration/test_mvp_validation.py`
- No external API calls or real credentials required

## Scenario Coverage
| ID | PRD Metric | Input Setup | Expected Result | Automated Test |
|---|---|---|---|---|
| MVP-001 | Market rules validation gate | `allow_entry=false` (used as market-rules gate output) | `SKIP`, reason `entry_not_allowed` | `test_market_rules_validation_gate_blocks_trade` |
| MVP-002 | No-trade when edge below threshold | `edge < min_edge`, `fee=0` | `SKIP`, reason `edge_below_threshold` | `test_no_trade_when_edge_below_threshold` |
| MVP-003 | Trade when edge positive and fee=0 | `edge > min_edge`, `fee=0`, fresh data | `PLACE` action | `test_trade_when_edge_positive_and_fee_zero` |
| MVP-004 | Kill-switch on stale data/non-zero fee | `data_age_s > max_data_age_s` OR `fee_rate != 0` | `KILL_SWITCH_TRIGGERED`, `kill_switch=true` | `test_kill_switch_on_stale_data_or_non_zero_fee` |

## Execution
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
