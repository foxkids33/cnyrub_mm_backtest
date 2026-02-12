# CNYRUB_TOM Market Making Backtest

Дано: `trades.parquet` со всеми сделками по CNYRUB_TOM за ~20 дней.

Нужно:
- пошаговый бэктест на этих сделках
- маркетмейкинговая стратегия (лимитные котировки)
- кумулятивный P&L (mark-to-market открытой позиции по медиане последних 5 сделок)

**Правило исполнения лимитки (из условия):** заявка считается исполненной, если по её цене совершилась хотя бы одна сделка.

Комиссии:
- maker: `0.000005`
- taker: `0.000045`

## Установка

```bash
pip install -r requirements.txt
```

## Запуск бэктеста

```bash
python3 -m mm_backtest.run \
  --trades trades.parquet --out_pnl pnl.csv \
  --base_spread_ticks 5 --min_spread_ticks 1 --max_spread_ticks 30 \
  --vol_window 50 --vol_mult 3.0 \
  --qty 300000 --inv_limit 3000000 --inv_skew_ticks 10 \
  --quote_every_n 1 \
  --flow_window 50 --flow_skew_ticks 2.0 \
  --enable_taker_unwind --unwind_to 1500000 --taker_slip_ticks 1 \
  --metric_freq 1s

```


## Выход

`pnl.csv` содержит:
- `datetime`
- `pnl` (cumulative)
- `cash`
- `position`
- `mark_price` (median last 5)
- `bid`, `ask`

