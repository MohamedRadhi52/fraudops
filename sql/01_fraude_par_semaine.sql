-- Weekly volume and fraud rate.
SELECT
    tx_time_days // 7 AS week,
    min(tx_datetime)::DATE AS first_day,
    count(*) AS transactions,
    sum(tx_fraud) AS frauds,
    round(100 * avg(tx_fraud), 2) AS fraud_rate_pct
FROM transactions
GROUP BY week
ORDER BY week;
