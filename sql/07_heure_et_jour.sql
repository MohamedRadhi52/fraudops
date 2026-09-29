-- Fraud rate at night (before 6 am) and on weekends.
SELECT
    dayofweek(tx_datetime) IN (0, 6) AS weekend,
    hour(tx_datetime) < 6 AS night,
    count(*) AS transactions,
    round(100 * avg(tx_fraud), 2) AS fraud_rate_pct
FROM transactions
GROUP BY weekend, night
ORDER BY weekend, night;
