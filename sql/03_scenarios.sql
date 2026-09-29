-- Frauds by scenario (0 is genuine): volume and amounts.
SELECT
    tx_fraud_scenario AS scenario,
    count(*) AS transactions,
    round(median(tx_amount), 2) AS median_amount,
    round(quantile_cont(tx_amount, 0.9), 2) AS p90_amount
FROM transactions
GROUP BY scenario
ORDER BY scenario;
