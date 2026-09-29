-- Amount compared with the customer's average spending over the previous 30 days.
WITH history AS (
    SELECT
        tx_fraud_scenario,
        tx_amount / avg(tx_amount) OVER (
            PARTITION BY customer_id
            ORDER BY tx_datetime
            RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 SECOND PRECEDING
        ) AS ratio_to_habit
    FROM transactions
)
SELECT
    tx_fraud_scenario AS scenario,
    round(median(ratio_to_habit), 2) AS median_ratio,
    round(quantile_cont(ratio_to_habit, 0.1), 2) AS p10_ratio
FROM history
WHERE ratio_to_habit IS NOT NULL
GROUP BY scenario
ORDER BY scenario;
