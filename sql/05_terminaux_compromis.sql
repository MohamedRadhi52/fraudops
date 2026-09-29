-- Fraud rate depending on whether the terminal already has a known fraud.
-- A label is known 7 days after the transaction, so the look-back stops 7 days before.
WITH history AS (
    SELECT
        tx_fraud,
        sum(tx_fraud) OVER (
            PARTITION BY terminal_id
            ORDER BY tx_datetime
            RANGE BETWEEN INTERVAL 37 DAYS PRECEDING AND INTERVAL 7 DAYS PRECEDING
        ) AS known_frauds
    FROM transactions
)
SELECT
    coalesce(known_frauds, 0) > 0 AS known_fraud_on_terminal,
    count(*) AS transactions,
    sum(tx_fraud) AS frauds,
    round(100 * avg(tx_fraud), 2) AS fraud_rate_pct,
    round(100 * sum(tx_fraud) / sum(sum(tx_fraud)) OVER (), 1) AS share_of_frauds_pct
FROM history
GROUP BY known_fraud_on_terminal
ORDER BY known_fraud_on_terminal;
