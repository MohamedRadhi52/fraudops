-- Amounts of genuine and fraudulent transactions.
SELECT
    tx_fraud,
    count(*) AS transactions,
    round(median(tx_amount), 2) AS median_amount,
    round(quantile_cont(tx_amount, 0.99), 2) AS p99_amount,
    max(tx_amount) AS max_amount,
    count(*) FILTER (WHERE tx_amount > 220) AS above_220
FROM transactions
GROUP BY tx_fraud
ORDER BY tx_fraud;
