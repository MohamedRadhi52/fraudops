-- Cards with at least one fraud each day, after the first 30 days of warm-up.
-- No model can find more fraudulent cards than there are: this caps Card Precision@100.
WITH daily AS (
    SELECT
        tx_time_days AS day,
        count(DISTINCT customer_id) FILTER (WHERE tx_fraud = 1) AS fraudulent_cards,
        count(DISTINCT customer_id) AS active_cards
    FROM transactions
    GROUP BY day
)
SELECT
    min(fraudulent_cards) AS min_fraudulent_cards,
    round(avg(fraudulent_cards), 1) AS mean_fraudulent_cards,
    max(fraudulent_cards) AS max_fraudulent_cards,
    round(avg(active_cards)) AS mean_active_cards
FROM daily
WHERE day >= 30;
