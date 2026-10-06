-- Leakage-safe transaction features. Every history window ends before the current row.
WITH base AS (
    SELECT
        t.*,
        c.home_country,
        c.customer_segment,
        CAST(strftime('%s', t.transaction_timestamp) AS INTEGER) AS event_epoch,
        CAST(strftime('%H', t.transaction_timestamp) AS INTEGER) AS transaction_hour,
        CAST(strftime('%w', t.transaction_timestamp) AS INTEGER) AS day_of_week
    FROM transactions t
    JOIN customers c USING (customer_id)
),
history AS (
    SELECT
        *,
        COUNT(*) OVER customer_before AS previous_transaction_count,
        AVG(amount) OVER customer_before AS avg_previous_amount,
        MAX(amount) OVER customer_before AS max_previous_amount,
        AVG(amount * amount) OVER customer_before AS avg_previous_amount_squared,
        LAG(event_epoch) OVER customer_order AS previous_event_epoch,
        COUNT(*) OVER previous_hour AS transactions_previous_1h,
        COUNT(*) OVER previous_day AS transactions_previous_24h,
        AVG(is_international) OVER customer_before AS historical_international_rate,
        COUNT(*) OVER device_before AS previous_device_uses,
        COUNT(*) OVER category_before AS previous_category_uses
    FROM base
    WINDOW
        customer_order AS (
            PARTITION BY customer_id ORDER BY event_epoch, transaction_id
        ),
        customer_before AS (
            PARTITION BY customer_id ORDER BY event_epoch, transaction_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        previous_hour AS (
            PARTITION BY customer_id ORDER BY event_epoch
            RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING
        ),
        previous_day AS (
            PARTITION BY customer_id ORDER BY event_epoch
            RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING
        ),
        device_before AS (
            PARTITION BY customer_id, device_id ORDER BY event_epoch, transaction_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        category_before AS (
            PARTITION BY customer_id, merchant_category ORDER BY event_epoch, transaction_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )
)
SELECT
    transaction_id,
    customer_id,
    transaction_timestamp,
    amount,
    merchant_category,
    merchant_country,
    transaction_channel,
    customer_segment,
    transaction_hour,
    day_of_week,
    is_international,
    is_card_present,
    previous_transaction_count,
    COALESCE(avg_previous_amount, amount) AS avg_previous_amount,
    COALESCE(max_previous_amount, amount) AS max_previous_amount,
    CASE
        WHEN previous_transaction_count > 1 THEN
            SQRT(MAX(0, avg_previous_amount_squared - avg_previous_amount * avg_previous_amount))
        ELSE 0
    END AS std_previous_amount,
    CASE
        WHEN previous_event_epoch IS NULL THEN NULL
        ELSE (event_epoch - previous_event_epoch) / 60.0
    END AS minutes_since_previous_transaction,
    transactions_previous_1h,
    transactions_previous_24h,
    COALESCE(historical_international_rate, 0) AS historical_international_rate,
    amount / MAX(COALESCE(avg_previous_amount, amount), 1.0) AS amount_vs_customer_average,
    CASE WHEN amount > 3 * MAX(COALESCE(avg_previous_amount, amount), 1.0) THEN 1 ELSE 0 END AS high_amount_indicator,
    CASE WHEN transaction_hour BETWEEN 0 AND 4 THEN 1 ELSE 0 END AS unusual_hour_indicator,
    CASE WHEN merchant_country <> home_country THEN 1 ELSE 0 END AS country_mismatch_indicator,
    CASE WHEN previous_transaction_count > 0 AND previous_device_uses = 0 THEN 1 ELSE 0 END AS new_device_indicator,
    CASE WHEN previous_transaction_count > 0 AND previous_category_uses = 0 THEN 1 ELSE 0 END AS new_category_indicator,
    is_fraud
FROM history
ORDER BY transaction_timestamp, transaction_id;
