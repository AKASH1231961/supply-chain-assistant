# Database schema

PostgreSQL schema used by the application. 

## `customers`

| Column | Type | Constraints / default |
| --- | --- | --- |
| `customer_id` | `SERIAL` | Primary key |
| `name` | `VARCHAR(10)` | `NOT NULL` |
| `email` | `VARCHAR(15)` | `UNIQUE`, `NOT NULL` |
| `created_at` | `TIMESTAMP` | Defaults to `CURRENT_TIMESTAMP` |

## `orders`

| Column | Type | Constraints / default |
| --- | --- | --- |
| `order_id` | `SERIAL` | Primary key |
| `customer_id` | `INTEGER` | `NOT NULL`, references `customers(customer_id)` |
| `status` | `VARCHAR(15)` | `NOT NULL`, defaults to `'pending'` |
| `order_date` | `TIMESTAMP` | Defaults to `CURRENT_TIMESTAMP` |
| `total_amount` | `NUMERIC(10,2)` | `NOT NULL`, defaults to `0` |

## `order_items`

| Column | Type | Constraints / default |
| --- | --- | --- |
| `item_id` | `SERIAL` | Primary key |
| `order_id` | `INTEGER` | `NOT NULL`, references `orders(order_id)` with `ON DELETE CASCADE` |
| `product` | `VARCHAR(100)` | `NOT NULL` |
| `quantity` | `INTEGER` | `NOT NULL`, defaults to `1` |
| `unit_price` | `NUMERIC(10,2)` | `NOT NULL` |

The order-status query selects `order_id`, `status`, `order_date`, and
`total_amount` from `orders`, filtered by `order_id` and `customer_id`.

Keep connection credentials in `.env`; do not add them to this document or
commit the `.env` file.

## Router database access

Run natural-language order questions through `router.py`. Generated SQL is
restricted to one `SELECT` over the three tables above, capped at 100 rows, and
executed in a read-only transaction with a statement timeout. Configure
`DATABASE_URL` to use a PostgreSQL account that has only `SELECT` privileges on
the data the assistant is allowed to expose. The read-only application check is
not a replacement for database permissions.

The live schema lookup is an internal database helper and is not exposed by the
user-facing router. Schema or system-configuration questions receive a fixed
refusal instead of querying or returning database metadata.

Generated SOP, database, and combined answers are checked in a second model
pass against the exact source passages or query rows used to create them.
Unsupported answers and malformed verification responses are not shown; the
router returns a fixed notice instead. This grounding check reduces unsupported
claims but is not a substitute for reviewing answers in high-stakes use.

Start the interactive router from the project directory with:

```bash
python router.py "What is the inventory ordering procedure?"
```

Replace the sample question with an order question or a question combining
order details and policy. Set `GROQ_API_KEY`, `DATABASE_URL`, and optionally
`GROQ_MODEL` in the local `.env` file. 