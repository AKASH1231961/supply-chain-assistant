import logging
import os

import psycopg
from psycopg.rows import dict_row
from sqlglot import ErrorLevel, exp, parse
from sqlglot.errors import ParseError, TokenError

from llm_client import get_groq_client, get_groq_model

logger = logging.getLogger(__name__)

SCHEMA = """
orders(order_id, customer_id, status, order_date, total_amount)
customers(customer_id, name, email, created_at)
order_items(item_id, order_id, product, quantity, unit_price)
"""

ALLOWED_TABLES = {"orders", "customers", "order_items"}


class SQLValidationError(ValueError):
    """Raised when generated SQL violates the read-only query policy."""


class SQLQueryError(RuntimeError):
    """Raised when generated SQL cannot be executed after retrying."""


def get_order_status(order_id: int, customer_id: int):
    database_url = os.environ["DATABASE_URL"]

    query = """
        SELECT
            order_id,
            status,
            order_date,
            total_amount
        FROM orders
        WHERE order_id = %s
          AND customer_id = %s
    """

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (order_id, customer_id))
            order = cur.fetchone()

    if order is None:
        return None

    return {
        "order_id": order["order_id"],
        "status": order["status"],
        "order_date": order["order_date"].isoformat(),
        "total_amount": float(order["total_amount"]),
    }


def question_to_sql(question: str, feedback: str = "") -> str:
    system_prompt = (
        "Convert the user's question into exactly one PostgreSQL "
        "SELECT statement. Use only the listed tables and columns. "
        "Do not use CTEs, comments, subqueries, or multiple "
        "statements. Use JOINs when combining tables. Add a LIMIT "
        "of at most 100 rows. Return SQL only.\n\n"
        f"Schema:\n{SCHEMA}"
    )
    if feedback:
        system_prompt += f"\n\nFeedback on your previous query:\n{feedback}"

    response = get_groq_client().chat.completions.create(
        model=get_groq_model(),
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
    )

    sql = response.choices[0].message.content
    if not sql:
        raise ValueError("The model did not return SQL.")
    return sql.strip()


def _validate_readonly_sql(sql: str) -> str:
    statement = sql.strip()
    if not statement:
        raise SQLValidationError("The generated SQL is empty.")

    try:
        expressions = parse(
            statement,
            read="postgres",
            error_level=ErrorLevel.RAISE,
        )
    except (ParseError, TokenError) as exc:
        raise SQLValidationError("The generated SQL is not valid PostgreSQL.") from exc

    if len(expressions) != 1 or expressions[0] is None:
        raise SQLValidationError("Only one uncommented SQL statement is allowed.")

    parsed_statement = expressions[0]
    if not isinstance(parsed_statement, exp.Select):
        raise SQLValidationError("Only SELECT statements are allowed.")

    nodes = list(parsed_statement.walk())
    if any(node.comments for node in nodes):
        raise SQLValidationError("Only one uncommented SQL statement is allowed.")
    if any(isinstance(node, exp.With) for node in nodes) or sum(
        isinstance(node, exp.Select) for node in nodes
    ) != 1:
        raise SQLValidationError(
            "Nested SELECT statements are not allowed; rewrite the query using JOINs."
        )
    if parsed_statement.args.get("into") or parsed_statement.args.get("locks"):
        raise SQLValidationError("The generated SQL contains a disallowed command.")

    tables = list(parsed_statement.find_all(exp.Table))
    if not tables:
        raise SQLValidationError(
            "The generated query must select from an approved table."
        )

    for table in tables:
        identifier = table.this
        if (
            not isinstance(identifier, exp.Identifier)
            or table.db
            or table.catalog
        ):
            raise SQLValidationError(
                "The generated query references a table outside the allowed schema."
            )
        table_name = table.name
        if identifier.args.get("quoted"):
            is_allowed = table_name in ALLOWED_TABLES
        else:
            is_allowed = table_name.lower() in ALLOWED_TABLES
        if not is_allowed:
            raise SQLValidationError(
                "The generated query references a table outside the allowed schema."
            )

    limit = parsed_statement.args.get("limit")
    if limit is None:
        parsed_statement.set(
            "limit",
            exp.Limit(expression=exp.Literal.number(100)),
        )
    else:
        limit_expression = limit.args.get("expression")
        if (
            not isinstance(limit_expression, exp.Literal)
            or limit_expression.is_string
            or not limit_expression.this.isascii()
            or not limit_expression.this.isdecimal()
        ):
            raise SQLValidationError(
                "The generated query may return at most 100 rows."
            )
        if int(limit_expression.this) > 100:
            raise SQLValidationError(
                "The generated query may return at most 100 rows."
            )

    return parsed_statement.sql(dialect="postgres")


def execute_readonly_sql(sql: str) -> list[dict[str, object]]:
    database_url = os.environ["DATABASE_URL"]
    safe_sql = _validate_readonly_sql(sql)

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET LOCAL statement_timeout = '5s'")
            with conn.cursor() as cursor:
                cursor.execute(safe_sql)
                return [dict(row) for row in cursor.fetchall()]


def get_database_schema() -> list[dict[str, object]]:
    database_url = os.environ["DATABASE_URL"]

    columns_query = """
        SELECT
            table_name,
            column_name,
            data_type,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            is_nullable
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = ANY(%s)
        ORDER BY table_name, ordinal_position
    """
    constraints_query = """
        SELECT
            source.relname AS table_name,
            source_column.attname AS column_name,
            constraint_info.contype AS constraint_type,
            target.relname AS foreign_table_name,
            target_column.attname AS foreign_column_name
        FROM pg_catalog.pg_constraint AS constraint_info
        JOIN pg_catalog.pg_class AS source
          ON source.oid = constraint_info.conrelid
        JOIN pg_catalog.pg_namespace AS source_schema
          ON source_schema.oid = source.relnamespace
        JOIN LATERAL unnest(constraint_info.conkey) WITH ORDINALITY
          AS source_key(attnum, position) ON TRUE
        JOIN pg_catalog.pg_attribute AS source_column
          ON source_column.attrelid = source.oid
         AND source_column.attnum = source_key.attnum
        LEFT JOIN pg_catalog.pg_class AS target
          ON target.oid = constraint_info.confrelid
        LEFT JOIN LATERAL unnest(constraint_info.confkey) WITH ORDINALITY
          AS target_key(attnum, position)
          ON target_key.position = source_key.position
        LEFT JOIN pg_catalog.pg_attribute AS target_column
          ON target_column.attrelid = target.oid
         AND target_column.attnum = target_key.attnum
        WHERE source_schema.nspname = current_schema()
          AND source.relname = ANY(%s)
          AND constraint_info.contype IN ('p', 'f', 'u')
        ORDER BY source.relname, source_column.attnum, constraint_info.contype
    """

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET LOCAL statement_timeout = '5s'")
            with conn.cursor() as cursor:
                table_names = sorted(ALLOWED_TABLES)
                cursor.execute(columns_query, (table_names,))
                columns = [dict(row) for row in cursor.fetchall()]
                cursor.execute(constraints_query, (table_names,))
                constraints = cursor.fetchall()

    if not columns:
        raise RuntimeError(
            "No approved application tables were found in the current database schema."
        )

    column_constraints: dict[tuple[object, object], list[str]] = {}
    for constraint in constraints:
        key = (constraint["table_name"], constraint["column_name"])
        constraint_types = {
            "p": "primary key",
            "f": "foreign key",
            "u": "unique",
        }
        description = constraint_types[str(constraint["constraint_type"])]
        if constraint["constraint_type"] == "f":
            description += (
                f" -> {constraint['foreign_table_name']}."
                f"{constraint['foreign_column_name']}"
            )
        column_constraints.setdefault(key, []).append(description)

    for column in columns:
        key = (column["table_name"], column["column_name"])
        column["constraints"] = column_constraints.get(key, [])

    return columns


def query_database(
    question: str,
    max_attempts: int = 3,
) -> tuple[str, list[dict[str, object]]]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    feedback = ""
    last_error: Exception | None = None
    for _ in range(max_attempts):
        sql = question_to_sql(question, feedback)
        try:
            rows = execute_readonly_sql(sql)
            return sql, rows
        except SQLValidationError as exc:
            last_error = exc
            feedback = (
                f"Your previous query was rejected: {exc}\n"
                f"Rejected query: {sql}\n"
                "Rewrite it as a single SELECT using JOINs instead of subqueries."
            )
        except psycopg.Error as exc:
            last_error = exc
            logger.exception("Generated SQL failed in PostgreSQL: %s", sql)
            feedback = (
                "Your previous query failed in PostgreSQL. "
                f"Database error: {exc}\n"
                f"Rejected query: {sql}\n"
                "Correct the SQL and return one read-only SELECT."
            )

    raise SQLQueryError(
        f"Could not generate an executable read-only query after {max_attempts} attempts."
    ) from last_error


if __name__ == "__main__":
    sql, rows = query_database("Show order 1 for customer 333")
    print("Generated SQL:", sql)
    print("Rows:", rows)