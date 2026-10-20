import argparse
import json
import re
from typing import Any, Literal

from sql_lookup import SQLQueryError, query_database
from llm_client import (
    answer_or_verification_notice,
    get_groq_client,
    get_groq_model,
)
from sop_rag import answer_from_sop, search_sop


Route = Literal["policy", "orders", "both", "schema", "policy+schema"]

_SCHEMA_QUESTION = re.compile(
    r"\b(schema|database structure|structure of "
    r"(?:(?:the|your|our|my) )?(?:data\s*base|db)|"
    r"(?:database|db) (?:tables?|columns?|fields?|types?|metadata)|"
    r"(?:tables?|columns?|fields?|data types?|constraints?) "
    r"(?:in|of|for) (?:the )?(?:database|data\s*base|db)|"
    r"(?:tables?|columns?|fields?|data types?|constraints?)"
    r".{0,40}(?:database|data\s*base|db)|"
    r"tables and columns)\b",
    flags=re.IGNORECASE,
)

_SCHEMA_REFUSAL = (
    "I can help with order information and SOP guidance, but I can't provide "
    "internal database structure or system configuration."
)


def _is_schema_question(question: str) -> bool:
    return _SCHEMA_QUESTION.search(question) is not None


def classify_question(question: str) -> Route:
    response = get_groq_client().chat.completions.create(
        model=get_groq_model(),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user's request for a supply-chain assistant. "
                    "Return exactly one label: policy, orders, both, schema, "
                    "or policy+schema. "
                    "Use policy for SOP/company policy/procedure questions. "
                    "Use orders for live customer, order, or order-item database "
                    "facts. Use schema when asked for database structure, tables, "
                    "columns, or their types. Use policy+schema when the question "
                    "asks for both SOP guidance and database structure. Use both "
                    "only when answering requires both SOP information and live "
                    "order data. If uncertain, choose both."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    label = response.choices[0].message.content
    if not label:
        raise RuntimeError("The model returned an empty routing decision.")

    route = label.strip().lower()
    routes: dict[str, Route] = {
        "policy": "policy",
        "orders": "orders",
        "both": "both",
        "schema": "schema",
        "policy+schema": "policy+schema",
    }
    if route not in routes:
        raise ValueError(f"Unexpected routing decision from the model: {route!r}")
    return routes[route]


def _answer_with_database(question: str, rows: list[dict[str, object]]) -> str:
    response = get_groq_client().chat.completions.create(
        model=get_groq_model(),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's database question using only the supplied "
                    "query results. Do not invent missing values. If there are no "
                    "rows, say no matching records were found."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nDatabase query results:\n"
                    f"{json.dumps(rows, default=str)}"
                ),
            },
        ],
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("The model returned an empty database answer.")
    return answer_or_verification_notice(question, answer.strip(), rows)


def _answer_with_both(
    question: str,
    rows: list[dict[str, object]],
    sources: list[dict[str, Any]],
) -> str:
    sop_context = "\n\n".join(
        (
            f"Document: {source['document']}; page: {source['page']}\n"
            f"Text: {source['text']}"
        )
        for source in sources
    )
    response = get_groq_client().chat.completions.create(
        model=get_groq_model(),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the live database results and SOP passages "
                    "below. Clearly distinguish current database facts from policy "
                    "guidance, cite SOP document and page, and do not invent facts. "
                    "If either source lacks needed information, say so."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nLive database results:\n"
                    f"{json.dumps(rows, default=str)}\n\nSOP passages:\n"
                    f"{sop_context or '(No matching SOP passages found.)'}"
                ),
            },
        ],
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("The model returned an empty combined answer.")
    return answer_or_verification_notice(
        question,
        answer.strip(),
        {"database_rows": rows, "sop_passages": sources},
    )


def answer_question(question: str, top_k: int = 5) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if _is_schema_question(question):
        return {"route": "restricted", "answer": _SCHEMA_REFUSAL}

    route = classify_question(question)
    if route in {"schema", "policy+schema"}:
        return {"route": "restricted", "answer": _SCHEMA_REFUSAL}

    if route == "policy":
        sources = search_sop(question, top_k=top_k)
        answer = answer_from_sop(question, sources)
        return {"route": route, "answer": answer, "sources": sources}

    try:
        sql, rows = query_database(question)
    except SQLQueryError:
        return {
            "route": route,
            "answer": (
                "I couldn't retrieve the requested order information. "
                "Please try rephrasing your question."
            ),
        }
    if route == "orders":
        return {
            "route": route,
            "answer": _answer_with_database(question, rows),
            "sql": sql,
            "rows": rows,
        }

    sources = search_sop(question, top_k=top_k)
    return {
        "route": route,
        "answer": _answer_with_both(question, rows, sources),
        "sql": sql,
        "rows": rows,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer questions using SOP documents and order data."
    )
    parser.add_argument("question", help="Question to ask the assistant")
    args = parser.parse_args()

    result = answer_question(args.question)
    print(f"Route: {result['route']}")
    print(f"\n{result['answer']}")

    for source in result.get("sources", []):
        print(
            f"\nSource: {source['document']} "
            f"(page {source['page']}, chunk {source['chunk_number']})"
        )


if __name__ == "__main__":
    main()
