# Supply Chain SOP and Order Assistant

This project is a command-line assistant for answering supply-chain questions
using two sources:

1. WHO supply-chain and inventory SOP documents.
2. Live customer, order, and order-item data in PostgreSQL.

The assistant routes each question to the appropriate source, asks a language
model to produce an answer grounded in that source, and runs a separate
verification pass before presenting the answer.

## What has been done so far

- Added three source SOP PDFs under [`documents/`](./documents/).
- Built notebook-based steps for extracting and cleaning PDF text,
  chunking that text, creating embeddings, and building a FAISS vector index.
- Saved extracted text under [`extracted_text/`](./extracted_text/), chunks
  under [`chunks/`](./chunks/), and the searchable index and metadata under
  [`vector_db/`](./vector_db/).
- Implemented SOP retrieval and answer generation in [`sop_rag.py`](./sop_rag.py).
- Implemented natural-language-to-SQL generation, validation, and read-only
  PostgreSQL querying in [`sql_lookup.py`](./sql_lookup.py).
- Implemented request routing, combined answers, the command-line interface,
  and a refusal for schema/configuration questions in [`router.py`](./router.py).
- Added a shared Groq client and evidence-based answer verification in
  [`llm_client.py`](./llm_client.py).
- Documented the PostgreSQL tables and the intended database access controls in
  [`DATABASE_SCHEMA.md`](./DATABASE_SCHEMA.md).

This repository currently provides a command-line workflow; it does not include
a web interface or API server.

## System flow

```mermaid
flowchart TD
    U[User runs router.py with a question] --> R[Validate question]
    R --> S{Looks like a schema or system-configuration question?}
    S -->|Yes| REF[Return fixed refusal]
    S -->|No| C[Groq classifies the question]

    C -->|policy| P[Embed question with all-MiniLM-L6-v2]
    P --> F[Search FAISS index for relevant SOP chunks]
    F --> PG[Groq drafts answer from SOP passages]
    PG --> PV[Groq verifies answer against those passages]
    PV --> PO[Return answer and source citations]

    C -->|orders| QSQL[Groq generates SQL]
    C -->|both| QSQL
    QSQL --> VAL[Parse and validate SQL]
    VAL -->|Invalid| RETRY[Retry SQL generation with validation feedback]
    RETRY --> QSQL
    VAL -->|Valid| DB[Execute read-only PostgreSQL query]
    DB -->|orders| OG[Groq drafts answer from query rows]
    OG --> OV[Groq verifies answer against query rows]
    OV --> OO[Return answer and query results]

    DB -->|both| BP[Embed question with all-MiniLM-L6-v2]
    BP --> BF[Search FAISS index for relevant SOP chunks]
    BF --> BG[Groq drafts answer from rows and SOP passages]
    BG --> BV[Groq verifies answer against both sources]
    BV --> BO[Return answer, query results, and SOP citations]

    PV -->|Unsupported or malformed verification| NOTICE[Return verification notice]
    OV -->|Unsupported or malformed verification| NOTICE
    BV -->|Unsupported or malformed verification| NOTICE
    DB -->|Query fails after retries| ERR[Return order retrieval error]
```

The classifier chooses among `policy`, `orders`, and `both`. A direct check in
the router catches common schema questions before classification; schema or
system-configuration requests classified later as `schema` are refused too.

### SOP document preparation

The notebook workflow prepares the searchable SOP knowledge base:

```text
documents/*.pdf
    │
    ▼
PDF text extraction and cleaning
    │
    ▼
extracted_text/*.txt
    │
    ▼
Text chunking
    │
    ▼
chunks/sop_chunks.json
    │
    ▼
Sentence-transformer embeddings and FAISS index
    │
    ├── vector_db/sop_faiss.index
    └── vector_db/sop_metadata.json
```

At runtime, [`sop_rag.py`](./sop_rag.py) loads the FAISS index and metadata,
embeds the question, and retrieves the nearest passages. The language model is
instructed to answer only from those passages and cite the document and page.
The loaded index and embedding model are cached for subsequent questions in the
same process.

### Order-data flow and safeguards

For an order-related question, [`sql_lookup.py`](./sql_lookup.py) asks the
language model to generate one PostgreSQL `SELECT` over the approved
`customers`, `orders`, and `order_items` tables. Before execution, the SQL is
parsed and checked to ensure that it:

- Contains only one `SELECT` statement and no comments, nested selects, or CTEs.
- Reads only from the approved tables.
- Has a row limit no greater than 100; a missing limit is set to 100.
- Runs in a read-only transaction with a five-second statement timeout.

Invalid SQL or a PostgreSQL query error is sent back to the model as feedback,
for up to three attempts by default. These application checks are defense in
depth: configure `DATABASE_URL` to use a database role with only the permissions
the assistant needs. Do not rely on generated-SQL validation as a substitute
for database permissions.

The user-facing router does not expose the internal schema lookup helper.
Schema questions are refused rather than sent to that helper.

### Answer verification

For SOP, database, and combined answers, [`llm_client.py`](./llm_client.py)
asks the model to verify the draft against the exact passages or rows used to
create it. An answer is returned only when the verifier produces the expected
supported result. If it rejects the answer or returns malformed output, the
assistant displays a verification notice instead. This can reduce unsupported
claims but is not a replacement for human review, especially for high-impact
decisions.

## Project structure

```text
.
├── documents/                 Source SOP PDFs
├── extracted_text/            Extracted and cleaned text
├── chunks/                    Chunked SOP text
├── vector_db/                 FAISS index and passage metadata
├── Data_Extracting.ipynb      PDF extraction and cleaning workflow
├── Data_Chunking.ipynb        Text chunking workflow
├── Embeddings.ipynb           Embedding workflow
├── FAISS_Vector_DB.ipynb      FAISS search and RAG experiments
├── router.py                  CLI entry point and request routing
├── sop_rag.py                 SOP retrieval and answer generation
├── sql_lookup.py              SQL generation, validation, and DB access
├── llm_client.py              Groq client and answer verification
└── DATABASE_SCHEMA.md         Database schema and access notes
```

## Configuration

Python code loads environment variables from a local `.env` file in the project
directory. Configure these variables locally:

```dotenv
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://user:password@host:port/database
# Optional; defaults to openai/gpt-oss-120b
GROQ_MODEL=openai/gpt-oss-120b
```

Do not commit `.env` or place real credentials in documentation. The
PostgreSQL database should contain the tables described in
[`DATABASE_SCHEMA.md`](./DATABASE_SCHEMA.md), and the configured account should
have suitably restricted, read-only access.

Install the runtime and notebook packages in your environment. For example:

```bash
python -m pip install groq "psycopg[binary]" sqlglot python-dotenv \
  sentence-transformers faiss-cpu pypdf numpy
```

The checked-in [`requirements_venv.txt`](./requirements_venv.txt) is primarily
an environment/package snapshot and may not include every package imported by
the runtime modules and notebooks.

The first SOP search may download the `all-MiniLM-L6-v2` model if it is not
already available in the local model cache.

## Run the assistant

Run commands from the project directory:

```bash
python router.py "What is the inventory ordering procedure?"
python router.py "Show the status of order 123 for customer 45"
python router.py "What does the SOP say about ordering, and what is the status of order 123?"
```

The command-line program prints the selected route and the answer, followed by
SOP source document/page/chunk details when applicable. When called from Python,
`answer_question` also includes generated SQL and matching rows for order
routes, and includes both database results and SOP sources for combined
answers.

The router can also be imported and called from Python:

```python
from router import answer_question

result = answer_question("What is the inventory ordering procedure?")
print(result["route"])
print(result["answer"])
```

## Database tables

The application expects these tables:

- `customers`: customer identifiers and customer details.
- `orders`: customer-linked orders, status, date, and total amount.
- `order_items`: products, quantities, and unit prices linked to orders.

The full columns, types, and relationships are listed in
[`DATABASE_SCHEMA.md`](./DATABASE_SCHEMA.md).
