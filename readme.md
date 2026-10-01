
# RAG Document Question Answering

This is a simple RAG pipeline that lets you ask questions about a document.

RAG means **Retrieval Augmented Generation**. The script reads a document, splits it into chunks, stores the chunks in Pinecone, retrieves the most relevant chunks for your question, and then uses OpenAI to answer based on those chunks.

## What This Project Does

The script can:

- Read a document from your computer
- Extract text from the document
- Split the document into smaller chunks
- Create embeddings for each chunk using OpenAI
- Store the embeddings in Pinecone
- Search for the most relevant chunks when you ask a question
- Generate an answer using only the document context

## Files

Use this file in your project:

```text
rag.py
````

## Install Dependencies

Run this in your terminal:

```bash
pip install openai pinecone python-dotenv pypdf
```

## Environment Variables

Create a `.env` file in the same folder as `rag.py`.

```env
OPENAI_API_KEY=your_openai_api_key_here

PINECONE_INDEX=rag-documents
PINECONE_NAMESPACE=document-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

The Pinecone API key is intentionally left blank inside the script:

```python
PINECONE_API_KEY = ""
```

Add your Pinecone key there when you are ready:

```python
PINECONE_API_KEY = "your_pinecone_api_key_here"
```

## Supported Document Types

The script supports:

* PDF
* TXT
* Markdown
* CSV
* JSON
* HTML

Supported extensions:

```text
.pdf
.txt
.md
.csv
.json
.html
.htm
```

## How To Run

### Ask One Question

```bash
python rag.py path/to/document.pdf --question "What is this document about?"
```

Example:

```bash
python rag.py ./docs/policy.pdf --question "What is the notice period for resignation?"
```

### Interactive Mode

Run:

```bash
python rag.py path/to/document.pdf
```

Then ask questions in the terminal:

```text
Question: What is the main idea of the document?
Question: Who is the document written for?
Question: What are the key requirements?
Question: quit
```

Type `quit` or `exit` to stop.

## Force Re-indexing

If you change the document and want to upload the chunks again, use:

```bash
python rag.py path/to/document.pdf --force
```

```
