# Vector Database
This project implements a simple vector database using FAISS (Facebook AI Similarity Search) and Sentence Transformers to store, search, update, and delete embeddings derived from text files. The database stores the embeddings, metadata, and a FAISS index for efficient similarity search. You can interact with the database through a command-line interface (CLI).

## Installation
You can install the dependencies using pip:
```Bash
pip install torch sentence-transformers faiss-cpu numpy
```

## Create

Add a new text entry by converting its content into embeddings and storing it in the database.
```python
python vector_db.py create /content/cartoon.txt --metadata '{"Topic": "Cartoon"}'
python vector_db.py create /content/dl.txt --metadata '{"Topic": "Deep learning"}'
python vector_db.py create /content/ml.txt --metadata '{"Topic": "Machine learning"}'
```

## Read

Perform a similarity search using a query file and retrieve top k most similar entries.
```python
python vector_db.py read /content/query.txt --top_k 3
```

## Update

Update the text content and metadata for an existing entry.
```python
python vector_db.py update 3 --new_metadata '{"Topic": "Query"}' /content/query.txt
```

## Delete

Delete an entry by its ID.
```python
python vector_db.py delete 3
```



