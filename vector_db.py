import argparse
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

class VectorDatabase:
    def __init__(self, model_name='all-mpnet-base-v2', storage_path='vector_db/'):
        self.model = SentenceTransformer(model_name)
        self.storage_path = storage_path
        self.embeddings = None
        self.metadata = []
        self.index = None

        # Create storage directory if it does not exist
        os.makedirs(self.storage_path, exist_ok=True)

        # Try to load existing database if available
        self.load()

    def create(self, file_path, metadata=None):
        """Create an embedding for the text content of a file and add it to the database."""
        with open(file_path, 'r') as file:
            text = file.read()

        embedding = self.model.encode([text])[0]
        self.metadata.append(metadata or {})
        self._add_embedding(embedding)
        self._save()  # Automatically save changes
        print(f"Entry created for file: {file_path}")

    def _add_embedding(self, embedding):
        """Add an embedding to the FAISS index."""
        if self.embeddings is None:
            self.embeddings = np.array([embedding])
            d = embedding.shape[0]
            self.index = faiss.IndexFlatL2(d)
        else:
            self.embeddings = np.vstack((self.embeddings, embedding))

        self.index.add(np.array([embedding]))

    def read(self, query_file_path, top_k=5):
        """Search for similar entries in the database using the text content of a file."""
        if self.index is None:
            print("No index found. Please load the database.")
            return

        with open(query_file_path, 'r') as file:
            query = file.read()

        query_embedding = self.model.encode([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        results = [(self.metadata[i], distances[0][j]) for j, i in enumerate(indices[0])]
        for result in results:
            print(f"Metadata: {result[0]}, Distance: {result[1]}")

    def update(self, text_id, new_file_path, new_metadata=None):
        """Update an existing text entry using the content from a new file."""
        if text_id >= len(self.metadata):
            print("Invalid text ID.")
            return

        with open(new_file_path, 'r') as file:
            new_text = file.read()

        # Recompute embedding for the updated text
        new_embedding = self.model.encode([new_text])[0]
        self.embeddings[text_id] = new_embedding
        self.metadata[text_id] = new_metadata or {}

        # Rebuild the FAISS index
        self._rebuild_index()
        self._save()  # Automatically save changes
        print(f"Entry {text_id} updated with file: {new_file_path}")

    def delete(self, text_id):
        """Delete an entry by its ID."""
        if text_id >= len(self.metadata):
            print("Invalid text ID.")
            return

        self.embeddings = np.delete(self.embeddings, text_id, axis=0)
        del self.metadata[text_id]

        # Rebuild the FAISS index
        self._rebuild_index()
        self._save()  # Automatically save changes
        print(f"Entry {text_id} deleted.")

    def _rebuild_index(self):
        """Rebuild the FAISS index after an update or delete operation."""
        if self.embeddings is not None and len(self.embeddings) > 0:
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(self.embeddings)
        else:
            self.index = None

    def _save(self):
        """Save embeddings, metadata, and index to disk."""
        if self.embeddings is not None:
            np.save(os.path.join(self.storage_path, 'embeddings.npy'), self.embeddings)
        with open(os.path.join(self.storage_path, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f)
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(self.storage_path, 'index.faiss'))
        print("Database saved to disk.")

    def load(self):
        """Load embeddings, metadata, and index from disk."""
        embeddings_path = os.path.join(self.storage_path, 'embeddings.npy')
        metadata_path = os.path.join(self.storage_path, 'metadata.json')
        index_path = os.path.join(self.storage_path, 'index.faiss')

        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
        else:
            self.embeddings = None

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = None

        print("Database loaded from disk.")

def parse_args():
    parser = argparse.ArgumentParser(description='Vector Database CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for 'create'
    parser_create = subparsers.add_parser('create', help='Create a new entry from a text file')
    parser_create.add_argument('file_path', type=str, help='Path to the text file')
    parser_create.add_argument('--metadata', type=json.loads, default='{}', help='Metadata as JSON')

    # Subparser for 'read'
    parser_read = subparsers.add_parser('read', help='Read/search entries using a query file')
    parser_read.add_argument('query_file_path', type=str, help='Path to the query text file')
    parser_read.add_argument('--top_k', type=int, default=5, help='Number of top results to return')

    # Subparser for 'update'
    parser_update = subparsers.add_parser('update', help='Update an existing entry using a new text file')
    parser_update.add_argument('text_id', type=int, help='ID of the text to update')
    parser_update.add_argument('new_file_path', type=str, help='Path to the new text file')
    parser_update.add_argument('--new_metadata', type=json.loads, default='{}', help='Updated metadata as JSON')

    # Subparser for 'delete'
    parser_delete = subparsers.add_parser('delete', help='Delete an entry')
    parser_delete.add_argument('text_id', type=int, help='ID of the text to delete')

    # Subparser for 'load'
    parser_load = subparsers.add_parser('load', help='Load the database from disk')

    return parser.parse_args()

def main():
    args = parse_args()
    db = VectorDatabase()

    if args.command == 'create':
        db.create(args.file_path, args.metadata)
    elif args.command == 'read':
        db.read(args.query_file_path, args.top_k)
    elif args.command == 'update':
        db.update(args.text_id, args.new_file_path, args.new_metadata)
    elif args.command == 'delete':
        db.delete(args.text_id)
    elif args.command == 'load':
        db.load()
    else:
        print("Unknown command")

if __name__ == "__main__":
    main()

