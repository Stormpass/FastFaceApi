import faiss
import numpy as np

def create_faiss_index(embeddings, ids=None):
    """Creates a FAISS index.
    
    Args:
        embeddings: An array of embedding vectors.
        ids: An optional array of IDs. If None, 0-based indices are used.
    
    Returns:
        A FAISS index object.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(index)
    if ids is None:
        ids = np.arange(embeddings.shape[0])
    index.add_with_ids(embeddings, ids)
    return index

def add_to_faiss_index(index, new_embeddings, new_ids):
    """Adds new vectors to an existing index.
    
    Args:
        index: The existing FAISS index.
        new_embeddings: An array of new embedding vectors.
        new_ids: An array of IDs corresponding to the new vectors.
    """
    index.add_with_ids(new_embeddings, new_ids)

def search_faiss_index(index, query_embedding, k=1):
    """Searches for similar vectors in the index.
    
    Args:
        index: The FAISS index.
        query_embedding: The query vector (single).
        k: The number of nearest neighbors to return.
    
    Returns:
        A tuple of (distances array, index ID array).
    """
    # Convert single vector to a 2D array
    query = np.array([query_embedding])
    distances, indices = index.search(query, k)
    return distances[0], indices[0]

def remove_from_faiss_index(index, id_to_remove):
    """Removes a vector with a specified ID from the FAISS index.
    
    Args:
        index: The FAISS index (must be of type IndexIDMap).
        id_to_remove: The ID to be removed.
    
    Returns:
        bool: True if removal is successful, False otherwise.
    """
    try:
        if not isinstance(index, faiss.IndexIDMap):
            print("Error: Index must be of type IndexIDMap")
            return False
            
        # Convert a single ID to an array
        if isinstance(id_to_remove, (int, np.integer)):
            ids_to_remove = np.array([id_to_remove], dtype=np.int64)
        else:
            ids_to_remove = np.array(id_to_remove, dtype=np.int64)
            
        index.remove_ids(ids_to_remove)
        return True
    except Exception as e:
        print(f"Error removing ID from FAISS index: {e}")
        return False