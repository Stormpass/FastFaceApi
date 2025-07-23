import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
import faiss
import sys
import numpy as np

from database import get_db_connection, create_table, get_all_users
from faiss_index import create_faiss_index
from api_routes import setup_routes
from config import FAISS_INDEX_PATH, HOST, PORT

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Application startup")
    yield
    # Shutdown
    print("Application shutdown")

app = FastAPI(lifespan=lifespan)


# Initialize the database
print("Initializing database connection...")
conn = get_db_connection()
if conn is not None:
    if create_table():
        print("Database table initialized successfully.")
    else:
        print("Warning: Database table initialization failed.")
else:
    print("Error: Could not create a database connection. The application will exit.")
    import sys
    sys.exit(1)

# Load or create the FAISS index
print("Loading FAISS index...")
try:
    index = faiss.read_index(FAISS_INDEX_PATH)
    # Get the number of vectors in the index
    num_vectors = index.ntotal
    print(f"Successfully loaded FAISS index with {num_vectors} vectors.")
except (RuntimeError, IOError) as e:
    print(f"Could not load FAISS index file ({str(e)}), creating a new one.")
    # If the index file does not exist or is corrupted, create a new one
    users = get_all_users()
    if users:
        try:
            # Create an array of embedding vectors from database users
            embeddings = np.array([np.frombuffer(user[2], dtype='float32') for user in users])
            user_ids = np.array([user[0] for user in users])  # Use the actual IDs from the database
            
            # Check the validity of the embedding vectors
            if embeddings.size == 0 or len(embeddings.shape) != 2:
                raise ValueError("Invalid embedding vector data.")
                
            index = create_faiss_index(embeddings, user_ids)
            faiss.write_index(index, FAISS_INDEX_PATH)
            print(f"Successfully created a new FAISS index with {len(users)} users.")
        except Exception as ex:
            print(f"Error creating FAISS index: {ex}")
            # Create an empty index as a fallback
            index = faiss.IndexFlatL2(512)  # Assuming embedding vector dimension is 512
            index = faiss.IndexIDMap(index)
            print("Created an empty FAISS index due to an error.")
    else:
        # If there are no users in the database, create an empty index
        index = faiss.IndexFlatL2(512)  # Assuming embedding vector dimension is 512
        index = faiss.IndexIDMap(index)
        print("No users in the database, created an empty FAISS index.")

# Set up API routes
setup_routes(app, index)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)