import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
import faiss
import asyncio

from database import get_db_connection, create_table
from api_routes import setup_routes
from config import HOST, PORT, FACE_EMBEDDING_DIMENSION
from update_faiss_index import update_index_from_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Application startup")
    # 初始化空的FAISS索引
    index = faiss.IndexFlatL2(FACE_EMBEDDING_DIMENSION)
    app.state.faiss_index = faiss.IndexIDMap(index)
    print("Initialized an empty FAISS index in memory.")

    # 初始化索引就绪状态标志
    app.state.INDEX_IS_READY = False
    print("Index is not ready yet.")

    # 启动后台任务来构建索引
    print("Scheduling FAISS index rebuild in the background.")
    asyncio.create_task(update_index_from_db(app))

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

# Set up API routes
# 将 index 的传递方式改为从 app.state 获取
setup_routes(app)

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)