import uvicorn
from fastapi import FastAPI
import faiss

from database import create_connection, create_table, get_all_users
from faiss_index import create_faiss_index
from api_routes import setup_routes
import numpy as np

app = FastAPI()

DATABASE = "users.db"
FAISS_INDEX = "face_index.bin"

# 初始化
conn = create_connection(DATABASE)
if conn is not None:
    create_table(conn)
else:
    print("Error! cannot create the database connection.")

# 加载或创建 FAISS 索引
try:
    index = faiss.read_index(FAISS_INDEX)
    # 获取索引中的向量数量
    num_vectors = index.ntotal
    print(f"Loaded FAISS index with {num_vectors} vectors.")
except RuntimeError:
    # 如果索引文件不存在, 则创建一个新的
    users = get_all_users(conn)
    if users:
        # 替换原来的embeddings生成代码
        embeddings = np.array([np.frombuffer(user[2], dtype='float32') for user in users])
        user_ids = np.array([user[0] for user in users])  # 使用数据库中的实际ID
        index = create_faiss_index(embeddings, user_ids)  # 需要修改create_faiss_index函数
        faiss.write_index(index, FAISS_INDEX)
        print("Created a new FAISS index.")
    else:
        # 如果数据库中没有用户, 创建一个空的索引
        index = faiss.IndexFlatL2(512)  # 假设嵌入向量维度为 512
        index = faiss.IndexIDMap(index)
        print("Created an empty FAISS index.")

# 设置API路由
setup_routes(app, index)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)