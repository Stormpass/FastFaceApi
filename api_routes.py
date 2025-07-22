from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io

from face_recognition import get_face_embedding
from database import create_connection, insert_user, get_all_users
from faiss_index import add_to_faiss_index, search_faiss_index

def setup_routes(app: FastAPI, index):
    @app.post("/register/")
    async def register(username: str = Form(...), image: UploadFile = File(...)):
        try:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            embedding = get_face_embedding(img)

            # 添加检查确保embedding是有效数组
            if not isinstance(embedding, np.ndarray) or embedding.size == 0:
                raise HTTPException(status_code=400, detail="Invalid face embedding.")

            # 检查是否已存在相似人脸
            distances, indices = search_faiss_index(index, embedding)
            print(len(distances))
            print(distances[0])
            if len(indices) > 0:
                print("已存在该人脸信息, 不再新增")
                raise HTTPException(status_code=409, detail="Similar face already exists in the system.")

            conn = create_connection("users.db")
            user_id = insert_user(conn, username, embedding.tobytes())
            add_to_faiss_index(index, np.array([embedding]), np.array([user_id]))

            return JSONResponse(content={"message": f"User {username} registered successfully.", "user_id": user_id})
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    @app.post("/search/")
    async def search(image: UploadFile = File(...)):
        try:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            query_embedding = get_face_embedding(img)

            if query_embedding is None:
                raise HTTPException(status_code=400, detail="No face detected in the image.")

            distances, indices = search_faiss_index(index, query_embedding)

            if len(indices) == 0:
                return JSONResponse(content={"message": "No similar faces found."})

            user_id = int(indices[0])
            conn = create_connection("users.db")
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()

            if result:
                return JSONResponse(content={"username": result[0]})
            else:
                return JSONResponse(content={"message": "User not found in database."})
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)