from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse
import faiss
import numpy as np
from PIL import Image
import io
import time
from config import FACE_RECOGNITION_THRESHOLD
from face_recognition import get_face_embedding
from database import get_db_connection, insert_user, get_all_users, check_username_exists, get_user_by_username, delete_user_by_username, get_users_paginated, get_user_by_id
from faiss_index import add_to_faiss_index, search_faiss_index, remove_from_faiss_index

def setup_routes(app: FastAPI):
    @app.post("/register")
    async def register(request: Request, username: str = Form(...), image: UploadFile = File(...)):
        if not request.app.state.INDEX_IS_READY:
            raise HTTPException(status_code=503, detail="Service is not ready yet, please try again later.")
        
        start_time = time.time()
        index = request.app.state.faiss_index
        try:
            # Validate username
            if not username or not username.strip():
                raise HTTPException(status_code=400, detail="Username cannot be empty")
            
            # Check if username already exists
            if check_username_exists(username):
                raise HTTPException(status_code=409, detail=f"Username '{username}' already exists")
                
            # Read and process the image
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            embedding = get_face_embedding(img)

            # Add a check to ensure the embedding is a valid array
            if not isinstance(embedding, np.ndarray) or embedding.size == 0:
                raise HTTPException(status_code=400, detail="Could not detect face or extract features")

            # Check for similar faces
            distances, indices = search_faiss_index(index, embedding)
            print(f"distances is {distances[0]}")
            if len(indices) > 0 and distances[0] < FACE_RECOGNITION_THRESHOLD:
                print(f"Similar face already exists, distance: {distances[0]}, not adding new user")
                raise HTTPException(status_code=409, detail="A similar face already exists in the system")


            # Use a shared database connection to insert the user
            user_id = insert_user(username=username, embedding=embedding.tobytes())
            if user_id is None:
                raise HTTPException(status_code=500, detail="Database insertion failed")
                
            # Update the FAISS index in memory
            add_to_faiss_index(index, np.array([embedding]), np.array([user_id]))

            process_time = time.time() - start_time
            print(f"User {username} registered successfully, processing time: {process_time:.2f} seconds")
            return JSONResponse(
                status_code=200,
                content={
                    "code": 200,
                    "message": "success",
                    "detail": {"user_id": user_id, "username": username}
                }
            )
        except HTTPException as he:
            # Pass through HTTP exceptions
            print(f"HTTP exception: {he.detail}")
            return JSONResponse(
                status_code=he.status_code,
                content={"code": he.status_code, "message": he.detail, "detail": None}
            )
        except Exception as e:
            # Log detailed error information
            import traceback
            error_details = traceback.format_exc()
            print(f"Error processing request: {str(e)}\n{error_details}")
            return JSONResponse(
                status_code=500,
                content={"code": 500, "message": f"Error processing request: {str(e)}", "detail": None}
            )

    @app.get("/user-list")
    async def get_users(pageNo: int = 1, pageSize: int = 10):
        try:
            conn = get_db_connection()
            if conn is None:
                raise HTTPException(status_code=500, detail="Database connection failed")
            
            data = get_users_paginated(pageNo, pageSize, conn)
            if data is None:
                raise HTTPException(status_code=500, detail="Failed to retrieve user data")
            
            # Convert database rows to a list of dictionaries
            users_list = [dict(user) for user in data['users']]
            
            return JSONResponse(
                status_code=200,
                content={
                    "code": 200,
                    "message": "success",
                    "detail": {
                        "total": data['total'],
                        "pageNo": pageNo,
                        "pageSize": pageSize,
                        "users": users_list
                    }
                }
            )
        except HTTPException as he:
            return JSONResponse(
                status_code=he.status_code,
                content={"code": he.status_code, "message": he.detail, "detail": None}
            )
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error getting user list: {str(e)}\n{error_details}")
            return JSONResponse(
                status_code=500,
                content={"code": 500, "message": f"Error getting user list: {str(e)}", "detail": None}
            )

    @app.post("/search")
    async def search(request: Request, image: UploadFile = File(...)):
        if not request.app.state.INDEX_IS_READY:
            raise HTTPException(status_code=503, detail="Service is not ready yet, please try again later.")

        start_time = time.time()
        index = request.app.state.faiss_index
        try:
            # Read and process the image
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            query_embedding = get_face_embedding(img)

            if query_embedding is None:
                raise HTTPException(status_code=400, detail="No face detected in the image")

            # Search for similar faces
            distances, indices = search_faiss_index(index, query_embedding)
            print(f"distances is {distances[0]}")
            if len(indices) == 0 or distances[0] > FACE_RECOGNITION_THRESHOLD:
                process_time = time.time() - start_time
                print(f"No similar faces found, processing time: {process_time:.2f} seconds")
                return JSONResponse(
                    status_code=404,
                    content={"code": 404, "message": "No similar faces found", "detail": None}
                )
            
            # Get the user ID and similarity of the most similar face
            user_id = int(indices[0])
            
            # Use a shared database connection to query user information
            conn = get_db_connection()
            if conn is None:
                raise HTTPException(status_code=500, detail="Database connection failed")
                
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()

            process_time = time.time() - start_time
            if result:
                return JSONResponse(
                    status_code=200,
                    content={
                        "code": 200,
                        "message": "success",
                        "detail": {
                            "username": result[0],
                            "distance": str(distances[0]),
                            "process_time": round(process_time, 2)
                        }
                    }
                )
            else:
                print(f"User ID {user_id} not found in the database, processing time: {process_time:.2f} seconds")
                return JSONResponse(
                    status_code=404,
                    content={"code": 404, "message": "Matching user not found in the database", "detail": None}
                )
        except HTTPException as he:
            # Pass through HTTP exceptions
            print(f"HTTP exception: {he.detail}")
            return JSONResponse(
                status_code=he.status_code,
                content={"code": he.status_code, "message": he.detail, "detail": None}
            )
        except Exception as e:
            # Log detailed error information
            import traceback
            error_details = traceback.format_exc()
            print(f"Error processing request: {str(e)}\n{error_details}")
            return JSONResponse(
                status_code=500,
                content={"code": 500, "message": f"Error processing request: {str(e)}", "detail": None}
            )
            
    @app.delete("/user-delete")
    async def delete_user(request: Request, username: str):
        if not request.app.state.INDEX_IS_READY:
            raise HTTPException(status_code=503, detail="Service is not ready yet, please try again later.")

        start_time = time.time()
        index = request.app.state.faiss_index
        try:
            # Validate username
            if not username or not username.strip():
                raise HTTPException(status_code=400, detail="Username cannot be empty")

            # Get user info before deleting from DB
            user = get_user_by_username(username)
            if not user:
                raise HTTPException(status_code=404, detail=f"User '{username}' not found")

            user_id_to_delete = user['id']

            # Delete user from database
            if not delete_user_by_username(username):
                raise HTTPException(status_code=500, detail="Failed to delete user from database")

            # Remove from FAISS index
            remove_from_faiss_index(index, user_id_to_delete)

            process_time = time.time() - start_time
            print(f"User {username} deleted successfully, processing time: {process_time:.2f} seconds")
            return JSONResponse(
                status_code=200,
                content={
                    "code": 200,
                    "message": "success",
                    "detail": {"username": username}
                }
            )
        except HTTPException as he:
            print(f"HTTP exception: {he.detail}")
            return JSONResponse(
                status_code=he.status_code,
                content={"code": he.status_code, "message": he.detail, "detail": None}
            )
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error deleting user: {str(e)}\n{error_details}")
            return JSONResponse(
                status_code=500,
                content={"code": 500, "message": f"Error deleting user: {str(e)}", "detail": None}
            )