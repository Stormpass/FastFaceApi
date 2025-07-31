# FastFaceAPI :camera_flash:

A high-performance face recognition system built with InsightFace and FAISS, providing RESTful APIs for face registration and search. Now with optimized database connection pool and enhanced error handling.

## Features :sparkles:

- :mag: Face embedding extraction using InsightFace
- :rocket: Fast similarity search with FAISS
- :floppy_disk: SQLite database with connection pooling
- :arrows_counterclockwise: Batch registration and real-time recognition
- :chart_with_upwards_trend: System status monitoring API
- :recycle: Automatic database reconnection
- :shield: Enhanced error handling and logging

## Tech Stack :computer:

- Python 3.11+
- FastAPI (Web Framework)
- InsightFace (Face Recognition)
- FAISS (Similarity Search)
- SQLite (Database)

## Quick Start :zap:

1. Clone the repository
```bash
git clone https://github.com/Stormpass/FastFaceApi.git
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the service
```bash
python main.py
```

## API Documentation :bookmark_tabs:

### Register a Face
`POST /register`
- **Parameters**:
  - `username` (string): User identifier
  - `image` (file): Face image file (jpg/png)
- **Response**:
  - `user_id`: Registered user ID
  - `message`: Registration result message

### Search for Similar Faces
`POST /search`
- **Parameters**:
  - `image` (file): Query face image file (jpg/png)
- **Response**:
  - `username`: Matched user's name
  - `similarity`: Match similarity score (0-1)
  - `process_time`: Processing time in seconds

### Delete a User
`DELETE /user-delete`
- **Parameters**:
  - `username` (string): User identifier to delete
- **Response**:
  - `status`: Operation status (success/error)
  - `message`: Result message
  - `processing_time`: Processing time in seconds

### Get Users (Paginated)
`GET /user-list`
- **Parameters**:
  - `pageNo` (integer, optional, default: 1): Page number
  - `pageSize` (integer, optional, default: 10): Number of users per page
- **Response**:
  - `total`: Total number of users
  - `pageNo`: Current page number
  - `pageSize`: Users per page
  - `users`: List of users, each with `id`, `username`, and `created_at`

## License :page_facing_up:

MIT License