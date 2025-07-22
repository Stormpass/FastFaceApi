# FastFaceAPI :camera_flash:

A high-performance face recognition system built with InsightFace and FAISS, providing RESTful APIs for face registration and search.

## Features :sparkles:

- :mag: Face embedding extraction using InsightFace
- :rocket: Fast similarity search with FAISS
- :floppy_disk: SQLite database for user management
- :arrows_counterclockwise: Batch registration and real-time recognition

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
`POST /register/`
- **Parameters**:
  - `username` (string): User identifier
  - `image` (file): Face image file (jpg/png)
- **Response**:
  - `user_id`: Registered user ID
  - `status`: Registration result

### Search for Similar Faces
`POST /search/`
- **Parameters**:
  - `image` (file): Query face image file (jpg/png)
- **Response**:
  - `user_id`: Most similar user ID
  - `confidence`: Match confidence score

## License :page_facing_up:

MIT License