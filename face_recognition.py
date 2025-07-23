import insightface
import numpy as np
from PIL import Image
import cv2

# Load the face recognition model
model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(image: Image.Image):
    """
    Extracts the face embedding from an image.

    Args:
        image (Image.Image): The input image in PIL format.

    Returns:
        numpy.ndarray: The face embedding vector, or None if no face is detected.
    """
    # Convert PIL Image to cv2 format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Get face information
    faces = model.get(img_cv)
    
    if len(faces) == 0:
        return None
    
    # Get the embedding from the first detected face
    embedding = faces[0].embedding

    # L2 normalization
    norm = np.linalg.norm(embedding)
    normalized_embedding = embedding / norm

    return normalized_embedding.astype(np.float32)