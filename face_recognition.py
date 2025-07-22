import insightface
import numpy as np
from PIL import Image
import cv2

# 加载人脸识别模型
model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(image: Image.Image):
    """_summary_

    Args:
        image (Image.Image): _description_

    Returns:
        _type_: _description_
    """
    # 将 PIL Image 转换为 cv2 格式
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 获取人脸信息
    faces = model.get(img_cv)
    
    if len(faces) == 0:
        return None
    
    # 返回第一个检测到的人脸的嵌入向量
    return faces[0].embedding