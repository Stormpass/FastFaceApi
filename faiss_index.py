import faiss
import numpy as np

def create_faiss_index(embeddings, ids=None):
    """创建FAISS索引
    
    Args:
        embeddings: 嵌入向量数组
        ids: 可选的ID数组，如果为None则使用0-based索引
    
    Returns:
        FAISS索引对象
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(index)
    if ids is None:
        ids = np.arange(embeddings.shape[0])
    index.add_with_ids(embeddings, ids)
    return index

def add_to_faiss_index(index, new_embeddings, new_ids):
    """向现有索引添加新向量
    
    Args:
        index: 现有FAISS索引
        new_embeddings: 新嵌入向量数组
        new_ids: 新向量对应的ID数组
    """
    index.add_with_ids(new_embeddings, new_ids)

def search_faiss_index(index, query_embedding, k=1):
    """在索引中搜索相似向量
    
    Args:
        index: FAISS索引
        query_embedding: 查询向量(单个)
        k: 返回的最近邻数量
    
    Returns:
        (距离数组, 索引ID数组)
    """
    # 将单向量转换为2D数组
    query = np.array([query_embedding])
    distances, indices = index.search(query, k)
    return distances[0], indices[0]