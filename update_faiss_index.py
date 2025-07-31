import asyncio
import logging
import math
import numpy as np
from database import get_db_connection

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def update_index_from_db(app):
    """
    异步地从数据库中分批读取用户数据并更新FAISS索引。
    这个函数应该在FastAPI应用的后台任务中运行。
    """
    logger.info("开始后台重建FAISS索引...")
    
    batch_size = 1000  # 每次从数据库读取的记录数
    
    try:
        # 使用项目共享的数据库连接
        conn = get_db_connection()
        if not conn:
            logger.error("无法获取数据库连接，索引重建失败。")
            app.state.INDEX_IS_READY = False
            return

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        total_users = cur.fetchone()[0]

        if total_users == 0:
            logger.info("数据库中没有用户，无需构建索引。")
            app.state.INDEX_IS_READY = True
            return

        num_batches = math.ceil(total_users / batch_size)
        logger.info(f"总用户数: {total_users}, 将分 {num_batches} 批处理.")

        for i in range(num_batches):
            offset = i * batch_size
            logger.info(f"正在处理批次 {i+1}/{num_batches} (offset: {offset})...")
            cur.execute("SELECT id, embedding FROM users LIMIT ? OFFSET ?", (batch_size, offset))
            users = cur.fetchall()
            
            if not users:
                break

            # 从数据库记录中提取ID和embeddings
            user_ids = np.array([user['id'] for user in users])
            embeddings = np.array([np.frombuffer(user['embedding'], dtype='float32') for user in users])
            
            # 确保embeddings是二维的
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(-1, app.state.faiss_index.d)

            # 将数据添加到内存中的FAISS索引
            app.state.faiss_index.add_with_ids(embeddings, user_ids)
            
            # 在处理大任务时给其他协程运行的机会，防止阻塞事件循环
            await asyncio.sleep(0.01)

        app.state.INDEX_IS_READY = True
        logger.info("FAISS索引后台重建完成。")

    except Exception as e:
        logger.error(f"FAISS索引重建过程中发生错误: {e}", exc_info=True)
        app.state.INDEX_IS_READY = False # 标记为未就绪

def run_update_in_background(app):
    """
    启动后台任务来更新索引。
    """
    logger.info("安排FAISS索引重建的后台任务...")
    asyncio.create_task(update_index_from_db(app))