import sqlite3
from sqlite3 import Error
import threading
import time
from config import DB_FILE

# Global variables for storing the database connection
_db_connection = None
_db_file = DB_FILE  # Default database filename
_connection_lock = threading.Lock()
_max_retry_attempts = 3
_retry_delay = 1  # Retry delay in seconds

def get_db_connection(db_file=None):
    """
    Gets the shared database connection. If the connection does not exist or is closed, a new one is created.
    Includes automatic reconnection logic.
    
    :param db_file: The path to the database file. If None, the default value is used.
    :return: A Connection object or None.
    """
    global _db_connection, _db_file
    
    if db_file is not None:
        _db_file = db_file
    
    with _connection_lock:
        # Check if the connection exists
        if _db_connection is None:
            return _create_new_connection()
        
        # Check if the connection is valid
        try:
            _db_connection.execute("SELECT 1")
            return _db_connection
        except (sqlite3.Error, sqlite3.ProgrammingError, sqlite3.OperationalError):
            print("Database connection lost, attempting to reconnect...")
            return _create_new_connection()

def _create_new_connection():
    """
    Creates a new database connection with retry logic.
    """
    global _db_connection, _db_file
    
    for attempt in range(_max_retry_attempts):
        try:
            # Close the old connection if it exists
            if _db_connection is not None:
                try:
                    _db_connection.close()
                except Exception:
                    pass
            
            # Create a new connection, allowing use in a multi-threaded environment
            _db_connection = sqlite3.connect(_db_file, check_same_thread=False)
            
            # Enable foreign key constraints
            _db_connection.execute("PRAGMA foreign_keys = ON")
            
            # Configure the connection to return rows as dictionaries
            _db_connection.row_factory = sqlite3.Row
            
            print(f"Successfully created a database connection to {_db_file}")
            return _db_connection
        except Error as e:
            print(f"Connection attempt {attempt+1}/{_max_retry_attempts} failed: {e}")
            if attempt < _max_retry_attempts - 1:
                time.sleep(_retry_delay)
    
    print(f"Could not connect to the database {_db_file} after {_max_retry_attempts} attempts")
    return None

def create_connection(db_file):
    """ 
    A function retained for backward compatibility that uses the shared connection pool.
    
    :param db_file: The database file.
    :return: A Connection object or None.
    """
    return get_db_connection(db_file)

def create_table(conn=None):
    """ 
    Creates the users table if it does not exist.
    
    :param conn: A Connection object. If None, the shared connection is used.
    :return: True on success, False on failure.
    """
    if conn is None:
        conn = get_db_connection()
        if conn is None:
            return False
    
    try:
        sql_create_users_table = """ CREATE TABLE IF NOT EXISTS users (
                                        id integer PRIMARY KEY,
                                        username text NOT NULL,
                                        embedding blob NOT NULL,
                                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                    ); """
        c = conn.cursor()
        c.execute(sql_create_users_table)
        conn.commit()
        return True
    except Error as e:
        print(f"Error creating table: {e}")
        return False

def insert_user(conn=None, username=None, embedding=None):
    """
    Inserts a new user into the database.
    
    :param conn: A Connection object. If None, the shared connection is used.
    :param username: The username.
    :param embedding: The face embedding vector (in binary format).
    :return: The ID of the newly inserted user, or None on failure.
    """
    if username is None or embedding is None:
        print("Error: Username and embedding cannot be None")
        return None
        
    if conn is None:
        conn = get_db_connection()
        if conn is None:
            return None
    
    try:
        sql = ''' INSERT INTO users(username,embedding) VALUES(?,?) '''
        cur = conn.cursor()
        cur.execute(sql, (username, embedding))
        conn.commit()
        return cur.lastrowid
    except Error as e:
        print(f"Error inserting user: {e}")
        return None

def get_all_users(conn=None):
    """
    Gets all users from the database.
    
    :param conn: A Connection object. If None, the shared connection is used.
    :return: A list of users, where each user is a tuple (id, username, embedding, created_at). Returns an empty list on failure.
    """
    if conn is None:
        conn = get_db_connection()
        if conn is None:
            return []
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users")
        rows = cur.fetchall()
        return rows
    except Error as e:
        print(f"Error getting users: {e}")
        return []

def check_username_exists(username, conn=None):
    """
    Checks if a username already exists.
    
    :param username: The username to check.
    :param conn: A Connection object. If None, the shared connection is used.
    :return: True if the username exists, False otherwise.
    """
    if username is None:
        return False
        
    if conn is None:
        conn = get_db_connection()
        if conn is None:
            return False
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
        result = cur.fetchone()
        return result[0] > 0 if result else False
    except Error as e:
        print(f"Error checking username: {e}")
        return False

def get_user_by_id(user_id, conn=None):
    """
    Gets user information by user ID.
    
    :param user_id: The user ID.
    :param conn: A Connection object. If None, the shared connection is used.
    :return: A dictionary of user information, or None if not found.
    """
    if user_id is None:
        return None
        
    if conn is None:
        conn = get_db_connection()
        if conn is None:
            return None
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    except Error as e:
        print(f"Error getting user by ID: {e}")
        return None

def get_user_by_username(username, conn=None):
    """
    Gets user information by username.
    
    :param username: The username.
    :param conn: A Connection object. If None, the shared connection is used.
    :return: A dictionary of user information, or None if not found.
    """
    if username is None:
        return None
        
    if conn is None:
        conn = get_db_connection()
        if conn is None:
            return None
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        return dict(row) if row else None
    except Error as e:
        print(f"Error getting user by username: {e}")
        return None

def get_users_paginated(page_no, page_size, conn=None):
    """
    Gets a paginated list of users.
    
    :param page_no: The page number (starting from 1).
    :param page_size: The size of each page.
    :param conn: A Connection object. If None, the shared connection is used.
    :return: A dictionary containing the total count and the list of users, or None on failure.
    """
    if conn is None:
        conn = get_db_connection()
        if conn is None:
            return None
    
    offset = (page_no - 1) * page_size
    
    try:
        cur = conn.cursor()
        
        # Get the total number of users
        cur.execute("SELECT COUNT(*) FROM users")
        total_users = cur.fetchone()[0]
        
        # Get paginated user data
        cur.execute("SELECT id, username, created_at FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?", (page_size, offset))
        users = cur.fetchall()
        
        return {"total": total_users, "users": users}
    except Error as e:
        print(f"Error getting paginated users: {e}")
        return None

def delete_user_by_username(username, conn=None):
    """
    根据用户名删除用户
    
    :param username: 要删除的用户名
    :param conn: Connection对象，如果为None则使用共享连接
    :return: 成功返回True，失败返回False
    """
    if username is None:
        return False
        
    if conn is None:
        conn = get_db_connection()
        if conn is None:
            return False
    
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        return cur.rowcount > 0  # 返回是否有行被删除
    except Error as e:
        print(f"删除用户时出错: {e}")
        return False