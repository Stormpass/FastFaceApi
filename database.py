import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn

def create_table(conn):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :return:
    """
    try:
        sql_create_users_table = """ CREATE TABLE IF NOT EXISTS users (
                                        id integer PRIMARY KEY,
                                        username text NOT NULL,
                                        embedding blob NOT NULL
                                    ); """
        c = conn.cursor()
        c.execute(sql_create_users_table)
    except Error as e:
        print(e)

def insert_user(conn, username, embedding):
    """_summary_

    Args:
        conn (_type_): _description_
        username (_type_): _description_
        embedding (_type_): _description_

    Returns:
        _type_: _description_
    """
    sql = ''' INSERT INTO users(username,embedding) VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, (username, embedding))
    conn.commit()
    return cur.lastrowid

def get_all_users(conn):
    """_summary_

    Args:
        conn (_type_): _description_

    Returns:
        _type_: _description_
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")

    rows = cur.fetchall()

    return rows