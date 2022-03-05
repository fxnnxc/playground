# pip install pymysql

import pymysql 


conn=None
cur=None

db_config = {
    "host":"test_ip",
    "user": 'root',
    'password':"password",
    'db':"pythonDB",
    'charset':"utf8"
}
conn = pymysql.connect(**db_config)
cur = conn.cursor()

# Create Table 
sql = "CREATE TABLE IF NOT EXISTS userTable (id char(4), userName char(10), email char(15), birthYear int"
cur.execute(sql)
conn.commit()
conn.close()

# Select 
sql = "SELECT * FROM userTable"
cur = execute(sql)
while (True):
    row = cur.fetchone()
    if row == None:
        break 
    print(row)


