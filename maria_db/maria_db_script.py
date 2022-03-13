# pip install pymysql
import pymysql 

conn=None
cur=None

db_config = {
    "host":"127.0.0.1",
    "user": 'bumjin',
    'password':"bumjinpark",
    'db':"mysql",
    'charset':"utf8"
}
conn = pymysql.connect(**db_config)
cur = conn.cursor()

# Create Table 
sql = """
        CREATE TABLE IF NOT EXISTS userTable (
            id char(4), 
            userName char(10),
            email char(15),
            birthYear int)
        """
cur.execute(sql)

# INSERT 
sql = """
        INSERT INTO userTable
            (id, userName, email, birthYear) VALUES(%s,%s,%s,%s)
        """

N=10
for id, name, email, birthYear in zip([f"id_{i}" for i in range(N)], 
                                      [f"name_{i}" for i in range(N)], 
                                      [f"email_{i}" for i in range(N)], 
                                      [{i} for i in range(N)]):
    cur.execute(sql, (id, name, email, birthYear))

conn.commit()

# Select 
sql = """
        SELECT * FROM userTable
      """
cur.execute(sql)
while (True):
    row = cur.fetchone()
    if row == None:
        break 
    print(row)


conn.close()
