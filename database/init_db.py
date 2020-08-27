import sqlite3

connection = sqlite3.connect('database.db')


with open('./schema.sql') as f:
    connection.executescript(f.read())
    
cur = connection.cursor()

cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ('chandra', 'chandra.b.suresh@gmail.com', 'abc')
            )

cur.execute("INSERT INTO links (title, link_hash, user_id) VALUES (?, ?, ?)",
            ('Physics MIT  Lecture 1', 'wWnfJ0-xXRE', 1)
            )

cur.execute("INSERT INTO links (title, link_hash, user_id) VALUES (?, ?, ?)",
            ('Physics MIT Lecture 2', 'GtOGurrUPmQ', 1)
            )

cur.execute("INSERT INTO lectures (file_name) VALUES (?)",
            ("test1.mp4",)
            )

cur.execute("INSERT INTO lectures (file_name) VALUES (?)",
            ("test2.mp4",)
            )

connection.commit()
connection.close()
