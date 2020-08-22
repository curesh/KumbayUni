import sqlite3

connection = sqlite3.connect('database.db')

with open('schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()
cur.execute("INSERT INTO links (title, link_hash) VALUES (?, ?)",
            ('Physics MIT  Lecture 1', 'wWnfJ0-xXRE')
            )

cur.execute("INSERT INTO links (title, link_hash) VALUES (?, ?)",
            ('Physics MIT Lecture 2', 'GtOGurrUPmQ')
            )

cur.execute("INSERT INTO lectures (title, file_name) VALUES (?, ?)",
            ("Video Test 1", "test1.mp4")
            )

cur.execute("INSERT INTO lectures (title, file_name) VALUES (?, ?)",
            ("Video Test 2", "test2.mp4")
            )

connection.commit()
connection.close()
