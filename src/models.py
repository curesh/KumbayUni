import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
class User(UserMixin):
    def __init__(self, user_id, name, email, password, active = True):
        self.id = user_id
        self.name = name
        self.email = email
        self.password_hash = password
        self.active = active

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

def get_db_connection():
    conn = sqlite3.connect('database/database.db')
    conn.row_factory = sqlite3.Row
    return conn


# @login.user_loader
# def load_user(user_id):
#     if not user_id:
#         return None
#     conn = get_db_connection()
#     c = conn.execute("SELECT * from users where user_id = (?)", [int(user_id)])
#     userrow = c.fetchone()
#     userid = userrow[0] # or whatever the index position is
#     if not userid:
#         return None
#     u = User(userid, userrow[1], userrow[2], userrow[3])
#     conn.close()
#     return u
