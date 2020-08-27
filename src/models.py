from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from src import app
# def set_password(self, password):
#     self.password_hash = generate_password_hash(password)
    
# def check_password(self, password):
#     return check_password_hash(self.password_hash, password)
class User(UserMixin):
    def __init__(self, user_id, name, email, password, active = True):
        self.user_id = user_id
        self.name = name
        self.email = email
        self.password_hash = password
        self.active = active

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login.user_loader
def load_user(user_id):
    if not user_id:
        return None
    conn = app.get_db_connection()
    c = conn.execute("SELECT * from users where user_id = (?)", [int(user_id)])
    userrow = c.fetchone()
    userid = userrow[0] # or whatever the index position is
    if not userid:
        return None
    u = User(userid, userrow[1], userrow[2], userrow[3])
    conn.close()
    return u
