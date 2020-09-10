import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
import redis
import rq
from flask import current_app

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

    def launch_task(self, name, description, *args, **kwargs):
        rq_job = current_app.task_queue.enqueue('src.tasks.' + name, self.id,
                                                *args, **kwargs)
        task = Task(task_id=rq_job.get_id(), name=name, description=description,
                    user_id=self.id)
        return task
    
    def get_tasks_in_progress(self):
        curr, conn = get_db_connection()
        curr.execute("SELECT * FROM tasks WHERE user_id = ? AND complete = ?",
                         (self.id, False))
        ret = curr.fetchall()
        conn.close()
        return ret

    def get_task_in_progress(self, name):
        curr, conn = get_db_connection()
        curr.execute("SELECT * FROM tasks WHERE name = ? AND user_id = ? AND complete = ?",
                         (name, self.id, False))
        ret = curr.fetchone()
        conn.close()
        return ret

# def is_open(path):
#     for proc in psutil.process_iter():
#         try:
#             files = proc.open_files()
#             if files:
#                 for _file in files:
#                     if _file.path == path:
#                         return True    
#         except psutil.NoSuchProcess as err:
#             print(err)
#     return False

def get_db_connection():
    conn = sqlite3.connect('database/database.db')
    conn.row_factory = sqlite3.Row
    curr = conn.cursor()
    return curr, conn


class Task():
    def __init__(self, task_id, name, description, user_id, complete=False):
        self.task_id = task_id
        curr, conn = get_db_connection()
        curr.execute("INSERT INTO tasks (task_id, name, description, complete, user_id) VALUES (?, ?, ?, ?, ?)",
                         (task_id, name, description, complete, user_id)
        )
        conn.commit()
        conn.close()

    def get_rq_job(self):
        try:
            rq_job = rq.job.Job.fetch(self.task_id, connection=current_app.redis)
        except (redis.exceptions.RedisError, rq.exceptions.NoSuchJobError):
            return None
        return rq_job

    def get_progress(self):
        job = self.get_rq_job()
        return job.meta.get('progress', 0) if job is not None else 100

