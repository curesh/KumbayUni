import time
from src.app import create_app
from rq import get_current_job
from src.models import Task, get_db_connection
import sys
from src.anon import Anon
from src.upload_youtube import upload
import os
import socket

app = create_app()
app.app_context().push()
def _set_task_progress(progress):
    job = get_current_job()
    if job:
        job.meta['progress'] = progress
        job.save_meta()
        if progress >= 100:
            curr, conn = get_db_connection()
            curr.execute("DELETE FROM tasks WHERE task_id = ?", (job.get_id(),))
            #c = conn.execute("INSERT INTO lectures WHERE task_id = ?", (
            conn.commit()
            conn.close()
#Sample video title:
# University of California, Los Angeles: Electrodynamics, Optics, and Special Relativity
# Physics 1CH Discussion 1 by Trevor Scheopner
# Winter 2020
def get_meta(lecture_row):
    curr, conn = get_db_connection()
    # print("lecture: ", [print(elem) for elem in lecture_row], "lecture after: ")
    curr.execute("SELECT * FROM users WHERE user_id =?", (lecture_row[10],))
    user = curr.fetchone()
    conn.close()
    # title = [university]: [course_name] [class_type] [lecture_num] by [first_name] [last_name]
    title = user[3] + ": " + lecture_row[4] + " " + lecture_row[7] + " " + str(lecture_row[9]) + " by " + user[1] + " " + user[2]
    description = "The official course name is " + lecture_row[3] + ". " + "This course was recorded for the " + lecture_row[5] + " " + str(lecture_row[6]) + " academic term. " + lecture_row[8]
    return title, description

def sort_table(row):
    return row[9]

def anonymize_video(user_id, orig_load_file, orig_save_file):
    # Things you need to do in task
    # Anonymize the video and save it in the processed folder
    # Upload the anonymized video to youtube using oath
    # Insert the hyperlink of this video into the links table
    # Insert this video into the lectures table
    # Delete the video?
    
    # Flag to check if databse connection is open
    open_db = False
    socket.setdefaulttimeout(600)  # set timeout to 10 minutes
    
    try:
        curr, conn = get_db_connection()
        c = curr.execute("SELECT * FROM lectures").fetchall()
        c.sort(key=sort_table)
        conn.close()
        for row in c:
            print("Current anonymization: ", row[2])
            _set_task_progress(0)
            load_file = os.path.join(orig_load_file, row[2])
            save_file = os.path.join(orig_save_file, row[2][:-3]+"mp4")
            title, description = get_meta(row)
            anon_obj = Anon(load_file, save_file)
            anon_obj.anon_static()
            # WARNING: You are rewriting the original video file here
            anon_obj.save_audio()
            link_hash = upload(save_file, title, description)
            open_db = True
            curr, conn = get_db_connection()
            curr.execute("INSERT INTO links (title, description, link_hash, user_id) VALUES (?, ?, ?, ?)",
                            (title, description, link_hash, user_id)
                            )
            curr.execute("DELETE FROM lectures WHERE file_name = (?)", (row[2],))
            open_db = False
            conn.commit()
            conn.close()
        
    except:
        if open_db:
            conn.close()
        app.logger.error('Unhandled exception', exc_info=sys.exc_info())
    finally:
        curr, conn = get_db_connection()
        curr.execute("DELETE FROM lectures")
        conn.commit()
        conn.close()
        filelist = [ f for f in os.listdir(orig_load_file)]
        for f in filelist:
            os.remove(os.path.join(orig_load_file,f))
        _set_task_progress(100)
