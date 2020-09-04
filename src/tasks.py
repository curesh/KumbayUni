import time
from src.app import create_app
from rq import get_current_job
from src.models import Task, get_db_connection
import sys
from src.anon import Anon
from src.upload_youtube import upload

app = create_app()
app.app_context().push()
def _set_task_progress(progress):
    job = get_current_job()
    if job:
        job.meta['progress'] = progress
        job.save_meta()
        if progress >= 100:
            conn = get_db_connection()
            c = conn.execute("DELETE FROM tasks WHERE task_id = ?", (job.get_id(),))
            #c = conn.execute("INSERT INTO lectures WHERE task_id = ?", (
            conn.commit()
            conn.close()

def anonymize_video(user_id, load_file, save_file, meta_video):
    # Things you need to do in task
    # Anonymize the video and save it in the processed folder
    # Upload the anonymized video to youtube using oath
    # Insert the hyperlink of this video into the links table
    # Insert this video into the lectures table
    # Delete the video?
    
    # Flag to check if databse connection is open
    open_db = False
    try:
        _set_task_progress(0)
        anon_obj = Anon(load_file)
        anon_obj.anon_static()
        # WARNING: You are rewriting the original video file here
        anon_obj.save_vid(save_file)
        link_hash = upload(save_file, meta_video)
        conn = get_db_connection()
        open_db = True
        c = conn.execute("INSERT INTO links (title, link_hash, user_id) VALUES (?, ?, ?)",
                         (meta_video[0], link_hash, user_id)
                         )
        open_db = False
        conn.commit()
        conn.close()
        
    except:
        if open_db:
            conn.close()
        app.logger.error('Unhandled exception', exc_info=sys.exc_info())
        return None
    finally:
        _set_task_progress(100)
