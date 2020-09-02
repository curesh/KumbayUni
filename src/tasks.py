import time
from src.app import create_app
from rq import get_current_job
from src.models import Task, get_db_connection
import sys
from src.anon import Anon

# def example(seconds):
#     job = get_current_job()
#     print('Starting task')
#     for i in range(seconds):
#         job.meta['progress'] = 100.0 * i / seconds
#         job.save_meta()
#         print(i)
#         time.sleep(1)
#     job.meta['progress'] = 100
#     job.save_meta()
#     print('Task completed')

app = create_app()
app.app_context().push()
def _set_task_progress(progress):
    job = get_current_job()
    if job:
        job.meta['progress'] = progress
        job.save_meta()
        if progress >= 100:
            conn = get_db_connection()
            c = conn.execute("UPDATE tasks SET complete = ? WHERE task_id = ?", (True, job.get_id()))
            conn.commit()
            conn.close()



def anonymize_video(user_id, load_file, save_file):
    try:
        _set_task_progress(0)
        anon_obj = Anon(load_file)
        
        anon_obj.anon_static()
        # WARNING: You are rewriting the original video file here
        anon.save_vid(save_file)
        
    except:
        app.logger.error('Unhandled exception', exc_info=sys.exc_info())
    finally:
        _set_task_progress(100)
