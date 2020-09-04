from datetime import datetime
from src.Google import Create_Service
from googleapiclient.http import MediaFileUpload
import os

def upload(path, meta_video):
    CLIENT_SECRET_FILE = os.getcwd() + '/static/assets/client_secret.json'
    # CLIENT_SECRET_FILE = 'client_secret.json'
    API_NAME = 'youtube'
    API_VERSION = 'v3'
    SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
    
    # Get the current date time without microseconds in the appropriate format for youtube uploading
    # upload_date_time = datetime.now().replace(microsecond=0).isoformat() + '.000Z'

    request_body = {
        'snippet': {
            'categoryI': 27,
            'title': meta_video[0],
            'description': meta_video[1],
            'tags': ['Lectures', 'Education']
        },
        'status': {
            'privacyStatus': 'unlisted',
            'selfDeclaredMadeForKids': True, 
        },
        'notifySubscribers': False
    }
    # media_file = MediaFileUpload(os.getcwd() + '/static/assets/test_data/vids/vids_test_load_func/test1.mp4')
    media_file = MediaFileUpload(path)
    print("media_file: ", media_file)
    response_upload = service.videos().insert(
        part='snippet,status',
        body=request_body,
        media_body=media_file
    ).execute()

    print("video_id: ", response_upload.get('id'))

    service.thumbnails().set(
        videoId=response_upload.get('id'),
        media_body=MediaFileUpload('static/assets/test_data/imgs/test1_frame.jpg')
    ).execute()
    
    return response_upload.get('id')
