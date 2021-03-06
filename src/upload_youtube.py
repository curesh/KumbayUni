from datetime import datetime
from src.Google import Create_Service
from googleapiclient.http import MediaFileUpload
import os
import socket

# YOUTUBE AUDITORS PLEASE LOOK AT THE COMMENT BELOW
# This function uploads the video to my own private youtube account (using the youtube data api) 
# and then displays the unlisted link on my web application
def upload(path, title, description):
    CLIENT_SECRET_FILE = os.getcwd() + '/static/assets/client_secret.json'
    # CLIENT_SECRET_FILE = 'client_secret.json'
    API_NAME = 'youtube'
    API_VERSION = 'v3'
    SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
    print("get socket timeout: ", socket.getdefaulttimeout())
    # Get the current date time without microseconds in the appropriate format for youtube uploading
    # upload_date_time = datetime.now().replace(microsecond=0).isoformat() + '.000Z'
    request_body = {
        'snippet': {
            'categoryI': 27,
            'title': title,
            'description': description,
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
    try:
        response_upload = service.videos().insert(
            part='snippet,status',
            body=request_body,
            media_body=media_file
        ).execute()
    except socket.timeout:
        print("Socket Timeout -CS")

    print("video_id: ", response_upload.get('id'))

    service.thumbnails().set(
        videoId=response_upload.get('id'),
        media_body=MediaFileUpload(os.getcwd() + '/static/assets/test_data/imgs/test1_frame.jpg')
    ).execute()
    
    return response_upload.get('id')
