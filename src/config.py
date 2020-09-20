import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://'    
    ELASTICSEARCH_URL = os.environ.get('ELASTICSEARCH_URL')
