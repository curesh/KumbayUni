from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename
import os
from src.forms import LoginForm, RegistrationForm
from src.config import Config
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, current_user, login_user, logout_user, UserMixin, login_required
from werkzeug.urls import url_parse
from src.models import get_db_connection, User
from redis import Redis
import rq
from google.oauth2 import service_account
import googleapiclient.discovery

# _________INIT___________
def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.redis = Redis.from_url(app.config['REDIS_URL'])
    app.task_queue = rq.Queue('src-tasks', connection=app.redis)
    return app

app = create_app(Config)

login = LoginManager(app)
login.login_view = 'login'

def create_service_account(project_id, name, display_name):
    """Creates a service account."""

    credentials = service_account.Credentials.from_service_account_file(
        filename=os.environ['GOOGLE_APPLICATION_CREDENTIALS'],
        scopes=['https://www.googleapis.com/auth/cloud-platform'])

    service = googleapiclient.discovery.build(
        'iam', 'v1', credentials=credentials)

    my_service_account = service.projects().serviceAccounts().create(
        name='projects/' + project_id,
        body={
            'accountId': name,
            'serviceAccount': {
                'displayName': display_name
            }
        }).execute()

    print('Created service account: ' + my_service_account['email'])
    return my_service_account

@login.user_loader
def load_user(user_id):
    if not user_id:
        return None
    conn = get_db_connection()
    c = conn.execute("SELECT * FROM users WHERE user_id = (?)", [int(user_id),])

    userrow = c.fetchone()
    conn.close()
    if userrow is None:
        return None
    userid = userrow[0] # or whatever the index position is

    if not userid:
        return None
    u = User(userid, userrow[1], userrow[2], userrow[3])
    if u:
        u.set_password(u.password_hash)
    return u

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        conn = get_db_connection()
        uid = conn.execute("SELECT user_id FROM users WHERE username = (?)", [form.username.data,]).fetchone()
        conn.close()
        if uid:
            uid = uid[0]
        user = load_user(uid)
        if not user or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(url_for('index'))
    
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        conn = get_db_connection()
        conn.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                     (form.username.data, form.email.data, form.password.data)
                     )
        conn.commit()
        conn.close()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/')
def index():
    conn = get_db_connection()
    links = conn.execute('SELECT * FROM links').fetchall()
    conn.close()
    return render_template('index.html', links=links)

def get_link(link_id):
    conn = get_db_connection()
    link = conn.execute('SELECT * FROM links WHERE link_id = ?',
                        (link_id,)).fetchone()
    conn.close()
    if link is None:
        abort(404)
    return link

@app.route('/<int:link_id>')
def link(link_id):
    link = get_link(link_id)
    return render_template('link.html', link=link)

app.config['UPLOAD_FOLDER'] = "/Users/bigboi01/Documents/CSProjects/kumbayuni/static/assets/test_data/vids"
# Maximum upload size is 100 mB
app.config['MAX_CONTENT_PATH'] = 100000000
app.config['ALLOWED_VID_EXTENSIONS'] = ["MP4", "MKV", "MOV", "WMV", "AVI"]

def allowed_images(filename):
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config['ALLOWED_VID_EXTENSIONS']:
        return True
    else:
        return False
    
@app.route('/create', methods=('GET', 'POST'))
@login_required
def create():
    if request.method == 'POST':
        f = request.files['file']
        print("f ", f)
        if not f:
            flash('File is required')
        title = request.form['title']
        load_file = os.path.join(app.config['UPLOAD_FOLDER'], "original")
        save_file = os.path.join(app.config['UPLOAD_FOLDER'], "processed", secure_filename(f.filename)[:-3]+"avi")
        print("filename: ", f.filename)
        if not title:
            flash('Title is required!')
        elif f.filename == "":
            flash('Video must have filename')
        elif not allowed_images(f.filename):
            flash('That video extension is not allowed')
        elif os.stat(load_file).st_size > app.config['MAX_CONTENT_PATH']:
            flash('Video is too large. Please, split into two videos.')
        else:
            curr_path = os.getcwd()
            os.chdir(load_file)
            f.save(secure_filename(f.filename))
            os.chdir(curr_path)
            print("File uploaded succesfully")
            load_file = os.path.join(load_file, secure_filename(f.filename))
            conn = get_db_connection()
            try:
                conn.execute('INSERT INTO lectures (file_name) VALUES (?)',
                             (f.filename,))
            except:
                print("Error in inserting lecture into database. That lecture is already there (UNIQUE failed). -CS")

            finally:
                conn.commit()
                conn.close()
                
            if current_user.get_task_in_progress('anonymize_video'):
                flash('A video is already being uploaded and anonymized')
            else:
                current_user.launch_task('anonymize_video', 'Uploading video...', load_file, save_file)

            return redirect(url_for('index'))
    return render_template('create.html')

# @app.route('/anon_vid')
# @login_required
# def anon_vid():
#     return redirect(url_for('index.html'))
