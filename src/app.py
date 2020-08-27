import sqlite3
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename
import os
from forms import LoginForm
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

def get_db_connection():
    conn = sqlite3.connect('database/database.db')
    conn.row_factory = sqlite3.Row
    return conn

app = Flask(__name__)
app.config.from_object(Config)
#app.config['SECRET_KEY'] = 'fjdalfiipyujsal7f34ks7a13lfnfa441ksivnor231'

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for user {}, remember_me={}'.format(
            form.username.data, form.remember_me.data))
        return redirect(url_for('index'))
    
    return render_template('login.html', title='Sign In', form=form)

@app.route('/')
def index():
    conn = get_db_connection()
    links = conn.execute('SELECT * FROM links').fetchall()
    conn.close()
    return render_template('index.html', links=links)

def get_link(link_id):
    conn = get_db_connection()
    link = conn.execute('SELECT * FROM links WHERE id = ?',
                        (link_id,)).fetchone()
    conn.close()
    if link is None:
        abort(404)
    return link

@app.route('/<int:link_id>')
def link(link_id):
    link = get_link(link_id)
    return render_template('link.html', link=link)

app.config['UPLOAD_FOLDER'] = "/Users/bigboi01/Documents/CSProjects/kumbayuni/assets/test_data/vids"
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
def create():
    if request.method == 'POST':
        f = request.files['file']

        print("f ", f)

        if not f:
            flash('File is required')

        title = request.form['title']
        full_file = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)) 
        if not title:
            flash('Title is required!')
        elif f.filename == "":
            flash('Video must have filename')
        elif not allowed_images(f.filename):
            flash('That video extension is not allowed')
        elif os.stat(full_file).st_size > app.config['MAX_CONTENT_PATH']:
            flash('Video is too large. Please, split into two videos.')
        else:
            f.save(full_file)
            conn = get_db_connection()
            conn.execute('INSERT INTO lectures (file_name) VALUES (?)',
                         (f.filename,))
            conn.commit()
            conn.close()
            return redirect(url_for('index'))
    return render_template('create.html')
