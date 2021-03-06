from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename
import os
from src.forms import LoginForm, RegistrationForm
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, current_user, login_user, logout_user, UserMixin, login_required
from werkzeug.urls import url_parse
from src.models import get_db_connection, User
from src import app
from google.oauth2 import service_account
import googleapiclient.discovery
from zipfile import ZipFile
import shutil
from elasticsearch import Elasticsearch

login = LoginManager(app)
login.login_view = 'login'

@login.user_loader
def load_user(user_id):
    if not user_id:
        return None
    curr, conn = get_db_connection()
    curr.execute("SELECT * FROM users WHERE user_id = (?)", [int(user_id),])

    userrow = curr.fetchone()
    conn.close()
    if userrow is None:
        return None
    userid = userrow[0] # or whatever the index position is

    if not userid:
        return None
    u = User(userid, userrow[4], userrow[5], userrow[6])
    if u:
        u.set_password(u.password_hash)
    return u

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        curr, conn = get_db_connection()
        uid = curr.execute("SELECT user_id FROM users WHERE username = (?)", [form.username.data,]).fetchone()
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
    form = RegistrationForm(request.form)
    if request.method == "POST":
        print("request: ", request.form['university'])
        if str(request.form['university']) == "0":
            flash("Please select a university")
            return render_template('register.html', title='Register', form=form)
        if form.validate_on_submit():
            print("validating: ")
            curr, conn = get_db_connection()
            curr.execute("INSERT INTO users (first_name, last_name, university, username, email, password_hash) VALUES (?, ?, ?, ?, ?, ?)",
                        (form.first_name.data, form.last_name.data, request.form['university'], form.username.data, form.email.data, form.password.data)
                        )
            conn.commit()
            conn.close()
            flash('Congratulations, you are now a registered user!')
            return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/')
def index():
    curr, conn = get_db_connection()
    links = curr.execute('SELECT * FROM links').fetchall()
    conn.close()
    return render_template('index.html', links=links)

def get_link(link_id):
    curr, conn = get_db_connection()
    link = curr.execute('SELECT * FROM links WHERE link_id = ?',
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
    
@app.route('/create', methods=('GET', 'POST'), )
@login_required
def create():
    if request.method == 'POST':
        print("request forms: ", request.form)
        load_file = os.path.join(app.config['UPLOAD_FOLDER'], "original")
        
        if 'NextButton' in request.form:
            # Check all input fields for errors
            if not request.files['file']:
                flash('File is required')
            elif not request.form['course_num']:
                flash('Course number is required')
            elif not request.form['course_name']:
                flash('Course name is required')
            else:
                # Actual Code
                f = request.files['file']
                load_file = os.path.join(app.config['UPLOAD_FOLDER'], "original")
                f.save(os.path.join(load_file, secure_filename(f.filename)))                
                # Handles zip file or video file submission
                if request.form["FileType"] == "Compressed zip file":
                    try:
                        with ZipFile(os.path.join(load_file,secure_filename(f.filename)), 'r') as zipObj:
                            zipObj.extractall(load_file)
                        os.remove(os.path.join(load_file,secure_filename(f.filename)))
                    except:
                        if os.path.exists(os.path.join(load_file, secure_filename(f.filename))):
                            os.remove(os.path.join(load_file, secure_filename(f.filename)))
                        flash('Please submit a zip file')
                        return render_template('create.html', file_names=None)
                
                    
                file_names = []
                incorrect_input = False
                curr, conn = get_db_connection()
                curr.execute("SELECT user_id FROM users WHERE username = (?)", [current_user.name,])
                user_id_curr = curr.fetchone()[0]
                for i, file_name in enumerate(os.listdir(load_file)):
                    if file_name == "__MACOSX":
                        shutil.rmtree(os.path.join(load_file,file_name))
                        continue
                    if not allowed_images(file_name) or file_name == "" or os.stat(os.path.join(load_file, file_name)).st_size > app.config['MAX_CONTENT_PATH']:
                        print("Incorrect_input: ", file_name)
                        incorrect_input = True
                        break
                    file_names.append(file_name)
                    print("Inserting: ", file_name)
                    curr.execute("INSERT INTO lectures (file_name, course_num, course_name, term, year, class_type, description, lecture_num, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (file_name, request.form['course_num'], request.form['course_name'], request.form['Term'], int(request.form['Year']), 
                                request.form['Instruction'], request.form['description'], i+1, user_id_curr))
                    conn.commit()
                if incorrect_input:
                    for f_remove in os.listdir(load_file):
                        print("removing from lectures")
                        try:
                            os.remove(os.path.join(load_file, f_remove))
                        except:
                            print("Couldn't remove ", f_remove)
                        curr.execute("DELETE FROM lectures WHERE file_name = (?)", (f_remove,))
                        flash("Remember to upload video files (mp4, mkv, mov, avi, wmv) that are less than 100 MB in size")
                    conn.commit()
                    conn.close()
                else:
                    conn.close()
                    return render_template('create.html', file_names=file_names)
        else:
            # If Submit button is clicked
            # TODO: Check how many videos there are in sql lecture table, to find out if its zip or not
            print("Nextbutton is not clicked")
            curr, conn = get_db_connection()
            num_lectures = len(curr.execute("SELECT * FROM lectures").fetchall())
            if num_lectures > 1:
                updated_order = request.form.getlist('lectures[]')
                print("updated order: ", updated_order)
                for i, lec_name in enumerate(updated_order):
                    #TODO Get the correct sqlite command for updating lectures table
                    curr.execute("UPDATE lectures SET lecture_num = (?) WHERE file_name = (?)", (i+1, lec_name))
            else:
                curr.execute("UPDATE lectures SET lecture_num = (?) WHERE lecture_id = (?)", (request.form['lecture_num'], 1))
            conn.commit()
            conn.close()
            save_file = os.path.join(app.config['UPLOAD_FOLDER'], "processed")
            if current_user.get_task_in_progress('anonymize_video'):
                flash('A video is already being uploaded and anonymized')
            else:
                print("task launching -CS")
                current_user.launch_task('anonymize_video', 'Uploading video...', load_file, save_file)
            return redirect(url_for('index'))

    return render_template('create.html')

# @app.route('/anon_vid')
# @login_required
# def anon_vid():
#     return redirect(url_for('index.html'))
if __name__ == '__main__':
    app.run()