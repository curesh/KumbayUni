import sqlite3
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename

def get_db_connection():
    conn = sqlite3.connect('database/database.db')
    conn.row_factory = sqlite3.Row
    return conn

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fjdalfiipyujsal7f34ks7a13lfnfa441ksivnor231'

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

app.config['UPLOAD_FOLDER'] = "assets/test_data/vids"
# Maximum upload size is 100 mB
app.config['MAX_CONTENT_PATH'] = 100000000

@app.route('/create', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        f = request.files['file']
        if not f:
            flash('File is required')
        title = request.form['title']
        if not title:
            flash('Title is required!')
        else:
            f.save(secure_filename(f.filename))
            conn = get_db_connection()
            conn.execute('INSERT INTO lectures (title, file) VALUES (?, ?)',
                         (title, f.filename))
            conn.commit()
            conn.close()
            return redirect(url_for('index'))
    return render_template('create.html')
