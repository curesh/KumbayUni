DROP TABLE IF EXISTS links;
DROP TABLE IF EXISTS lectures;
DROP TABLE IF EXISTS tasks;
DROP TABLE IF EXISTS users;

CREATE TABLE links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    link_hash TEXT NOT NULL UNIQUE,
    user_id INTEGER NOT NULL,
    FOREIGN KEY (user_id)
    	REFERENCES users (user_id) 
    	    ON DELETE CASCADE 
            ON UPDATE NO ACTION
);

CREATE TABLE lectures (
    lecture_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    file_name TEXT NOT NULL UNIQUE,
    course_num TEXT NOT NULL,
    course_name TEXT NOT NULL,
    term TEXT NOT NULL,
    year INTEGER NOT NULL,
    class_type TEXT NOT NULL,
    description TEXT,
    lecture_num INTEGER,
    user_id INTEGER NOT NULL,
    FOREIGN KEY (user_id)
    	REFERENCES users (user_id) 
    	    ON DELETE CASCADE 
            ON UPDATE NO ACTION
);

CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    university TEXT NOT NULL,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL
);


CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    complete INTEGER,
    user_id INTEGER NOT NULL,
    FOREIGN KEY (user_id)
    	REFERENCES users (user_id)
  	    ON DELETE CASCADE
	    ON UPDATE NO ACTION
);

