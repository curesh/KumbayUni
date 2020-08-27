DROP TABLE IF EXISTS links;
DROP TABLE IF EXISTS lectures;
DROP TABLE IF EXISTS tasks;
DROP TABLE IF EXISTS users;

CREATE TABLE links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    title TEXT NOT NULL,
    link_hash TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    FOREIGN KEY (user_id)
       REFERENCES users (user_id) 
         ON DELETE CASCADE 
         ON UPDATE NO ACTION
);

CREATE TABLE lectures (
    lecture_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    file_name TEXT NOT NULL UNIQUE
);

CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL
);

/*
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    user_id INTEGER 
);
*/
