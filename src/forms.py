from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, validators
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo
from src.models import User, get_db_connection

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField(
        'Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = (?)", [username.data,]).fetchone()
        if user is not None:
            raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = (?)", [email.data,]).fetchone()
        if user is not None:
            raise ValidationError('Please use a different email address.')
