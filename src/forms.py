from flask_wtf import FlaskForm
from wtforms import SelectField, SelectMultipleField, StringField, PasswordField, BooleanField, SubmitField, validators
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo
from src.models import User, get_db_connection

class NonValidatingSelectField(SelectField):
    def pre_validate(self, form):
        pass

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    first_name = StringField('First Name', validators=[DataRequired()])
    last_name = StringField('Last Name', validators=[DataRequired()])
    # university = StringField('University', validators=[DataRequired()])
    university = NonValidatingSelectField(u"University", [],
            choices=[("0", "")],
            description="Select your university",
            render_kw={}, coerce=str)
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField(
        'Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        curr, conn = get_db_connection()
        user = curr.execute("SELECT * FROM users WHERE username = (?)", [username.data,]).fetchone()
        if user is not None:
            raise ValidationError('Please use a different username.')
        conn.close()

    def validate_email(self, email):
        curr, conn = get_db_connection()
        user = curr.execute("SELECT * FROM users WHERE email = (?)", [email.data,]).fetchone()
        if user is not None:
            raise ValidationError('Please use a different email address.')
        conn.close()
