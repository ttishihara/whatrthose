from app import application
from .helpers import *
from .config import *

from fastai.vision import Path, load_learner, open_image

from flask import render_template, redirect, url_for, flash, send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from werkzeug import secure_filename
import os

class UploadFileForm(FlaskForm):
    """Class for uploading file when submitted"""
    file_selector = FileField('File', validators=[FileRequired(),
                                                  FileAllowed(['jpg', 'jpe', 'jpeg', 'png', 'svg', 'gif', 'bmp'],
                                                              "Image files only!")]
                              )
    submit = SubmitField('Submit')


@application.route('/favicon.ico')
def favicon():
    """Display favicon in browser tab"""
    return send_from_directory(os.path.join(application.root_path, 'static'),
                              'favicon.ico', mimetype='image/vnd.microsoft.icon')


@application.route('/index', methods=['GET', 'POST'])
@application.route('/', methods=['GET', 'POST'])
def index():
    """Index Page : Renders index.html where users can upload files"""

    file = UploadFileForm()  # file : UploadFileForm class instance
    if file.validate_on_submit():  # Check if it is a POST request and if it is valid.
        # upload_destination = s3_upload(file.file_selector, bucket, 'images')
        # print(upload_destination)

        path = Path("app/models/cnn_classifier/")
        classifier = load_learner(path)

        img = open_image(file.file_selector.data)
        pred_class, pred_idx, outputs = classifier.predict(img)
        print(str(pred_class).replace("_", " "))
        print(max(outputs))

        return redirect(url_for('index'))  # Redirect to / (/index) page.
    else:
        flash_errors(file)
    return render_template("index.html",  form=file)

@application.route('/about', methods=['GET'])
def about():
    """
    About Us Page: Renders about.html where introduces the company and the team
    """
    return render_template("about.html")
