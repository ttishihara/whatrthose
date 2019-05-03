import base64
import os
from werkzeug import secure_filename
from werkzeug.datastructures import FileStorage
from wtforms import SubmitField
from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask_wtf import FlaskForm
from app import application
from .helpers import *
from .config import *
from flask import render_template, redirect, url_for, flash
from flask import send_from_directory, request, jsonify


class UploadFileForm(FlaskForm):
    """
    Class for Flask-WTF form for uploading photo.
    """
    file_selector = FileField('File',
                              validators=[FileRequired(),
                                          FileAllowed(['jpg', 'jpe',
                                                       'jpeg', 'png',
                                                       'svg', 'gif',
                                                       'bmp'],
                                                      "Image files only!")]
                              )
    submit = SubmitField('Submit')


@application.route('/favicon.ico')
def favicon():
    """Display favicon in browser tab"""
    return send_from_directory(os.path.join(application.root_path, 'static'),
                               'favicon.ico',
                               mimetype='image/vnd.microsoft.icon')


@application.route('/index', methods=['GET', 'POST'])
@application.route('/', methods=['GET', 'POST'])
def index():
    """Index Page : Renders index.html where users can upload files"""

    file = UploadFileForm()  # File : UploadFileForm class instance
    if file.validate_on_submit():
        # upload_destination = s3_upload(file.file_selector, bucket, 'images')
        destination_filename = save_photo(file.file_selector.data)
        destination = os.path.join('app/static/img/tmp', destination_filename)
        pred_class, pred_idx, outputs = classify_photo(pic=file.file_selector.data, destination=destination)

        classes = ["Addidas_Dame_5", "Addidas_Harden", "Addidas_Superstar",
                   "Addidas_Ultraboost",
                   "Nike_Air_Force_1", "Nike_Air_Max_1", "Nike_Air_Max_90",
                   "Nike_Air_Jordan_1"]

        # If probability of classifying the image is less than 92%, ask user to
        # resubmit a different picture.
        if max(outputs) < 0.92:
            print(f"{pred_class}: {max(outputs)}")
            flash(
                "We are unsure about What Those R. Please try another image.",
                "form-warning"
            )
            return redirect(url_for('index'))

        else:
            return render_template('results.html',
                                   pred_class=str(
                                       pred_class).replace('_', ' '),
                                   pred_prob=round(
                                       max(outputs).item()*100, 4),
                                   img=os.path.join(
                                       'img/tmp',
                                       destination_filename)
                                   )
    else:
        flash_errors(file)
    return render_template("index.html",  form=file)


@application.route('/webcam_submit', methods=['POST'])
def webcam_submit():
    """Handles photo submissions via webcam with POST method."""

    # Base64 string of image.
    pic_64 = request.form['file'].partition('base64,')[2]

    # Convert base64 string to bytes object.
    pic = base64.b64decode(pic_64)

    # Save bytes object to storage and predict.
    destination_filename = save_photo(pic)
    destination = os.path.join('app/static/img/tmp', destination_filename)
    pred_class, pred_idx, outputs = classify_photo(destination=destination)

    # If probability of classifying the image is less than 92%, ask user to
    # resubmit a different picture.
    if max(outputs) < 0.92:
        print(f"{pred_class}: {max(outputs)}")
        flash(
            "We are unsure about What Those R. Please try another image.",
            "form-warning"
        )
        return jsonify({"redirect": url_for('index')})

    else:
        return jsonify({"results":
                        url_for('results',
                                pred_class=str(pred_class).replace('_', ' '),
                                pred_prob=round(max(outputs).item()*100, 4),
                                img=os.path.join(
                                    'img/tmp',
                                    destination_filename)
                                )
                        })


@application.route('/about', methods=['GET'])
def about():
    """
    About Us Page: Renders about.html where introduces the company and the team
    """
    return render_template("about.html")


@application.route('/results', methods=['GET', 'POST'])
def results():
    """
    Results Page: Renders results.html where users see
    the results of their search
    """
    pred_class = request.args.get("pred_class")
    pred_prob = request.args.get("pred_prob")
    img = request.args.get("img")

    return render_template("results.html", pred_class=pred_class,
                           pred_prob=pred_prob, img=img)
