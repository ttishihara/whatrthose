import os
import cv2
import tensorflow as tf
from werkzeug import secure_filename
from wtforms import SubmitField
from flask_wtf.file import FileField, FileRequired, FileAllowed
from flask_wtf import FlaskForm
from app import application
from .helpers import *
from .config import *
from fastai.vision import Path, load_learner, open_image
from flask import render_template, redirect, url_for, flash
from flask import send_from_directory, request
from app.models.frcnn_detector import frcnn
from app.download_weights import *

# Download Weights
print('Downloading weights...')
weight_path = 'app/models/frcnn_detector/weights'
weight_name = 'model_frcnn_vgg2.hdf5'
assert os.path.isdir(weight_path), f'{weight_path} does not exist, please create'
download_frcnn_weights(os.path.join(weight_path,weight_name))
print('Done!')


# Load Detector Model
assert os.path.isfile(os.path.join(weight_path,weight_name)), f'File was not downloaded, please retry'
detector_weights = "app/models/frcnn_detector/weights/model_frcnn_vgg2.hdf5"
detector = frcnn.detector_model()
detector.load_model(detector_weights) 
global_graph = tf.get_default_graph() 

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
        # Check if it is a POST request and if it is valid.
        # Upload_destination = s3_upload(file.file_selector, bucket, 'images')
        source_filename = secure_filename(file.file_selector.data.filename)
        # e.g. '.png', '.jpg'
        source_extension = os.path.splitext(source_filename)[1]
        destination_filename = uuid4().hex + source_extension
        destination = os.path.join('app/static/img/tmp', destination_filename)
        file.file_selector.data.save(destination)

        classifier_path = Path("app/models/cnn_classifier/")
        classifier = load_learner(classifier_path)

        #################################################
        # Detector Model Start 
        # Detector model saves image to save_path for classfiier to read
        img_data = cv2.imread(destination)

        with global_graph.as_default():
            img_paths = detector.predict(img_data, destination)
        img_paths.append(destination)
        
        # Detector Model End 
        #################################################
        classifier_outputs = []

        for path in img_paths:
            img = open_image(path)
            pred_class, pred_idx, outputs = classifier.predict(img)
            classifier_outputs.append([max(outputs), pred_class, pred_idx, outputs])
        classifier_outputs.sort(key = lambda x: x[0], reverse=True)

        print(classifier_outputs)

        pred_class = classifier_outputs[0][1]
        pred_idx = classifier_outputs[0][2]
        outputs = classifier_outputs[0][3]

        classes = ['Air_Force_1', 'Air_Max_1', 'Air_Max_90', 'Air_Jordan_1']

        # If probability of classifying the image is less than 92%, ask user to
        # Resubmit a different picture.
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


@application.route('/about', methods=['GET'])
def about():
    """
    About Us Page: Renders about.html where introduces the company and the team
    """
    return render_template("about.html")


@application.route('/results', methods=['POST'])
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
