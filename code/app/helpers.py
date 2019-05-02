import os.path
from uuid import uuid4

import boto3
from fastai.vision import Path, load_learner, open_image
from flask import flash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage


def flash_errors(form):
    """
    Flashes Flask-WTF form errors.

    :param form: Form object that contains errors.
    :type form: UploadFileForm
    :return: None
    """
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"%s" % (error), 'form-error')


def s3_upload(source_file, bucket_name='s3://whatrthose', upload_dir=None,
              acl='public-read'):
    """
    Uploads data within Flask-WTForm File Objects to Amazon S3.

    :param source_file: File to be uploaded to s3 bucket.
    :type source_file: flask_wtf.file.FileField
    :param bucket_name: Name of s3 bucket.
    :type bucket_name: str
    :param upload_dir: Directory within the s3 bucket to upload the file into.
    :type upload_dir: str
    :param acl: Access Control List.
    :type acl: str

    :return: Location of uploaded s3 object.
    """

    if upload_dir is None:
        upload_dir = "images"

    source_filename = secure_filename(source_file.data.filename)
    source_extension = os.path.splitext(source_filename)[
        1]  # e.g. '.png', '.jpg'

    destination_filename = uuid4().hex + source_extension

    # Connect to s3 and upload file.
    s3 = boto3.client('s3')

    try:
        s3.upload_fileobj(source_file.data,
                          bucket_name,
                          destination_filename,
                          ExtraArgs={
                              "ACL": acl
                          })

    except Exception as e:
        print(e)

    return "s3://" + os.path.join(bucket_name, destination_filename)


def save_photo(pic):
    """
    Stores a picture to the app/static/img/tmp directory with a randomly
    generated name.

    :param pic: Picture of a sneakers to be classified.
    :type pic: werkzeug.datastructures.FileStorage or bytes object

    :return: Filename of saved picture.
    """
    if isinstance(pic, FileStorage):
        source_filename = secure_filename(pic.filename)
        source_extension = os.path.splitext(source_filename)[1]
        destination_filename = uuid4().hex + source_extension
        destination = os.path.join('app/static/img/tmp', destination_filename)
        pic.save(destination)
    elif isinstance(pic, bytes):
        destination_filename = uuid4().hex + '.jpeg'
        destination = os.path.join('app/static/img/tmp', destination_filename)
        with open(destination, 'wb') as f:
            f.write(pic)

    return destination_filename


def classify_photo(pic):
    """
    Feeds a picture through the sneaker detection pipeline.

    :param pic: Picture of a sneakers to be classified.
    :type pic: werkzeug.datastructures.FileStorage or Pathstr

    :return: Prediction class, Prediction class index, Output probabilities.
    """
    img = open_image(pic)
    classifier_path = "app/models/cnn_classifier/"
    classifier = load_learner(classifier_path)
    pred_class, pred_idx, outputs = classifier.predict(img)

    return pred_class, pred_idx, outputs
