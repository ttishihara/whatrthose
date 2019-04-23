from uuid import uuid4
from flask import flash
from werkzeug.utils import secure_filename
import boto3
import os.path



def flash_errors(form):
    """
    Flashes form errors.
    Choices are 'form-error', 'form-warning', 'form-info' and 'form-success'.
    """
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"%s" % (error), 'form-error')
    return


def s3_upload(source_file, bucket_name='s3://whatrthose', upload_dir=None, acl='public-read'):
    """
    Uploads WTForm File Objects to Amazon S3.
    """
    
    if upload_dir is None:
        upload_dir = "images"

    source_filename = secure_filename(source_file.data.filename)
    source_extension = os.path.splitext(source_filename)[1]  # e.g. '.png', '.jpg'

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
