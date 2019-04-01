import os

from flask import Flask

application = Flask(__name__)
application.secret_key = os.urandom(24)

from app import routes
