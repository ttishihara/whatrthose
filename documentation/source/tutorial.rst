Setup Tutorial
==============
The following will walk through the setup, deployment, and execution of the application on a remote server. These steps are performed after the requirements have been met.

Deployment
------------
1. Be sure all the required configurations are met from the requirements page
2. Clone the repo to your local machine
3. Install the conda environment found in the root directory
4. Run "python deploy.py" from the code directory. This will:
  * Stand up an S3 bucket for photo logging
  * Push the current code.zip file to an Elastic Beanstalk deployment
