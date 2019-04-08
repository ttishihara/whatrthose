Requirements
============
Below list the requirements needed to run the application.

The full Github repository is located at `here <https://github.com/MSDS698/whatrthose>`_ where all necessary files for this application resides.

**Amazon Web Services Setup**


1. An EC2 instance must be stood up with:
  * Git installed and user configured
  * Anaconda installed
  * An IAM role that allows full S3 and ElasticBeanstalk access
2. The following must be configured in code/user_definition.py:
  * ec2_address: The DNS name of the EC2 described above
  * user: Which user to log into the EC2 with
  * key_file: The identity file used to log into the EC2 with
  * git_repo_owner: Owner of the git repo of the application
  * git_repo_name: Name of the git repo of the application
  * git_user_id: User ID configured in the EC2 mentioned above

