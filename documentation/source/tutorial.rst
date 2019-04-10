Setup Tutorial
==============
The following will walk through the setup, deployment, and execution of the application on a remote server. These steps are performed after the requirements have been met.

Deployment
------------
1. On AWS, create an IAM role:
 * The role should be from EC2 as the trusted entity
 * It should have full S3, ElasticBeanstalk and Lambda access
 * No tags are necessary
 * Name it whatever you want - just remember the name
2. Launch EC2 instance.  Any instance type is fine.  
3. The EC2 should be setup with:
 * Git installed
 * Git user credentials stored with access to repo 
 * Conda installed
 * IAM role created in 1) must be attached (this can be done during launch or after)
 
4. Download the whatrthose repo to your local machine
5. Update user_definition.py in the code folder with:
 * EC2 address (ec2_address)
 * EC2 user (user)
 * PEM Key file (key_file)
 * Git user ID with credentials stored on EC2 (git_user_id)
6. Activate a virtual environment with paramiko installed (whatrthose in environment.yml will work)
7. Run deploy.py in the code directory
