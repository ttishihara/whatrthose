Setup Tutorial
==============
The following will walk through the setup, deployment, and execution of the application on a remote server. These steps
are performed after the requirements have been met.

Deployment
----------
1. On AWS, create an IAM role:

 * The role should be from EC2 as the trusted entity
 * It should have full S3, ElasticBeanstalk, ElasticLoadBalancing and Lambda access
 * No tags are necessary
 * Name it whatever you want - just remember the name

2. Attach new policy for s3 access to the default AWS elastic beanstalk ec2 IAM role:

 * Create a new policy (Service: S3, Actions: List(All), Read(All), Write(All), Permissions Management(PutObjectAcl))
 * Attach new policy to IAM role aws-elasticbeanstalk-ec2-role

3. Launch EC2 instance. Make sure to add at least 50GB of storage.  Any instance type is fine (Anaconda with Python3 AMI recommended).

4. The EC2 should be setup with:

 * Git installed
 * Git user credentials stored with access to repo 
 * Conda installed
 * IAM role created in 1) must be attached

5. Update user_definition.py in the code folder with:

 * EC2 address (ec2_address)
 * EC2 user (user)
 * PEM Key file (key_file)
 * Git user ID with credentials stored on EC2 (git_user_id)

6. On the local machine, activate the What R Those virtual environment by executing :code:`conda activate whatrthose`.

7. Run :code:`python deploy.py` on the local machine's terminal in the code directory.

This deployment will take several minutes while it waits for Elastic Beanstalk to allocate resources.  When finished,
the script will output the URL to the app:


.. image:: output_url.png
   :width: 1000pt
