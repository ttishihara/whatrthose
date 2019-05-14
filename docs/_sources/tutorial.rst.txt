Setup Tutorial
==============
The following will walk through the setup, deployment, and execution of the application on a remote server. These steps
are performed **after** the requirements have been met.

Deployment
----------

1. Launch EC2 instance. Make sure to add at least 50GB of storage.  Any instance type is fine (Anaconda with Python3 AMI recommended).

2. The EC2 should be setup with:

 * Git installed
 * Git user credentials stored with access to repo 
 * Conda installed
 * Port 5000 opened inbound and outbound (in security group)

3. Update user_definition.py in the code folder with:

 * EC2 address (ec2_address)
 * EC2 user (user)
 * PEM Key file (key_file)
 * Git user ID with credentials stored on EC2 (git_user_id)

4. On the local machine, activate the What R Those virtual environment by executing :code:`conda activate whatrthose`.

5. Run :code:`python deploy.py` on the local machine's terminal in the code directory.

This deployment will take several minutes while it waits for the application to be pushed to the EC2.  When finished, the application will be available on port 5000 at the EC2's address.
