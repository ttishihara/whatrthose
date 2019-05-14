Requirements
============
In order to deploy this product, it requires a remote computer setup with Amazon Web Services and a local computer.
Follow the steps below before starting the tutorial.

The full Github repository is located at `here <https://github.com/MSDS698/whatrthose>`_ where all necessary files for
this application resides.

Amazon Web Services Setup
-------------------------
The process for deployment is detailed in the tutorial.  Here is what is required:

1. An EC2 instance must be stood up with:
 
  * t2.medium preferred
  * At least 50GB of storage 
  * Git installed and user configured
  * Anaconda installed
  * Port 5000 opened inbound and outbound

2. The following must be configured in code/user_definition.py:

  * ec2_address: The DNS name of the EC2 described above
  * user: Which user to log into the EC2 with
  * key_file: The identity file used to log into the EC2 with
  * git_repo_owner: Owner of the git repo of the application
  * git_repo_name: Name of the git repo of the application
  * git_user_id: User ID configured in the EC2 mentioned above

Local Computer Setup
-------------------
1. Clone the What R Those `GitHub repository <https://github.com/MSDS698/whatrthose>`_ to your local machine, using
:code:`git clone` on the terminal.

2. Create the What R Those virtual environment by running :code:`conda create -f environment.yml` on the terminal on
the local machine, inside the `whatrthose` directory.
