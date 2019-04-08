import paramiko
from os.path import expanduser
from user_definition import *


# ## Assumption : Anaconda, Git (configured)

def ssh_client():
    """
    Return SSH client object

    :return: Paramiko SSH Client object
    """
    return paramiko.SSHClient()


def ssh_connection(ssh, ec2_address, user, key_file):
    """
    Estabish SSH connection with AWS EC2 instance

    :param ssh: SSH object
    :param ec2_address: EC2 Public DNS address
    :type ec2_address: str
    :param user: EC2 username
    :type user: str
    :param key_file: Path to PEM key file
    :type key_file: str
    :return: SSH object with credentials for connection
    """
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ec2_address, username=user,
                key_filename=expanduser("~") + key_file)
    return ssh


def create_or_update_environment(ssh):
    """
    Create or update environment within EC2 instance

    :param ssh: SSH object
    :return: None
    """
    create_command = f"conda env create -f ~/{git_repo_name}/environment.yml"
    _, _, stderr = ssh.exec_command(create_command)
    if b'already exists' in stderr.read():
        update_command = f"conda env update -f " \
                         f"~/{git_repo_name}/environment.yml"
        _, _, _ = ssh.exec_command(update_command)


def git_clone(ssh):
    """
    Clone Github repository of application or update if already exists

    :param ssh: SSH object
    :return: None
    """
    stdin, stdout, stderr = ssh.exec_command("git --version")
    if b"" is stderr.read():
        git_clone_command = f"git clone https://github.com/" \
                            f"{git_repo_owner}/{git_repo_name}.git"
        stdin, stdout, stderr = ssh.exec_command(git_clone_command)

    if b"already exists" in stderr.read():
        git_pull_command = f"cd {git_repo_name}; git pull https://" \
                           f"github.com/{git_repo_owner}/{git_repo_name}"
        _, _, _ = ssh.exec_command(git_pull_command)

def deploy(ssh):
    """
    Run deploy bash script from EC2

    :param ssh: SSH object
    :return: None
    """
    stdin, stdout, stderr = ssh.exec_command("cd")
    stdin, stdout, stderr = ssh.exec_command("conda activate whatrthose")
    #print(stderr.read())
    deploy_command = f"bash ~/{git_repo_name}/code/deploy.sh -bucket whatrthose" \
                     f" -region us-west-2 -ebname whatrthose-dev"
    stdin, stdout, stderr = ssh.exec_command(deploy_command)
    print(str(stdout.read()))

def main():
    """
    With AWS EC2 crendentials, establish SSH connection and setup
    development environment on EC2 instance. Then add crontab job
    to execute calculate_driving_time.py every minute.

    :return: None
    """
    ssh = ssh_client()
    ssh_connection(ssh, ec2_address, user, key_file)
    git_clone(ssh)
    create_or_update_environment(ssh)
    deploy(ssh)

    # Logout
    ssh.close()


if __name__ == '__main__':
    main()
