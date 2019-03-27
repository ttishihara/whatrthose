import paramiko
from os.path import expanduser
from user_definition import *


# ## Assumption : Anaconda, Git (configured)

def ssh_client():
    """Return ssh client object"""
    return paramiko.SSHClient()


def ssh_connection(ssh, ec2_address, user, key_file):
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ec2_address, username=user,
                key_filename=expanduser("~") + key_file)
    return ssh


def create_or_update_environment(ssh):
    stdin, stdout, stderr = \
        ssh.exec_command("conda env create -f "
                         "~/"+git_repo_name+"/environment.yml")
    if (b'already exists' in stderr.read()):
        stdin, stdout, stderr = \
            ssh.exec_command("conda env update -f "
                             "~/"+git_repo_name+"/environment.yml")


def git_clone(ssh):
    # ---- HOMEWORK ----- #
    stdin, stdout, stderr = ssh.exec_command("git --version")
    if (b"" is stderr.read()):
        git_clone_command = "git clone https://github.com/" + \
                            git_repo_owner + "/" + git_repo_name + ".git"
        stdin, stdout, stderr = ssh.exec_command(git_clone_command)
    if ("already exists" in str(stderr.read())):
        git_pull_command = "cd " + git_repo_name + ";" + \
            " git pull https://github.com/" + git_repo_owner \
            + "/" + git_repo_name
        stdin, stdout, stderr = ssh.exec_command(git_pull_command)


def add_crontab(ssh, filename):
    command = 'echo "* * * * * ' +\
              '~/miniconda3/envs/whatrthose/bin/python ' +\
              '~/'+git_repo_name+'/'+filename+'" > mycron'
    stdin, stdout, stderr = ssh.exec_command(command)
    stdin, stdout, stderr = ssh.exec_command('crontab mycron')
    stdin, stdout, stderr = ssh.exec_command('rm mycron')


def main():
    ssh = ssh_client()
    ssh_connection(ssh, ec2_address, user, key_file)
    git_clone(ssh)
    create_or_update_environment(ssh)
    add_crontab(ssh, 'calculate_driving_time.py')


if __name__ == '__main__':
    main()
