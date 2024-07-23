# Setup for remote server

These ansible playbooks are used to setup rapidly and easily a jupyter environment for marchine learning on a remote server. This readme will explain set by set what to do in order to do that. Do note that it will setup pyTorch, scikit-learn and jupyter notebook. TensorFlow isn't setup here but you could set it up easily from there since this ansible will setup the GPU and Cuda.

# Prerequisities

Here is a list of tools you need before starting:

- A VM server

# Setup

### SSH

You first need to setup the SSH connection to that server.

```bash
    ssh root@{ip_adress}
    apt-install vim
    vim /etc/ssh/sshd_config
```

Open the port 22 and permit ssh connections as root, then restart the ssh service.

```bash
    service ssh restart
    exit
```

Locally, copy your ssh id onto the distant server for the root and your user.

```bash
    ssh-copy-id root@{ip_adress}
    ssh-copy-id {user_name}@{ip_adress}
```

### Ansible

Launch the ansible/commands/edit_vault.sh file

```bash
    ./ansible/commands/edit_vault.sh
```

Put in the root password and the user name.
For instance:

```
    ansible_become_pass: root
    user: dany
```

If root does not have any password, do a

```bash
    sudo passwd root
```

Once done, change the inventory ip adress so that it fits the one from your server. After that, run the following command:

```bash
    ./ansible/commands/setup_jupyter.sh
```

Once it's over, wait for the server to reboot. After rebooting, run the following command:

```bash
    ./ansible/commands/setup_nvidia.sh
```

Your server is now setup for using pyTorch, Cuda, Nvidia and jupyter notebooks.

# How do use

Now that it's setup, you have to ssh onto the server and run the jupyter_server_start.sh file in the root folder of the user.

```bash
#/home/{user_name}/
./jupyter_server_start.sh
```

You can now connect remotely onto the jupyter server either on your browser or on your IDE of choice. And IDE would be better since you get better code highlighting and you can use copilot.
