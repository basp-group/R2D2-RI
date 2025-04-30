# Conda Installation
If you need to configure the ``conda`` command to setup your environment.
- Download the bash file online. You should find the latest Linux version on https://www.anaconda.com/download#downloads.
- Copy the installer link and in the *bash terminal* of your server, enter 
```
    wget <yourlink>

    [example] 
    wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
```
- Once the ``file.sh`` has been correctly downloaded, install it with
```
    bash <filename.sh>

    [example]
    bash Anaconda3-2023.09-0-Linux-x86_64.sh
```
- Accept to modify the ``~/.bashrc`` and ``~/.bash_profile`` files to add the ``conda.exe`` to your PATH variable.
- You should now be able to use the ``conda`` command in a terminal.

# Conda Environment
To activate this environment, use
```
conda activate RI-meas-op / source activate RI-meas-op
```
To deactivate an active environment, use
```
conda deactivate / source deactivate
```
where ``source`` might be needed for recent versions of conda.

---
If you want another environment name, write
```
conda env create -f requirements_conda.yml --name <your_env_name>
```