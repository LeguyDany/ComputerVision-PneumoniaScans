
# Zoidberg

This is a project about machine learning, more precisely image recognition. The goal is to create or use several models, finetune and classify medical images. The main topic is pneumonia, and the images are lung scannings.

The project can be setup through a GCP compute engine instance rapidly thanks to Terraform and ansible. Terraform is used to create a new vm instance on debian 12, while Ansible sets up the environment to code and run the code. The code can be run locally with conda after installing the dependances as well.

### Main technologies
Exploration and training:
- PyTorch
- Scikit-learn
- Numpy
- Plotly
- Matplotlib

Deployment and development environments:
- Ansible
- Terraform
- Google Cloud Platform
- Azure Storage Accounts
- Amazon SageMaker

### Environment
Google Cloud Platform:
- GPU: Tesla T4
- CPU: 2 cores
- Memory: 13Gb
- Disk storage: 50Gb
# Setup

### Cloud environment
You can follow the instructions in ./cloud/ansible/readme.md to setup the environment if you want to setup in a cloud instance.

### Local environment
Make sure you have conda installed on your machine. Once it's setup, you can do the following commands at the project's root:

```bash
conda activate
conda env update -f ./environment.yaml
```

If you have an NVIDIA gpu on your machine with the drivers already setup, you should be able to run the training with your GPU after installing the following dependencies:

```bash
conda activate
conda install -y pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y -c 'nvidia/label/cuda-11.7.0' cuda-toolkit
```
# How do use

You can train different models within the project or do some exploration with notebooks.


## Exploration
### Cloud environment

You can follow the instructions in ./cloud/ansible/readme.md to run a jupyter notebook.

### Local environment

You can run a jupyter server on your IDE or run it with the following command line:

```bash
conda jupyter-notebook --ip=localhost --port=8888 --no-browser --allow-root
```

## Training

There are scripts located in ./src/bin. Before training, you should download the dataset first through this link :

[Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)

Once it's done, run this command from the project's root:

```bash
python3 ./src/bin/repartition_script.py
```

Then enter the path to the chest_Xray file. It will create a better_repartition folder. Place that folder in the parent folder of the root and rename it "dataset".

To run the training, run the following sript:

```bash
python3 ./src/bin/train_model.py
```

To train a different model, change the following line in the train_model.py file by selecting the model you want to use:

```python
model = DeviceDataLoader.to_device(ResNet14(3, 3), DeviceDataLoader.get_default_device())
```