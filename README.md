# Assignment 1: ML Configuration Management

This is the GitHub repo for assignment 1 of CS4295 Release Engineering for Machine Learning Applications. The submission is in the branch setup-pipeline.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/remla24-team7/project.git
    ```

2. Navigate into the project directory:

    ```bash
    cd your-repository
    ```

3. Install the required dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

The submission for assignment 1 is within the branch setup pipeline. After the installation above, checkout to setup-pipeline branch.

Having checked out, run the command: 

    dvc repro --pull
    
Running the command will pull the required datasets onto your local machine and run the pipeline or get cached outputs of the trained model and evaluation. 

To train or evaluate the model yourself, run the project/src/models/model_train.py file to train the model and project/src/models/model_evaluate.py to evaluate the model.


## Dataset 

The dataset we use was open for public use by kaggle. It can be found here: [dataset](https://www.kaggle.com/datasets/aravindhannamalai/dl-dataset/download?datasetVersionNumber=1)
