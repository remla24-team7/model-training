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

## Mutamorphic testing:
Create dataset through:
creation of the dataset is still reliant on the tokenizer being pressent in the repro, this should be fixed through the lib-ml package as a TODO:

```
dvc repro make_mutamorphic_data
```

Eval on mutamorphic data

```
dvc repro mutamorphic_eval
```



lint-results.txt:

```
The lint scores will be shown here. The Before score it the score before code improvements are made and the after score
is the score after doing the suggested improvements.

Pylint scores with initial configuration:

make_dataset.py
Before: 0.00/10
After:  10.00/10

model_evaluate.py
Before: 6.82/10
After:  10.00/10

model_train.py:
Before: 4.86/10
After:  10.00/10


flake8 evaluation:
flake8 was run AFTER max scores with pylint was achieved.
flake8 was configured to follow same line length rile as pyline i.e. line of length 100 os allowed:
This tool gives no scores, so ill just show what issues showed up if they did.

make_dataset.py
no issues

model_evaluate.py
no issues

model_train.py:
Before:
model_train.py:16:24: E127 continuation line over-indented for visual indent
model_train.py:65:17: E128 continuation line under-indented for visual indent
model_train.py:66:17: E128 continuation line under-indented for visual indent
model_train.py:67:17: E128 continuation line under-indented for visual indent
model_train.py:68:17: E128 continuation line under-indented for visual indent
model_train.py:69:17: E124 closing bracket does not match visual indentation
After:
no issues

Bandit evaluation:
Bandit is run after flake8 shows no issues.

make_dataset.py

    Test results:
    >> Issue: [B106:hardcoded_password_funcarg] Possible hardcoded password: '-n-'
       Severity: Low   Confidence: Medium
       CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
       More Info: https://bandit.readthedocs.io/en/1.7.8/plugins/b106_hardcoded_password_funcarg.html
       Location: make_dataset.py:26:12
    25
    26      tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    27      tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)

    --------------------------------------------------

    Code scanned:
            Total lines of code: 37
            Total lines skipped (#nosec): 0

    Run metrics:
            Total issues (by severity):
                    Undefined: 0
                    Low: 1
                    Medium: 0
                    High: 0
            Total issues (by confidence):
                    Undefined: 0
                    Low: 0
                    Medium: 1
                    High: 0
    Files skipped (0):

    Explanation by me: No fix is made here as there is no password or private information being stored. This can be
    seen as a false positive and can be ignored as an issue.



model_evaluate.py

    Test results:
            No issues identified.

    Code scanned:
            Total lines of code: 37
            Total lines skipped (#nosec): 0

    Run metrics:
            Total issues (by severity):
                    Undefined: 0
                    Low: 0
                    Medium: 0
                    High: 0
            Total issues (by confidence):
                    Undefined: 0
                    Low: 0
                    Medium: 0
                    High: 0
    Files skipped (0):

model_train.py:

    Test results:
            No issues identified.

    Code scanned:
            Total lines of code: 54
            Total lines skipped (#nosec): 0

    Run metrics:
            Total issues (by severity):
                    Undefined: 0
                    Low: 0
                    Medium: 0
                    High: 0
            Total issues (by confidence):
                    Undefined: 0
                    Low: 0
                    Medium: 0
                    High: 0
    Files skipped (0):


Final words: Having run three separate linters, and having no issue show up by any of them gives confidence in our codes
lack of code smells. There was an attempt to make dslinter work as it would show ML specific code smells, but this was
not possible due to an error which occurs while using an older version of pylint which is a requirement of dslinter
(the issue happens within the pylint library, so I am unable to fix it in reasonable time).
```