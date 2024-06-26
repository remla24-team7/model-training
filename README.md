# model-training

This project is using Poetry for dependency management, so make sure it is installed on your system.

To set up the project, run `poetry install`.

Tip: to avoid prefixing subsequent commands with `poetry run`, launch a Poetry-aware shell with `poetry shell`.

To sync the latest artifacts (including the dataset) from our remote, run `dvc pull`. The original dataset is also available on [Kaggle](https://www.kaggle.com/datasets/aravindhannamalai/dl-dataset/download?datasetVersionNumber=1). Note that even though our remote is public, DVC must be logged into your Google Drive account to work.

To reproduce the pipeline, run `dvc repro`. The preprocess and training stages reuse existing artifacts unless specified otherwise (`--force`).

Afterwards, you can review evaluation metrics and plots using `dvc metrics show` and `dvc plots show`.

Once the project is setup, one can run the tests throug the following command: `pytest`.

## Project structure

Earlier iterations of the project used the [Cookiecutter data science template](https://cookiecutter-data-science.drivendata.org/), but ultimately we pruned unused files and decided to maintain the following project structure:

```
├── .dvc
│   └── config                                  -- DVC remote configuration
├── .github
│   └── workflow.yml                            -- GitHub Action (Workflow) to lint, reproduce, and test
├── dataset                                     -- the raw (DVC-managed) dataset
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── notebooks
│   └── phishing-detection-cnn.ipynb            -- the original notebook
├── outputs                                     -- (DVC-managed) stage outputs
│   ├── preprocess
│   │   ├── encoder.joblib                      -- serialized LabelEncoder
│   │   ├── tokenizer.joblib                    -- serializer Tokenizer
│   │   ├── x_train.joblib                      
│   │   ├── y_train.joblib
│   │   ├── x_val.joblib
│   │   ├── y_val.joblib
│   │   ├── x_test.joblib
│   │   └── y_test.joblib
│   ├── train
│   │   ├── history.json                        -- training metrics report
│   │   └── model.keras                         -- model file
│   ├── evaluate
│   │   ├── conf_matrix.png                     -- confusion matrix plot
│   │   ├── metrics.json                        -- classification metrics report
│   │   └── roc_curve.png                       -- ROC curve plot
│   └── mutamorphic
│       ├── conf_matrix.png
│       ├── metrics.json
│       ├── roc_curve.png
│       └── x_test.joblib                       -- mutated (and preprocessed) testing data
├── scripts                                     -- project scripts
│   ├── mutamorphic                             -- mutamorphic testing scripts
│   │   ├── __init__.py
│   │   ├── preprocess.py                       -- script to mutate and preprocess testing data
│   │   └── evaluate.py                         -- script to evaluate the model on the mutamorphic testing data
│   ├── __init__.py
│   ├── preprocess.py                           -- script to preprocess the entire dataset
│   ├── train.py                                -- script to train the model
│   └── evaluate.py                             -- script to evaluate the model on the testing data
├── tests                                       -- pytest tests
│   ├── __init__.py
│   ├── test_data.py                            -- tests the dataset contents
│   ├── test_model_train.py                     -- tests whether training is functioning as expected
│   └── test_model_evaluate.py                  -- tests whether evaluation is functioning as expected
├── conftest.py                                 -- pytest fixture for the (DVC) params.yaml
├── dataset.dvc                                 -- contains the (DVC-managed) dataset location on the remote
├── dvc.lock                                    -- DVC reproduction pipeline lock file
├── dvc.yaml                                    -- DVC reproduction pipeline
├── params.yaml                                 -- DVC parameters used in pipeline stages
├── poetry.lock                                 -- Poetry project dependency lock file
├── pyproject.toml                              -- Poetry project configuration
└── README.md                                   -- the current README file
```

## Code Quality

In our [GitHub workflow](.github/workflows/workflow.yml) we use [Flake8](https://flake8.pycqa.org/en/latest/), [Bandit](https://bandit.readthedocs.io/en/latest/), and [Pylint](https://pylint.readthedocs.io/en/stable/) for code quality and style analysis before reproducing the pipeline, and run tests using [Pytest](https://docs.pytest.org/en/8.2.x/) afterwards.

A chart of the average pylint scores can be seen if you go into the latest actions and look at the artifacts. the chart will be called pylint_score.png

To verify the code quality manually: 

First do: 
```
pip install poetry 
```

then install the dependencies using poetry: 

```
poetry install 
```

Having installed the dependencies, you can manually run the lints as follows: 

```
pylint scripts/ tests/

bandit -c pyproject.toml -r .

flake8
```
