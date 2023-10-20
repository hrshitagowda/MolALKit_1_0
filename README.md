
![ActiveLearningBenchmark2](https://user-images.githubusercontent.com/127516906/230432117-43e7f5b6-db93-4688-a2de-13d55fe3b0e9.png)

# MolALKit: A Toolkit for Active Learning of molecular data.
This package is a toolkit for active learning of molecular data.

## Installation
```commandline
conda env create -f environment.yml
conda activate molalkit
```

## Usage
[scripts](https://github.com/RekerLab/MolAlKit/tree/main/scripts) contains executable files to perform 
active learning jobs.

[examples](https://github.com/RekerLab/MolAlKit/tree/main/examples) contains multiple examples to use this package.

[model_config](https://github.com/RekerLab/MolAlKit/tree/main/model_config) contains the config files of machine 
learning models.


### Models
#### Random Forest with Morgan fingerprints
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

#### Logistic Regression with Morgan fingerprints
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/LogisticRegression_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

#### Gaussian Process Regression with Morgan fingerprints
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionValue_DotProductKerneL_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionUncertainty_DotProductKerneL_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

Use posterior uncertainty for classification.
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionUncertainty_DotProductKerneL_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

#### Multi-layer perceptron (MLP) with Morgan fingerprints
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/MLP_BinaryClassification_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/MLP_Regression_MVE_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/MLP_Regression_Evidential_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

#### D-MPNN
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/DMPNN_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/DMPNN_Regression_MVE_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/DMPNN_Regression_Evidential_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

#### D-MPNN + rdkit features
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/DMPNN_RDKIT_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/DMPNN_RDKIT_Regression_MVE_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/DMPNN_RDKIT_Regression_Evidential_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

#### GPR-MGK
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionUncertainty_MarginalizedGraphKerneL_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al --n_jobs 6
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionValue_MarginalizedGraphKerneL_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al --n_jobs 6
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionUncertainty_MarginalizedGraphKerneL_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

#### SVM
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/SupportVectorMachine_DotProductKerneL_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

## Yoked Learning
Yoked Learning uses different models for selection and evaluation.
```commandline
python3 ActiveLearning.py --data_path alb/data/bace.csv --pure_columns mol --target_columns Class --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al --model_config_extra_evaluators model_config/MLP_BinaryClassification_Morgan_Config
```

## Checkpoint file, continue run, and extensions
Stop at 10% and write checkpoint file every 10 iterations of active learning.
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al --stop_ratio 0.1 --save_cpt_stride 10
```
Continue the active learning to 20%.
```commandline
python3 ALContinue.py --save_dir test_al --stop_ratio 0.2
```

## Reevaluate a new model using an existing active learning trajectory
```commandline
python3 ReEvaluate.py --data_public bace --save_dir test_al --model_config_evaluator model_config/RandomForest_RdkitNorm_Config --evaluator_id 0 --evaluate_stride 10 --metrics roc-auc mcc accuracy precision recall f1_score 
```
