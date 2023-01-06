# ActiveLearningBenchmark
Benchmark for molecular active learning.

## Installation
```commandline
conda env create -f environment.yml
conda activate alb
```
[GPU-enabled PyTorch](https://pytorch.org/get-started/locally/).
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```
## Usage. BACE and freesolv datasets as examples.
## Models
### Random Forest on Morgan fingerprints
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

### Logistic Regression on Morgan fingerprints
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/LogisticRegression_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

### Gaussian Process Regression on Morgan fingerprints
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionValue_DotProductKerneL_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionUncertainty_DotProductKerneL_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

Use posterior uncertainty for classification.
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionUncertainty_DotProductKerneL_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

### Multi-layer perceptron (MLP) on Morgan fingerprints
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/MLP_BinaryClassification_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/MLP_Regression_MVE_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/MLP_Regression_Evidential_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

### D-MPNN
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/DMPNN_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/DMPNN_Regression_MVE_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/DMPNN_Regression_Evidential_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

### D-MPNN + rdkit features
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/DMPNN_RDKIT_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/DMPNN_RDKIT_Regression_MVE_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/DMPNN_RDKIT_Regression_Evidential_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

### GPR-MGK
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionUncertainty_MarginalizedGraphKerneL_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al --n_jobs 6
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionValue_MarginalizedGraphKerneL_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al --n_jobs 6
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionUncertainty_MarginalizedGraphKerneL_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

### SVM
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/SupportVectorMachine_DotProductKerneL_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al
```

## Yoked Learning
using different models for selection and evaluation.
```commandline
python3 ActiveLearning.py --data_path alb/data/bace.csv --pure_columns mol --target_columns Class --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al --model_config_extra_evaluators model_config/MLP_BinaryClassification_Morgan_Config
```

## checkpoint file, continue run, and extensions.
stop at 10% and write checkpoint file every 10 iterations of active learning.
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al --stop_ratio 0.1 --save_cpt_stride 10
```
continue the active learning to 20%.
```commandline
python3 ALContinue.py --save_dir test_al --stop_ratio 0.2
```

## ReEvaluate a new model using existed active learning trajectory
```commandline
python3 ReEvaluate.py --data_public bace --save_dir test_al --model_config_evaluator model_config/RandomForest_RdkitNorm_Config --evaluator_id 0 --evaluate_stride 10 --metrics roc-auc mcc accuracy precision recall f1_score 
```