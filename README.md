
# MolALKit: A Toolkit for Active Learning in Molecular Data.
This software package serves as a robust toolkit designed for the active learning of molecular data.

## Installation
```commandline
pip install numpy==1.22.3 git+https://gitlab.com/Xiangyan93/graphdot.git@feature/xy git+https://github.com/bp-kelley/descriptastorus git+https://github.com/Xiangyan93/chemprop.git@molalkit
pip install mgktools molalkit
```

## Data
**MolALKit** currently supports active learning exclusively for single-task datasets, which can be either classification or regression tasks.

### Custom Dataset
The data file must be in CSV format with a header row, structured as follows:
```
smiles,p_np
[Cl].CC(C)NCC(O)COc1cccc2ccccc12,1
C(=O)(OC(C)(C)C)CCCc1ccc(cc1)N(CCCl)CCCl,1
...
```
The following arguments are required to run the active learning 
```
--data_path <dataset.csv> --pure_columns <smiles> --target_columns <target> --dataset_type <classification/regression>
```

### Public Dataset
The toolkit incorporates several popular public datasets, such as MoleculeNet and TDC, which can be used directly `--data_public <dataset name>`.

Here is the list of available datasets:
```commandline
>>> from molalkit.data.datasets import AVAILABLE_DATASETS
>>> print(AVAILABLE_DATASETS)
['CYP1A2_Veith.csv', 'CYP2C19_Veith.csv', 'CYP2C9_Substrate_CarbonMangels.csv', 'CYP2C9_Veith.csv', 'CYP2D6_Substrate_CarbonMangels.csv', 
'CYP2D6_Veith.csv', 'CYP3A4_Substrate_CarbonMangels.csv', 'CYP3A4_Veith.csv', 'Clearance_Hepatocyte_AZ.csv', 'Half_Life_Obach.csv', 
'ames.csv', 'bace.csv', 'bbb_martins.csv', 'bbbp.csv', 'bioavailability_ma.csv', 'caco2_wang.csv', 'carcinogens_lagunin.csv', 'clintox.csv', 
'delaney.csv', 'dili.csv', 'freesolv.csv', 'herg.csv', 'hia_hou.csv', 'hiv.csv', 'ld50_zhu.csv', 'lipo.csv', 'pdbbind_full.csv', 
'pdbbind_refined.csv', 'pgp_broccatelli.csv', 'ppbr_az.csv', 'skin.csv', 'solubility_aqsoldb.csv', 'vdss_lombardo.csv']
```

### ActiveLearning/Validation Split
Our code supports several methods of splitting data into an active learning set and a validation set. 
The active learning is used for active learning and the validation set is used for evaluating the performance of the active learning model.
* **random**  The data will be split randomly.
* **scaffold_order** With this approach, the data is split based on molecular scaffolds, ensuring that the same scaffold never appears in both the active learning and validation sets. 
The scaffold containing the most molecules is placed in the active learning set. This method aligns with the implementation in DeepChem and is independent of random seeds.
* **scaffold_random** In this method, the placement of scaffolds in either the active learning set or the validation set is done randomly. 
This split is dependent on random seeds and introduces an element of randomness into the scaffold split.

The following arguments are required for data split:
```
--split_type <random/scaffold_order/scaffold_random> --split_sizes <active learning set ratio> <validation set ratio> --seed <random seed>
```

## Surrogate Model
The surrogate model used in this package is described in a json config file. 
For example, a random forest using RDKit features is:
```angular2html
{
    "data_format": "fingerprints",
    "model": "random_forest",
    "fingerprints_class": [
        "rdkit_2d_normalized"
    ]
}
```
Here is the list of built-in surrogate models:
```commandline
>>> from molalkit.models.configs import AVAILABLE_MODELS
>>> print(AVAILABLE_MODELS)
['2Mol_GaussianProcessClassification_MarginalizedGraphKernel+DotProductKernelMorgan_Config', 
'2Mol_GaussianProcessClassification_MarginalizedGraphKernel+RBFKernelRDKitNorm_Config', 
'2Mol_GaussianProcessClassification_MarginalizedGraphKernel_Config', 
'2Mol_GaussianProcessRegressionDecisionBoundaryUncertainty_MarginalizedGraphKernel+RBFKernelRDKitNorm_Config', 
'2Mol_GaussianProcessRegressionDecisionBoundaryUncertainty_MarginalizedGraphKernel_Config', 
'2Mol_GaussianProcessRegressionPosteriorUncertainty_MarginalizedGraphKernel+RBFKernelRDKitNorm_Config', 
'2Mol_GaussianProcessRegressionPosteriorUncertainty_MarginalizedGraphKernel_Config', 
'2Mol_RandomForest_Morgan_concat_Config', '2Mol_RandomForest_Morgan_mean_Config', 
'DMPNN+Morgan_BinaryClassification_Config', 'DMPNN+Morgan_Regression_Config', 'DMPNN+Morgan_Regression_Evidential_Config', 
'DMPNN+Morgan_Regression_MVE_Config', 'DMPNN+RDKitNorm_BinaryClassification_Config', 'DMPNN+RDKitNorm_Regression_Config', 
'DMPNN+RDKitNorm_Regression_Evidential_Config', 'DMPNN+RDKitNorm_Regression_MVE_Config', 'DMPNN_BinaryClassification_Config', 
'DMPNN_Regression_Config', 'DMPNN_Regression_Evidential_Config', 'DMPNN_Regression_MVE_Config', 
'GaussianProcessClassification_DotProductKernelMorgan_Config', 'GaussianProcessClassification_MarginalizedGraphKernel+DotProductKernelMorgan_Config', 
'GaussianProcessClassification_MarginalizedGraphKernel+RBFKernelRDKitNorm_Config', 'GaussianProcessClassification_MarginalizedGraphKernel_Config', 
'GaussianProcessClassification_RBFKernelRDKitNorm_Config', 'GaussianProcessRegressionDecisionBoundaryUncertainty_DotProductKernelMorgan_Config', 
'GaussianProcessRegressionDecisionBoundaryUncertainty_MarginalizedGraphKernel+DotProductKernelMorgan_Config', 
'GaussianProcessRegressionDecisionBoundaryUncertainty_MarginalizedGraphKernel+RBFKernelRDKitNorm_Config', 
'GaussianProcessRegressionDecisionBoundaryUncertainty_MarginalizedGraphKernel_Config', 
'GaussianProcessRegressionDecisionBoundaryUncertainty_RBFKernelRDKitNorm_Config', 
'GaussianProcessRegressionPosteriorUncertainty_DotProductKernelMorgan_Config', 
'GaussianProcessRegressionPosteriorUncertainty_MarginalizedGraphKernel+DotProductKernelMorgan_Config', 
'GaussianProcessRegressionPosteriorUncertainty_MarginalizedGraphKernel+RBFKernelRDKitNorm_Config', 
'GaussianProcessRegressionPosteriorUncertainty_MarginalizedGraphKernel_Config', 'GaussianProcessRegressionPosteriorUncertainty_RBFKernelRDKitNorm_Config', 
'LogisticRegression_Morgan_Config', 'LogisticRegression_RDkitNorm_Config', 'MLP_Morgan_BinaryClassification_Config', 
'MLP_Morgan_Regression_Config', 'MLP_Morgan_Regression_Evidential_Config', 'MLP_Morgan_Regression_MVE_Config', 
'MLP_RDKitNorm_BinaryClassification_Config', 'MLP_RDKitNorm_Regression_Config', 'MLP_RDKitNorm_Regression_Evidential_Config', 
'MLP_RDKitNorm_Regression_MVE_Config', 'NaiveBayes_Morgan_Config', 'NaiveBayes_RDKitNorm_Config', 'RandomForest_Morgan_Config', 
'RandomForest_RDKitNorm_Config', 'RandomForest_RDKitNorm_Morgan_Config', 'SupportVectorMachine_DotProductKernelMorgan_Config', 
'SupportVectorMachine_MarginalizedGraphKernel+DotProductKernelMorgan_Config', 'SupportVectorMachine_MarginalizedGraphKernel+RBFKernelRDKitNorm_Config', 
'SupportVectorMachine_MarginalizedGraphKernel_Config', 'SupportVectorMachine_RBFKernelRDKitNorm_Config']
```
The following arguments are required for choosing a surrogate model:
```
--model_config_selector <model_config_file>
```

## First Example
Here's an example of running active learning using MolALKit with the BACE dataset, a 50:50 scaffold split, and Random Forest as the surrogate model:
```commandline
molalkit_run --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir bace
```

## Usage
More examples can be found at [examples](https://github.com/RekerLab/MolAlKit/tree/main/examples).
