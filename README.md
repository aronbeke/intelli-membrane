# intelli-membrane
Repository for explainable transfer learning in membrane science.

This codebase is released to complement the manuscript "Energy-efficient membrane-assisted catalytic systems enabled by explainable transfer learning". The manuscript is currently under peer review. The DOI number will be referenced here upon publication.

This repository contains code to reproduce the training and testing of machine learning models for rejection prediction.

## Installation

The codebase was developed and tested on **Linux**. To ensure reproducibility, all required dependencies are included in the provided `env.yml` file.  

To set up the environment, first ensure that [Conda](https://docs.conda.io/en/latest/) is installed. Then, create the environment using:

```bash
conda env create -f env.yml
```

Check available environments:
```bash
conda env list
```

Activate the environment with:
```bash
conda activate intelli-membrane
```
The environment includes all necessary Python packages and versions used in development and testing. If you encounter any issues with package versions, running `conda env update -f env.yml` may help.

## Example prediction
An example prediction script can be run with the `python model_predict.py` command from the command line (from the repository root), which performs rejection prediction for the metal complex inputs in `example/prediction_input.csv`, producing an `example/prediction_output.csv` file with the predicted rejection values.

## Model retraining
The checkpoints in `checkpoints_mcmpnn/combi` correspond to the final specialized MCMPNN models presented in the manuscript. The 'v42a' models are targeted for metal complexes, 'v42c' for ligands, and 'v42d' for organocatalysts.
To retrain the models, use the 'v42ar', v42cr', and 'v42dr' tags, and the following commands:
```bash
python model_train.py mcmpnn combi 'v42ar'
python model_train.py mcmpnn combi 'v42cr'
python model_train.py mcmpnn combi 'v42dr'
```
The `suffix` parameter in the main function in `model_predict.py` can be modified accordingly, to test the retrained models on input features provided in `example/prediction_input.csv`. For information on how to formulate the input SMILES strings please refer to the Supplementary Methods section of the Supplementary Information.

Training is automatically performed on the best detected hardware (GPU, if available). Depending on system architecture, the training of all 5 models in the ensemble can take between 2-5 hours.

## Dataset and results
The OSN Catalyst Dataset is available in the osncat.csv file. Further results and data demonstrated in the manuscript can be found in the corresponding subfolders of the results folder.