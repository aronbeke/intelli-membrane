# intelli-membrane
Repository for explainable transfer learning in membrane science.

This codebase is to complement the manuscript "Energy-efficient membrane-assisted catalytic systems enabled by explainable transfer learning". The manuscript is currently under peer review.

## Installation

The codebase was developed and tested on **Linux**. To ensure reproducibility, all required dependencies are included in the provided `env.yml` file.  

To set up the environment, first ensure that [Conda](https://docs.conda.io/en/latest/) is installed. Then, create the environment using:

```bash
conda env create -f env.yml
```
Activate the environment with:
```bash
conda activate intelli-membrane
```
The environment includes all necessary Python packages and versions used in development and testing. If you encounter any issues with package versions, running `conda env update -f env.yml` may help.

## Example prediction
An example prediction script can be run with the python model_predict.py command from the command line, which performs rejection prediction for the inputs in example/prediction_input.csv, producing an example/prediction_output.csv file with the predicted rejection values.

The OSN Catalyst Dataset is available in the osncat.csv file.