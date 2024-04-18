#TradeOffRelations

Supplementary materials for the "Ab initio framework for deciphering trade-off relationships in multi-component alloys" paper.

## Overview

This repository contains supplementary materials related to the research project titled "Ab initio framework for deciphering trade-off relationships in multi-component alloys". The project focuses on developing a computational framework to analyze and understand trade-off relationships in alloys composed of multiple components.

## Data

The `data` directory contains the results of the Multi-objective optimization (MOO). Specifically, you can find the results in the files `moo_monbti.json` and `moo_monbtita.json`.

## Potentials

The `potentials` directory includes potential files and configurations used in the computational simulations. It contains the following files and directories:
- `INCAR`: Configuration file for VASP.
- `mlip_params.yaml`: YAML file containing parameters for the machine learning interatomic potentials (MLIP).
- `training_data_monbti`: Training data for the MLIP related to the `monbti` alloy system.
- `training_data_monbtita`: Training data for the MLIP related to the `monbtita` alloy system.

## Model

The `model` directory contains scripts for the VBA model used in the research. It includes the following files:
- `model_fitter.py`: Python script for fitting the VBA model.
- `model.py`: Python script containing the VBA model implementation.

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). See the [LICENSE](LICENSE) file for details.

