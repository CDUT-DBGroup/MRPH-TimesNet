# Physics-informed multi-scale temporal learning for rainfall–inflow prediction in complex underground mine groundwater systems

This repository contains the implementation of a physics-informed multi-scale temporal learning framework for rainfall-inflow prediction in complex underground mine groundwater systems. The project is developed based on the Time-Series-Library framework and extends it with a hydrology-oriented MRPH-TimesNet model, rainfall-inflow experiments, uncertainty analysis, extreme rainfall event analysis, hydrological lag analysis, physical consistency analysis, and ablation studies.

## Project Structure

```text
MRPH-TimesNet/
├── data_provider/
├── dataset/
├── exp/
├── layers/
├── models/
├── scripts/
├── utils/
├── LICENSE
├── requirements.txt
├── run.py
├── run_mrph_timesnet_610.py
├── run_mrph_timesnet_670.py
├── run_dl_comparison_610.py
├── run_dl_comparison_670.py
├── run_tradition_comparision_610.py
├── run_tradition_comparision_670.py
├── traditional_comparison_common.py
├── run_ablation.py
├── sensitivity_analysis_610.py
├── sensitivity_analysis_670.py
├── uncertainty_analysis_610.py
├── uncertainty_analysis_670.py
├── hydro_mechanism_analysis_610.py
├── hydro_mechanism_analysis_670.py
├── extreme_rainfall_analysis_610.py
├── extreme_rainfall_analysis_670.py
├── physical_consistency_analysis_610.py
├── physical_consistency_analysis_670.py
├── plot_test_fit_subplots.py
└── plot_test_fit_individual.py
```

## File and Directory Description

### Core Directories

| Path | Description |
| --- | --- |
| `models/` | Model definitions. It contains the proposed `MRPH_TimesNet.py` and baseline models such as `TimesNet.py`, `Autoformer.py`, `Informer.py`, `DLinear.py`, `PatchTST.py`, `Transformer.py`, `LSTM.py`, and `CNN_LSTM.py`. |
| `models/MRPH_TimesNet.py` | The proposed physics-informed multi-scale rainfall-inflow prediction model. It extends TimesNet with hydrological temporal embedding and multi-scale rainfall lag attention. |
| `layers/` | Neural network building blocks used by TimesNet and other deep learning models, including embedding layers, attention layers, convolution blocks, auto-correlation blocks, and transformer encoder-decoder components. |
| `exp/` | Experiment classes inherited from the Time-Series-Library framework. These files define training, validation, testing, and prediction workflows for different time series tasks. |
| `data_provider/` | Data loading and preprocessing modules. `data_loader.py` and `data_factory.py` are used to build datasets and dataloaders for model training and evaluation. |
| `utils/` | Utility functions, including metrics, early stopping, learning rate scheduling, time feature encoding, data augmentation, masking, DTW calculation, and argument printing. |
| `dataset/` | Rainfall-inflow datasets used in the experiments. The current project uses `water_timeseries_610.csv` and `water_timeseries_670.csv`. |
| `scripts/` | Original shell scripts from Time-Series-Library for benchmark experiments. They are kept mainly for framework completeness and reference. |

### Configuration and Entry Files

| File | Description |
| --- | --- |
| `requirements.txt` | Python dependencies required by the project. PyTorch should be installed according to the local CUDA or CPU environment. |
| `LICENSE` | Open-source license file. The repository follows the license inherited from the original Time-Series-Library base. |
| `run.py` | General training and evaluation entry inherited from Time-Series-Library. It can run supported time series models through command-line arguments. |

### Main Experiment Scripts

| File | Description |
| --- | --- |
| `run_mrph_timesnet_610.py` | Trains and evaluates the proposed MRPH-TimesNet model on the 610 rainfall-inflow dataset. |
| `run_mrph_timesnet_670.py` | Trains and evaluates the proposed MRPH-TimesNet model on the 670 rainfall-inflow dataset. |
| `run_dl_comparison_610.py` | Runs deep learning baseline comparisons on the 610 dataset. |
| `run_dl_comparison_670.py` | Runs deep learning baseline comparisons on the 670 dataset. |
| `run_tradition_comparision_610.py` | Runs traditional machine learning baseline comparisons on the 610 dataset. |
| `run_tradition_comparision_670.py` | Runs traditional machine learning baseline comparisons on the 670 dataset. |
| `traditional_comparison_common.py` | Shared functions for traditional baseline models, feature construction, training, prediction, and metric calculation. |
| `run_ablation.py` | Runs ablation experiments for evaluating the contribution of different MRPH-TimesNet components. |

### Analysis Scripts

| File | Description |
| --- | --- |
| `sensitivity_analysis_610.py` | Performs parameter sensitivity analysis on the 610 dataset. |
| `sensitivity_analysis_670.py` | Performs parameter sensitivity analysis on the 670 dataset. |
| `uncertainty_analysis_610.py` | Performs uncertainty analysis for the 610 dataset, including interval-related statistics. |
| `uncertainty_analysis_670.py` | Performs uncertainty analysis for the 670 dataset. |
| `hydro_mechanism_analysis_610.py` | Analyzes rainfall-inflow hydrological lag mechanisms on the 610 dataset. |
| `hydro_mechanism_analysis_670.py` | Analyzes rainfall-inflow hydrological lag mechanisms on the 670 dataset. |
| `extreme_rainfall_analysis_610.py` | Evaluates model behavior under extreme rainfall events on the 610 dataset. |
| `extreme_rainfall_analysis_670.py` | Evaluates model behavior under extreme rainfall events on the 670 dataset. |
| `physical_consistency_analysis_610.py` | Checks physical consistency of prediction results on the 610 dataset. |
| `physical_consistency_analysis_670.py` | Checks physical consistency of prediction results on the 670 dataset. |

### Plotting Scripts

| File | Description |
| --- | --- |
| `plot_test_fit_subplots.py` | Plots multiple test-set fitting curves in subplots for model comparison. |
| `plot_test_fit_individual.py` | Plots individual test-set fitting curves for selected models or datasets. |

### Dataset Files

| File | Description |
| --- | --- |
| `dataset/water_timeseries_610.csv` | Rainfall, cumulative rainfall, water level, and water inflow time series for the 610 dataset. |
| `dataset/water_timeseries_670.csv` | Rainfall, cumulative rainfall, water level, and water inflow time series for the 670 dataset. |

### Model Files

| File | Description |
| --- | --- |
| `models/MRPH_TimesNet.py` | Proposed MRPH-TimesNet model for physics-informed rainfall-inflow forecasting. |
| `models/TimesNet.py` | Original TimesNet baseline model. |
| `models/Autoformer.py`, `models/Informer.py`, `models/Transformer.py`, `models/FEDformer.py`, `models/Reformer.py` | Transformer-family forecasting baselines. |
| `models/DLinear.py`, `models/LightTS.py`, `models/PatchTST.py`, `models/iTransformer.py`, `models/TimeMixer.py`, `models/TimeXer.py` | Representative deep time series forecasting baselines. |
| `models/LSTM.py`, `models/CNN_LSTM.py` | Recurrent and hybrid neural network baselines used for rainfall-inflow comparison. |
| Other files in `models/` | Additional models inherited from Time-Series-Library and retained for reproducibility or future comparison. |

## Environment

The project is implemented in Python and PyTorch. A CUDA-enabled GPU is recommended for deep learning experiments.

Recommended environment:

```text
Python >= 3.8
PyTorch >= 1.10
CUDA-compatible GPU recommended
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch installation fails or CUDA is not available, install PyTorch manually according to the official instruction:

```text
https://pytorch.org/get-started/locally/
```

## Deployment and Reproduction

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd MRPH-TimesNet
```

### 2. Create Python Environment

Using conda:

```bash
conda create -n mrph-timesnet python=3.8
conda activate mrph-timesnet
pip install -r requirements.txt
```

Or using venv:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare Data

Place the rainfall-inflow datasets under `dataset/`:

```text
dataset/water_timeseries_610.csv
dataset/water_timeseries_670.csv
```

The expected data format is a time series CSV file with a `date` column, rainfall-related input variables, water level variables, and the target variable `water_inflow`.

### 4. Train and Test MRPH-TimesNet

Run the 610 dataset experiment:

```bash
python run_mrph_timesnet_610.py
```

Run the 670 dataset experiment:

```bash
python run_mrph_timesnet_670.py
```

Training checkpoints are normally saved to `checkpoints/`, and prediction or metric outputs are generated during the experiment process.

### 5. Run Deep Learning Baseline Comparison

```bash
python run_dl_comparison_610.py
python run_dl_comparison_670.py
```

These scripts compare MRPH-TimesNet with deep learning baselines such as TimesNet, Autoformer, Informer, DLinear, PatchTST, Transformer, LSTM, and CNN-LSTM.

### 6. Run Traditional Baseline Comparison

```bash
python run_tradition_comparision_610.py
python run_tradition_comparision_670.py
```

These scripts evaluate traditional machine learning models for rainfall-inflow forecasting and generate comparison metrics.

### 7. Run Ablation Study

```bash
python run_ablation.py
```

This script evaluates the contribution of different MRPH-TimesNet components.

### 8. Run Additional Analyses

```bash
python sensitivity_analysis_610.py
python sensitivity_analysis_670.py

python uncertainty_analysis_610.py
python uncertainty_analysis_670.py

python hydro_mechanism_analysis_610.py
python hydro_mechanism_analysis_670.py

python extreme_rainfall_analysis_610.py
python extreme_rainfall_analysis_670.py

python physical_consistency_analysis_610.py
python physical_consistency_analysis_670.py
```

These scripts generate results for parameter sensitivity, uncertainty estimation, hydrological lag mechanism analysis, extreme rainfall event evaluation, and physical consistency verification.

### 9. Plot Prediction Results

```bash
python plot_test_fit_subplots.py
python plot_test_fit_individual.py
```

The generated figures can be used for visual comparison of predicted and observed inflow values.

## Notes for Reproducibility

- The `checkpoints/` directory stores trained model weights and is usually not recommended for direct GitHub upload.
- If pretrained weights are required for reproducing analysis scripts, please provide them through GitHub Releases, Zenodo, Hugging Face, or another external storage service.
- If the original rainfall-inflow data cannot be publicly released, provide a sample dataset and describe the data format in `dataset/README.md`.

## Acknowledgement

This project is built upon the excellent open-source project [Time-Series-Library](https://github.com/thuml/Time-Series-Library) developed by THUML. We sincerely thank the authors and contributors of Time-Series-Library for providing a comprehensive and extensible framework for advanced time series analysis.

