# SGCRN
## Smart Graph Convolutional Recurrent Network

SGCRN is a traffic forecasting model that combines **causal graph initialization** (via DAGMA) with **adaptive spatiotemporal learning**.  
Unlike models that start from random adjacency matrices, SGCRN uses causal priors to improve convergence, robustness, and interpretability.

---

## Repository Structure

- **data/**  
  Benchmark datasets (**PEMSD4** and **PEMSD8**) used in our experiments.  
  Original datasets are available from [ASTGCN](https://github.com/Davidham3/ASTGCN/tree/master/data).

- **lib/**  
  Utility modules for dataset loading, preprocessing, normalization, and evaluation metrics.

- **model/**  
  Implementation of the SGCRN architecture.

- **run/**  
  Example scripts to reproduce experiments or run SGCRN on custom datasets.

---

## Requirements

- Python 3.11.10  
- PyTorch 2.2.1  
- NumPy 1.24.4  
- DAGMA 1.1.1  
- argparse  
- configparser  

Install all dependencies:
```bash
pip install -r requirements.txt
