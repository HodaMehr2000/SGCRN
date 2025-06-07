# SGCRN_RD
## Architecture Overview

This model initializes the graph adjacency matrix using DAGMA (leveraging causal priors instead of random initialization) and employs an adaptive graph structure that evolves during training. A GCRN-based residual separator decomposes traffic signals into normal patterns (modeled via spatiotemporal layers) and anomalous residuals, fusing both for robust predictions in dynamic scenarios.

## Structure:

* data: including PEMSD4 and PEMSD8 dataset used in our experiments, which are released by and available at  [ASTGCN](https://github.com/Davidham3/ASTGCN/tree/master/data).

* lib: contains self-defined modules for our work, such as data loading, data pre-process, normalization, and evaluate metrics.

* model: implementation of our SGCRN_RD model

## Requirements

Python 3.11.10, Pytorch 2.2.1, Numpy 1.24.4, Dagma 1.1.1, argparse and configparser



To replicate the results on PEMSD4 and PEMSD8 datasets, you can run the codes in the **run** file directly.
If you want to use the model for your own dataset, please load your dataset by checking "load_dataset" in the **lib** folder and remember to set the learning rate and embedding dimensions.



