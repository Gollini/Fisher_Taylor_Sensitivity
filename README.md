### **Fishing For Cheap And Efficient Pruners At Initialization**

This is the official repository for the **Preprint** "Fishing For Cheap And Efficient Pruners At Initialization." Link: ArXiV


## **Abstract**
Pruning offers a promising solution to mitigate the associated costs and environmental impact of deploying large deep neural networks (DNNs). Traditional approaches rely on computationally expensive trained models or time-consuming iterative prune-retrain cycles, undermining their utility in resource-constrained settings. To address this issue, we build upon the established principles of saliency (LeCun et al., 1989) and connection sensitivity (Lee et al., 2018) to tackle the challenging problem of one-shot pruning neural networks (NNs) before training (PBT) at initialization. 

We introduce **Fisher-Taylor Sensitivity (FTS)**, a computationally cheap and efficient pruning criterion based on the empirical Fisher Information Matrix (FIM) diagonal, offering a viable alternative for integrating first- and second-order information to identify a model’s structurally important parameters. Although the FIM-Hessian equivalency only holds for convergent models that maximize the likelihood, recent studies (Karakida et al., 2019) suggest that, even at initialization, the FIM captures essential geometric information of parameters in overparameterized NNs, providing the basis for our method. 

Finally, we demonstrate empirically that **layer collapse**, a critical limitation of data-dependent pruning methodologies, is easily overcome by pruning within a single training epoch after initialization. We perform experiments on ResNet18 and VGG19 with CIFAR-10 and CIFAR-100, widely used benchmarks in pruning research. Our method achieves competitive performance against state-of-the-art techniques for one-shot PBT, even under extreme sparsity conditions.

---

## **Installation & Setup**
To reproduce the experiments, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Gollini/Fisher_Taylor_Sensitivity.git
   ```

2. **Create and activate a Conda environment:**
   ```bash 
   conda create --name pruning_env python=3.12.4-y
   conda activate pruning_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the configuration file:**
   - Ensure that your **configuration JSON files** are in the correct location (e.g., `exp_configs/`).
   - The script automatically picks up configurations for **batch execution of experiments**.

5. **Run experiments automatically:**
   ```bash
   python main.py --experiment pbt --config exp_configs/
   ```

---

- Experiments were runned with three **random seeds**: 0, 42, 123.
- We share ready to run json files.

---

## **Compressor Codes**
The table below summarizes the different compression methods available and their corresponding codes for running experiments.

| **Compressor Name** | **Code** |
|---------------------|----------|
| Baseline | `none` |
| Random | `random` |
| Magnitude Pruning | `magnitude` |
| Gradient Norm (GN) | `grad_norm` |
| SNIP | `snip` |
| GraSP | `grasp` |
| Fisher Diagonal (FD) | `fisher_diag` |
| Fisher Pruning (FP) | `fisher_pruner` |
| Fisher-Taylor Sensitivity (FTS) | `fts` |
| Fisher Brain Surgeon Sensitivity (FBSS) | `fbss` |

To specify a compressor in your JSON configuration file, update the `"compressor"` field:
```json
"compressor": {
    "class": "none",
    "mask": "global",
    "sparsity": 0,
    "warmup": 0,
    "batch_size": 1
}
```

---

## **Table 1: Performance of Different Pruning Methods on ResNet18 with CIFAR-10**

| Sparsity | Random | Magnitude | GN | SNIP | GraSP | FD | FP | FTS | FBSS |
|-------------|---------|-----------|----|------|------|----|----|----|----|
| 0.80 | 90.78 ± 0.08 | 91.10 ± 0.12 | 90.95 ± 0.35 | 90.74 ± 0.10 | 87.18 ± 0.51 | 90.95 ± 0.11 | 91.08 ± 0.06 | 90.94 ± 0.22 | 90.73 ± 0.33 |
| 0.90 | 89.35 ± 0.13 | 89.88 ± 0.28 | 90.39 ± 0.23 | 90.36 ± 0.34 | 86.60 ± 0.51 | 90.04 ± 0.21 | 90.20 ± 0.08 | 90.55 ± 0.23 | 89.22 ± 0.30 |
| 0.95 | 87.59 ± 0.11 | 89.23 ± 0.19 | 89.00 ± 0.05 | 89.31 ± 0.17 | 86.50 ± 0.05 | 88.61 ± 0.28 | 89.50 ± 0.18 | 89.47 ± 0.32 | 87.58 ± 0.25 |
| 0.98 | 83.47 ± 0.20 | 85.70 ± 0.33 | 86.43 ± 0.05 | 87.26 ± 0.28 | 85.99 ± 0.08 | 85.61 ± 0.20 | 86.97 ± 0.22 | 87.24 ± 0.32 | 83.40 ± 0.74 |
| 0.99 | 78.28 ± 0.45 | 71.99 ± 0.28 | 83.47 ± 0.15 | 84.54 ± 0.04 | 84.56 ± 0.46 | 82.13 ± 0.28 | 83.74 ± 0.48 | 84.85 ± 0.18 | 77.60 ± 1.02 |

---

## **Contact & Issues**
For any issues, please open a discussion or create an issue in this repository.
