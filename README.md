# HLD-DDoSDN Dataset Analysis and Preprocessing

## Overview

This repository analyzes and highlights major inconsistencies within the **HLD-DDoSDN** dataset, which is designed for DDoS attack detection using machine learning. It includes both **binary** and **multiclass classification** versions.

#### References

* Original article: Bahashwan AA, Anbar M, Manickam S, Issa G, Aladaileh MA, et al. (2024) HLD-DDoSDN: High and low-rates dataset-based DDoS attacks against SDN. PLOS ONE 19(2): e0297548. https://doi.org/10.1371/journal.pone.0297548 

#### Dataset availability
The HLD-DDoSDN can be accessed on

```
https://sites.google.com/view/hld-ddosdn-datasets/home
```

#### Jupyter notebook file:
You can access the jupyter notebook file located in this repository or copy the URL provided below.

```
https://github.com/nqmn/hld-ddosdn_ddos_dataset/blob/main/hldddosdn_preparationv2.ipynb
```

## Identified Issues

### 1. Binary Dataset Inconsistencies

Load both files for binary dataset:

```python
# Read multiple CSV files from folder
df = pd.concat(map(pd.read_csv, glob.glob(path + "/*.csv"))) #output: (2413314, 72)
print("Dataset loaded...")
```

Check protocol distribution per label:

```python
df.groupby("Label")["Protocol"].value_counts().unstack(fill_value=0)
```

**Output:**

![image](https://github.com/user-attachments/assets/617e57b4-e358-43dd-b511-d9303dacf13e)

### Analysis:

```
TL;DR: All DDoS protocol types in this case are TCP-based.
```
The author mentioned:

> "This study includes both binary and multiclass classifications. In a binary experiment, the normal class is assigned a value of 1, and the malicious traffic is assigned a value of 0." -Article

The binary dataset are divided into two files (one for H-DDoS and another for L-DDoS), which contains 4,950,080 samples and 72 features including the label.

Label 0 (representing DDoS traffic) contains 2,475,040 samples, all of which are TCP-based (Protocol 6) regardless of the DDoS type.
This is inconsistent with the intended dataset design, which should include ICMP and UDP DDoS traffic.

Meanwhile, label 1 (normal traffic) contains a realistic mix of protocols:
- ICMP: 1,059,226 samples
- TCP: 808,918 samples
- UDP: 606,896 samples

This discrepancy indicates a serious labeling or extraction issue in the dataset, where attack traffic does not represent protocol diversity, thereby undermining its suitability for training robust, real-world DDoS detection models.

All traffic for label 0 (DDoS traffic) uses **Protocol 6 (TCP)**, which contradicts the expected protocol types:
- **ICMP DDoS** → Protocol `0`
- **TCP DDoS** → Protocol `6`
- **UDP DDoS** → Protocol `17`

This contradicts the definitions stated in the associated article.

---

### 2. Multiclass Classification Inconsistencies

Load all files for multiclass dataset:

```
# Read multiple CSV files from folder
df = pd.concat(map(pd.read_csv, glob.glob(path + "/*.csv"))) #output: (2413314, 72)
print("Dataset loaded...")
```

Check protocol distribution per label:

```python
df.groupby("Label")["Protocol"].value_counts().unstack(fill_value=0)
```

**Output:**

![image](https://github.com/user-attachments/assets/69b0f9a0-d92f-4567-a89e-108b4ba095d1)


### Analysis

```
TL;DR: Protocol mismatch with the attack type
```

The multiclass datasets contains 2,000,000 samples with 72 features, including the label.

The author mentioned:

> "In the multiclass experiment, every class is given a unique value. For example, 0, 1, 2, and 3 represent SDN normal traffic, ICMP, TCP, and UDP DDoS flooding attacks, respectively." -Article

The ICMP DDoS dataset (H-ICMP_...) assigns label 0 (DDoS) to TCP traffic and label 1 (Normal) to ICMP traffic, which presents an illogical protocol-label mapping. Similarly, the UDP DDoS dataset (H-UDP_...) maps label 0 (DDoS) to TCP and label 1 (Normal) to UDP, indicating a clear protocol mismatch with the intended attack type. In contrast, the TCP DDoS dataset (H-TCP_...) maintains consistent labeling and protocol alignment, but it lacks protocol diversity, limiting its applicability for generalized DDoS detection.


- **ICMP DDoS Dataset (`H-ICMP_...`)**:  
  - Label `0` (DDoS) → TCP  
  - Label `1` (Normal) → ICMP  
  **⚠ Illogical protocol-label mapping**

- **UDP DDoS Dataset (`H-UDP_...`)**:  
  - Label `0` (DDoS) → TCP  
  - Label `1` (Normal) → UDP  
  **⚠ Protocol mismatch with the attack type**

- **TCP DDoS Dataset (`H-TCP_...`)**:  
  Consistent labeling and protocol use, but **lacks protocol diversity**.




These inconsistencies may severely impact model training and generalization.

## Data Summary

```python
df.info()
```

* **72 columns**:

  * 71 features (input)
  * 1 label (target)

### Label Mapping:

* **Binary**:

  * `0`: Malicious (DDoS)
  * `1`: Normal
* **Multiclass**:

  * `0`: Normal
  * `1`: ICMP DDoS
  * `2`: TCP DDoS
  * `3`: UDP DDoS

---

## Feature Selection

Due to inconsistent values, the following features are dropped:

```python
features_to_drop = [
    'Flow ID', 'Src IP', 'Src Port',
    'Dst IP', 'Dst Port', 'Timestamp', 'Protocol'
]
new_df = df.drop(features_to_drop, axis=1)
```

> `Protocol` was dropped due to inconsistencies across attack types.

---

## Export Cleaned Dataset

```python
new_df.to_csv('../ds/hldddosdn_hlddos_combined_binary_cleaned_0d1n.csv', index=False)
```

---

## Conclusion

The **HLD-DDoSDN** dataset presents significant labeling and protocol inconsistencies. Caution is advised when using it for training machine learning models. Carefully validating class-label mappings and traffic characteristics is essential before drawing conclusions from models trained on this dataset.

---

# Output for Random Forest classifications:

```
Dataset: ../ds/hldddosdn_hlddos_combined_binary_cleaned_0d1n.csv
Dataset counts: 4950080 samples, 64 features

✅ Cleaning dataset...
Removed 9 constant numeric features
Total features removed: 9
Retained 55 clean numeric features

✅ Normalizing data...
Scaler used: MinMaxScaler()

✅ Running Conventional VarianceThreshold (VT)...
Selected features (VT): 6
Selected feature names (VT): ['Bwd IAT Min', 'Bwd PSH Flags', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'Subflow Fwd Pkts']
VT completed in 4.1901 seconds

✅ Running Dynamic Feature Analysis (DFA)...
Feature analysis completed in 11.7023 seconds

Total features involved: 55
Selected features: 4
Selected feature indices: [28, 30, 38, 49]
Selected feature names (DFA): ['Bwd IAT Min', 'Bwd PSH Flags', 'Pkt Len Std', 'Subflow Fwd Pkts']

✅ Training classifier on full features...
✅ Training classifier on VT features...
✅ Training classifier on selected features...
✅ Training classifier on aggregated feature...

Model: RandomForestClassifier(n_jobs=-1)

Result Summary:
===========================================================================
Method         Feature   Accuracy  ROC-AUC   MSE       R2        Train Time (s) Test Time (s)  
===========================================================================
Full Features  55        1.000000  1.000000  0.000000  1.000000  95.732413      2.042903       
VT             6         1.000000  1.000000  0.000000  1.000000  40.571136      1.435399       
DFA Sel        4         1.000000  1.000000  0.000000  1.000000  40.507255      1.247946       
DFA Agg        1         1.000000  1.000000  0.000000  1.000000  42.490553      1.639931       
===========================================================================
Evaluation Result for: Full Feature

Classification Report (Full Feature):
              precision    recall  f1-score   support

           0   1.000000  1.000000  1.000000    495008
           1   1.000000  1.000000  1.000000    495008

    accuracy                       1.000000    990016
   macro avg   1.000000  1.000000  1.000000    990016
weighted avg   1.000000  1.000000  1.000000    990016

[[495008      0]
 [     0 495008]]
======================================================================

Evaluation Result for: VT Feature

Classification Report (Selected Feature):
              precision    recall  f1-score   support

           0   1.000000  1.000000  1.000000    495008
           1   1.000000  1.000000  1.000000    495008

    accuracy                       1.000000    990016
   macro avg   1.000000  1.000000  1.000000    990016
weighted avg   1.000000  1.000000  1.000000    990016

[[495008      0]
 [     0 495008]]
======================================================================
Evaluation Result for: Selected Feature

Classification Report (Selected Feature):
              precision    recall  f1-score   support

           0   1.000000  1.000000  1.000000    495008
           1   1.000000  1.000000  1.000000    495008

    accuracy                       1.000000    990016
   macro avg   1.000000  1.000000  1.000000    990016
weighted avg   1.000000  1.000000  1.000000    990016

[[495008      0]
 [     0 495008]]
======================================================================
Evaluation Result for: Aggregated Feature

Classification Report (Aggregated Feature):
              precision    recall  f1-score   support

           0   1.000000  1.000000  1.000000    495008
           1   1.000000  1.000000  1.000000    495008

    accuracy                       1.000000    990016
   macro avg   1.000000  1.000000  1.000000    990016
weighted avg   1.000000  1.000000  1.000000    990016

[[495008      0]
 [     0 495008]]

✅ Evaluate Aggregated Feature (X_agg)...

Average Silhouette Score: 0.6027
Analysis complete!
```

## Analysis

The results show a **near-perfect classification performance** across all evaluation methods, but given earlier dataset inconsistencies, these results are **likely misleading**. Let's break this down:

---

## 🔍 Summary of Observations

### 1. **Dataset Size & Features**

* **Samples**: 4,950,080
* **Initial features**: 64
* **Final features** after cleaning:

  * Removed **9 constant** features
  * **55** usable numeric features

---

### 2. **Feature Selection Techniques**

#### ✅ **Variance Threshold (VT)**:

* Selected **6 features**
* Quick and unsupervised
* Features:
  `['Bwd IAT Min', 'Bwd PSH Flags', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'Subflow Fwd Pkts']`

#### ✅ **Dynamic Feature Analysis (DFA)**:

* Selected **4 features**
  `['Bwd IAT Min', 'Bwd PSH Flags', 'Pkt Len Std', 'Subflow Fwd Pkts']`
* Overlaps with VT, suggesting robustness

#### ✅ **DFA Aggregated Feature**:

* A composite single feature likely combining most discriminative traits

---

### 3. **Model & Performance**

* Model used: `RandomForestClassifier(n_jobs=-1)`
* **All configurations** (Full, VT, DFA, DFA-Aggregated) yielded:

  * Accuracy: **100%**
  * ROC-AUC: **1.0**
  * F1-score: **1.0**
  * Zero misclassifications

#### Evaluation Time:

| Method        | Train Time (s) | Test Time (s) |
| ------------- | -------------- | ------------- |
| Full Features | 95.7           | 2.0           |
| VT            | 40.6           | 1.4           |
| DFA (4 feat)  | 40.5           | 1.2           |
| DFA Agg. (1)  | 42.5           | 1.6           |

---

### 4. **Clustering Insight**

* **Silhouette Score** of **0.6027** for the aggregated feature suggests:

  * **Moderate separation** between clusters (i.e., classes)
  * The feature is capturing meaningful signal

---

## ⚠️ Critical Concerns

Despite these perfect results, there are **red flags** that suggest the model might be **overfitting to dataset artifacts**:

1. **Label-Protocol Mismatch**

   * The dataset labels attacks as DDoS but retains only TCP protocol across all attack samples.
   * This introduces **protocol imbalance** between classes (malicious = TCP-only, normal = ICMP/UDP/TCP).
   * It’s very likely that the model has **learned to identify TCP-only traffic as DDoS**, not actual attack patterns.

2. **No Noise, No Errors**

   * Perfect metrics across all feature sets are highly suspicious in real-world cybersecurity datasets.
   * This suggests a **data leakage** or spurious correlations.

3. **Feature Overperformance**

   * Even with just 1 aggregated feature, you get perfect classification.
   * This confirms that the feature separation is too clean to be realistic.

---

## ✅ Recommendations

1. **Investigate Class Imbalance Beyond Label Count**

   * Check the **protocol distribution** after cleaning. If DDoS = TCP-only and normal = diverse, the model is detecting protocol, not DDoS behavior.

2. **Rebuild Dataset with Correct Labeling**

   * Consider fixing the mislabeled samples or filtering out TCP-only bias.

3. **Introduce Adversarial Noise or Validate on External Data**

   * Try testing your model on a different dataset (e.g., CIC-IDS, UNSW-NB15) to confirm generalization.

4. **Avoid Publishing Results Based on Current Data**

   * These results do not reflect real-world performance and could be misleading.

---

## ✅ Final Verdict

Fixing the data should be **priority #1** before trusting or deploying these models.

