# HLD-DDoSDN Dataset Analysis & Preprocessing

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

## Overview

This repository analyzes and presents solutions based on the HLD-DDoSDN dataset, which is designed for DDoS attack detection using machine learning. It includes both binary and multiclass classification approaches.

## Getting Started

```bash
git clone https://github.com/nqmn/HLD-DDOSDN_ddos_dataset.git
cd HLD-DDOSDN_ddos_dataset
pip install -r requirements.txt
jupyter lab
```

#### Datasets & Paper

- Dataset availability
The HLD-DDoSDN dataset can be accessed on https://sites.google.com/view/hld-ddosdn-datasets/home
- Article: Bahashwan AA, Anbar M, Manickam S, Issa G, Aladaileh MA, et al. (2024) HLD-DDoSDN: High and low-rates dataset-based DDoS attacks against SDN. PLOS ONE 19(2): e0297548. https://doi.org/10.1371/journal.pone.0297548 

#### Jupyter Notebook Files
You can access the jupyter notebook file located in this repository or copy the URL provided below.

```
https://github.com/nqmn/hld-ddosdn_ddos_dataset/blob/main/hldddosdn_preparationv2.ipynb
https://github.com/nqmn/HLD-DDOSDN_ddos_dataset/blob/main/hldddosdn_stats.ipynb
```

## Identified Issues

### 1. Binary Dataset

Load both files for binary dataset:

```python
# Read multiple CSV files from folder
df = pd.concat(map(pd.read_csv, glob.glob(path + "/*.csv")))
```

Check protocol distribution per label:

```python
df.groupby("Label")["Protocol"].value_counts().unstack(fill_value=0)
```

**Output:**

![image](https://github.com/user-attachments/assets/617e57b4-e358-43dd-b511-d9303dacf13e)

### Analysis:

```
TL;DR: Recommend to exclude Protocol feature from model training.
```
The author mentioned:

> "This study includes both binary and multiclass classifications. In a binary experiment, the normal class is assigned a value of 1, and the malicious traffic is assigned a value of 0." -Article

The binary dataset are divided into two files (one for H-DDoS and another for L-DDoS), which contains 4,950,080 samples and 72 features including the label.

Label 0 (representing DDoS traffic) contains 2,475,040 samples, all of which are recorded as `TCP` (Protocol `6`), regardless of the actual DDoS type (`ICMP`, `TCP`, or `UDP`).
This is likely due to **OpenFlow** encapsulation, where attack packets are transported over the controller channel using TCP.
Meanwhile, label 1 (normal traffic) contains a realistic mix of protocols:
- ICMP: 1,059,226 samples
- TCP: 808,918 samples
- UDP: 606,896 samples

As a result, the `Protocol` feature may becomes ineffective for distinguishing between classes.

### 2. Multiclass Classification

Load all files for multiclass dataset:

```
# Read multiple CSV files from folder
df = pd.concat(map(pd.read_csv, glob.glob(path + "/*.csv")))
```

Check protocol distribution per label:

```python
df.groupby("Label")["Protocol"].value_counts().unstack(fill_value=0)
```

**Output:**

![image](https://github.com/user-attachments/assets/69b0f9a0-d92f-4567-a89e-108b4ba095d1)


### Analysis

```
TL;DR: Protocol mismatch with the attack type.
```

The multiclass datasets contains 2,000,000 samples with 72 features, including the label.

The author mentioned:

> "In the multiclass experiment, every class is given a unique value. For example, 0, 1, 2, and 3 represent SDN normal traffic, ICMP, TCP, and UDP DDoS flooding attacks, respectively." -Article

In both High-Rate and Low-Rate “All Attacks” datasets, the authors define four classes—Normal, ICMP DDoS, TCP DDoS, and UDP DDoS, each with 250 000 samples. Critically, each attack class should carry its own protocol (ICMP→1, TCP→6, UDP→17) in the Protocol field. This is also likely due to **OpenFlow** encapsulation, where attack packets are transported over the controller channel using TCP.

As a result, the `Protocol` feature may becomes ineffective for distinguishing between classes.

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

The HLD-DDoSDN dataset contains significant inconsistencies, so caution is advised when using it for training machine learning models.
The `Protocol` feature is recommended to be excluded from model training across all dataset variants.
If not removed, there is a risk of data leakage, as all non-zero labels use Protocol `6`, which could inadvertently reveal label information during training.

---
### This is another part of analysis.

# ML Classifications:

The output of analysis:

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
VT completed in 4.0957 seconds

✅ Running Dynamic Feature Analysis (DFA)...
DFA completed in 3.0301 seconds
Selected features: 4 | Indices: [28, 30, 38, 49]
Selected feature names (DFA): ['Bwd IAT Min', 'Bwd PSH Flags', 'Pkt Len Std', 'Subflow Fwd Pkts']


✅ Training classifier on full features with 5-Fold CV...
✅ Training classifier on VT features with 5-Fold CV...
✅ Training classifier on selected features with 5-Fold CV...
✅ Training classifier on aggregated features with 5-Fold CV...

Model: DecisionTreeClassifier(random_state=42)

Result Summary (5-Fold CV Average):
===============================================================================================
Method         Feature   Accuracy  ROC-AUC   MSE       R2        Train Time (s) Test Time (s)  
===============================================================================================
Full Features  55        1.000000  1.000000  0.000000  1.000000  19.245008      0.253589       
VT             6         1.000000  1.000000  0.000000  1.000000  1.128329       0.077748       
DFA Sel        4         1.000000  1.000000  0.000000  1.000000  0.959719       0.062975       
DFA Agg        1         1.000000  1.000000  0.000000  1.000000  0.757206       0.060614       
===============================================================================================
Evaluation Result for: Full Features

Classification Report:
              precision    recall  f1-score   support

           0   1.000000  1.000000  1.000000   2475040
           1   1.000000  1.000000  1.000000   2475040

    accuracy                       1.000000   4950080
   macro avg   1.000000  1.000000  1.000000   4950080
weighted avg   1.000000  1.000000  1.000000   4950080

Confusion Matrix:
[[2475040       0]
 [      0 2475040]]
===============================================================================================
Evaluation Result for: VT

Classification Report:
              precision    recall  f1-score   support

           0   1.000000  1.000000  1.000000   2475040
           1   1.000000  1.000000  1.000000   2475040

    accuracy                       1.000000   4950080
   macro avg   1.000000  1.000000  1.000000   4950080
weighted avg   1.000000  1.000000  1.000000   4950080

Confusion Matrix:
[[2475040       0]
 [      0 2475040]]
===============================================================================================
Evaluation Result for: DFA Sel

Classification Report:
              precision    recall  f1-score   support

           0   1.000000  1.000000  1.000000   2475040
           1   1.000000  1.000000  1.000000   2475040

    accuracy                       1.000000   4950080
   macro avg   1.000000  1.000000  1.000000   4950080
weighted avg   1.000000  1.000000  1.000000   4950080

Confusion Matrix:
[[2475040       0]
 [      0 2475040]]
===============================================================================================
Evaluation Result for: DFA Agg

Classification Report:
              precision    recall  f1-score   support

           0   1.000000  1.000000  1.000000   2475040
           1   1.000000  1.000000  1.000000   2475040

    accuracy                       1.000000   4950080
   macro avg   1.000000  1.000000  1.000000   4950080
weighted avg   1.000000  1.000000  1.000000   4950080

Confusion Matrix:
[[2475040       0]
 [      0 2475040]]
===============================================================================================

✅ Evaluate Aggregated Feature (X_agg)...

![image](https://github.com/user-attachments/assets/203a3dc3-915e-4779-993f-901e4831ded2)

Average Silhouette Score: 0.6027

🎉 Analysis complete!
```

## Analysis

The results show a **near-perfect classification performance** across all evaluation methods, but given earlier dataset inconsistencies, these results are **likely misleading**. The findings also suggest the presence of **data leakage** which could stem from the fixed packet sending rates of 0.2 seconds and 0.03 seconds, which corresponding to high-rate and low-rate UDP DDoS flooding attacks, respectively.

> As indicated by [25], the sending packet rate of 0.2 (s) and 0.03 (s) correspond to high-rate and low-rate UDP DDoS flooding attacks, respectively. -Article

Let's break this down:

## Summary of Observations

### 1. **Dataset Size & Features**

* **Samples**: 4,950,080
* **Initial features**: 64 (after feature dropped as mentioned previously)
* **Final features** after cleaning:

  * Removed **9 constant** features
  * **55** usable numeric features

---

### 2. **Feature Selection Techniques**

#### **Variance Threshold (VT)**:

* Selected **6 features**
* Quick and unsupervised
* Features:
  `['Bwd IAT Min', 'Bwd PSH Flags', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'Subflow Fwd Pkts']`

#### **Dynamic Feature Analysis (DFA)**:

* Selected **4 features**
  `['Bwd IAT Min', 'Bwd PSH Flags', 'Pkt Len Std', 'Subflow Fwd Pkts']`
* The new unsupervised dynamic feature selection based on traffic distributions
* Overlaps with VT, suggesting robustness

#### **DFA Aggregated Feature**:

* Aggregate the selected feature into 1 single feature.
* The new unsupervised dynamic feature selection with weighted aggregation
* A composite single feature likely combining most discriminative traits

---

### 3. **Model & Performance**

* Model used: `DecisionTreeClassifier()`
* **All configurations** (Full, VT, DFA, DFA-Aggregated) yielded:

  * Accuracy: **100%**
  * ROC-AUC: **1.0**
  * F1-score: **1.0**
  * Zero misclassifications

#### Evaluation Time:

| Method        | Feature | Train Time (s) | Test Time (s) |
| ------------- | ------- | -------------- | ------------- |
| Full Features |    55   | 19.2           | 0.253         |
| VT            |    6    | 1.13           | 0.078         |
| DFA Sel.      |    4    | 0.96           | 0.063         |
| DFA Agg.      |    1    | 0.76           | 0.061         |

---

### 4. **Clustering Insight**

![image](https://github.com/user-attachments/assets/203a3dc3-915e-4779-993f-901e4831ded2)

* **Silhouette Score** of **0.6027** for the aggregated feature suggests:
  * **Moderate separation** between clusters (i.e., classes)
  * The feature is capturing meaningful signal

---

## Critical Concerns

Despite these perfect results, there are **red flags** that suggest the model might be **overfitting to dataset artifacts**:

1. **Label-Protocol Mismatch**

   * The dataset labels attacks as DDoS but retains only TCP protocol across all attack samples.
   * This introduces **protocol imbalance** between classes (malicious = TCP-only, normal = ICMP/UDP/TCP).
   * It’s very likely that the model has **learned to identify TCP-only traffic as DDoS**, not actual attack patterns.

2. **No Noise, No Errors**

   * Perfect metrics across all feature sets are highly suspicious in real-world cybersecurity datasets.
   * This suggests a **data leakage** or spurious correlations.
   * Features like `Bwd IAT Min` and `Pkt Len Std` indicate that the classifier likely learned the timing patterns associated with different DDoS attack rates (e.g., 0.2s vs 0.03s).

3. **Feature Overperformance**

   * Even with just 1 aggregated feature, you get perfect classification.
   * This confirms that the feature separation is too clean to be realistic.

---

## Recommendations

1. **Rebuild Dataset with Correct Labeling**

   * Consider fixing the mislabeled samples or filtering out TCP-only bias.

2. **Avoid Publishing Results Based on Current Data**

   * These results do not reflect real-world performance and could be misleading.

---

## Final Verdict

Fixing the data should be **priority #1** before trusting or deploying these models.

