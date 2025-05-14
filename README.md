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
Removed 9 constant numeric features: ['Active Max', 'Active Mean', 'Active Min', 'Active Std', 'Bwd Pkt Len Std', 'FIN Flag Cnt', 'Fwd PSH Flags', 'Init Fwd Win Byts', 'RST Flag Cnt']
Removed 3 features constant within each class: ['Bwd PSH Flags', 'PSH Flag Cnt', 'ACK Flag Cnt']
Total features removed: 12
Retained 52 clean numeric features

✅ Normalizing data...
Scaler used: MinMaxScaler()

✅ Running Conventional VarianceThreshold (VT)...
Selected features (VT): 3
Selected feature names (VT): ['Bwd Header Len', 'SYN Flag Cnt', 'Init Bwd Win Byts']
VT completed in 4.2609 seconds

✅ Running Dynamic Feature Analysis (DFA)...
DFA completed in 6.0969 seconds
Selected features: 6 | Indices: [0, 2, 29, 37, 38, 46]
Selected feature names (DFA): ['Flow Duration', 'Tot Bwd Pkts', 'Bwd Header Len', 'SYN Flag Cnt', 'Down/Up Ratio', 'Init Bwd Win Byts']


✅ Training classifier on full features with 5-Fold CV...
✅ Training classifier on VT features with 5-Fold CV...
✅ Training classifier on selected features with 5-Fold CV...
✅ Training classifier on aggregated features with 5-Fold CV...

Model: RandomForestClassifier(n_jobs=-1)

Result Summary (5-Fold CV Average):
==============================================================================================================
Method         Feature   Accuracy  ROC-AUC   MSE       R2        Train Time (s) Test Time (s)  Inference time (μs)
==============================================================================================================
Full Features  52        1.000000  1.000000  0.000000  1.000000  129.602940     2.123005       0.428883       
VT             3         1.000000  1.000000  0.000000  1.000000  70.677141      1.448538       0.292629       
DFA Sel        6         1.000000  1.000000  0.000000  1.000000  91.299638      1.744394       0.352397       
DFA Agg        1         1.000000  1.000000  0.000000  0.999999  103.955865     1.558262       0.314795       
==============================================================================================================
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
==============================================================================================================
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
==============================================================================================================
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
==============================================================================================================
Evaluation Result for: DFA Agg

Classification Report:
              precision    recall  f1-score   support

           0   1.000000  1.000000  1.000000   2475040
           1   1.000000  1.000000  1.000000   2475040

    accuracy                       1.000000   4950080
   macro avg   1.000000  1.000000  1.000000   4950080
weighted avg   1.000000  1.000000  1.000000   4950080

Confusion Matrix:
[[2475039       1]
 [      0 2475040]]
==============================================================================================================

✅ Evaluate Aggregated Feature (X_agg)...

Average Silhouette Score: 0.1792

🎉 Analysis complete!
```

## Analysis

This dataset analysis summary shows a highly effective and clean machine learning pipeline for binary classification (likely attack detection, given the context and feature names).
Here's a breakdown of the key points and what they suggest:

### Dataset Overview
- Samples: 4,950,080 (massive scale)
- Initial Features: 64
- Post-cleaning Features: 52
  - Removed 12 non-informative features:
    - 9 constant across the dataset (e.g., Active Max, FIN Flag Cnt)
    - 3 constant per class (likely redundant under binary classification)

⚙️ Preprocessing
Scaling: MinMaxScaler() — ensures uniform scaling across features, especially important for algorithms sensitive to feature magnitudes.

🧠 Feature Selection Techniques
1. Variance Threshold (VT)
- Selected only 3 features: Bwd Header Len, SYN Flag Cnt, Init Bwd Win Byts
- Fast, simple method — relies on selecting features with non-zero variance.

2. Dynamic Feature Analysis (DFA)
- Selected 6 features, more diverse: includes Flow Duration, Tot Bwd Pkts, and Down/Up Ratio
- The new unsupervised dynamic feature selection based on traffic distributions

3. DFA Aggregated (DFA Agg)
- Reduced features to just 1 while maintaining perfect performance — implies heavy feature engineering or dimensionality reduction.
- The new unsupervised dynamic feature selection with weighted aggregation

---

### 3. **Model & Performance**

* Model used: `RandomForestClassifier()`
- All feature subsets yield perfect classification — very likely due to clearly separable classes.
- Inference time and training time drop significantly with fewer features — ideal for deployment.
- Slight trade-off in training time with DFA Agg — possibly due to added preprocessing or transformation complexity.

---

### 4. **Clustering Insight**

- Measures clustering tightness and separability.
- 0.1792 is low → while classification is perfect, the intrinsic cluster structure may be weak, suggesting that:
- The classes are linearly separable but not well-clustered.
- The aggregated feature might be synthetic and lacks natural grouping.

---

## Final Verdict

Ensure to do proper pre-processing before classification.
