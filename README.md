# Company-Level Layoff Risk Forecasting

Company-level monthly layoff risk forecasting using LLM-extracted signals, temporal decay modeling, and LSTM-based walk-forward evaluation.

---

## 📌 Overview

This project builds a company-level monthly time-series forecasting system to predict whether a company will conduct a **material layoff in the next month**.

The system combines:

- Structured layoff data from Layoffs.fyi
- LLM-extracted qualitative event signals
- Temporal decay modeling
- Financial severity scoring
- Industry and geographic pressure features
- 12-month rolling sequence modeling
- LSTM-based probabilistic forecasting
- Walk-forward time-based evaluation

The final output is a probability:

P(Material Layoff in Month t+1 | Company State at Month t)

---

## 📊 Data Source

Primary dataset:
- Layoffs.fyi (structured global tech layoff tracker)

The raw layoff dataset was transformed into a monthly company-level panel.

Additional structured features (event flags such as restructuring, cost-cutting, acquisition, macroeconomic stress, etc.) were extracted using a Large Language Model (LLM).  
The LLM extraction code is not included in this repository; the modeling pipeline begins from the structured dataset.

---

## 🧠 Problem Definition

For each company-month, the model predicts:

Target = 1  
If a material layoff occurs in the next month

Target = 0  
Otherwise

The target is created using forward-shifting to ensure leakage-free prediction.

---

## 🏗 Pipeline Architecture

1. Monthly panel construction (company × month)
2. Event aggregation at company-month level
3. Temporal signal decay modeling
4. Material layoff severity scoring
5. Company-level historical feature engineering
6. Industry and geographic pressure features
7. Momentum and signal delta features
8. Target shifting (next-month forecasting)
9. 12-month rolling sequence generation
10. LSTM model training
11. Expanding window walk-forward validation (2025)

---

## 🔥 Feature Engineering

### 1️⃣ LLM-Derived Event Flags
Binary indicators capturing:

- Restructuring
- Cost-cutting
- Acquisition
- Revenue decline
- Profitability pressure
- Runway issues
- Macroeconomic stress
- Industry downturn
- Regulatory pressure
- Workforce-specific impact signals

---

### 2️⃣ Temporal Decay Modeling

Event signals are transformed using exponential decay:

Signal_t = Flag_t + α * Signal_(t-1)

Different half-lives are used for different signal categories to model persistence.

---

### 3️⃣ Material Severity Modeling

Not all layoffs are equal.

A material risk score is computed using:

- Absolute scale of layoffs
- Relative workforce impact
- Financial fragility (funding normalization)
- Capital buffer adjustment for VC-backed firms

Binary material_layoff_event is created using a threshold.

---

### 4️⃣ Company Historical Features

- Layoff frequency (3m, 6m, 12m)
- Maximum severity (6m, 12m)
- Months since last layoff

---

### 5️⃣ Contextual Ecosystem Pressure

Industry-level:
- Rolling layoff counts
- Pressure index (normalized by active firms)

Geographic-level:
- Country layoff pressure metrics

---

### 6️⃣ Momentum Features

- Change in macroeconomic signals
- Change in industry pressure

---

## 🔁 Sequence Construction

Each training sample consists of:

12 months of company history  
→ Predict layoff risk for next month

Shape:

(Samples, 12, Features)

---

## 🤖 Model Architecture

- Input Layer
- Masking Layer
- LSTM (64 units)
- Dropout (0.3)
- Dense(1, Sigmoid)

Loss: Binary Crossentropy  
Optimizer: Adam  
Metrics: AUC, Recall  

Class imbalance is handled using dynamically computed class weights.

---

## 📈 Evaluation Strategy

Walk-forward expanding window validation:

For each month in 2025:
- Train on all data before month t
- Predict month t
- Evaluate performance

Metrics:

- ROC-AUC
- PR-AUC
- F1 Score
- Precision / Recall
- Confusion Matrix

Threshold optimization is performed globally across 2025 predictions.

---

## 🚀 Output

For each company-month, the model outputs:

Predicted Risk ∈ [0, 1]

Example:

Company A → 0.82 (High Risk)  
Company B → 0.17 (Low Risk)

---

## 📂 Repository Structure

```
.
├── layoff.ipynb               # Data preparation & EDA
├── ver3.ipynb                 # Full modeling + walk-forward pipeline
├── dataset.csv                # Structured LLM-enhanced dataset
├── README.md
├── requirements.txt
```

---

## 🧪 Reproducibility

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run:
   - layoff.ipynb (data preparation)
   - ver3.ipynb (feature engineering + modeling)

---

## ⚠ Notes

- LLM feature extraction code is not included.
- The repository focuses on structured modeling and forecasting.
- The system is leakage-free and time-aware.

---

## 🎯 Applications

- Venture Capital portfolio risk monitoring
- Corporate distress early warning systems
- Macro labor market analysis
- Credit risk modeling
- Workforce planning analytics

---

## 📌 Summary

This project builds a fully temporal, probabilistic, and leakage-free layoff forecasting system integrating:

- Structured layoff data
- LLM-derived qualitative signals
- Decay-based narrative modeling
- Financial fragility assessment
- Ecosystem contagion effects
- Sequential deep learning
- Walk-forward validation

It demonstrates advanced time-series feature engineering and real-world ML deployment design.
