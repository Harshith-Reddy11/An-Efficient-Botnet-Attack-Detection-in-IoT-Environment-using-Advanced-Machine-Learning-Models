# IoT Botnet Attack Detection Using Hybrid Machine Learning

A machine learning-based intrusion detection system designed to identify and classify malicious network traffic in Internet of Things (IoT) environments using a hybrid ensemble of Random Forest, Support Vector Machine (SVM), and Logistic Regression.

## Overview

The increasing adoption of Internet of Things (IoT) devices has introduced significant network security challenges, particularly from botnet attacks and malicious network activities. Traditional signature-based intrusion detection systems often struggle to identify evolving attack patterns.

This project implements a hybrid machine learning approach that combines multiple classification algorithms to detect malicious network traffic. By leveraging the complementary capabilities of individual classifiers, the system aims to improve detection reliability and handle imbalanced network traffic data.

## Key Features

- Hybrid machine learning architecture combining Random Forest, SVM, and Logistic Regression.
- Majority-voting mechanism for final classification.
- Network traffic preprocessing and feature engineering.
- Class imbalance handling using SMOTE.
- Feature selection using correlation analysis and Random Forest feature importance.
- Performance evaluation using standard classification metrics.
- Flask-based web application for interactive traffic prediction.

## System Architecture

The detection pipeline consists of the following stages:



1. **Data preprocessing:** Removal of duplicate and missing records, categorical encoding, and feature normalization.
2. **Class balancing:** Application of SMOTE to improve representation of minority attack categories.
3. **Feature selection:** Identification of relevant features using correlation analysis and Random Forest feature importance.
4. **Model training:** Independent training of Random Forest, Support Vector Machine, and Logistic Regression classifiers.
5. **Ensemble prediction:** Combination of individual classifier predictions through majority voting.
6. **Performance evaluation:** Assessment using accuracy, precision, recall, F1-score, and confusion matrices.

## Machine Learning Models

| Algorithm              | Purpose                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| Random Forest          | Captures complex feature relationships and identifies important network traffic attributes |
| Support Vector Machine | Separates network traffic classes using classification boundaries                          |
| Logistic Regression    | Provides a probabilistic classification approach                                           |
| Hybrid Ensemble        | Combines predictions from all three classifiers using majority voting                      |

## Dataset

The project uses IoT network intrusion data containing normal traffic and multiple categories of malicious activity.

Dataset files are available in the `Dataset/` directory:

- `UNSW_NB15.csv`
- `testData.csv`

The classification task considers network traffic characteristics and attack categories such as DoS, DDoS, reconnaissance, backdoor, and other malicious activities.

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix



## Repository Structure

```text
.
├── Dataset/
│   ├── UNSW_NB15.csv
│   └── testData.csv
│
├── classical_ml/
│   ├── train_evaluate.py
│   ├── requirements.txt
│   ├── README.md
│   └── results/
│
├── model/
│   └── Trained model artifacts
│
├── paper/
│   └── Research document
│
├── static/
│   └── Application static files
│
├── templates/
│   └── Flask HTML templates
│
├── BotnetDetection.ipynb
├── BotnetDetection.html
├── test.py
├── attention.py
├── database.txt
├── requirements.txt
└── README.md
```

The repository also retains deep-learning notebooks and model artifacts from the broader project development. The `classical_ml/` directory contains the separate implementation of the three-classifier ensemble.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Harshith-Reddy11/An-Efficient-Botnet-Attack-Detection-in-IoT-Environment-using-Advanced-Machine-Learning-Models.git
```

### 2. Navigate to the project

```bash
cd An-Efficient-Botnet-Attack-Detection-in-IoT-Environment-using-Advanced-Machine-Learning-Models
```

### 3. Install dependencies

```bash
pip install -r classical_ml/requirements.txt
```

### 4. Run the machine learning pipeline

```bash
python classical_ml/train_evaluate.py
```

### 5. View evaluation outputs

Generated metrics, classification reports, and confusion matrices are stored in:

```text
classical_ml/results/
```

## Web Application

The project includes a Flask-based web interface that provides:

- User registration and authentication.
- Interactive traffic prediction.
- Integration with the intrusion detection functionality.
- A user-friendly interface for exploring network traffic classifications.

## Future Enhancements

- Deployment and evaluation on real-time IoT edge devices.
- Lightweight model optimization for resource-constrained environments.
- Federated learning for distributed intrusion detection.
- Explainable AI for interpreting classification decisions.
- Improved robustness against evolving network attack patterns.

## Technologies Used

**Programming:** Python

**Machine Learning:** Scikit-learn, Random Forest, SVM, Logistic Regression

**Data Processing:** Pandas, NumPy

**Imbalance Handling:** SMOTE

**Web Development:** Flask, HTML, CSS, JavaScript

**Dataset:** UNSW-NB15

## Contributors

- Sathwika Golanakonda
- Sritha Nallamala
- Harshith Reddy Rajalingari

Department of Computer Science and Engineering (AI & ML)
CMR Institute of Technology, Hyderabad, India.
