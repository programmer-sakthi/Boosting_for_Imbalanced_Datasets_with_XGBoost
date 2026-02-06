# Boosting for Imbalanced Datasets with XGBoost

## 📌 Overview

Handling imbalanced datasets is one of the biggest challenges in machine
learning. Traditional models like Support Vector Machines and Random
Forest often get biased toward majority classes, leading to poor
performance on minority classes.

This project demonstrates how **XGBoost**, a powerful gradient boosting
algorithm, can be used to improve classification performance on
imbalanced datasets using techniques such as:

-   Weighted loss functions\
-   SMOTE (Synthetic Minority Oversampling Technique)\
-   Hyperparameter tuning\
-   Evaluation metrics designed for imbalanced classification

------------------------------------------------------------------------

## 🚀 Project Goals

-   Apply **XGBoost** for classification on imbalanced datasets\
-   Handle class imbalance using:
    -   Class weights
    -   SMOTE oversampling
-   Tune model hyperparameters for optimal performance\
-   Evaluate using metrics suited for imbalance:
    -   Precision-Recall Curve\
    -   ROC-AUC Score\
    -   F1 Score\
    -   Confusion Matrix

------------------------------------------------------------------------

## 🧠 Why XGBoost for Imbalanced Data?

XGBoost helps because: - It focuses on misclassified samples during
boosting iterations\
- Supports class weighting using `scale_pos_weight`\
- Works well with tabular structured data\
- Highly customizable and efficient

------------------------------------------------------------------------

## 🏗️ Project Structure

    Boosting_for_Imbalanced_Datasets_with_XGBoost/
    │
    ├── data/                     # Dataset files
    ├── notebooks/                # Experiment notebooks (if any)
    ├── src/
    │   ├── preprocessing.py      # Data preprocessing & SMOTE
    │   ├── model.py              # XGBoost model training
    │   ├── tuning.py             # Hyperparameter tuning
    │   └── evaluation.py         # Metrics & plots
    │
    ├── requirements.txt
    ├── README.md
    └── main.py                   # Entry point

------------------------------------------------------------------------

## ⚙️ Installation

### 1️⃣ Clone Repository

    git clone https://github.com/programmer-sakthi/Boosting_for_Imbalanced_Datasets_with_XGBoost.git
    cd Boosting_for_Imbalanced_Datasets_with_XGBoost

### 2️⃣ Create Virtual Environment (Recommended)

    python -m venv venv
    source venv/bin/activate   # Linux / Mac
    venv\Scripts\activate      # Windows

### 3️⃣ Install Dependencies

    pip install -r requirements.txt

------------------------------------------------------------------------

## 📊 Class Imbalance Handling Techniques

### ✔ Class Weights

Used XGBoost parameter:

    scale_pos_weight = (negative_class_count / positive_class_count)

Helps the model give more importance to minority class samples.

------------------------------------------------------------------------

### ✔ SMOTE (Synthetic Minority Oversampling Technique)

SMOTE generates synthetic samples for the minority class to balance
dataset distribution.

Benefits: - Reduces overfitting compared to random oversampling\
- Improves recall for minority class

------------------------------------------------------------------------

## 🤖 Model Training

XGBoost hyperparameters tuned: - `learning_rate` - `max_depth` -
`n_estimators` - `subsample` - `colsample_bytree`

Tuning methods used: - Grid Search / Random Search (depending on
implementation)

------------------------------------------------------------------------

## 📈 Performance Evaluation

Since accuracy is misleading for imbalanced datasets, we use:

### 🔹 Precision-Recall Curve

Best when: - Dataset is highly imbalanced - Focus is on minority class
detection

### 🔹 ROC-AUC Score

Measures overall classification ability across thresholds.

### 🔹 F1 Score

Balances precision and recall.

------------------------------------------------------------------------

## 📌 Results

✅ Improved minority class detection\
✅ Better Precision-Recall balance\
✅ Higher ROC-AUC compared to baseline models

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Python\
-   XGBoost\
-   Scikit-learn\
-   Imbalanced-learn (SMOTE)\
-   NumPy\
-   Pandas\
-   Matplotlib / Seaborn

------------------------------------------------------------------------

## 📚 Key Learnings

-   Handling imbalance is as important as model selection\
-   Boosting methods are highly effective for skewed datasets\
-   Evaluation metrics must match real-world goals

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Try ensemble with LightGBM / CatBoost\
-   Add cross-validation pipeline\
-   Deploy model using FastAPI or Flask\
-   Add experiment tracking (MLflow / Weights & Biases)

------------------------------------------------------------------------

## 🤝 Contributing

Contributions are welcome!\
If you'd like to improve the project: - Fork the repo\
- Create a feature branch\
- Submit a pull request

------------------------------------------------------------------------

## 📜 License

This project is open-source and available under the MIT License.

------------------------------------------------------------------------

## 👨‍💻 Author

**Sakthi Palanisamy**\
GitHub: https://github.com/programmer-sakthi
