# Elevate-labs
AI/ML Virtual Internship @ Elevate Labs — hands-on projects covering data cleaning, EDA, feature engineering, model building, and evaluation using Python, Pandas, and Scikit-learn.

# Elevate-Labs

AI/ML Virtual Internship — Elevate Labs

This repository contains the tasks, notebooks, and project work completed as part of the **Elevate Labs AI/ML Virtual Internship**. Each task focuses on a core concept in the machine learning workflow, from raw data handling to model building, with hands-on implementation in Python.

## 📌 About the Internship

Elevate Labs' AI/ML internship is a project-based program designed to build practical skills across the ML pipeline — data cleaning, exploratory data analysis, feature engineering, model training, and evaluation — using real datasets and industry-standard tools.

## 🛠️ Tools & Technologies

- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Environment:** Jupyter Notebook

## 📂 Repository Structure

```
Elevate-Labs/
│
├── Task-1-Data-Cleaning-Preprocessing/
│   ├── titanic_cleaning_eda.ipynb      # Full cleaning + EDA notebook
│   ├── Titanic-Dataset.csv             # Raw dataset
│   ├── titanic_cleaned.csv             # Final cleaned & preprocessed dataset
│   └── README.md                       # Task-specific notes (optional)
│
└── README.md                           # This file
```

*(Folder names above are suggestions — update to match how you've organized your files.)*

## ✅ Tasks Completed

### Task 1: Data Cleaning & Preprocessing
**Objective:** Learn how to clean and prepare raw data for machine learning.

**Dataset:** [Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

**What was done:**
- Explored dataset structure, data types, and missing values
- Performed exploratory data analysis (EDA) with visualizations — survival trends by sex, class, and embarkation port, age/fare distributions, and a correlation heatmap
- Handled missing values using median imputation (`Age`) and mode imputation (`Embarked`); dropped columns too sparse to impute (`Cabin`) or with no predictive value (`Name`, `Ticket`, `PassengerId`)
- Encoded categorical features — Label Encoding for `Sex`, One-Hot Encoding for `Embarked`
- Detected and removed outliers in `Age` and `Fare` using the IQR method, with before/after boxplots
- Normalized/standardized numerical features (`Age`, `Fare`, `SibSp`, `Parch`) using `StandardScaler`
- Exported the final cleaned, encoded, and scaled dataset ready for model training

**Key skills demonstrated:** data exploration, missing value imputation, categorical encoding, outlier detection/removal, feature scaling, data visualization

---

*More tasks will be added here as the internship progresses.*

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Elevate-Labs.git
   cd Elevate-Labs
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Launch Jupyter and open the notebook for the relevant task:
   ```bash
   jupyter notebook
   ```

## 👤 Author

Internship project completed as part of the Elevate Labs AI/ML Virtual Internship program.

