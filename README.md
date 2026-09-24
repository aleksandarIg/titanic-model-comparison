# 🚢 Titanic Survival Prediction — Model Comparison

An extended version of the [Titanic Survival Prediction](https://github.com/aleksandarIg/titanic-survival-prediction) project. Instead of a single model, this project trains and compares **four different classifiers** to see which one predicts survival best, and looks at which features matter most.

## 📋 What it does

- Loads the [Titanic dataset](https://www.kaggle.com/c/titanic/data) and prints basic exploratory info (`head`, `info`, `describe`, missing values)
- Analyzes survival rate by gender and passenger class
- Engineers a new `FamilySize` feature (`SibSp` + `Parch` + 1)
- Prepares features (`Age`, `Sex`, `Pclass`, `Fare`, `FamilySize`):
  - Fills missing `Age` values with the median
  - One-hot encodes the `Sex` column
- Splits the data into train/test sets (80/20, stratified) and scales features with `StandardScaler`
- Trains and compares **4 models**:
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
- Evaluates each model with **Accuracy, Precision, Recall, and F1-score**
- Plots a confusion matrix for every model
- Compares **feature importances** between Decision Tree and Random Forest
- Shows correlation of each numeric feature with survival
- Saves every chart as a `.png` file to a `plots/` folder (no pop-up windows — the script runs start to finish on its own)

## ⚙️ Prerequisites

- Python 3.8+
- The `Titanic-Dataset.csv` file (download from [Kaggle](https://www.kaggle.com/c/titanic/data)) placed in the project root

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/aleksandarIg/titanic-model-comparison.git
cd titanic-model-comparison
```

2. (Optional but recommended) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Make sure `Titanic-Dataset.csv` is in the same folder as `main.py`, then run:

```bash
python main.py
```

The script prints all results (model comparison table, feature importances, correlations) to the console, and saves every chart to the `plots/` folder:

- `age_distribution.png`
- `class_distribution_train.png`
- `confusion_matrix_logistic_regression.png`
- `confusion_matrix_knn.png`
- `confusion_matrix_decision_tree.png`
- `confusion_matrix_random_forest.png`

## 📁 Project structure

```
titanic-model-comparison/
├── main.py
├── Titanic-Dataset.csv    # dataset, add it yourself (see below)
├── plots/                 # generated charts, not pushed to GitHub
├── requirements.txt
├── .gitignore
└── README.md
```

## 📦 Dataset

This project uses the **Titanic - Machine Learning from Disaster** dataset from Kaggle:
👉 https://www.kaggle.com/c/titanic/data

Download `Titanic-Dataset.csv` (or `train.csv` renamed) and place it in the project root before running the script.

## 📝 License

Free to use and modify as needed.
