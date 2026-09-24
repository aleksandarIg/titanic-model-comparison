import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_path = "./Titanic-Dataset.csv"
df = pd.read_csv(csv_path)

print("\nHead\n")
print(df.head())
print("\nInfo\n")
print(df.info())
print("\nDescribe\n")
print(df.describe())
print("\nSum of NULL\n")
print(df.isnull().sum())

print("\nNumber of survived:")
survived = (df["Survived"]==1)
print(survived.sum())

print("\nGender of survived:")
percentage = df[survived]["Sex"].value_counts(normalize=True)*100
print(percentage.round(2))


print("\nClasses of survived on diagram:")
classes = df[survived]["Pclass"].value_counts()
print(classes)

print("\n Ages of survived:")
plt.figure(figsize=(10, 6))

plt.hist(
    df['Age'].dropna(),
    bins=30,
    color='skyblue',
    edgecolor='black'
)

plt.title('Age Distribution of Passengers', fontsize=14, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig(os.path.join(OUTPUT_DIR, "age_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

X = df[["Age", "Sex", "Pclass", "Fare", "FamilySize"]].copy()
y = df["Survived"]

median_age = X["Age"].median()
X["Age"] = X["Age"].fillna(median_age)


encoder = OneHotEncoder(sparse_output=False)
sex_encoded = encoder.fit_transform(X[["Sex"]])
sex_df = pd.DataFrame(
    sex_encoded, columns=encoder.get_feature_names_out(["Sex"]), index=X.index
)

X = X.drop(columns=["Sex"])
X = pd.concat([X, sex_df], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


counts = y_train.value_counts()

plt.figure(figsize=(7, 5))
plt.bar(
    ["Did Not Survive (0)", "Survived (1)"],
    counts.values,
    color=["crimson", "mediumseagreen"],
)
plt.title("Class Distribution in y_train")
plt.ylabel("Number of Samples")
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution_train.png"), dpi=150, bbox_inches="tight")
plt.close()

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns =X_train.columns,
    index = X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index = X_test.index
)
print("\nX_train After scaling\n ")
print(X_train_scaled.head(3))

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled,y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test,y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%\n")

single_instance = X_test_scaled.iloc[[1]]

print("Prediction:", model.predict(single_instance))
original_values = scaler.inverse_transform(single_instance)
original_df = pd.DataFrame(
    original_values,
    columns=X_test_scaled.columns, 
    index=single_instance.index
)
print(original_df.T)
print("Value:", y_test.iloc[1])




print(" -- Survival rate by gender -- ")
print(df.groupby("Sex")["Survived"].mean() * 100)

print("Comparing different ML Models")

print(f"\nAccuracy of Logistic Regression: {acc * 100:.2f}%\n")

knc = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
knc.fit(X_train_scaled,y_train)
y_pred_knc = knc.predict(X_test_scaled)
acc_knc = accuracy_score(y_test,y_pred_knc)
print(f"\nAccuracy Of KNeighborsClassifier: {acc_knc * 100:.2f}%\n")

dtc = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dtc.fit(X_train_scaled,y_train)
y_pred_dtc = dtc.predict(X_test_scaled)
acc_dtc = accuracy_score(y_test,y_pred_dtc)
print(f"\nAccuracy Of Decision Tree Classifier: {acc_dtc * 100:.2f}%\n")

rfc = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42,n_jobs=-1)
rfc.fit(X_train_scaled,y_train)
y_pred_rfc = rfc.predict(X_test_scaled)
acc_rfc = accuracy_score(y_test,y_pred_rfc)
print(f"\nAccuracy Of Random Forest classifier: {acc_rfc * 100:.2f}%\n")

cm = confusion_matrix(y_test,y_pred)
print("Confusion matrix LR: ")
print(cm)
disp= ConfusionMatrixDisplay (confusion_matrix=cm, display_labels=["Did Not Survive (0)", "Survived (1)"] )
disp.plot(cmap="Blues")
plt.title("Confusion matrix - Logistic Regression")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_logistic_regression.png"), dpi=150, bbox_inches="tight")
plt.close()

cm_knc = confusion_matrix(y_test,y_pred_knc)
print("Confusion matrix KNN: ")
print(cm_knc)
disp= ConfusionMatrixDisplay (confusion_matrix=cm_knc, display_labels=["Did Not Survive (0)", "Survived (1)"] )
disp.plot(cmap="Blues")
plt.title("Confusion matrix - KNeighbors Classifier")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_knn.png"), dpi=150, bbox_inches="tight")
plt.close()

cm_dtc= confusion_matrix(y_test,y_pred_dtc)
print("Confusion matrix DTC: ")
print(cm_dtc)
disp= ConfusionMatrixDisplay (confusion_matrix=cm_dtc, display_labels=["Did Not Survive (0)", "Survived (1)"] )
disp.plot(cmap="Blues")
plt.title("Confusion matrix - DecisionTree Classifier")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_decision_tree.png"), dpi=150, bbox_inches="tight")
plt.close()

cm_rfc= confusion_matrix(y_test,y_pred_rfc)
print("Confusion matrix RFC: ")
print(cm_rfc)
disp= ConfusionMatrixDisplay (confusion_matrix=cm_rfc, display_labels=["Did Not Survive (0)", "Survived (1)"] )
disp.plot(cmap="Blues")
plt.title("Confusion matrix - RandomForest Classifier")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_random_forest.png"), dpi=150, bbox_inches="tight")
plt.close()

models_data = {
    "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_knc),
        accuracy_score(y_test, y_pred_dtc),
        accuracy_score(y_test, y_pred_rfc)
    ],
    "Precision": [
        precision_score(y_test, y_pred),
        precision_score(y_test, y_pred_knc),
        precision_score(y_test, y_pred_dtc),
        precision_score(y_test, y_pred_rfc)
    ],
    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test, y_pred_knc),
        recall_score(y_test, y_pred_dtc),
        recall_score(y_test, y_pred_rfc)
    ],
    "F1": [
        f1_score(y_test, y_pred),
        f1_score(y_test, y_pred_knc),
        f1_score(y_test, y_pred_dtc),
        f1_score(y_test, y_pred_rfc)
    ]
}
results_df = pd.DataFrame(models_data)
results_df = results_df.round(4)
print("\n-- Comparing different models --")
print(results_df.to_string(index=False))

correlations = df.select_dtypes(include=['number']).corr()["Survived"].sort_values(ascending=False)

print("-- Correlations with Survived --")
print(correlations)

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Decision Tree": dtc.feature_importances_,
    "Random Forest": rfc.feature_importances_
}).sort_values(by="Random Forest", ascending=False)

print("\n-- Comparing feature importances --")
print(importance_df.round(4).to_string(index=False))

print(f"\nAll plots saved to the '{OUTPUT_DIR}/' folder.")
