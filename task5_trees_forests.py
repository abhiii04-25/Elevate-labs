"""
Task 5: Decision Trees and Random Forests
Dataset: heart.csv (Heart Disease Classification)

Objective: Learn tree-based models for classification.
Steps:
1. Train a Decision Tree Classifier and visualize the tree.
2. Analyze overfitting and control tree depth.
3. Train a Random Forest and compare accuracy.
4. Interpret feature importances.
5. Evaluate using cross-validation.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------
df = pd.read_csv("heart.csv")
print("Shape:", df.shape)
n_dupes = df.duplicated().sum()
print(f"Duplicate rows: {n_dupes} out of {len(df)}")
if n_dupes > 0:
    print(f"-> Dropping duplicates to prevent train/test leakage. "
          f"Unique rows: {len(df.drop_duplicates())}")
    df = df.drop_duplicates().reset_index(drop=True)
print("Shape after dedup:", df.shape)

print("\nClass balance (target):")
print(df["target"].value_counts())

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------
# 1. Train a Decision Tree Classifier and visualize the tree
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 1: Decision Tree Classifier (unrestricted depth)")
print("=" * 60)

dt_full = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_full.fit(X_train, y_train)

train_acc_full = accuracy_score(y_train, dt_full.predict(X_train))
test_acc_full = accuracy_score(y_test, dt_full.predict(X_test))
print(f"Train accuracy: {train_acc_full:.4f}")
print(f"Test accuracy : {test_acc_full:.4f}")
print(f"Tree depth reached: {dt_full.get_depth()}, Leaves: {dt_full.get_n_leaves()}")

# Visualize the full (likely overfit) tree - limited to depth 3 for readability
plt.figure(figsize=(20, 10))
plot_tree(
    dt_full,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    max_depth=3,  # only display top levels for readability; model itself is unrestricted
    fontsize=9,
)
plt.title("Decision Tree (unrestricted depth, showing top 3 levels)")
plt.tight_layout()
plt.savefig("01_decision_tree_full.png", dpi=150)
plt.close()
print("Saved: 01_decision_tree_full.png")

# ---------------------------------------------------------------
# 2. Analyze overfitting and control tree depth
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Overfitting Analysis - Accuracy vs max_depth")
print("=" * 60)

depths = range(1, 21)
train_scores, test_scores = [], []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
    test_scores.append(accuracy_score(y_test, clf.predict(X_test)))

for d, tr, te in zip(depths, train_scores, test_scores):
    print(f"max_depth={d:2d}  train_acc={tr:.4f}  test_acc={te:.4f}")

plt.figure(figsize=(9, 6))
plt.plot(list(depths), train_scores, marker="o", label="Train Accuracy")
plt.plot(list(depths), test_scores, marker="s", label="Test Accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Overfitting Analysis (Train vs Test Accuracy)")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(list(depths))
plt.tight_layout()
plt.savefig("02_overfitting_depth_curve.png", dpi=150)
plt.close()
print("Saved: 02_overfitting_depth_curve.png")

best_depth = int(np.array(test_scores).argmax()) + 1
print(f"\nBest max_depth by test accuracy: {best_depth} (test_acc={max(test_scores):.4f})")

# Train the "pruned" / depth-controlled tree using best depth
dt_pruned = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)
dt_pruned.fit(X_train, y_train)
print(f"Pruned tree -> Train acc: {accuracy_score(y_train, dt_pruned.predict(X_train)):.4f}, "
      f"Test acc: {accuracy_score(y_test, dt_pruned.predict(X_test)):.4f}")

plt.figure(figsize=(20, 10))
plot_tree(
    dt_pruned,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title(f"Decision Tree (max_depth={best_depth}, controlled to reduce overfitting)")
plt.tight_layout()
plt.savefig("03_decision_tree_pruned.png", dpi=150)
plt.close()
print("Saved: 03_decision_tree_pruned.png")

# ---------------------------------------------------------------
# 3. Train a Random Forest and compare accuracy
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Random Forest Classifier")
print("=" * 60)

rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

train_acc_rf = accuracy_score(y_train, rf.predict(X_train))
test_acc_rf = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest -> Train acc: {train_acc_rf:.4f}, Test acc: {test_acc_rf:.4f}")

print("\nModel comparison (Test Accuracy):")
comparison = {
    "Decision Tree (unrestricted)": test_acc_full,
    f"Decision Tree (max_depth={best_depth})": max(test_scores),
    "Random Forest (n_estimators=200)": test_acc_rf,
}
for name, acc in comparison.items():
    print(f"  {name:38s}: {acc:.4f}")

plt.figure(figsize=(8, 5))
names = list(comparison.keys())
vals = list(comparison.values())
bars = plt.bar(names, vals, color=["#e07a5f", "#f2cc8f", "#3d5a80"])
plt.ylabel("Test Accuracy")
plt.title("Model Comparison: Test Accuracy")
plt.ylim(0, 1.05)
plt.xticks(rotation=15, ha="right")
for bar, v in zip(bars, vals):
    plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig("04_model_comparison.png", dpi=150)
plt.close()
print("Saved: 04_model_comparison.png")

print("\nClassification report - Random Forest:")
print(classification_report(y_test, rf.predict(X_test), target_names=["No Disease", "Disease"]))

cm = confusion_matrix(y_test, rf.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Random Forest - Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("05_confusion_matrix_rf.png", dpi=150)
plt.close()
print("Saved: 05_confusion_matrix_rf.png")

# ---------------------------------------------------------------
# 4. Interpret feature importances
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Feature Importances")
print("=" * 60)

importances_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances_dt = pd.Series(dt_pruned.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nRandom Forest feature importances:")
print(importances_rf.round(4))

print("\nDecision Tree (pruned) feature importances:")
print(importances_dt.round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
importances_rf.plot(kind="barh", ax=axes[0], color="#3d5a80")
axes[0].invert_yaxis()
axes[0].set_title("Random Forest - Feature Importance")
axes[0].set_xlabel("Importance")

importances_dt.plot(kind="barh", ax=axes[1], color="#e07a5f")
axes[1].invert_yaxis()
axes[1].set_title(f"Decision Tree (depth={best_depth}) - Feature Importance")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("06_feature_importances.png", dpi=150)
plt.close()
print("Saved: 06_feature_importances.png")

# ---------------------------------------------------------------
# 5. Evaluate using cross-validation
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Cross-Validation (5-fold)")
print("=" * 60)

cv_dt_full = cross_val_score(DecisionTreeClassifier(random_state=RANDOM_STATE), X, y, cv=5)
cv_dt_pruned = cross_val_score(DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE), X, y, cv=5)
cv_rf = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), X, y, cv=5)

print(f"Decision Tree (unrestricted) CV scores: {np.round(cv_dt_full, 4)} | mean={cv_dt_full.mean():.4f} std={cv_dt_full.std():.4f}")
print(f"Decision Tree (depth={best_depth}) CV scores:   {np.round(cv_dt_pruned, 4)} | mean={cv_dt_pruned.mean():.4f} std={cv_dt_pruned.std():.4f}")
print(f"Random Forest CV scores:                 {np.round(cv_rf, 4)} | mean={cv_rf.mean():.4f} std={cv_rf.std():.4f}")

cv_summary = pd.DataFrame({
    "Decision Tree (full)": cv_dt_full,
    f"Decision Tree (depth={best_depth})": cv_dt_pruned,
    "Random Forest": cv_rf,
})
cv_summary.index = [f"Fold {i+1}" for i in range(5)]
cv_summary.loc["Mean"] = cv_summary.mean()
cv_summary.loc["Std"] = cv_summary.std()
cv_summary.to_csv("cv_results_summary.csv")
print("\nSaved: cv_results_summary.csv")

plt.figure(figsize=(8, 6))
data_to_plot = [cv_dt_full, cv_dt_pruned, cv_rf]
labels = ["Decision Tree\n(full)", f"Decision Tree\n(depth={best_depth})", "Random Forest"]
bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
colors = ["#e07a5f", "#f2cc8f", "#3d5a80"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
plt.ylabel("Cross-Validation Accuracy")
plt.title("5-Fold Cross-Validation Accuracy Comparison")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("07_cv_boxplot.png", dpi=150)
plt.close()
print("Saved: 07_cv_boxplot.png")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Model':35s}{'Test Acc':>10s}{'CV Mean':>10s}{'CV Std':>10s}")
print(f"{'Decision Tree (unrestricted)':35s}{test_acc_full:10.4f}{cv_dt_full.mean():10.4f}{cv_dt_full.std():10.4f}")
print(f"{'Decision Tree (depth='+str(best_depth)+')':35s}{max(test_scores):10.4f}{cv_dt_pruned.mean():10.4f}{cv_dt_pruned.std():10.4f}")
print(f"{'Random Forest':35s}{test_acc_rf:10.4f}{cv_rf.mean():10.4f}{cv_rf.std():10.4f}")

print("\nAll outputs saved in current directory.")
