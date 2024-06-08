from read_and_preprocess_data import read_and_preprocess_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

_, X, y = read_and_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a single figure and axis matrix for subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Adjust the grid if necessary
fig.suptitle('Confusion Matrices for Various Models', fontsize=16)

models = [
    (LogisticRegression(solver='liblinear'), 'Logistic Regression'),
    (GaussianNB(), 'Naive Bayes'),
    (DecisionTreeClassifier(random_state=42), 'Decision Tree'),
    (RandomForestClassifier(n_estimators=100, random_state=42), 'Random Forest'),
    (SVC(kernel='rbf', C=1.0, random_state=42), 'SVM'),
    (KNeighborsClassifier(n_neighbors=5), 'KNN')  # KNN with 5 neighbors
]

accuracies = []

for i, (model, title) in enumerate(models):
    ax = axes[i//3, i%3]  # Determine the position on the grid
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"{title} Accuracy: {accuracy}")
    print(f"{title} Classification Report:\n{classification_report(y_test, y_pred)}")

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('all_confusion_matrices.png', format='png', dpi=300)

# Create a bar plot for accuracies
plt.figure(figsize=(10,6))
model_names = [name for _, name in models]
plt.bar(model_names, accuracies, color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
plt.savefig('model_accuracies.png', format='png', dpi=300)