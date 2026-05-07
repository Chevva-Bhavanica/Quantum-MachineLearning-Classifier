from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# =========================================================
# FUNCTION TO RETURN ALL CLASSICAL MODELS
# =========================================================

def get_classical_models():

    models = {

        # =================================================
        # Support Vector Machine
        # =================================================
        "SVM": SVC(
            kernel='rbf',
            probability=True
        ),

        # =================================================
        # Logistic Regression
        # =================================================
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        # =================================================
        # Decision Tree
        # =================================================
        "Decision Tree": DecisionTreeClassifier(
            criterion='gini',
            max_depth=5,
            random_state=42
        ),

        # =================================================
        # Random Forest
        # =================================================
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),

        # =================================================
        # K-Nearest Neighbors
        # =================================================
        "KNN": KNeighborsClassifier(
            n_neighbors=3
        ),

        # =================================================
        # Naive Bayes
        # =================================================
        "Naive Bayes": GaussianNB(),

        # =================================================
        # AdaBoost Classifier
        # =================================================
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            random_state=42
        ),

        # =================================================
        # Gradient Boosting
        # =================================================
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),

        # =================================================
        # Neural Network / MLP
        # =================================================
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=42
        )
    }

    return models

