
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
 # AutoML Functionality
def auto_ml_churn_prediction(data):
        # Prepare the features and target variable
        X = data[['Contract_Type', 'Monthly_Charges', 'Tenure']]
        y = data['Churn_Flag']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Define the models to evaluate
        models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        best_model = None
        best_accuracy = 0

        # Train and evaluate each model
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)

            print(f'{name}: Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

        print(f'\nBest Model: {best_model}')
        return best_model

    # Call the AutoML function
    best_model = auto_ml_churn_prediction(data)