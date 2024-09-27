from month_to_month import MonthToMonthContract
from one_year import OneYearContract
from two_year import TwoYearContract

if __name__ == "__main__":
    # Load the dataset
    dataset_path = "C:/Users/preks/OneDrive/Desktop/CA1/churn_data.csv"
    
    # Initialize each contract class
    mtm_contract = MonthToMonthContract(dataset_path)
    one_year_contract = OneYearContract(dataset_path)
    two_year_contract = TwoYearContract(dataset_path)

    # Calculate churn rates
    try:
        churn_rates = {
            "Month-to-Month": mtm_contract.calculate_churn_rate(),  # Uses churn_flag
            "One-Year": 0,  # Placeholder since there's no method in OneYearContract
            "Two-Year": two_year_contract.analyze_long_term_behavior()  # Method references churn
        }
        print("Churn Rates:")
        for contract_type, rate in churn_rates.items():
            print(f"{contract_type}: {rate:.2%}")
    except KeyError as e:
        print(f"KeyError: {e} - Check if the required columns exist in your dataset.")

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