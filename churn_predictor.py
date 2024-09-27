import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ChurnPredictor:
    def __init__(self, data):
        """
        Initialize ChurnPredictor with dataset
        """
        self.data = data
        self.model = None

    def train_model(self):
        """
        Train the Random Forest model on the data
        """
        # Prepare the features and target variable
        X = pd.get_dummies(self.data[['Contract_Type', 'Monthly_Charges', 'Tenure']], drop_first=True)
        y = self.data['Churn_Flag']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train the Random Forest model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        # Optional: evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model trained with accuracy: {accuracy:.2f}')

    def predict_churn(self, new_customer):
        """
        Predict churn for a new customer
        :param new_customer: List of features for a new customer
        :return: Churn prediction (0 or 1)
        """
        # Convert the new customer's features to a DataFrame
        new_customer_df = pd.DataFrame([new_customer], columns=['Monthly_Charges', 'Tenure', 'Contract_Type_One-Year', 'Contract_Type_Month-to-Month'])
        
        # Predict churn
        return self.model.predict(new_customer_df)[0]

    def retention_rate(self):
        """
        Calculate retention rate based on the dataset
        :return: Retention rate as a float
        """
        total_customers = len(self.data)
        retained_customers = total_customers - self.data['Churn_Flag'].sum()
        return retained_customers / total_customers

