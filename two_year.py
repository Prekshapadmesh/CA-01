import pandas as pd

class TwoYearContract:
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)

    def analyze_long_term_behavior(self):
        # Analyze churn for Two-Year contracts
        two_year_data = self.data[self.data['Contract'] == 'Two-Year']
        churned_customers = two_year_data['Churn'].sum()
        total_customers = two_year_data.shape[0]
        return churned_customers / total_customers if total_customers > 0 else 0

    def predict_churn(self):
        # Predict churn based on payment method and interactions
        churn_prediction = self.data[(self.data['Contract'] == 'Two-Year') & 
                                     (self.data['Payment Method'] == 'Credit Card') & 
                                     (self.data['Customer Support Interactions'] < 3)]
        return churn_prediction[['Customer ID', 'Payment Method', 'Customer Support Interactions']]
