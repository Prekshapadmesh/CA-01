import pandas as pd

class MonthToMonthContract:
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)  # Use the dataset_path parameter

    def calculate_churn_rate(self):
        # Calculate churn rate for Month-to-Month contracts
        churned_customers = self.data[self.data['Contract'] == 'Month-to-Month']['Churn'].sum()
        total_customers = self.data[self.data['Contract'] == 'Month-to-Month'].shape[0]
        return churned_customers / total_customers if total_customers > 0 else 0

    def retention_strategy(self):
        # Identify at-risk customers with low support interaction
        at_risk_customers = self.data[(self.data['Contract'] == 'Month-to-Month') & 
                                       (self.data['Customer Support Interactions'] < 2)]
        return at_risk_customers[['Customer ID', 'Customer Support Interactions']]
