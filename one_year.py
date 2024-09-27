import pandas as pd

class OneYearContract:
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)

    def analyze_renewal_rates(self):
        # Calculate renewal rates for One-Year contracts
        one_year_data = self.data[self.data['Contract'] == 'One-Year']
        renewed_customers = one_year_data['Renewed'].sum()
        total_customers = one_year_data.shape[0]
        return renewed_customers / total_customers if total_customers > 0 else 0

    def evaluate_loyalty_program(self):
        # Check Customer Support Interactions for One-Year contracts
        loyalty_effect = self.data[(self.data['Contract'] == 'One-Year') & 
                                   (self.data['Customer Support Interactions'] > 3)]
        return loyalty_effect[['Customer ID', 'Customer Support Interactions']]
