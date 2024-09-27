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
