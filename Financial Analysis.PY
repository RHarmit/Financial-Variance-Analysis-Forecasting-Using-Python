import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Sample Financial Data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
budget = np.random.randint(80000, 120000, size=12)
actual_expense = budget + np.random.randint(-10000, 10000, size=12)

# Create DataFrame
data = pd.DataFrame({'Date': dates, 'Budget': budget, 'Actual_Expense': actual_expense})
data['Variance'] = data['Actual_Expense'] - data['Budget']

def display_analysis(df):
    print("\nBudget vs. Actual Expense:")
    print(df[['Date', 'Budget', 'Actual_Expense', 'Variance']])
    
    # Calculate and print average change in variance
    df['Variance_Change'] = df['Variance'].diff()
    avg_variance_change = df['Variance_Change'].mean()
    print(f"\nAverage Change in Variance: {avg_variance_change:.2f}")

    # Plot Budget vs Actual Expense
    plt.figure(figsize=(8,4))
    plt.plot(df['Date'], df['Budget'], label='Budget', marker='o', linestyle='--')
    plt.plot(df['Date'], df['Actual_Expense'], label='Actual Expense', marker='s')
    plt.xlabel('Date')
    plt.ylabel('Amount ($)')
    plt.title('Budget vs. Actual Expense')
    plt.legend()
    plt.grid()
    plt.show()

def forecast_expenses(df):
    model = ExponentialSmoothing(df['Actual_Expense'], trend='add', seasonal=None, initialization_method='estimated').fit(smoothing_level=0.8, smoothing_trend=0.2)
    forecast_values = model.forecast(3)
    
    print("\nForecasted Expenses for Next 3 Months:")
    for i, value in enumerate(forecast_values, start=1):
        print(f"Month {i}: ${value:.2f}")

# Run Analysis
display_analysis(data)
forecast_expenses(data)
