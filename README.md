# simple_markowitz

This Python package provides functions for analyzing stock portfolios, including downloading stock data, calculating performance metrics, and optimizing portfolios.

## Installation

```bash
pip install -e .
```

## Usage

```python
import portfolio_analysis as pa

# Read data from an Excel file
stocks = pa.read_data('path_to_excel_file.xlsx')

# Download stock data from Yahoo Finance
symbols = "SPCE MSFT F TSLA PFE ^GSPC"
start_date = "2023-01-01"
end_date = "2023-12-31"
stocks = pa.download_data(symbols, start_date, end_date)

# Calculate logarithmic returns
log_returns = pa.log_data(stocks)

# Calculate performance metrics
perf = pa.performance(log_returns)
print(perf)

# Calculate covariance matrix
cov_matrix = pa.cov_matrix(log_returns)
print(cov_matrix)

# Calculate optimal portfolio
optimal_portfolio = pa.optimal_portfolio(perf, cov_matrix)
print(optimal_portfolio)

# Plot Markowitz bullet
pa.bullet_plot(random_set, optimal_set)

# Write data to an Excel file
pa.write_data('output.xlsx', 'sheet_name', [2, 1], stocks)
```


