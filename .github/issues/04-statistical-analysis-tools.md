# Add Statistical Analysis Tools to MCP Server

## Overview
Implement basic statistical analysis tools for data analysis.

## Tools to Implement

### 1. descriptive_stats
Calculate comprehensive descriptive statistics.
- **Input**: `data` (array of numbers, 1-10000 items)
- **Output**: Object with mean, median, mode, range, variance, std dev
- **Calculations**:
  - Mean (average)
  - Median (middle value)
  - Mode (most frequent)
  - Range (max - min)
  - Variance
  - Standard deviation

### 2. correlation
Calculate Pearson correlation coefficient between two datasets.
- **Input**: `x` and `y` (arrays of equal length, 2-1000 items each)
- **Output**: Correlation coefficient (-1 to 1) with interpretation
- **Interpretation**: 
  - 1.0: Perfect positive correlation
  - 0.0: No correlation
  - -1.0: Perfect negative correlation

### 3. percentile
Calculate percentile or find value at specific percentile.
- **Input**: `data` (array), `percentile` (0-100)
- **Output**: Value at given percentile
- **Example**: 50th percentile = median

### 4. outliers
Identify outliers using IQR method.
- **Input**: `data` (array of numbers)
- **Output**: Array of outlier values and their indices
- **Method**: Values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

## Implementation Requirements
- Use numpy-like efficient calculations (pure Python)
- Handle edge cases (empty data, single value, all same values)
- Provide human-readable interpretations
- Include formulas in docstrings
- Add data validation (check for non-numeric values)
- Update list_tools() and call_tool() handlers

## Acceptance Criteria
- [ ] All 4 statistical tools implemented
- [ ] Accurate calculations verified against known datasets
- [ ] Full code annotations explaining formulas
- [ ] Clear error messages for invalid input
- [ ] Works correctly in Claude Desktop
- [ ] README documentation with examples

## Example Usage
```
User: "Analyze this data: [23, 45, 12, 67, 34, 89, 23, 56]"
Tool: descriptive_stats with data=[23, 45, 12, 67, 34, 89, 23, 56]
Result: {
  "mean": 43.625,
  "median": 39.5,
  "mode": 23,
  "range": 77,
  "variance": 642.73,
  "std_dev": 25.35
}

User: "What's the correlation between [1,2,3,4,5] and [2,4,6,8,10]?"
Tool: correlation with x=[1,2,3,4,5], y=[2,4,6,8,10]
Result: "Perfect positive correlation (r = 1.00)"
```

## References
- [Descriptive Statistics](https://en.wikipedia.org/wiki/Descriptive_statistics)
- [Pearson Correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
- [Percentile](https://en.wikipedia.org/wiki/Percentile)
- [Outlier Detection](https://en.wikipedia.org/wiki/Outlier)

## Labels
enhancement, statistics, data-analysis
