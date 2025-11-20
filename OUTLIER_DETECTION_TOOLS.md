# Advanced Outlier Detection Tools

## Overview

The stats server now includes 6 advanced outlier detection methods designed for industrial automation, data quality control, sensor validation, and anomaly detection in process manufacturing. These tools provide statistical rigor and are suitable for different data types, sample sizes, and detection requirements.

## Tools Summary

| Tool | Purpose | Best For | Sample Size |
|------|---------|----------|-------------|
| `z_score_detection` | Standard/Modified Z-score | Normally distributed data | Any (min 3) |
| `grubbs_test` | Statistical significance test | Single outlier with p-value | 7+ points |
| `dixon_q_test` | Quick test for small samples | Laboratory measurements | 3-30 points |
| `isolation_forest` | ML-based anomaly detection | Complex/multivariate patterns | 10+ points |
| `mahalanobis_distance` | Multivariate outliers | Correlated variables | 10+ points |
| `streaming_outlier_detection` | Real-time detection | Continuous sensor streams | 10-1000 window |

## When to Use Which Method

### By Data Characteristics
- **Normally distributed data**: `z_score_detection` (standard), `grubbs_test`
- **Non-normal or skewed data**: `z_score_detection` (modified)
- **Small samples (< 30)**: `dixon_q_test`, `grubbs_test`
- **Large datasets**: `z_score_detection`, `isolation_forest`
- **Multivariate data**: `mahalanobis_distance`, `isolation_forest`
- **Real-time streams**: `streaming_outlier_detection`

### By Application
- **Sensor validation**: `z_score_detection` (modified), `streaming_outlier_detection`
- **Laboratory QC**: `dixon_q_test`, `grubbs_test`
- **Process monitoring**: `streaming_outlier_detection`, `mahalanobis_distance`
- **Equipment health**: `mahalanobis_distance`, `isolation_forest`
- **Regulatory compliance**: `grubbs_test`, `dixon_q_test`

## Detailed Tool Documentation

---

### 1. `z_score_detection`

Detect outliers using Z-score methods (standard or modified).

**Parameters:**
- `data` (array, required): Measurements to check (min 3 items)
- `method` (string, optional): "standard" or "modified" (default: "modified")
  - **standard**: Uses mean and standard deviation
  - **modified**: Uses median and MAD (more robust to outliers)
- `threshold` (number, optional): Z-score threshold (default: 3.0)
  - Typical: 3.0 for standard, 3.5 for modified
- `two_tailed` (boolean, optional): Detect on both sides (default: true)

**Example Usage:**
```
"Check these temperature readings for outliers: [200.1, 200.3, 199.8, 250.5, 200.2, 199.9]"

"Use modified Z-score to detect sensor faults in pressure data"

"Validate these flow meter readings using Z-score with threshold 3.5"
```

**Industrial Use Cases:**
- **Sensor Validation**: Detect faulty temperature, pressure, or flow sensors
- **Quality Control**: Screen production measurements for defects
- **Process Control**: Identify out-of-control process variables
- **Batch Comparison**: Compare batch results to historical norms

**Output Interpretation:**
- **Z-score > 3**: Likely outlier (99.7% confidence for normal distribution)
- **Severity levels**: "moderate" (3-4.5σ), "extreme" (>4.5σ)
- **Method choice**: Modified is preferred when outliers might affect mean/std

**Example Output:**
```
Modified Z-Score Outlier Detection:
⚠ Found 1 outlier(s):
  • Index 3: Value 250.5
    Z-score: 169.81, Severity: extreme

Statistics:
  Median: 200.15
  MAD: 0.20
  Total Points: 8
  Outlier Percentage: 12.5%

Interpretation: 1 outliers detected (12.5%). Modified Z-score method used for robustness.
```

---

### 2. `grubbs_test`

Statistical test for detecting single outliers in normally distributed data.

**Parameters:**
- `data` (array, required): Dataset to test (min 7 items)
- `alpha` (number, optional): Significance level (default: 0.05)
  - Common values: 0.05 (95% confidence), 0.01 (99% confidence)
- `method` (string, optional): "max", "min", or "two_sided" (default: "two_sided")

**Example Usage:**
```
"Is 15.8 an outlier in these calibration measurements: [10.2, 10.3, 10.1, 15.8, 10.4, 10.2, 10.3, 10.1]?"

"Test if the highest value is a statistically significant outlier at 99% confidence"

"Validate laboratory test results using Grubbs test"
```

**Industrial Use Cases:**
- **Calibration Validation**: Reject suspicious calibration points
- **Laboratory QC**: Validate test results with statistical rigor
- **Regulatory Compliance**: FDA/ISO requirements for outlier rejection
- **Precision Measurements**: Critical measurements requiring statistical justification

**Output Interpretation:**
- **Test Statistic (G)**: Calculated from (value - mean) / std_dev
- **Critical Value**: Threshold from t-distribution for given α and sample size
- **P-value**: Probability of observing this value if not an outlier
- **Conclusion**: Reject null hypothesis if G > critical value

**Example Output:**
```
Grubbs Test for Outliers:

Suspected Outlier:
  Value: 15.8000
  Index: 3
  Side: maximum

Test Results:
  Test Statistic (G): 2.8246
  Critical Value: 2.2150
  P-value: 0.0230

Conclusion:
  Reject null hypothesis - value is a significant outlier at α=0.05

Recommendation:
  Remove point 3 (value: 15.8000) - statistically significant outlier
```

---

### 3. `dixon_q_test`

Quick test for outliers in small datasets (3-30 points).

**Parameters:**
- `data` (array, required): Small dataset (3-30 points)
- `alpha` (number, optional): Significance level (default: 0.05)

**Example Usage:**
```
"Check if 34.2 is an outlier in these 8 measurements: [10.2, 10.3, 10.1, 34.2, 10.4, 10.2, 10.3, 10.1]"

"Validate duplicate measurements using Dixon Q test"

"Test pilot plant trial data for outliers"
```

**Industrial Use Cases:**
- **Laboratory QC**: Small sample quality control
- **Pilot Plants**: Limited trial data validation
- **Expensive Tests**: When retesting is costly
- **Duplicate/Triplicate Validation**: Check measurement consistency
- **Shift Samples**: Validate small sample batches

**Output Interpretation:**
- **Q Statistic**: gap / range (larger Q indicates stronger outlier)
- **Q Critical**: Threshold value from Dixon Q tables for sample size and α
- **Position**: "high" or "low" (which end of the distribution)

**Advantages:**
- Designed specifically for small samples
- Simple calculation
- Well-established critical value tables
- Quick screening tool

**Example Output:**
```
Dixon Q Test for Outliers:

Suspected Outlier:
  Value: 15.8000
  Index: 3
  Position: high

Test Results:
  Q Statistic: 0.9474
  Q Critical: 0.5540

Conclusion:
  Outlier detected - reject value (Q=0.947 > Q_critical=0.554)

Recommendation:
  Repeat measurement or investigate cause for value 15.8000 at position high
```

---

### 4. `isolation_forest`

Machine learning-based anomaly detection for complex datasets.

**Parameters:**
- `data` (array, required): Univariate `[x1, x2, ...]` or multivariate `[[x1, y1, z1], ...]` (min 10 items)
- `contamination` (number, optional): Expected outlier proportion 0-0.5 (default: 0.1)
- `n_estimators` (integer, optional): Number of isolation trees 50-500 (default: 100)

**Example Usage:**
```
"Detect anomalies in these sensor readings using machine learning: [1, 2, 3, 4, 5, 6, 100]"

"Find unusual combinations in temperature, pressure, and flow data"

"Identify equipment failure patterns in multivariate sensor data"
```

**Industrial Use Cases:**
- **Multivariate Anomalies**: Multiple sensors with complex interactions
- **Equipment Failure Prediction**: Unusual parameter combinations
- **Cyber Security**: Detect unusual patterns in process data
- **Complex Processes**: When simple statistical methods fail
- **Pattern Recognition**: Learn normal behavior automatically

**How It Works:**
- Creates random decision trees that isolate anomalies
- Anomalies are easier to isolate (fewer splits needed)
- Doesn't assume any data distribution
- Handles both univariate and multivariate data

**Output Interpretation:**
- **Anomaly Score**: Higher = more isolated = more likely outlier
- **Severity**: "high" (>0.6), "medium" (0.4-0.6), "low" (<0.4)
- **Contamination**: Actual percentage of outliers found

**Example Output:**
```
Isolation Forest Anomaly Detection:

⚠ Found 1 anomalies:
  1. Index 10: Score 0.7234, Severity: high

Statistics:
  Contamination Rate: 9.1%
  N Estimators: 100
  Total Data Points: 11

Interpretation: 1 anomalies detected (9.1% of data). Isolation scores indicate how isolated each point is from normal patterns.
```

---

### 5. `mahalanobis_distance`

Multivariate outlier detection considering correlations between variables.

**Parameters:**
- `data` (array, required): Multivariate data `[[x1, y1, z1], [x2, y2, z2], ...]` (min 10 items)
- `threshold` (number, optional): Chi-square percentile 0.9-0.999 (default: 0.975)

**Example Usage:**
```
"Detect unusual combinations in temperature, pressure, and flow data"

"Find multivariate outliers in process monitoring data with 5 variables"

"Check if any points are abnormal considering all sensor correlations"
```

**Industrial Use Cases:**
- **Process State Monitoring**: Multiple correlated parameters
- **Equipment Health**: Temperature, vibration, pressure together
- **Quality Control**: Multiple quality measurements
- **Sensor Networks**: Correlated sensor validation
- **Multivariate SPC**: Statistical process control with multiple variables

**How It Works:**
- Calculates distance from multivariate center (mean)
- Accounts for variable correlations via covariance matrix
- Uses chi-square distribution for threshold
- Identifies points unusual in combined context

**Key Advantage:**
A point might be normal individually but abnormal in combination with other variables.

**Example:**
- Temperature: 100°C (normal)
- Pressure: 5 bar (normal)
- Together: Unusual (usually high temp → high pressure)

**Output Interpretation:**
- **Mahalanobis Distance**: Distance from center accounting for correlations
- **P-value**: Probability of observing this combination
- **Variable Contributions**: Which variable(s) contribute most to outlier status

**Example Output:**
```
Mahalanobis Distance Multivariate Outlier Detection:

⚠ Found 2 multivariate outliers:
  • Index 23: Distance 12.5000, p-value: 0.0060
  • Index 67: Distance 10.8000, p-value: 0.0120

Configuration:
  Dimensions: 3
  Threshold Distance: 9.3500
  Degrees of Freedom: 3

Variable Contributions (for outliers):
  Index 23: Primary variable: variable_0, Contribution: 65.0%

Interpretation: 2 multivariate outliers detected. These points are unusual in the combined context of all 3 variables.
```

---

### 6. `streaming_outlier_detection`

Real-time outlier detection for continuous sensor streams.

**Parameters:**
- `current_value` (number, required): New measurement to evaluate
- `historical_window` (array, required): Recent values (10-1000 items)
- `method` (string, optional): "ewma", "cusum", or "adaptive_threshold" (default: "ewma")
- `sensitivity` (integer, optional): Detection sensitivity 1-10 (default: 5)
  - Higher = more sensitive = detects smaller deviations

**Example Usage:**
```
"Is the current temperature reading 145.8°C an outlier given recent readings?"

"Monitor reactor pressure in real-time with sensitivity level 7"

"Detect sensor faults using EWMA method for streaming data"
```

**Industrial Use Cases:**
- **SCADA Alarming**: Real-time alarm generation
- **Edge Devices**: On-device data validation
- **Continuous Monitoring**: 24/7 process surveillance
- **High-Frequency Sensors**: Fast sampling rates (1-second intervals)
- **Telemetry**: Remote equipment monitoring

**Methods:**
- **EWMA** (Exponentially Weighted Moving Average): Good general purpose, reacts to trends
- **CUSUM** (Cumulative Sum): Detects sustained shifts, good for drift
- **Adaptive Threshold**: Adjusts to recent volatility, handles changing conditions

**Output Interpretation:**
- **is_outlier**: Boolean flag for alarm triggering
- **Severity**: "normal", "warning", "critical"
- **Expected Range**: Normal operating range
- **Deviation (σ)**: How many standard deviations from expected
- **Trend**: "increasing", "decreasing", "stable"
- **Rate of Change**: How fast the value is changing

**Example Output:**
```
Streaming Outlier Detection (Real-time):

Current Value Assessment:
  Value: 145.8000
  Is Outlier: YES
  Severity: WARNING

Analysis:
  Method: EWMA
  Expected Range: [95.2000, 105.8000]
  Deviation: 40.0000
  Deviation (σ): 3.20
  Trend: increasing
  Rate of Change: 5.2000

Interpretation:
  Current value 145.80 exceeds expected range [95.20, 105.80] by 3.2 standard deviations

Recommendation:
  Investigate sensor or process condition
```

---

## Comparison of Methods

### Performance Characteristics

| Method | Speed | Statistical Rigor | Multivariate | Real-time |
|--------|-------|-------------------|--------------|-----------|
| Z-score | Fast | Medium | No | Possible |
| Grubbs | Fast | High | No | No |
| Dixon Q | Fast | High | No | No |
| Isolation Forest | Medium | Medium | Yes | No |
| Mahalanobis | Medium | High | Yes | No |
| Streaming | Very Fast | Medium | No | Yes |

### Strengths and Limitations

**Z-score Detection:**
- ✓ Fast, simple, intuitive
- ✓ Modified version robust to outliers
- ✗ Assumes distribution shape
- ✗ Sensitive to multiple outliers (standard version)

**Grubbs Test:**
- ✓ Statistical p-values
- ✓ Regulatory acceptance
- ✗ One outlier at a time
- ✗ Requires normal distribution

**Dixon Q Test:**
- ✓ Designed for small samples
- ✓ Simple and quick
- ✗ Limited to 3-30 points
- ✗ One outlier at a time

**Isolation Forest:**
- ✓ No distribution assumption
- ✓ Handles multivariate data
- ✓ Learns complex patterns
- ✗ Requires training
- ✗ Less interpretable

**Mahalanobis Distance:**
- ✓ Accounts for correlations
- ✓ True multivariate detection
- ✓ Statistical foundation
- ✗ Requires sufficient sample size
- ✗ Sensitive to covariance matrix estimation

**Streaming Detection:**
- ✓ Real-time capability
- ✓ Adapts to trends
- ✓ Very fast
- ✗ Needs historical context
- ✗ Tuning required

---

## Industrial Application Examples

### Example 1: Temperature Sensor Validation

**Scenario:** Reactor temperature sensor showing unusual reading

**Data:** [200.1, 200.3, 199.8, 250.5, 200.2, 199.9, 200.4, 200.1]°C

**Approach:**
1. Use `z_score_detection` (modified) for robust initial screening
2. Confirm with `grubbs_test` for statistical validation
3. Monitor with `streaming_outlier_detection` going forward

**Result:**
```
Point 4 (250.5°C) is outlier:
- Modified Z-score: 169.81 (extreme severity)
- Grubbs test: G=2.82 > critical=2.21 (p<0.05)
- Likely sensor fault
- Expected range: 199-201°C
```

**Action:** Replace sensor, investigate cause

---

### Example 2: Multivariate Process Monitoring

**Scenario:** Chemical reactor with correlated variables

**Data:** Temperature, Pressure, Flow measurements over time

**Approach:**
1. Use `mahalanobis_distance` to detect unusual combinations
2. Use `isolation_forest` to learn normal patterns
3. Identify which variable(s) contribute to anomaly

**Result:**
```
Anomaly at 14:30:
- High temperature (210°C, individually normal)
- Low flow (50 L/min, individually normal)
- Together: abnormal (Mahalanobis distance = 12.5)
- Normal operation: high temp → high flow
```

**Action:** Check heat exchanger for fouling

---

### Example 3: Quality Control Lab Data

**Scenario:** 8 replicate measurements from quality lab

**Data:** [10.2, 10.3, 10.1, 15.8, 10.4, 10.2, 10.3, 10.1]

**Approach:**
1. Use `dixon_q_test` (designed for small samples)
2. Confirm with `grubbs_test`
3. Document statistical justification for rejection

**Result:**
```
Value 15.8 rejected:
- Dixon Q: 0.947 > Q_critical=0.554 (α=0.05)
- Grubbs test: confirms (p<0.05)
- Likely measurement error or contamination
```

**Action:** Repeat test, investigate lab procedure

---

## Best Practices

### 1. Data Preparation
- Check for missing values first
- Understand your data distribution
- Consider data transformations if severely skewed
- Document normal operating ranges

### 2. Method Selection
- Start with simpler methods (Z-score, Grubbs)
- Use appropriate test for sample size
- Consider multivariate methods for related variables
- Use streaming for real-time applications

### 3. Threshold Selection
- Conservative for critical processes (higher threshold)
- Aggressive for data cleansing (lower threshold)
- Document and justify threshold choices
- Consider false positive vs false negative costs

### 4. Interpretation
- Always investigate detected outliers
- Don't automatically remove outliers
- Consider if outlier represents true anomaly or measurement error
- Document decisions and rationale

### 5. Validation
- Test methods on historical data
- Calculate false positive/negative rates
- Compare multiple methods on same data
- Validate assumptions (normality, independence)

### 6. Integration
- Connect outlier detection to alarm systems
- Log all detections for audit trail
- Provide clear action recommendations
- Consider cascading detection (multiple methods)

---

## Common Pitfalls

1. **Assuming Normality**: Not all data is normally distributed
   - Solution: Use modified Z-score or Isolation Forest

2. **Multiple Testing**: Running many tests increases false positives
   - Solution: Adjust significance levels or use Bonferroni correction

3. **Masking Effect**: Multiple outliers can hide each other
   - Solution: Use modified Z-score or remove outliers iteratively

4. **Swamping Effect**: Non-outliers flagged as outliers
   - Solution: Use robust methods (modified Z-score, MAD)

5. **Insufficient Data**: Small samples with powerful tests
   - Solution: Use Dixon Q test for small samples

6. **Ignoring Context**: Outliers might be valid extreme events
   - Solution: Always investigate, don't auto-delete

---

## Integration with Existing Tools

The outlier detection tools integrate seamlessly with other stats server tools:

```
Example workflow:
1. descriptive_stats - Get data summary
2. z_score_detection - Quick screening
3. grubbs_test - Statistical validation
4. detect_outliers - IQR method (existing tool)
5. moving_average - Smooth cleaned data
6. detect_trend - Analyze cleaned trend
```

---

## References and Further Reading

### Statistical Methods
- Grubbs, F. E. (1969). "Procedures for Detecting Outlying Observations in Samples"
- Dixon, W. J. (1953). "Processing Data for Outliers"
- Iglewicz, B., & Hoaglin, D. C. (1993). "Volume 16: How to Detect and Handle Outliers"

### Machine Learning
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest"
- Mahalanobis, P. C. (1936). "On the Generalized Distance in Statistics"

### Industrial Applications
- Montgomery, D. C. (2009). "Statistical Quality Control"
- ASTM E178 - Standard Practice for Dealing With Outlying Observations

---

## Support and Feedback

For questions, issues, or suggestions about these outlier detection tools, please refer to the main repository documentation and issue tracker.

**Remember:** Outlier detection is a tool to assist decision-making, not replace expert judgment. Always investigate detected outliers and understand their cause before taking action.
