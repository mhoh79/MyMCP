# Time Series Analysis Tools

## Overview

The stats server now includes 6 comprehensive time series analysis tools designed for industrial automation, process monitoring, trend detection, and predictive maintenance. These tools help analyze sensor data, detect equipment degradation, identify process cycles, and monitor production trends.

## Tools Summary

| Tool | Purpose | Use Cases |
|------|---------|-----------|
| `moving_average` | Smooth noisy data | Sensor noise filtering, trend visualization |
| `detect_trend` | Identify trends | Equipment degradation, efficiency decline |
| `autocorrelation` | Find patterns | Batch cycles, seasonality detection |
| `change_point_detection` | Detect changes | Process modifications, equipment changes |
| `rate_of_change` | Monitor rates | Startup monitoring, leak detection |
| `rolling_statistics` | Continuous monitoring | SCADA displays, stability tracking |

## Detailed Tool Documentation

### 1. `moving_average`

Calculate Simple, Exponential, or Weighted Moving Averages for smoothing time series data.

**Parameters:**
- `data` (array, required): Time series data (min 2 items)
- `window_size` (integer, required): Number of periods (2-1000)
- `ma_type` (string, optional): "simple", "exponential", or "weighted" (default: "simple")
- `alpha` (number, optional): Smoothing factor for EMA (0-1), defaults to 2/(window_size+1)

**Example Usage:**
```
"Calculate a 5-period simple moving average for hourly temperature readings: [100, 102, 98, 105, 103, 107, 110]"

"Apply an exponential moving average with window size 3 to smooth pressure sensor data"

"Use a weighted moving average to filter flow meter readings"
```

**Industrial Use Cases:**
- **HVAC Systems**: Smooth temperature sensor readings to reduce false alarms
- **Flow Monitoring**: Filter high-frequency noise from flow meters
- **Pressure Control**: Identify underlying pressure trends in process systems
- **Quality Control**: Smooth measurement data for trend analysis

**Algorithm Types:**
- **Simple (SMA)**: Equal weight to all values in window - best for general smoothing
- **Exponential (EMA)**: More weight to recent values - better for tracking rapid changes
- **Weighted (WMA)**: Linear weighting - compromise between SMA and EMA

---

### 2. `detect_trend`

Identify and quantify trends using linear regression analysis.

**Parameters:**
- `data` (array, required): Time series values (min 3 items)
- `timestamps` (array, optional): Time indices (defaults to 0, 1, 2, ...)
- `method` (string, optional): "linear" or "polynomial" (default: "linear")
- `degree` (integer, optional): Polynomial degree 2-5 (for polynomial method)

**Example Usage:**
```
"Analyze the trend in bearing temperature readings over the last 30 days"

"Is compressor efficiency declining? Here are the daily efficiency values..."

"Detect if catalyst activity is decreasing using these monthly measurements"
```

**Industrial Use Cases:**
- **Predictive Maintenance**: Detect bearing temperature increases before failure
- **Equipment Monitoring**: Track compressor efficiency decline
- **Process Optimization**: Identify catalyst deactivation trends
- **Energy Management**: Analyze long-term energy consumption patterns

**Output Interpretation:**
- **Slope**: Rate of change per time period (positive = increasing, negative = decreasing)
- **R²**: Fit quality (>0.9 excellent, >0.7 good, >0.5 fair)
- **Predictions**: Forecasts for next 5 time periods
- **Confidence Interval**: Statistical uncertainty in the slope estimate

---

### 3. `autocorrelation`

Calculate autocorrelation function (ACF) to identify repeating patterns and cycles.

**Parameters:**
- `data` (array, required): Time series data (min 10 items)
- `max_lag` (integer, optional): Maximum lag to calculate (1-500), defaults to min(len(data)//2, 50)

**Example Usage:**
```
"Find repeating patterns in batch cycle times"

"Detect if there's a weekly pattern in energy consumption"

"Analyze production data for cyclic behavior"
```

**Industrial Use Cases:**
- **Batch Processing**: Identify optimal batch cycle times
- **Production Planning**: Detect repeating patterns in output
- **Energy Management**: Find daily/weekly consumption cycles
- **Maintenance Scheduling**: Identify equipment performance patterns

**Output Interpretation:**
- **ACF Values**: Range from -1 to 1 (values near 0 = no correlation)
- **Significant Lags**: Lags with |ACF| > 0.2 indicate strong patterns
- **Pattern Detection**: Repeating significant lags suggest cyclic behavior

---

### 4. `change_point_detection`

Identify significant changes in process behavior (regime changes, upsets, modifications).

**Parameters:**
- `data` (array, required): Time series data (min 10 items)
- `method` (string, optional): "cusum", "standard_deviation", or "mean_shift" (default: "cusum")
- `threshold` (number, optional): Sensitivity (0.1-10, higher = less sensitive, default: 1.5)
- `min_size` (integer, optional): Minimum segment size (min 2, default: 5)

**Example Usage:**
```
"When did the process improvement take effect? Here's production data..."

"Detect upsets in reactor temperature over the last month"

"Find when equipment behavior changed in this pressure data"
```

**Industrial Use Cases:**
- **Process Validation**: Verify when process changes became effective
- **Upset Detection**: Identify when disturbances occurred
- **Equipment Changes**: Detect when equipment behavior shifted
- **Shift Analysis**: Find differences between operating shifts

**Detection Methods:**
- **CUSUM**: Best for detecting small, persistent changes
- **Standard Deviation**: Good for variance changes
- **Mean Shift**: Simple method for obvious level changes

---

### 5. `rate_of_change`

Calculate rate of change to detect acceleration or deceleration in processes.

**Parameters:**
- `data` (array, required): Time series values (min 2 items)
- `time_intervals` (array, optional): Time between measurements (default: uniform)
- `method` (string, optional): "simple" or "smoothed" (default: "simple")
- `smoothing_window` (integer, optional): Window for smoothing (min 2, default: 3)

**Example Usage:**
```
"How fast is the reactor temperature rising during startup?"

"Calculate the rate of pressure change to detect potential leaks"

"Monitor production rate changes over the last shift"
```

**Industrial Use Cases:**
- **Startup Monitoring**: Track temperature ramp rates
- **Safety Systems**: Detect abnormally rapid pressure changes
- **Production Tracking**: Monitor rate changes in output
- **Level Control**: Track tank level change rates

**Safety Applications:**
- **Rapid Pressure Drop**: May indicate leak or rupture
- **Fast Temperature Rise**: Could signal runaway reaction
- **Sudden Flow Changes**: Might indicate valve failures
- **Quick Level Changes**: May suggest pump problems

---

### 6. `rolling_statistics`

Calculate rolling/windowed statistics for continuous monitoring.

**Parameters:**
- `data` (array, required): Time series data (min 2 items)
- `window_size` (integer, required): Rolling window size (2-1000)
- `statistics` (array, optional): Statistics to calculate - "mean", "std", "min", "max", "median", "range", "variance" (default: ["mean", "std"])

**Example Usage:**
```
"Calculate rolling mean and standard deviation with a 10-point window for process monitoring"

"Show rolling statistics (mean, min, max) for the last 50 pressure readings"

"Monitor process stability using rolling variance with a 20-point window"
```

**Industrial Use Cases:**
- **SCADA Displays**: Show recent averages and trends
- **Process Stability**: Monitor rolling standard deviation
- **Quality Control**: Track recent performance metrics
- **Alarm Systems**: Use rolling statistics for adaptive thresholds

**Recommended Statistics:**
- **Mean + Std**: Best for process stability monitoring
- **Min + Max**: Good for range control
- **Median**: Better than mean for data with outliers
- **Variance**: Useful for detecting instability

## Industrial Application Examples

### Example 1: Bearing Temperature Monitoring

**Scenario**: Monitor bearing temperature to predict failure

```
Step 1: Use moving_average to smooth daily temperature variations
"Calculate 24-hour moving average of bearing temperature"

Step 2: Use detect_trend to identify degradation
"Analyze trend in smoothed bearing temperature over 30 days"

Step 3: Use rate_of_change to detect rapid changes
"Calculate rate of temperature change - alert if > 0.5°C/day"
```

**Expected Results**:
- Smooth trend showing gradual increase
- Slope indicates 0.3-0.5°C/day increase
- Recommendation: Schedule inspection within 2 weeks

### Example 2: Batch Process Optimization

**Scenario**: Optimize batch cycle times and detect improvements

```
Step 1: Use rolling_statistics to track recent performance
"Show rolling mean cycle time over last 20 batches"

Step 2: Use change_point_detection to find improvements
"Detect when process change reduced cycle time"

Step 3: Use autocorrelation to find patterns
"Identify if cycle time repeats every N batches"
```

**Expected Results**:
- Change point at batch 342 where cycle time dropped 15 minutes
- Pattern repeats every 8 batches
- Optimization saved 2 hours per day

### Example 3: Energy Consumption Analysis

**Scenario**: Reduce energy costs through pattern analysis

```
Step 1: Use moving_average to smooth daily variations
"Calculate 7-day exponential moving average of energy use"

Step 2: Use autocorrelation to detect patterns
"Find weekly or monthly patterns in consumption"

Step 3: Use detect_trend for long-term analysis
"Analyze if energy consumption is increasing or decreasing"
```

**Expected Results**:
- Strong weekly pattern (weekends 30% lower)
- Overall trend down 2% annually
- Identified opportunities for off-peak scheduling

### Example 4: Compressor Performance Monitoring

**Scenario**: Detect compressor efficiency degradation

```
Step 1: Use detect_trend on efficiency ratio
"Analyze compressor efficiency over 90 days"

Step 2: Use change_point_detection for sudden drops
"Detect if efficiency suddenly decreased"

Step 3: Use rolling_statistics for real-time monitoring
"Show rolling 7-day average efficiency on SCADA"
```

**Expected Results**:
- Gradual efficiency decline of 0.5% per month
- No sudden changes (rules out mechanical failure)
- Recommendation: Schedule maintenance based on trend

## Performance Targets

All tools are optimized for industrial-scale data:

| Tool | Data Points | Typical Time | Max Recommended |
|------|-------------|--------------|-----------------|
| moving_average | 10,000 | <100ms | 100,000 |
| detect_trend | 10,000 | <500ms | 100,000 |
| autocorrelation | 1,000 | <200ms | 10,000 |
| change_point_detection | 10,000 | <1s | 100,000 |
| rate_of_change | 10,000 | <100ms | 100,000 |
| rolling_statistics | 10,000 | <200ms | 100,000 |

## Edge Cases and Best Practices

### Data Quality

**Handle Missing Data**: Remove or interpolate before analysis
```
# Not recommended - will cause errors
data_with_gaps = [100, 102, None, 105, 107]

# Better approach
data_filled = [100, 102, 103.5, 105, 107]  # Interpolated
```

**Outlier Treatment**: Consider removing extreme outliers first
```
# Use detect_outliers before time series analysis
"First detect outliers in temperature data, then analyze trend"
```

### Window Size Selection

**Moving Average Windows**:
- Short window (3-5): Responsive to changes, some noise remains
- Medium window (10-20): Good balance for most applications  
- Long window (50+): Very smooth, slower to respond

**Rolling Statistics Windows**:
- Rule of thumb: Use 5-10% of total data points
- Minimum: 5 points for meaningful statistics
- Maximum: Don't exceed 20% of data length

### Threshold Selection

**Change Point Detection**:
- threshold=1.5: Standard sensitivity (recommended starting point)
- threshold=0.5-1.0: High sensitivity (may detect minor variations)
- threshold=2.0-3.0: Low sensitivity (only major changes)

**Autocorrelation**:
- Significant if |ACF| > 0.2 (moderate correlation)
- Strong correlation if |ACF| > 0.5
- Perfect correlation at lag 0 (always 1.0)

## Integration with SCADA Systems

### Real-Time Monitoring

1. **Configure data collection interval** based on process dynamics
2. **Use rolling_statistics** for dashboard displays
3. **Apply moving_average** to reduce alarm noise
4. **Set up rate_of_change** alerts for safety-critical parameters

### Alarm Configuration

```
Example: Bearing Temperature Alarm
- Normal range: 60-80°C (direct measurement)
- Trend alarm: Slope > 0.5°C/day (detect_trend)
- Rate alarm: Rate > 2°C/hour (rate_of_change)
- Pattern alarm: ACF shows unusual cycling (autocorrelation)
```

## Troubleshooting

### "Data must contain at least N items"
- **Cause**: Insufficient data for analysis
- **Solution**: Collect more data or reduce window size

### "Cannot calculate... zero variance"
- **Cause**: All data points are identical
- **Solution**: Check sensor, use longer time period

### "window_size cannot be larger than data length"
- **Cause**: Window exceeds available data
- **Solution**: Reduce window size or provide more data

### Poor R² value in trend detection
- **Cause**: Data doesn't follow linear trend
- **Solution**: Data may be stable (no trend) or need polynomial fit

## Summary

These time series analysis tools provide industrial-grade capabilities for:
- ✅ Process monitoring and optimization
- ✅ Predictive maintenance
- ✅ Quality control
- ✅ Energy management
- ✅ Production planning
- ✅ Safety monitoring

All tools are designed to work with real-world industrial data, including noisy sensor readings, missing values, and large datasets. They provide actionable insights for maintenance, operations, and process engineering teams.

For more information about the statistical server, see the main README.md file.
