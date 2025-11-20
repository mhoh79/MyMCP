# Add Time Series Analysis Tools to Stats Server

## Overview
Implement comprehensive time series analysis tools for process monitoring, trend detection, and predictive maintenance in industrial automation and processing plants.

## Motivation
Time series analysis is critical for:
- **Process Variable Trending**: Monitor temperature, pressure, flow rates over time
- **Predictive Maintenance**: Detect degradation patterns before equipment failure
- **Production Cycle Analysis**: Optimize batch processing and cycle times
- **Energy Management**: Analyze consumption patterns and identify savings opportunities
- **Sensor Drift Detection**: Identify calibration issues early
- **Anomaly Detection**: Real-time detection of process deviations

## Tools to Implement

### 1. `moving_average`
Calculate Simple, Exponential, and Weighted Moving Averages for smoothing process data.

**Use Cases:**
- Smooth noisy sensor readings (temperature, pressure sensors)
- Trend visualization in SCADA systems
- Filter out high-frequency noise from flow meters
- Identify underlying process trends

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Time series data (e.g., hourly temperature readings)",
      "minItems": 2
    },
    "window_size": {
      "type": "integer",
      "description": "Number of periods for moving average",
      "minimum": 2,
      "maximum": 1000
    },
    "ma_type": {
      "type": "string",
      "enum": ["simple", "exponential", "weighted"],
      "default": "simple",
      "description": "Type of moving average"
    },
    "alpha": {
      "type": "number",
      "description": "Smoothing factor for EMA (0-1), default 2/(window_size+1)",
      "minimum": 0,
      "maximum": 1
    }
  },
  "required": ["data", "window_size"]
}
```

**Example Output:**
```json
{
  "original_data": [100, 102, 98, 105, 103, 107, 110],
  "moving_average": [100.0, 100.0, 101.33, 103.0, 105.0, 106.67],
  "ma_type": "simple",
  "window_size": 3,
  "data_points": 7,
  "smoothed_points": 6
}
```

### 2. `detect_trend`
Identify and quantify trends in process data using linear or polynomial regression.

**Use Cases:**
- Equipment degradation trends (bearing temperature increasing)
- Process efficiency decline over time
- Catalyst deactivation in reactors
- Compressor performance degradation
- Heat exchanger fouling detection

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Time series values",
      "minItems": 3
    },
    "timestamps": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Optional time indices (defaults to 0, 1, 2...)"
    },
    "method": {
      "type": "string",
      "enum": ["linear", "polynomial"],
      "default": "linear"
    },
    "degree": {
      "type": "integer",
      "description": "Polynomial degree (only for polynomial method)",
      "minimum": 2,
      "maximum": 5,
      "default": 2
    }
  },
  "required": ["data"]
}
```

**Example Output:**
```json
{
  "trend": "increasing",
  "slope": 0.245,
  "slope_interpretation": "Increasing by 0.245 units per time period",
  "r_squared": 0.87,
  "fit_quality": "good",
  "coefficients": [100.5, 0.245],
  "equation": "y = 0.245x + 100.5",
  "confidence_interval": [0.21, 0.28],
  "prediction_next_5": [102.7, 102.95, 103.2, 103.45, 103.7]
}
```

### 3. `autocorrelation`
Calculate autocorrelation function (ACF) to identify repeating patterns and cycles.

**Use Cases:**
- Detect cyclic patterns in batch processes
- Identify production cycle times
- Find optimal sampling intervals for data collection
- Detect seasonality in energy consumption
- Validate process control loop tuning

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Time series data",
      "minItems": 10
    },
    "max_lag": {
      "type": "integer",
      "description": "Maximum lag to calculate",
      "minimum": 1,
      "maximum": 500
    }
  },
  "required": ["data"]
}
```

### 4. `change_point_detection`
Identify significant changes in process behavior (regime changes, upsets, modifications).

**Use Cases:**
- Detect when process modifications were effective
- Identify process upsets or disturbances
- Find when equipment behavior changed
- Detect shift changes affecting production
- Identify when maintenance activities impacted performance

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Time series data",
      "minItems": 10
    },
    "method": {
      "type": "string",
      "enum": ["cusum", "standard_deviation", "mean_shift"],
      "default": "cusum"
    },
    "threshold": {
      "type": "number",
      "description": "Sensitivity threshold (higher = less sensitive)",
      "minimum": 0.1,
      "maximum": 10,
      "default": 1.5
    },
    "min_size": {
      "type": "integer",
      "description": "Minimum segment size between change points",
      "minimum": 2,
      "default": 5
    }
  },
  "required": ["data"]
}
```

**Example Output:**
```json
{
  "change_points": [45, 123, 287],
  "change_point_timestamps": ["2025-01-15 14:30", "2025-01-18 08:15", "2025-01-22 16:45"],
  "number_of_segments": 4,
  "segments": [
    {"start": 0, "end": 45, "mean": 100.5, "std": 2.3},
    {"start": 45, "end": 123, "mean": 95.2, "std": 1.8},
    {"start": 123, "end": 287, "mean": 98.7, "std": 2.1},
    {"start": 287, "end": 350, "mean": 102.1, "std": 2.5}
  ],
  "largest_change": {"index": 45, "magnitude": 5.3, "direction": "decrease"}
}
```

### 5. `rate_of_change`
Calculate rate of change over time to detect acceleration or deceleration in processes.

**Use Cases:**
- Monitor how fast temperature is rising during startup
- Detect rapid pressure changes indicating leaks
- Track production rate changes
- Monitor tank level changes
- Identify abnormal ramp rates

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Time series values",
      "minItems": 2
    },
    "time_intervals": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Time intervals between measurements (default: uniform)"
    },
    "method": {
      "type": "string",
      "enum": ["simple", "smoothed"],
      "default": "simple"
    },
    "smoothing_window": {
      "type": "integer",
      "description": "Window size for smoothed rate of change",
      "minimum": 2,
      "default": 3
    }
  },
  "required": ["data"]
}
```

### 6. `rolling_statistics`
Calculate rolling/windowed statistics for continuous monitoring.

**Use Cases:**
- Monitor rolling averages on SCADA displays
- Track process stability with rolling standard deviation
- Calculate recent performance metrics
- Implement sliding window quality checks
- Real-time process capability monitoring

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Time series data",
      "minItems": 2
    },
    "window_size": {
      "type": "integer",
      "description": "Rolling window size",
      "minimum": 2,
      "maximum": 1000
    },
    "statistics": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["mean", "std", "min", "max", "median", "range", "variance"]
      },
      "default": ["mean", "std"]
    }
  },
  "required": ["data", "window_size"]
}
```

## Implementation Requirements

### Algorithms
- Simple Moving Average (SMA): arithmetic mean over window
- Exponential Moving Average (EMA): weighted average favoring recent data
- Weighted Moving Average (WMA): linearly weighted
- Linear Regression: least squares method
- CUSUM: cumulative sum algorithm for change detection
- ACF: autocorrelation using FFT for efficiency

### Dependencies
- No new dependencies required (use existing math libraries)
- Consider numpy for performance optimization (optional)

### Performance Targets
- Handle time series up to 100,000 data points
- Moving average: < 100ms for 10,000 points
- Trend detection: < 500ms for 10,000 points
- Change point detection: < 1s for 10,000 points

## Industrial Application Examples

### Example 1: Bearing Temperature Monitoring
```
Input: Hourly bearing temperature readings over 30 days
Tools Used:
1. moving_average (window=24) - smooth daily variations
2. detect_trend - identify if temperature is increasing
3. rate_of_change - alert if temperature rising too fast

Output: "Bearing temperature trending up 0.5°C/day. Recommend inspection within 2 weeks."
```

### Example 2: Batch Process Optimization
```
Input: Batch cycle times for 500 batches
Tools Used:
1. rolling_statistics (window=20) - track recent performance
2. change_point_detection - identify when process improvement occurred
3. autocorrelation - find cycle patterns

Output: "Process improvement at batch 342 reduced cycle time by 15 minutes. Pattern repeats every 8 batches."
```

### Example 3: Energy Consumption Analysis
```
Input: Daily energy consumption for 1 year
Tools Used:
1. moving_average (exponential) - smooth daily variations
2. autocorrelation - detect weekly patterns
3. detect_trend - identify long-term trends

Output: "Energy consumption down 2% annually. Strong weekly pattern detected (weekends 30% lower)."
```

## Acceptance Criteria

- [ ] All 6 tools implemented with comprehensive docstrings
- [ ] Input validation with appropriate range checks
- [ ] Handle edge cases (insufficient data, constant values, NaN handling)
- [ ] Return rich output with interpretations
- [ ] Include confidence metrics where applicable
- [ ] Performance meets targets for large datasets
- [ ] Integration tests with industrial sample data
- [ ] Documentation includes processing plant examples
- [ ] Error messages are clear and actionable

## Testing Requirements

**Test Data Sets:**
1. Synthetic trend data (linear, polynomial)
2. Noisy sensor data with outliers
3. Cyclic batch process data
4. Step changes (process upsets)
5. Real-world equipment degradation data

**Edge Cases:**
- Very small datasets (< 10 points)
- Constant values (no variation)
- Missing data / NaN values
- Extreme outliers
- Very large datasets (100k+ points)

## Documentation Requirements

- Tool descriptions with industrial context
- Parameter selection guidelines
- Interpretation guides for outputs
- Common pitfalls and how to avoid them
- Integration examples with SCADA/historian systems
- Best practices for different process types

## Labels
`enhancement`, `stats-server`, `time-series`, `industrial-automation`, `tier-1-priority`

## Priority
**Tier 1 - Highest Priority** - Essential for industrial process monitoring

## Estimated Effort
6-8 hours for implementation and testing
