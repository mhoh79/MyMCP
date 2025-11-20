# Add Advanced Outlier Detection Methods to Stats Server

## Overview
Extend the current IQR-based outlier detection with multiple sophisticated methods suitable for different data types and industrial applications.

## Motivation
Current outlier detection (IQR method) is limited:
- **Multiple Methods Needed**: Different processes require different detection approaches
- **Multivariate Detection**: Industrial processes have correlated variables requiring multivariate methods
- **Real-time Detection**: Need methods suitable for streaming sensor data
- **Robust to Contamination**: Some methods handle high outlier percentages better
- **Statistical Rigor**: Need methods with statistical significance testing

Industrial applications:
- Sensor validation and fault detection
- Quality control and defect identification
- Anomaly detection in equipment behavior
- Data cleansing for historian databases
- Real-time alarm generation

## Tools to Implement

### 1. `z_score_detection`
Standard and Modified Z-score methods for outlier detection.

**Use Cases:**
- Normally distributed process variables (temperature, pressure)
- Quick screening of large datasets
- Real-time sensor validation
- Quality control measurements
- Batch-to-batch comparison

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Measurements to check for outliers",
      "minItems": 3
    },
    "method": {
      "type": "string",
      "enum": ["standard", "modified"],
      "default": "modified",
      "description": "Standard (mean/std) or Modified (median/MAD) - modified is more robust"
    },
    "threshold": {
      "type": "number",
      "description": "Z-score threshold (typical: 3.0 for standard, 3.5 for modified)",
      "minimum": 1.0,
      "maximum": 10.0,
      "default": 3.0
    },
    "two_tailed": {
      "type": "boolean",
      "description": "Detect outliers on both sides",
      "default": true
    }
  },
  "required": ["data"]
}
```

**Example Output:**
```json
{
  "method": "modified_z_score",
  "threshold": 3.5,
  "outliers": {
    "indices": [12, 45, 67],
    "values": [
      {"index": 12, "value": 156.3, "z_score": 4.2, "severity": "extreme"},
      {"index": 45, "value": 145.8, "z_score": 3.7, "severity": "moderate"},
      {"index": 67, "value": 38.2, "z_score": -3.9, "severity": "extreme"}
    ]
  },
  "statistics": {
    "median": 100.5,
    "mad": 5.3,
    "total_points": 100,
    "outlier_count": 3,
    "outlier_percentage": 3.0
  },
  "cleaned_data": [...],
  "interpretation": "3 outliers detected (3.0%). Modified Z-score used for robustness."
}
```

### 2. `grubbs_test`
Statistical test for single outliers in normally distributed data.

**Use Cases:**
- Reject suspicious calibration points
- Validate laboratory test results
- Quality control for precise measurements
- Statistical rigor for critical decisions
- Regulatory compliance (FDA, ISO)

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Dataset to test for outliers",
      "minItems": 7
    },
    "alpha": {
      "type": "number",
      "description": "Significance level (typical: 0.05 or 0.01)",
      "minimum": 0.001,
      "maximum": 0.1,
      "default": 0.05
    },
    "method": {
      "type": "string",
      "enum": ["max", "min", "two_sided"],
      "default": "two_sided",
      "description": "Test for maximum outlier, minimum outlier, or both"
    }
  },
  "required": ["data"]
}
```

**Example Output:**
```json
{
  "test": "Grubbs test",
  "alpha": 0.05,
  "sample_size": 50,
  "suspected_outlier": {
    "value": 156.3,
    "index": 12,
    "side": "maximum"
  },
  "test_statistic": 3.82,
  "critical_value": 3.16,
  "p_value": 0.023,
  "conclusion": "Reject null hypothesis - value is a significant outlier",
  "recommendation": "Remove point 12 (value: 156.3) - statistically significant at α=0.05"
}
```

### 3. `dixon_q_test`
Quick test for outliers in small datasets (3-30 points).

**Use Cases:**
- Laboratory quality control (small sample sizes)
- Pilot plant trials (limited data)
- Expensive test results validation
- Duplicate/triplicate measurement validation
- Shift sample validation

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Small dataset (3-30 points)",
      "minItems": 3,
      "maxItems": 30
    },
    "alpha": {
      "type": "number",
      "description": "Significance level",
      "default": 0.05
    }
  },
  "required": ["data"]
}
```

**Example Output:**
```json
{
  "test": "Dixon Q test",
  "sample_size": 8,
  "suspected_outlier": {
    "value": 34.2,
    "index": 5,
    "position": "high"
  },
  "q_statistic": 0.68,
  "q_critical": 0.526,
  "conclusion": "Outlier detected - reject value",
  "recommendation": "Repeat measurement or investigate cause"
}
```

### 4. `isolation_forest`
Machine learning-based anomaly detection for complex datasets.

**Use Cases:**
- Multivariate anomaly detection (multiple sensors)
- Complex process behavior patterns
- Equipment failure prediction
- Cyber security (unusual patterns)
- Unstructured anomaly patterns

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "description": "Univariate or multivariate data [[x1, x2, x3], ...]",
      "minItems": 10
    },
    "contamination": {
      "type": "number",
      "description": "Expected proportion of outliers (0-0.5)",
      "minimum": 0.0,
      "maximum": 0.5,
      "default": 0.1
    },
    "n_estimators": {
      "type": "integer",
      "description": "Number of isolation trees",
      "minimum": 50,
      "maximum": 500,
      "default": 100
    }
  },
  "required": ["data"]
}
```

**Example Output:**
```json
{
  "method": "Isolation Forest",
  "anomalies": {
    "indices": [12, 34, 56, 89],
    "anomaly_scores": [0.72, 0.68, 0.65, 0.63],
    "severity": ["high", "high", "medium", "medium"]
  },
  "contamination": 0.04,
  "interpretation": "4 anomalies detected (4% of data). Isolation scores indicate how isolated each point is from normal patterns."
}
```

### 5. `mahalanobis_distance`
Multivariate outlier detection considering correlations between variables.

**Use Cases:**
- Multiple correlated sensor detection
- Process state monitoring (temperature, pressure, flow together)
- Multivariate quality control
- Equipment health monitoring (multiple parameters)
- Pattern recognition in complex processes

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "description": "Multivariate data [[x1, y1, z1], [x2, y2, z2], ...]",
      "minItems": 10
    },
    "threshold": {
      "type": "number",
      "description": "Chi-square threshold percentile (0-1)",
      "minimum": 0.9,
      "maximum": 0.999,
      "default": 0.975
    }
  },
  "required": ["data"]
}
```

**Example Output:**
```json
{
  "method": "Mahalanobis Distance",
  "dimensions": 3,
  "outliers": {
    "indices": [23, 67],
    "distances": [
      {"index": 23, "distance": 12.5, "p_value": 0.006},
      {"index": 67, "distance": 10.8, "p_value": 0.012}
    ]
  },
  "threshold_distance": 9.35,
  "degrees_of_freedom": 3,
  "interpretation": "2 multivariate outliers detected. These points are unusual in the combined context of all variables.",
  "variable_contributions": [
    {"index": 23, "primary_variable": "temperature", "contribution": 0.65}
  ]
}
```

### 6. `streaming_outlier_detection`
Real-time outlier detection for continuous sensor streams.

**Use Cases:**
- Real-time SCADA alarming
- Edge device data validation
- Continuous process monitoring
- High-frequency sensor data (1-second intervals)
- Telemetry data validation

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "current_value": {
      "type": "number",
      "description": "New measurement to evaluate"
    },
    "historical_window": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Recent historical values for context",
      "minItems": 10,
      "maxItems": 1000
    },
    "method": {
      "type": "string",
      "enum": ["ewma", "cusum", "adaptive_threshold"],
      "default": "ewma"
    },
    "sensitivity": {
      "type": "number",
      "description": "Detection sensitivity (1-10, higher = more sensitive)",
      "minimum": 1,
      "maximum": 10,
      "default": 5
    }
  },
  "required": ["current_value", "historical_window"]
}
```

**Example Output:**
```json
{
  "current_value": 145.8,
  "is_outlier": true,
  "severity": "warning",
  "method": "ewma",
  "expected_range": [95.2, 105.8],
  "deviation": 40.0,
  "deviation_sigma": 3.2,
  "interpretation": "Current value exceeds expected range by 3.2 standard deviations",
  "recommendation": "Investigate sensor or process condition",
  "trend": "increasing",
  "rate_of_change": 5.2
}
```

## Implementation Requirements

### Algorithms
- Z-score: standard and modified (MAD-based)
- Grubbs test: maximum normed residual test with critical values
- Dixon Q test: gap ratio test with lookup tables
- Isolation Forest: tree-based isolation algorithm
- Mahalanobis distance: multivariate distance with covariance matrix
- EWMA: exponentially weighted moving average for streaming
- CUSUM: cumulative sum for shift detection

### Dependencies
- scipy.stats for statistical distributions
- numpy for efficient calculations (optional)
- No sklearn required (implement isolation forest from scratch if needed)

### Performance Targets
- Z-score/IQR: < 10ms for 10,000 points
- Grubbs/Dixon: < 50ms for 100 points
- Mahalanobis: < 100ms for 1,000 points (10 dimensions)
- Streaming: < 5ms per point (real-time requirement)

## Industrial Application Examples

### Example 1: Temperature Sensor Validation
```
Input: Reactor temperature readings: [200.1, 200.3, 199.8, 250.5, 200.2, ...]
Tools Used:
1. z_score_detection (modified) - robust to single outliers
2. grubbs_test - statistical validation
3. streaming_outlier_detection - real-time monitoring

Output: "Point 4 (250.5°C) is outlier (Z=4.5, p<0.01). Likely sensor fault. Expected range: 199-201°C."
```

### Example 2: Multivariate Process Monitoring
```
Input: [Temperature, Pressure, Flow] measurements over time
Tools Used:
1. mahalanobis_distance - detect unusual combinations
2. isolation_forest - learn normal patterns

Output: "Anomaly at 14:30 - unusual combination of high temp (normal) + low flow (normal individually, but together abnormal). Check heat exchanger."
```

### Example 3: Quality Control Lab Data
```
Input: 8 replicate measurements: [10.2, 10.3, 10.1, 15.8, 10.4, 10.2, 10.3, 10.1]
Tools Used:
1. dixon_q_test - small sample size
2. grubbs_test - statistical confirmation

Output: "Value 15.8 rejected (Dixon Q=0.82 > Q_crit=0.526, α=0.05). Likely measurement error. Repeat test."
```

## Acceptance Criteria

- [ ] All 6 outlier detection methods implemented
- [ ] Statistical rigor with p-values where applicable
- [ ] Handle univariate and multivariate data
- [ ] Real-time streaming detection capability
- [ ] Clear severity levels (mild, moderate, extreme)
- [ ] Recommendations for each detected outlier
- [ ] Comparison of methods in documentation
- [ ] Edge case handling (all outliers, no outliers)
- [ ] Integration tests with industrial sensor data

## Testing Requirements

**Test Data:**
1. Clean data with no outliers
2. Data with known outliers (single, multiple)
3. Multivariate correlated data
4. Small datasets (n<10)
5. Large datasets (n>10,000)
6. Streaming data with sudden changes

**Validation:**
- Compare with established statistical software
- Verify critical values for Grubbs and Dixon tests
- Test multivariate methods with known patterns

## Documentation Requirements

- Method selection guide (when to use which method)
- Comparison of methods (strengths/weaknesses)
- Interpretation of outputs
- False positive/negative rates
- Industrial case studies
- Integration with alarm systems
- Data quality implications

## Labels
`enhancement`, `stats-server`, `outlier-detection`, `data-quality`, `industrial-automation`, `tier-1-priority`

## Priority
**Tier 1 - Highest Priority** - Critical for data quality and sensor validation

## Estimated Effort
8-10 hours for implementation and testing
