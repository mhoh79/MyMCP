# Add Statistical Process Control (SPC) Toolkit to Stats Server

## Overview
Implement Statistical Process Control tools essential for manufacturing quality control, Six Sigma initiatives, and continuous process monitoring in processing plants.

## Motivation
SPC is fundamental for:
- **Quality Control**: Monitor product specifications and catch defects early
- **Process Stability**: Ensure processes operate within control limits
- **Six Sigma Programs**: Calculate capability indices for improvement projects
- **Regulatory Compliance**: Meet ISO 9001, FDA, and industry standards
- **Cost Reduction**: Reduce scrap and rework through early detection
- **Continuous Improvement**: Data-driven decision making

## Tools to Implement

### 1. `control_limits`
Calculate control chart limits (UCL, LCL, centerline) for X-bar, R, and S charts.

**Use Cases:**
- Monitor critical quality parameters (viscosity, pH, concentration)
- Track process variables (temperature, pressure, flow rate)
- Control product dimensions and weights
- Monitor cycle times and production rates
- Detect process shifts before they cause defects

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Process measurements or subgroup averages",
      "minItems": 5
    },
    "chart_type": {
      "type": "string",
      "enum": ["x_bar", "individuals", "range", "std_dev", "p", "np", "c", "u"],
      "description": "Type of control chart"
    },
    "subgroup_size": {
      "type": "integer",
      "description": "Size of subgroups (for X-bar and R charts)",
      "minimum": 2,
      "maximum": 25,
      "default": 5
    },
    "sigma_level": {
      "type": "number",
      "description": "Number of standard deviations for limits",
      "minimum": 1,
      "maximum": 6,
      "default": 3
    }
  },
  "required": ["data", "chart_type"]
}
```

**Example Output:**
```json
{
  "chart_type": "x_bar",
  "centerline": 100.5,
  "ucl": 103.2,
  "lcl": 97.8,
  "sigma": 0.9,
  "subgroup_size": 5,
  "data_points": 30,
  "out_of_control_points": [12, 23],
  "out_of_control_details": [
    {"index": 12, "value": 103.5, "violation": "above_ucl"},
    {"index": 23, "value": 97.1, "violation": "below_lcl"}
  ],
  "process_status": "Out of Control - 2 points beyond limits",
  "recommendations": "Investigate special cause variation at points 12 and 23"
}
```

### 2. `process_capability`
Calculate process capability indices (Cp, Cpk, Pp, Ppk) to assess process performance.

**Use Cases:**
- Evaluate if process meets customer specifications
- Compare process performance before/after improvements
- Assess supplier quality capability
- Determine if process can achieve Six Sigma levels
- Support process qualification and validation
- Make data-driven equipment purchase decisions

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Process measurements",
      "minItems": 30
    },
    "usl": {
      "type": "number",
      "description": "Upper specification limit"
    },
    "lsl": {
      "type": "number",
      "description": "Lower specification limit"
    },
    "target": {
      "type": "number",
      "description": "Target value (optional, defaults to midpoint)"
    }
  },
  "required": ["data", "usl", "lsl"]
}
```

**Example Output:**
```json
{
  "sample_size": 100,
  "mean": 100.2,
  "std_dev": 0.85,
  "usl": 103.0,
  "lsl": 97.0,
  "target": 100.0,
  "cp": 2.35,
  "cpk": 2.18,
  "pp": 2.28,
  "ppk": 2.15,
  "cp_interpretation": "Excellent - Process spread is much smaller than tolerance",
  "cpk_interpretation": "Excellent - Process is well-centered and capable",
  "sigma_level": 6.54,
  "percent_within_spec": 99.9999,
  "estimated_ppm_defects": 0.001,
  "process_performance": "Six Sigma capable",
  "centering": "Slightly off-center by 0.2 units",
  "recommendations": "Process is capable. Consider tightening specifications or cost reduction.",
  "capability_chart_data": {
    "distribution_curve": [...],
    "spec_limits": {"lsl": 97.0, "usl": 103.0}
  }
}
```

### 3. `western_electric_rules`
Apply Western Electric run rules to detect non-random patterns in control charts.

**Use Cases:**
- Early warning of process shifts before out-of-control points
- Detect systematic patterns indicating assignable causes
- Identify tool wear, shift changes, or raw material variation
- Supplement traditional control limits
- Automated process monitoring and alarming

**Rules Implemented:**
1. One point beyond 3σ
2. Two out of three consecutive points beyond 2σ (same side)
3. Four out of five consecutive points beyond 1σ (same side)
4. Eight consecutive points on same side of centerline
5. Six points in a row steadily increasing or decreasing
6. Fifteen points in a row within 1σ of centerline (both sides)
7. Fourteen points in a row alternating up and down
8. Eight points in a row beyond 1σ (either side)

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Process measurements in time order",
      "minItems": 15
    },
    "centerline": {
      "type": "number",
      "description": "Process centerline (mean)"
    },
    "sigma": {
      "type": "number",
      "description": "Process standard deviation"
    },
    "rules_to_apply": {
      "type": "array",
      "items": {"type": "integer", "minimum": 1, "maximum": 8},
      "description": "Which rules to check (default: all)",
      "default": [1, 2, 3, 4, 5, 6, 7, 8]
    }
  },
  "required": ["data", "centerline", "sigma"]
}
```

**Example Output:**
```json
{
  "violations": [
    {
      "rule": 1,
      "rule_name": "One point beyond 3σ",
      "indices": [45],
      "severity": "critical",
      "description": "Point 45 (value: 106.2) is 3.4σ above centerline"
    },
    {
      "rule": 4,
      "rule_name": "Eight consecutive points same side",
      "indices": [12, 13, 14, 15, 16, 17, 18, 19],
      "severity": "warning",
      "description": "8 consecutive points above centerline indicates process shift"
    }
  ],
  "total_violations": 2,
  "process_status": "Out of Statistical Control",
  "action_required": "Investigate special cause at point 45 and sustained shift starting at point 12",
  "pattern_detected": "Upward trend and outlier"
}
```

### 4. `cusum_chart`
Implement CUSUM (Cumulative Sum) chart for detecting small persistent shifts.

**Use Cases:**
- Detect gradual process drift
- Monitor slow equipment degradation
- Identify small but persistent quality shifts
- More sensitive than traditional control charts for small shifts (<1.5σ)
- Track process improvements over time

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Process measurements in time order",
      "minItems": 5
    },
    "target": {
      "type": "number",
      "description": "Target process value"
    },
    "k": {
      "type": "number",
      "description": "Reference value (typically 0.5σ)",
      "default": 0.5
    },
    "h": {
      "type": "number",
      "description": "Decision interval (typically 4σ or 5σ)",
      "default": 4
    },
    "sigma": {
      "type": "number",
      "description": "Process standard deviation"
    }
  },
  "required": ["data", "target"]
}
```

**Example Output:**
```json
{
  "cusum_positive": [0, 0.2, 0.5, 1.1, 1.8, 2.6, 3.5, 4.2],
  "cusum_negative": [0, 0, 0, 0, 0, 0, 0, 0],
  "upper_limit": 4.0,
  "lower_limit": -4.0,
  "signals": [
    {
      "index": 7,
      "type": "positive_shift",
      "cusum_value": 4.2,
      "estimated_change_point": 4,
      "magnitude_estimate": 1.2
    }
  ],
  "process_status": "Shift detected at point 7",
  "estimated_new_level": 101.2,
  "recommendation": "Process has shifted upward by approximately 1.2 units starting around point 4"
}
```

### 5. `ewma_chart`
Implement EWMA (Exponentially Weighted Moving Average) chart for detecting small shifts.

**Use Cases:**
- Monitor processes where small shifts are critical
- Detect shifts faster than X-bar charts
- Smooth noisy process data
- Balance between Shewhart and CUSUM charts
- Track chemical composition or pharmaceutical potency

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Process measurements",
      "minItems": 5
    },
    "target": {
      "type": "number",
      "description": "Target process mean"
    },
    "lambda": {
      "type": "number",
      "description": "Weighting factor (0-1, typical: 0.2)",
      "minimum": 0.01,
      "maximum": 1.0,
      "default": 0.2
    },
    "sigma": {
      "type": "number",
      "description": "Process standard deviation"
    },
    "l": {
      "type": "number",
      "description": "Control limit factor (typical: 3)",
      "default": 3
    }
  },
  "required": ["data", "target", "sigma"]
}
```

## Implementation Requirements

### Algorithms
- A2, D3, D4 constants for control charts (tabulated values)
- Cp/Cpk calculations with bias correction for small samples
- CUSUM: V-mask or tabular method
- EWMA: recursive calculation with proper initialization
- Western Electric rules: pattern matching algorithms

### Dependencies
- scipy.stats for normal distribution calculations
- No additional external dependencies

### Performance Targets
- Control limits: < 50ms for 1000 points
- Process capability: < 100ms for 1000 points
- CUSUM/EWMA: < 200ms for 1000 points
- Western Electric rules: < 100ms for 1000 points

## Industrial Application Examples

### Example 1: Chemical Reactor pH Control
```
Input: pH measurements from reactor (every 5 minutes, 24 hours)
Tools Used:
1. control_limits (individuals chart) - establish control limits
2. western_electric_rules - detect early warning signs
3. ewma_chart - smooth noisy pH readings

Output: "pH shifted 0.2 units upward at 14:30. EWMA detected change 30 minutes before out-of-control point."
```

### Example 2: Packaging Line Weight Control
```
Input: Package weights (n=5 samples every hour)
Tools Used:
1. control_limits (X-bar and R chart) - monitor average and variability
2. process_capability - verify meeting label weight requirements
3. western_electric_rules - detect trends indicating machine wear

Output: "Cpk = 1.82 (Good). Warning: Rule 3 violation - process trending downward. Check filler mechanism."
```

### Example 3: Compressor Performance Monitoring
```
Input: Compressor efficiency measurements (daily over 90 days)
Tools Used:
1. cusum_chart - detect gradual efficiency loss
2. control_limits - overall monitoring
3. process_capability - compare to design specifications

Output: "CUSUM detected 2% efficiency drop starting day 45. Estimated current efficiency 88% vs. 90% target. Schedule maintenance."
```

## Acceptance Criteria

- [ ] All 5 tools implemented with industrial-grade accuracy
- [ ] Proper handling of control chart constants (A2, D3, D4, etc.)
- [ ] Comprehensive input validation
- [ ] Clear, actionable output messages
- [ ] Visual data for plotting (coordinates for charts)
- [ ] Handle edge cases (insufficient data, zero variation)
- [ ] Integration tests with manufacturing data
- [ ] Documentation with SPC terminology
- [ ] Examples for different process types (discrete, continuous, batch)

## Testing Requirements

**Test Data:**
1. In-control process data (random variation only)
2. Out-of-control data (step change, trend, cycle)
3. Real manufacturing data sets
4. Edge cases (zero variation, single outlier)
5. Long-run data (1000+ points)

**Validation:**
- Compare results with Minitab/JMP for standard datasets
- Verify capability indices match hand calculations
- Test Western Electric rules against known patterns
- Validate CUSUM/EWMA detection speeds

## Documentation Requirements

- SPC terminology glossary
- Interpretation guides for each chart type
- When to use which tool
- Common process patterns and causes
- Integration with SCADA/MES systems
- Case studies from processing plants
- Regulatory compliance notes (FDA, ISO)

## Labels
`enhancement`, `stats-server`, `spc`, `quality-control`, `industrial-automation`, `tier-1-priority`

## Priority
**Tier 1 - Highest Priority** - Critical for quality control and Six Sigma programs

## Estimated Effort
8-10 hours for implementation and testing
