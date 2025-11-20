# Add Regression Analysis Toolkit to Stats Server

## Overview
Implement comprehensive regression analysis tools for equipment performance modeling, predictive maintenance, calibration, and energy optimization in processing plants.

## Motivation
Regression analysis enables:
- **Equipment Performance Modeling**: Predict output based on operating conditions
- **Energy Optimization**: Model energy consumption vs. production rate
- **Calibration**: Create calibration curves for instruments
- **Predictive Maintenance**: Model degradation and predict failures
- **Process Optimization**: Understand relationships between variables
- **Cost Modeling**: Predict maintenance and operating costs

## Tools to Implement

### 1. `linear_regression`
Perform simple and multiple linear regression with comprehensive diagnostics.

**Use Cases:**
- Equipment efficiency vs. load relationship
- Energy consumption vs. production rate
- Pump performance curves (head vs. flow)
- Temperature vs. pressure relationships
- Vibration amplitude vs. bearing wear
- Production rate vs. raw material feed rate

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "x": {
      "type": "array",
      "description": "Independent variable(s) - single array or array of arrays for multiple regression",
      "minItems": 3
    },
    "y": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Dependent variable (response)",
      "minItems": 3
    },
    "confidence_level": {
      "type": "number",
      "description": "Confidence level for intervals (0-1)",
      "minimum": 0.5,
      "maximum": 0.999,
      "default": 0.95
    },
    "include_diagnostics": {
      "type": "boolean",
      "description": "Include full regression diagnostics",
      "default": true
    }
  },
  "required": ["x", "y"]
}
```

**Example Output:**
```json
{
  "coefficients": {
    "intercept": 45.2,
    "slopes": [0.85],
    "equation": "y = 0.85x + 45.2"
  },
  "statistics": {
    "r_squared": 0.94,
    "adj_r_squared": 0.93,
    "rmse": 2.34,
    "mae": 1.89,
    "f_statistic": 234.5,
    "p_value": 0.0001
  },
  "confidence_intervals": {
    "intercept": [43.1, 47.3],
    "slopes": [[0.82, 0.88]]
  },
  "prediction_intervals": {
    "confidence_level": 0.95,
    "example_prediction": {
      "x": 100,
      "predicted_y": 130.2,
      "confidence_interval": [128.5, 131.9],
      "prediction_interval": [125.1, 135.3]
    }
  },
  "diagnostics": {
    "residuals": [...],
    "standardized_residuals": [...],
    "leverage_points": [12, 45],
    "influential_points": [],
    "durbin_watson": 1.85,
    "condition_number": 12.3
  },
  "interpretation": "Strong positive linear relationship (R² = 0.94). Model explains 94% of variance. No influential outliers detected.",
  "warnings": []
}
```

### 2. `polynomial_regression`
Fit polynomial curves for non-linear relationships.

**Use Cases:**
- Compressor performance curves (non-linear)
- Valve characteristics (flow vs. position)
- Catalyst activity decline over time
- Temperature profiles in heat exchangers
- Motor torque vs. speed curves
- Pump efficiency curves

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "x": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Independent variable",
      "minItems": 5
    },
    "y": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Dependent variable",
      "minItems": 5
    },
    "degree": {
      "type": "integer",
      "description": "Polynomial degree (2=quadratic, 3=cubic)",
      "minimum": 2,
      "maximum": 6,
      "default": 2
    },
    "auto_select_degree": {
      "type": "boolean",
      "description": "Automatically select best degree based on AIC/BIC",
      "default": false
    }
  },
  "required": ["x", "y"]
}
```

**Example Output:**
```json
{
  "degree": 2,
  "coefficients": [0.005, 1.2, 50.3],
  "equation": "y = 0.005x² + 1.2x + 50.3",
  "r_squared": 0.97,
  "rmse": 1.8,
  "turning_points": [
    {"x": 120, "y": 122.3, "type": "maximum"}
  ],
  "optimal_x": 120,
  "optimal_y": 122.3,
  "interpretation": "Parabolic relationship with maximum at x=120",
  "goodness_of_fit": "Excellent (R² = 0.97)"
}
```

### 3. `residual_analysis`
Comprehensive analysis of regression residuals to validate model assumptions.

**Use Cases:**
- Validate regression model assumptions
- Detect non-linearity or missing variables
- Identify outliers and influential points
- Check for heteroscedasticity
- Verify normality of errors
- Detect autocorrelation in time series regressions

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "actual": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Actual observed values",
      "minItems": 10
    },
    "predicted": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Model predicted values",
      "minItems": 10
    },
    "x_values": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Independent variable values (optional)",
      "minItems": 10
    }
  },
  "required": ["actual", "predicted"]
}
```

**Example Output:**
```json
{
  "residuals": [...],
  "standardized_residuals": [...],
  "tests": {
    "normality": {
      "shapiro_wilk_statistic": 0.985,
      "p_value": 0.234,
      "conclusion": "Residuals are normally distributed (p > 0.05)"
    },
    "homoscedasticity": {
      "breusch_pagan_statistic": 3.45,
      "p_value": 0.178,
      "conclusion": "Constant variance assumption met (p > 0.05)"
    },
    "autocorrelation": {
      "durbin_watson": 1.92,
      "conclusion": "No significant autocorrelation"
    }
  },
  "outliers": {
    "indices": [23, 67],
    "values": [{"index": 23, "residual": 3.8, "standardized": 3.2}]
  },
  "patterns_detected": [],
  "overall_assessment": "Model assumptions are satisfied. Residuals show random scatter.",
  "recommendations": []
}
```

### 4. `prediction_with_intervals`
Generate predictions with confidence and prediction intervals.

**Use Cases:**
- Forecast equipment performance at specific operating conditions
- Estimate production output with uncertainty bounds
- Predict energy consumption with confidence intervals
- Estimate maintenance costs with uncertainty
- Calculate expected valve position for desired flow
- Predict process yield with tolerance bands

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "model": {
      "type": "object",
      "description": "Regression model from linear_regression or polynomial_regression",
      "properties": {
        "coefficients": {"type": "array"},
        "rmse": {"type": "number"},
        "degree": {"type": "integer"}
      }
    },
    "x_new": {
      "type": "array",
      "description": "New x values for prediction",
      "items": {"type": "number"}
    },
    "confidence_level": {
      "type": "number",
      "description": "Confidence level (0-1)",
      "default": 0.95
    }
  },
  "required": ["model", "x_new"]
}
```

**Example Output:**
```json
{
  "predictions": [
    {
      "x": 150,
      "predicted_y": 172.5,
      "confidence_interval": [170.2, 174.8],
      "prediction_interval": [166.3, 178.7],
      "interpretation": "At x=150, predicted y is 172.5 (95% CI: 170.2-174.8)"
    }
  ],
  "extrapolation_warning": false,
  "reliability": "high"
}
```

### 5. `multivariate_regression`
Multiple linear regression with multiple independent variables.

**Use Cases:**
- Chiller efficiency vs. multiple variables (load, ambient temp, condenser flow)
- Production yield vs. temperature, pressure, catalyst age
- Energy consumption vs. production rate, ambient conditions, equipment age
- Compressor power vs. suction pressure, discharge pressure, flow rate
- Product quality vs. multiple process parameters

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "X": {
      "type": "array",
      "description": "Matrix of independent variables [[x1_1, x2_1, x3_1], [x1_2, x2_2, x3_2], ...]",
      "minItems": 5
    },
    "y": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Dependent variable",
      "minItems": 5
    },
    "variable_names": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Names for each independent variable",
      "default": ["X1", "X2", "X3"]
    },
    "standardize": {
      "type": "boolean",
      "description": "Standardize variables for coefficient comparison",
      "default": false
    }
  },
  "required": ["X", "y"]
}
```

**Example Output:**
```json
{
  "coefficients": {
    "intercept": 25.3,
    "load": 0.45,
    "ambient_temp": -0.32,
    "condenser_flow": 0.18,
    "equation": "Efficiency = 25.3 + 0.45*Load - 0.32*Ambient_Temp + 0.18*Condenser_Flow"
  },
  "r_squared": 0.91,
  "adj_r_squared": 0.89,
  "variable_importance": [
    {"variable": "load", "coefficient": 0.45, "p_value": 0.001, "significance": "***"},
    {"variable": "ambient_temp", "coefficient": -0.32, "p_value": 0.003, "significance": "**"},
    {"variable": "condenser_flow", "coefficient": 0.18, "p_value": 0.042, "significance": "*"}
  ],
  "vif": {
    "load": 1.2,
    "ambient_temp": 1.5,
    "condenser_flow": 1.3
  },
  "multicollinearity": "Low - all VIF < 5",
  "interpretation": "Load has strongest positive effect. Ambient temperature negatively impacts efficiency. All variables are significant."
}
```

## Implementation Requirements

### Algorithms
- Ordinary Least Squares (OLS) using normal equations or QR decomposition
- Coefficient standard errors and t-statistics
- F-statistic for overall model significance
- Durbin-Watson test for autocorrelation
- Breusch-Pagan test for heteroscedasticity
- Shapiro-Wilk test for normality
- VIF (Variance Inflation Factor) for multicollinearity
- AIC/BIC for model selection

### Dependencies
- scipy.stats for statistical tests
- numpy for matrix operations (optional, can use pure Python)

### Performance Targets
- Simple regression: < 50ms for 1000 points
- Multiple regression: < 200ms for 1000 points, 10 variables
- Residual analysis: < 100ms for 1000 points

## Industrial Application Examples

### Example 1: Pump Energy Optimization
```
Input: 
- x: Flow rates (100, 150, 200, 250, 300 m³/h)
- y: Power consumption (45, 62, 82, 105, 131 kW)

Tools Used:
1. polynomial_regression (degree=2) - model power curve
2. residual_analysis - validate model
3. prediction_with_intervals - predict power at 225 m³/h

Output: "Power = 0.0013*Flow² + 0.15*Flow + 30 (R²=0.998). At 225 m³/h: predicted 93.8 kW (CI: 92.1-95.5 kW)"
```

### Example 2: Chiller Efficiency Model
```
Input:
- X: [[80, 30, 1200], [85, 28, 1250], ...] # [load%, ambient_temp°C, condenser_flow_gpm]
- y: [4.8, 5.1, 4.5, ...] # COP values

Tools Used:
1. multivariate_regression - model with all variables
2. residual_analysis - check assumptions
3. VIF analysis - check multicollinearity

Output: "COP = 8.2 + 0.03*Load - 0.08*Ambient + 0.001*Flow (R²=0.89). Ambient temp most critical factor."
```

### Example 3: Bearing Wear Prediction
```
Input:
- x: Operating hours (0, 1000, 2000, ..., 10000)
- y: Vibration amplitude (2.1, 2.3, 2.8, ..., 5.6 mm/s)

Tools Used:
1. polynomial_regression - model wear curve
2. prediction_with_intervals - predict at 12000 hours
3. residual_analysis - validate exponential-like behavior

Output: "Vibration = 0.00004*Hours² + 0.01*Hours + 2.0. At 12000h: predicted 8.4 mm/s (PI: 7.1-9.7). Recommend replacement."
```

## Acceptance Criteria

- [ ] All 5 tools implemented with statistical rigor
- [ ] Comprehensive regression diagnostics
- [ ] Proper handling of edge cases (multicollinearity, rank deficiency)
- [ ] Clear interpretation messages
- [ ] Confidence and prediction intervals calculated correctly
- [ ] VIF analysis for multivariate models
- [ ] Residual plots data returned for visualization
- [ ] Integration tests with engineering datasets
- [ ] Documentation with process engineering examples

## Testing Requirements

**Test Data:**
1. Perfect linear relationships (R²=1.0)
2. Noisy real-world data
3. Non-linear data requiring polynomial fits
4. Multicollinear data for VIF testing
5. Heteroscedastic data
6. Autocorrelated time series data

**Validation:**
- Compare with scipy.stats.linregress
- Verify against hand calculations
- Test edge cases (vertical lines, zero variance)

## Documentation Requirements

- Regression terminology for engineers
- When to use linear vs. polynomial
- Interpreting R², p-values, confidence intervals
- Multicollinearity detection and remedies
- Model validation best practices
- Common pitfalls (overfitting, extrapolation)
- Integration with SCADA/historian data

## Labels
`enhancement`, `stats-server`, `regression`, `predictive-modeling`, `industrial-automation`, `tier-2-priority`

## Priority
**Tier 2 - Important** - Essential for predictive maintenance and optimization

## Estimated Effort
6-8 hours for implementation and testing
