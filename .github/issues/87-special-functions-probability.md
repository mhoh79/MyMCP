# Issue #87: Implement Special Functions & Probability Tools

**Priority**: Medium  
**Dependencies**: #79 (Engineering Math)  
**Labels**: enhancement, builtin-server, special-functions, probability, advanced-math  
**Estimated Effort**: 1-2 weeks

## Overview

Add special mathematical functions and advanced probability distributions to enhance the math toolbox for scientific computing, statistics, physics, and engineering applications. Complements the stats server with more specialized probability tools.

## Objectives

- Provide special functions not in standard libraries
- Enable advanced probability calculations
- Support physics and engineering applications
- Facilitate statistical modeling

## Scope

### Special Functions & Probability Tools (8-10 tools in Engineering Math Server)

#### 1. `special_functions`

**Mathematical Special Functions**:
- Gamma function, log-gamma
- Beta function
- Bessel functions (J, Y, I, K)
- Error function, complementary error function
- Exponential integrals (Ei, E1)
- Elliptic integrals
- Zeta function

**Features**:
```python
special_functions(
    function="bessel_j",
    order=2,                   # n for Jn(x)
    argument=5.0,
    derivative=0               # 0 for function, 1+ for derivatives
)
```

**Applications**: Wave propagation, heat transfer, electromagnetic theory

#### 2. `probability_distributions`

**Advanced Distributions** (beyond stats server basics):
- Weibull (reliability)
- Gamma, Inverse Gamma
- Beta (Bayesian priors)
- Log-normal
- Cauchy
- Student's t
- F-distribution
- Chi-squared
- Pareto
- Rayleigh (wind speed)
- Rice (communications)

**Features**:
```python
probability_distributions(
    distribution="weibull",
    parameters={"shape": 2.0, "scale": 100},
    operation="pdf",           # pdf, cdf, quantile, random
    x=50                       # or percentile for quantile
)
```

**Applications**: Reliability analysis, risk assessment, Bayesian inference

#### 3. `reliability_analysis`

**Reliability Engineering**:
- Failure rate (hazard function)
- Reliability function R(t) = 1 - F(t)
- Mean Time To Failure (MTTF)
- Bathtub curve analysis
- Series/parallel system reliability

**Features**:
```python
reliability_analysis(
    distribution="weibull",
    parameters={"shape": 2.5, "scale": 10000},  # hours
    time=5000,
    calculation="reliability"  # or "hazard_rate", "mttf"
)
```

**System Reliability**:
```python
reliability_analysis(
    system_type="series",      # or "parallel", "k_out_of_n"
    component_reliabilities=[0.95, 0.98, 0.92]
)
```

**Applications**: Equipment reliability, maintenance planning

#### 4. `bayesian_inference`

**Bayesian Statistical Tools**:
- Prior distributions
- Likelihood functions
- Posterior calculations (conjugate priors)
- Credible intervals
- Bayes factors

**Features**:
```python
bayesian_inference(
    prior={"distribution": "beta", "parameters": {"alpha": 2, "beta": 2}},
    likelihood={"distribution": "binomial", "successes": 7, "trials": 10},
    calculation="posterior"    # Returns posterior distribution
)
```

**Conjugate Pairs**:
- Beta-Binomial
- Gamma-Poisson
- Normal-Normal

**Applications**: Parameter estimation, A/B testing, decision making

#### 5. `monte_carlo_simulation`

**Stochastic Simulation**:
- Random sampling from distributions
- Uncertainty propagation
- Sensitivity analysis
- Probabilistic design

**Features**:
```python
monte_carlo_simulation(
    variables=[
        {"name": "load", "distribution": "normal", "mean": 1000, "std": 100},
        {"name": "strength", "distribution": "lognormal", "mean": 1500, "std": 200}
    ],
    function="lambda load, strength: strength - load",  # Safety margin
    samples=10000,
    output_statistics=["mean", "std", "percentiles", "probability_of_failure"]
)
```

**Applications**: Risk analysis, tolerance analysis, probabilistic design

#### 6. `confidence_intervals`

**Advanced Confidence Intervals**:
- Bootstrap confidence intervals
- Parametric CIs (normal, t, chi-squared, F)
- Non-parametric CIs
- Proportions (Wilson score, Clopper-Pearson)
- Ratio estimation

**Features**:
```python
confidence_intervals(
    method="bootstrap",
    data=sample_data,
    statistic="mean",          # or custom function
    confidence_level=0.95,
    bootstrap_samples=10000
)
```

**Applications**: Statistical inference, hypothesis testing

#### 7. `optimization_algorithms`

**Advanced Optimization** (beyond Engineering Math basics):
- Simulated annealing
- Genetic algorithms
- Particle swarm optimization
- Differential evolution
- Multi-objective optimization (Pareto fronts)

**Features**:
```python
optimization_algorithms(
    algorithm="genetic_algorithm",
    objective_function="lambda x: x[0]**2 + x[1]**2",
    bounds=[[-10, 10], [-10, 10]],
    constraints=[...],
    population_size=100,
    generations=50
)
```

**Applications**: Non-convex optimization, combinatorial problems

#### 8. `time_series_analysis`

**Time Series Tools** (beyond stats server basics):
- ARIMA model fitting
- Exponential smoothing
- Seasonal decomposition
- Trend analysis
- Autocorrelation (ACF, PACF)

**Features**:
```python
time_series_analysis(
    data=time_series,
    method="arima",
    parameters={"p": 1, "d": 1, "q": 1},  # ARIMA(1,1,1)
    forecast_steps=10
)
```

**Applications**: Forecasting, anomaly detection, trend analysis

#### 9. `survival_analysis`

**Survival & Censored Data**:
- Kaplan-Meier estimator
- Log-rank test
- Cox proportional hazards
- Survival function estimation
- Hazard ratios

**Features**:
```python
survival_analysis(
    times=[5, 10, 15, 20, 25],
    events=[1, 0, 1, 1, 0],    # 1=event occurred, 0=censored
    method="kaplan_meier",
    confidence_level=0.95
)
```

**Applications**: Medical studies, reliability, customer churn

#### 10. `experimental_design`

**Design of Experiments (DOE)**:
- Factorial designs (2^k, 3^k)
- Response surface methodology
- Latin hypercube sampling
- Orthogonal arrays
- Sample size calculations

**Features**:
```python
experimental_design(
    design_type="full_factorial",
    factors={
        "temperature": [20, 40, 60],
        "pressure": [1, 2, 3],
        "catalyst": ["A", "B"]
    },
    replicates=3
)
```

**Output**: Design matrix, randomization order, analysis framework

**Applications**: Process optimization, quality improvement

## Technical Implementation

### Integration with Engineering Math Server

These tools will be **added to the existing Engineering Math Server** rather than creating a separate server:

```
src/builtin/engineering_math_server/
├── tools/
│   ├── ...existing tools...
│   ├── special_functions.py     # Tool 1
│   ├── probability_advanced.py  # Tools 2-4
│   ├── monte_carlo.py           # Tool 5
│   ├── inference.py             # Tool 6
│   ├── optimization_advanced.py # Tool 7
│   ├── time_series.py           # Tool 8
│   ├── survival.py              # Tool 9
│   └── experimental_design.py   # Tool 10
```

### Dependencies
```python
# Additional requirements
scipy.special           # Already available
scipy.stats             # Already available
statsmodels>=0.14.0    # Time series, survival analysis
pyDOE2>=1.3.0          # Experimental design
```

### Coordinate with Stats Server

**Stats Server** (existing): Basic distributions, hypothesis tests, descriptive stats, correlation  
**Engineering Math Server** (enhanced): Special functions, advanced distributions, reliability, Bayesian

Documentation should cross-reference:
```python
"""
For basic probability distributions (normal, uniform, exponential), 
see stats-server tools.

For advanced distributions (Weibull, Gamma, Beta) and reliability analysis,
use engineering-math-server probability tools.
"""
```

## Key Application Examples

### Example 1: Reliability Prediction
```python
# Component follows Weibull distribution
reliability_analysis(
    distribution="weibull",
    parameters={"shape": 2.5, "scale": 10000},  # hours
    time=5000,
    calculation="reliability"
)

# System reliability (3 components in series)
reliability_analysis(
    system_type="series",
    component_reliabilities=[0.95, 0.93, 0.97]
)
# Output: System reliability = 0.95 × 0.93 × 0.97 = 0.857
```

### Example 2: Bayesian A/B Test
```python
# Prior: Beta(2, 2) - neutral prior
# Data: 45 successes out of 100 trials (group A)
bayesian_inference(
    prior={"distribution": "beta", "parameters": {"alpha": 2, "beta": 2}},
    likelihood={"distribution": "binomial", "successes": 45, "trials": 100},
    calculation="posterior"
)
# Output: Posterior is Beta(47, 57)
# Credible interval: [0.37, 0.54]
```

### Example 3: Monte Carlo Tolerance Analysis
```python
# Part dimensions have tolerances
monte_carlo_simulation(
    variables=[
        {"name": "length", "distribution": "uniform", "low": 99.8, "high": 100.2},
        {"name": "width", "distribution": "uniform", "low": 49.9, "high": 50.1}
    ],
    function="lambda length, width: length * width",  # Area
    samples=10000
)
# Output: Mean area, std dev, probability of exceeding 5010 mm²
```

### Example 4: ARIMA Forecasting
```python
time_series_analysis(
    data=monthly_sales,
    method="arima",
    parameters={"p": 1, "d": 1, "q": 1},
    forecast_steps=12          # Forecast next 12 months
)
```

## Testing Requirements

### Unit Tests
- Special functions vs. reference values
- Distribution PDF/CDF calculations
- Reliability formulas
- Bayesian conjugate priors

### Validation Tests
- Compare with R packages (survival, stats)
- Verify against textbook examples
- Cross-check with commercial software

## Deliverables

- [ ] 10 new tools added to Engineering Math Server
- [ ] Integration with scipy.special and scipy.stats
- [ ] statsmodels integration for time series
- [ ] Comprehensive test suite
- [ ] Documentation with examples
- [ ] Cross-references with stats server

## Success Criteria

- ✅ All special functions accurate
- ✅ Probability distributions validated
- ✅ Reliability calculations correct
- ✅ Bayesian inference working
- ✅ Monte Carlo simulations functional
- ✅ Clear documentation on when to use vs. stats server

## Timeline

**Week 1**: Special functions, advanced probability distributions  
**Week 2**: Reliability, Bayesian inference, Monte Carlo, testing

## Related Issues

- Requires: #79 (Engineering Math foundation)
- Related: Stats Server (basic probability)
- Enhancement of: Engineering Math Server

## References

- Numerical Recipes (Press et al.)
- Statistical Distributions (Forbes et al.)
- Reliability Engineering (Ebeling)
- Bayesian Data Analysis (Gelman et al.)
- Introduction to Statistical Quality Control (Montgomery)
