# Split Statistical Tools into Separate MCP Server

## Overview
Extract the statistical analysis tools from the main mathematical tools MCP server into a dedicated statistical analysis MCP server. This separation will improve modularity, maintainability, and allow for specialized statistical functionality expansion.

## Motivation
- **Separation of Concerns**: Statistical analysis is a distinct domain from mathematical calculations
- **Focused Development**: Easier to extend statistical capabilities without affecting math tools
- **Performance**: Lighter weight servers with specific purposes
- **Dependency Management**: Statistical tools may require additional dependencies (numpy, scipy, pandas) that aren't needed for basic math
- **Easier Testing**: Isolated testing of statistical functionality
- **Deployment Flexibility**: Users can choose to run only the servers they need

## Current Statistical Tools to Extract

The following tools should be moved to the new statistical server:

1. **descriptive_stats** - Calculate mean, median, mode, standard deviation, variance, min, max, quartiles
2. **correlation** - Pearson correlation coefficient and covariance analysis between two datasets
3. **percentile** - Calculate specific percentiles from a dataset
4. **detect_outliers** - Identify outliers using IQR method with configurable threshold

## Proposed Architecture

### New Repository Structure
```
MyMCP/
├── src/
│   ├── math_server/          # Existing mathematical tools
│   │   └── server.py
│   └── stats_server/         # New statistical tools server
│       ├── __init__.py
│       └── server.py
├── requirements/
│   ├── math.txt              # Dependencies for math server
│   └── stats.txt             # Dependencies for stats server (may include numpy, scipy)
└── README.md
```

### Configuration for Claude Desktop
Users would configure both servers:
```json
{
  "mcpServers": {
    "math-tools": {
      "command": "path/to/venv/Scripts/python.exe",
      "args": ["path/to/MyMCP/src/math_server/server.py"]
    },
    "stats-tools": {
      "command": "path/to/venv/Scripts/python.exe",
      "args": ["path/to/MyMCP/src/stats_server/server.py"]
    }
  }
}
```

## Implementation Tasks

### Phase 1: Create New Statistical Server
- [ ] Create `src/stats_server/` directory
- [ ] Create `src/stats_server/__init__.py` with version info
- [ ] Create `src/stats_server/server.py` with MCP server boilerplate
- [ ] Copy statistical functions from main server:
  - `descriptive_stats(data: list[float]) -> dict[str, Any]`
  - `correlation(x: list[float], y: list[float]) -> dict[str, Any]`
  - `percentile(data: list[float], p: float) -> float`
  - `detect_outliers(data: list[float], threshold: float) -> dict[str, Any]`
- [ ] Implement tool registration in `list_tools()`
- [ ] Implement tool handlers in `call_tool()`
- [ ] Add comprehensive logging

### Phase 2: Update Main Math Server
- [ ] Remove statistical functions from `src/fibonacci_server/server.py`
- [ ] Remove statistical tool registrations from `list_tools()`
- [ ] Remove statistical tool handlers from `call_tool()`
- [ ] Update server name to "math-calculator" for clarity

### Phase 3: Dependencies and Requirements
- [ ] Create `requirements/math.txt` with math server dependencies
- [ ] Create `requirements/stats.txt` with stats server dependencies
- [ ] Update root `requirements.txt` to reference both files or provide installation instructions
- [ ] Consider adding optional dependencies for enhanced stats (numpy, scipy, pandas)

### Phase 4: Documentation
- [ ] Update main README.md with dual-server setup instructions
- [ ] Create `src/stats_server/README.md` with statistical tools documentation
- [ ] Update `src/math_server/README.md` (rename from fibonacci_server)
- [ ] Add configuration examples for both servers
- [ ] Document migration path for existing users

### Phase 5: Testing and Validation
- [ ] Test statistical server independently
- [ ] Test math server after removal of statistical tools
- [ ] Verify both servers can run simultaneously
- [ ] Test in Claude Desktop with dual configuration
- [ ] Ensure all statistical tools work correctly in new server

## Tool Specifications for Stats Server

### descriptive_stats
```json
{
  "name": "descriptive_stats",
  "description": "Calculate comprehensive descriptive statistics for a dataset including mean, median, mode, standard deviation, variance, min, max, quartiles, and range.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "data": {
        "type": "array",
        "items": {"type": "number"},
        "description": "Array of numerical values to analyze",
        "minItems": 1
      }
    },
    "required": ["data"]
  }
}
```

### correlation
```json
{
  "name": "correlation",
  "description": "Calculate Pearson correlation coefficient and covariance between two datasets. Returns correlation coefficient, covariance, p-value, and interpretation.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "x": {
        "type": "array",
        "items": {"type": "number"},
        "description": "First dataset",
        "minItems": 2
      },
      "y": {
        "type": "array",
        "items": {"type": "number"},
        "description": "Second dataset (must be same length as x)",
        "minItems": 2
      }
    },
    "required": ["x", "y"]
  }
}
```

### percentile
```json
{
  "name": "percentile",
  "description": "Calculate the value at a specific percentile in a dataset using linear interpolation method.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "data": {
        "type": "array",
        "items": {"type": "number"},
        "description": "Array of numerical values",
        "minItems": 1
      },
      "p": {
        "type": "number",
        "description": "Percentile to calculate (0-100)",
        "minimum": 0,
        "maximum": 100
      }
    },
    "required": ["data", "p"]
  }
}
```

### detect_outliers
```json
{
  "name": "detect_outliers",
  "description": "Detect outliers in a dataset using the Interquartile Range (IQR) method with configurable threshold multiplier.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "data": {
        "type": "array",
        "items": {"type": "number"},
        "description": "Array of numerical values to check for outliers",
        "minItems": 4
      },
      "threshold": {
        "type": "number",
        "description": "IQR multiplier (default: 1.5 for standard outliers, 3.0 for extreme outliers)",
        "minimum": 0.1,
        "maximum": 10,
        "default": 1.5
      }
    },
    "required": ["data"]
  }
}
```

## Future Statistical Enhancements

Once the separation is complete, the stats server can be extended with:

- **Hypothesis Testing**: t-tests, chi-square tests, ANOVA
- **Distributions**: Normal distribution, binomial, Poisson calculations
- **Regression Analysis**: Linear regression, multiple regression
- **Time Series**: Moving averages, trend analysis, seasonality
- **Sampling**: Random sampling, stratified sampling, bootstrap
- **Data Transformation**: Normalization, standardization, log transforms
- **Advanced Correlation**: Spearman, Kendall rank correlation
- **Statistical Power**: Sample size calculations, power analysis

## Acceptance Criteria

- [ ] New `stats_server` runs independently as a valid MCP server
- [ ] All 4 statistical tools work correctly in the new server
- [ ] Math server no longer contains statistical code
- [ ] Both servers can run simultaneously without conflicts
- [ ] Documentation updated with dual-server setup instructions
- [ ] Claude Desktop can connect to both servers
- [ ] All existing statistical functionality preserved
- [ ] Code is well-documented with comprehensive docstrings
- [ ] Logging implemented for debugging
- [ ] Error handling covers all edge cases

## Migration Notes for Users

Existing users will need to:
1. Pull the latest code from the repository
2. Update their Claude Desktop configuration to include both servers
3. Restart Claude Desktop to load both server configurations
4. No changes to how they use the tools - same tool names and interfaces

## Technical Notes

- Maintain consistent error handling patterns across both servers
- Use the same MCP SDK version for both servers
- Keep logging format consistent
- Consider shared utility functions in a common module if needed
- Ensure both servers handle stdio transport correctly
- Test with MCP Inspector for validation

## Labels
`enhancement`, `refactoring`, `architecture`

## Priority
Medium - This is a structural improvement that will benefit long-term maintainability

## Estimated Effort
2-3 hours for implementation and testing
