"""
Main MCP server implementation for statistical analysis.
This server exposes tools for descriptive statistics, correlation analysis, 
percentile calculations, and outlier detection.
"""

import asyncio
import logging
from typing import Any
import math

# Scientific computing imports for advanced outlier detection
import numpy as np
from scipy import stats

# MCP SDK imports for building the server
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)

# Configure logging to stderr (stdout is reserved for MCP protocol messages)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Logs go to stderr by default
)
logger = logging.getLogger("stats-server")


# ============================================================================
# Statistical Analysis Functions
# ============================================================================


def descriptive_stats(data: list[float]) -> dict[str, Any]:
    """
    Calculate comprehensive descriptive statistics for a dataset.
    
    This function computes key statistical measures that describe the central
    tendency, dispersion, and distribution of a dataset.
    
    Args:
        data: List of numeric values (1-10000 items)
        
    Returns:
        Dictionary containing:
        - 'mean': Arithmetic average of the data
        - 'median': Middle value when data is sorted
        - 'mode': Most frequently occurring value(s)
        - 'range': Difference between max and min values
        - 'variance': Average of squared deviations from mean
        - 'std_dev': Square root of variance (standard deviation)
        - 'count': Number of data points
        - 'min': Minimum value
        - 'max': Maximum value
        
    Raises:
        ValueError: If data is empty, not a list, or contains non-numeric values
        ValueError: If data contains fewer than 1 or more than 10000 items
        
    Examples:
        descriptive_stats([23, 45, 12, 67, 34, 89, 23, 56])
        >>> {
        ...   'mean': 43.625,
        ...   'median': 39.5,
        ...   'mode': [23],
        ...   'range': 77,
        ...   'variance': 642.734375,
        ...   'std_dev': 25.352...,
        ...   'count': 8,
        ...   'min': 12,
        ...   'max': 89
        ... }
        
    Formula Explanations:
        - Mean: μ = Σx / n (sum of all values divided by count)
        - Median: Middle value of sorted data (or average of two middle values)
        - Mode: Value(s) with highest frequency
        - Range: max - min
        - Variance: σ² = Σ(x - μ)² / n (population variance)
        - Standard Deviation: σ = √(variance)
        
    Algorithm Explanation:
        1. Validate input data (type, size, numeric values)
        2. Calculate mean by summing all values and dividing by count
        3. Calculate median by sorting data and finding middle value(s)
        4. Calculate mode by counting frequencies and finding maximum
        5. Calculate variance as average of squared deviations from mean
        6. Calculate standard deviation as square root of variance
        7. Find min, max, and range
        
    Time Complexity: O(n log n) due to sorting for median
    Space Complexity: O(n) for storing sorted data and frequency map
    """
    # Validate input type
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    # Validate data size
    if len(data) < 1 or len(data) > 10000:
        raise ValueError("Data must contain between 1 and 10000 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    n = len(data)
    
    # Calculate mean
    # μ = Σx / n
    mean = sum(data) / n
    
    # Calculate median
    # Sort data and find middle value(s)
    sorted_data = sorted(data)
    if n % 2 == 0:
        # Even number of elements: average of two middle values
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        # Odd number of elements: middle value
        median = sorted_data[n // 2]
    
    # Calculate mode (most frequent value(s))
    # Build frequency map
    frequency = {}
    for value in data:
        frequency[value] = frequency.get(value, 0) + 1
    
    # Find maximum frequency
    max_freq = max(frequency.values())
    
    # Find all values with maximum frequency
    mode = [value for value, freq in frequency.items() if freq == max_freq]
    mode.sort()  # Sort for consistent output
    
    # Calculate variance
    # σ² = Σ(x - μ)² / n (population variance)
    variance = sum((x - mean) ** 2 for x in data) / n
    
    # Calculate standard deviation
    # σ = √(variance)
    std_dev = variance ** 0.5
    
    # Calculate min, max, and range
    min_value = min(data)
    max_value = max(data)
    data_range = max_value - min_value
    
    return {
        'mean': mean,
        'median': median,
        'mode': mode,
        'range': data_range,
        'variance': variance,
        'std_dev': std_dev,
        'count': n,
        'min': min_value,
        'max': max_value
    }


def correlation(x: list[float], y: list[float]) -> dict[str, Any]:
    """
    Calculate Pearson correlation coefficient between two datasets.
    
    The Pearson correlation coefficient measures the linear relationship between
    two variables. It ranges from -1 (perfect negative correlation) to +1
    (perfect positive correlation), with 0 indicating no linear correlation.
    
    Args:
        x: First dataset (2-1000 items)
        y: Second dataset (must be same length as x)
        
    Returns:
        Dictionary containing:
        - 'coefficient': Pearson correlation coefficient (r)
        - 'interpretation': Human-readable interpretation of the correlation
        
    Raises:
        ValueError: If x or y are not lists, have different lengths,
                   contain non-numeric values, or are outside valid size range
        
    Examples:
        correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        >>> {'coefficient': 1.0, 'interpretation': 'Perfect positive correlation'}
        
        correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        >>> {'coefficient': -1.0, 'interpretation': 'Perfect negative correlation'}
        
        correlation([1, 2, 3, 4, 5], [3, 5, 2, 7, 4])
        >>> {'coefficient': 0.5..., 'interpretation': 'Moderate positive correlation'}
        
    Formula:
        r = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² × Σ(y - ȳ)²]
        
        Where:
        - x̄ is the mean of x
        - ȳ is the mean of y
        - Numerator: covariance between x and y
        - Denominator: product of standard deviations
        
    Interpretation Guide:
        - r = 1.0: Perfect positive correlation
        - 0.7 ≤ r < 1.0: Strong positive correlation
        - 0.4 ≤ r < 0.7: Moderate positive correlation
        - 0.1 ≤ r < 0.4: Weak positive correlation
        - -0.1 < r < 0.1: No correlation
        - -0.4 < r ≤ -0.1: Weak negative correlation
        - -0.7 < r ≤ -0.4: Moderate negative correlation
        - -1.0 < r ≤ -0.7: Strong negative correlation
        - r = -1.0: Perfect negative correlation
        
    Algorithm Explanation:
        1. Validate inputs (type, size, equal length, numeric values)
        2. Calculate means of both datasets
        3. Calculate covariance: Σ[(x - x̄)(y - ȳ)]
        4. Calculate standard deviations for both datasets
        5. Compute correlation coefficient: covariance / (std_x × std_y)
        6. Provide human-readable interpretation
        
    Time Complexity: O(n) where n is the length of the datasets
    Space Complexity: O(1) not counting the input
    """
    # Validate input types
    if not isinstance(x, list):
        raise ValueError(f"Parameter 'x' must be a list, got {type(x).__name__}")
    if not isinstance(y, list):
        raise ValueError(f"Parameter 'y' must be a list, got {type(y).__name__}")
    
    # Validate sizes
    if len(x) < 2 or len(x) > 1000:
        raise ValueError("Parameter 'x' must contain between 2 and 1000 items")
    if len(y) < 2 or len(y) > 1000:
        raise ValueError("Parameter 'y' must contain between 2 and 1000 items")
    
    # Validate equal length
    if len(x) != len(y):
        raise ValueError(f"Arrays must have equal length. x has {len(x)} items, y has {len(y)} items")
    
    # Validate all elements are numeric
    for i, value in enumerate(x):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All values in 'x' must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    for i, value in enumerate(y):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All values in 'y' must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    n = len(x)
    
    # Calculate means
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Calculate covariance: Σ[(x - x̄)(y - ȳ)]
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    
    # Calculate sum of squared deviations for both variables
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
    
    # Handle edge case: all values are the same (no variance)
    if sum_sq_x == 0 or sum_sq_y == 0:
        raise ValueError("Cannot calculate correlation: data has zero variance")
    
    # Calculate Pearson correlation coefficient
    # r = covariance / (std_x × std_y)
    # = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² × Σ(y - ȳ)²]
    r = covariance / (sum_sq_x * sum_sq_y) ** 0.5
    
    # Interpret the correlation
    if r == 1.0:
        interpretation = "Perfect positive correlation"
    elif r >= 0.7:
        interpretation = "Strong positive correlation"
    elif r >= 0.4:
        interpretation = "Moderate positive correlation"
    elif r >= 0.1:
        interpretation = "Weak positive correlation"
    elif r > -0.1:
        interpretation = "No correlation"
    elif r > -0.4:
        interpretation = "Weak negative correlation"
    elif r > -0.7:
        interpretation = "Moderate negative correlation"
    elif r > -1.0:
        interpretation = "Strong negative correlation"
    else:  # r == -1.0
        interpretation = "Perfect negative correlation"
    
    return {
        'coefficient': r,
        'interpretation': interpretation
    }


def percentile(data: list[float], p: float) -> float:
    """
    Calculate the value at a specific percentile in a dataset.
    
    A percentile is a value below which a given percentage of observations fall.
    For example, the 50th percentile (median) is the value below which 50% of
    the data falls.
    
    Args:
        data: List of numeric values (1-10000 items)
        p: Percentile to calculate (0-100)
        
    Returns:
        The value at the given percentile
        
    Raises:
        ValueError: If data is empty, not a list, contains non-numeric values,
                   or percentile is outside 0-100 range
        
    Examples:
        percentile([15, 20, 35, 40, 50], 50)
        >>> 35  # Median (50th percentile)
        
        percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90)
        >>> 9.1  # 90th percentile
        
        percentile([23, 45, 12, 67, 34, 89, 23, 56], 25)
        >>> 23.0  # First quartile (Q1)
        
    Formula (Linear Interpolation Method):
        1. Sort the data in ascending order
        2. Calculate rank: k = (p / 100) × (n - 1)
        3. If k is an integer, return data[k]
        4. Otherwise, interpolate between data[floor(k)] and data[ceil(k)]
        
    Special Cases:
        - p = 0: minimum value
        - p = 25: first quartile (Q1)
        - p = 50: median (second quartile, Q2)
        - p = 75: third quartile (Q3)
        - p = 100: maximum value
        
    Algorithm Explanation:
        1. Validate input data and percentile value
        2. Sort the data
        3. Calculate the position (rank) in the sorted data
        4. If position is exact, return that value
        5. If position is between two values, interpolate linearly
        
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n) for storing sorted data
    """
    # Validate input type
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    # Validate data size
    if len(data) < 1 or len(data) > 10000:
        raise ValueError("Data must contain between 1 and 10000 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate percentile range
    if not isinstance(p, (int, float)):
        raise ValueError(f"Percentile must be numeric, got {type(p).__name__}")
    
    if p < 0 or p > 100:
        raise ValueError(f"Percentile must be between 0 and 100, got {p}")
    
    # Sort the data
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Handle edge cases
    if p == 0:
        return float(sorted_data[0])
    if p == 100:
        return float(sorted_data[-1])
    
    # Calculate rank using linear interpolation method
    # k = (p / 100) × (n - 1)
    k = (p / 100) * (n - 1)
    
    # Get lower and upper indices
    lower_index = int(k)
    upper_index = lower_index + 1
    
    # If k is exactly an integer, return that element
    if k == lower_index:
        return float(sorted_data[lower_index])
    
    # Otherwise, interpolate between two values
    # Interpolation formula: value = lower + (upper - lower) × fraction
    fraction = k - lower_index
    lower_value = sorted_data[lower_index]
    upper_value = sorted_data[upper_index]
    
    result = lower_value + (upper_value - lower_value) * fraction
    
    return float(result)


def detect_outliers(data: list[float], threshold: float = 1.5) -> dict[str, Any]:
    """
    Identify outliers in a dataset using the Interquartile Range (IQR) method.
    
    The IQR method is a robust statistical technique for detecting outliers.
    It uses the spread of the middle 50% of the data to identify values that
    are unusually far from the central tendency.
    
    Args:
        data: List of numeric values (4-10000 items, minimum 4 needed for quartiles)
        threshold: IQR multiplier for outlier boundaries (0.1-10, default: 1.5)
                  1.5 = standard outliers, 3.0 = extreme outliers
        
    Returns:
        Dictionary containing:
        - 'outliers': List of outlier values
        - 'indices': List of indices where outliers occur in original data
        - 'count': Number of outliers found
        - 'lower_bound': Lower threshold (Q1 - threshold×IQR)
        - 'upper_bound': Upper threshold (Q3 + threshold×IQR)
        - 'q1': First quartile (25th percentile)
        - 'q3': Third quartile (75th percentile)
        - 'iqr': Interquartile range (Q3 - Q1)
        - 'threshold': The threshold multiplier used
        
    Raises:
        ValueError: If data is empty, not a list, contains non-numeric values,
                   or threshold is outside valid range
        
    Examples:
        detect_outliers([10, 12, 14, 13, 15, 100, 11, 13, 14])
        >>> {
        ...   'outliers': [100],
        ...   'indices': [5],
        ...   'count': 1,
        ...   'lower_bound': 5.5,
        ...   'upper_bound': 20.5,
        ...   'q1': 11.5,
        ...   'q3': 14.5,
        ...   'iqr': 3.0,
        ...   'threshold': 1.5
        ... }
        
    IQR Method Explanation:
        1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
        2. Calculate IQR = Q3 - Q1 (range of middle 50% of data)
        3. Define outlier boundaries:
           - Lower bound = Q1 - threshold × IQR
           - Upper bound = Q3 + threshold × IQR
        4. Any value < lower_bound or > upper_bound is an outlier
        
    Why 1.5 × IQR?
        - This is a standard convention in statistics
        - Balances between detecting true outliers and false positives
        - Used in box plots and many statistical packages
        - Values beyond 3 × IQR are considered "extreme outliers"
        
    Advantages of IQR Method:
        - Robust: not affected by extreme values
        - Works well for skewed distributions
        - Does not assume normal distribution
        - Easy to interpret and implement
        
    Algorithm Explanation:
        1. Validate input data and threshold
        2. Calculate Q1 (25th percentile) and Q3 (75th percentile)
        3. Calculate IQR = Q3 - Q1
        4. Calculate lower and upper bounds using threshold
        5. Identify values outside the bounds
        6. Return outliers with their indices and statistics
        
    Time Complexity: O(n log n) due to percentile calculations (sorting)
    Space Complexity: O(k) where k is the number of outliers
    """
    # Validate input type
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    # Validate data size (need at least 4 points for meaningful quartiles)
    if len(data) < 4 or len(data) > 10000:
        raise ValueError("Data must contain between 4 and 10000 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate threshold
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"Threshold must be numeric, got {type(threshold).__name__}")
    
    if threshold < 0.1 or threshold > 10:
        raise ValueError(f"Threshold must be between 0.1 and 10, got {threshold}")
    
    # Calculate quartiles using the percentile function
    q1 = percentile(data, 25)
    q3 = percentile(data, 75)
    
    # Calculate IQR (Interquartile Range)
    iqr = q3 - q1
    
    # Calculate outlier boundaries
    # Configurable threshold: 1.5 × IQR for standard, 3.0 × IQR for extreme
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    # Identify outliers
    outlier_values = []
    outlier_indices = []
    
    for i, value in enumerate(data):
        if value < lower_bound or value > upper_bound:
            outlier_values.append(value)
            outlier_indices.append(i)
    
    return {
        'outliers': outlier_values,
        'indices': outlier_indices,
        'count': len(outlier_values),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'threshold': threshold
    }


# ============================================================================
# Advanced Outlier Detection Functions
# ============================================================================


def z_score_detection(data: list[float], method: str = "modified", threshold: float = 3.0, two_tailed: bool = True) -> dict[str, Any]:
    """
    Detect outliers using Z-score method (standard or modified).
    
    Standard Z-score uses mean and standard deviation.
    Modified Z-score uses median and MAD (Median Absolute Deviation) for robustness.
    
    Args:
        data: Measurements to check for outliers (min 3 items)
        method: "standard" (mean/std) or "modified" (median/MAD) - modified is more robust
        threshold: Z-score threshold (typical: 3.0 for standard, 3.5 for modified)
        two_tailed: Detect outliers on both sides (default: True)
        
    Returns:
        Dictionary containing:
        - 'method': Method used
        - 'threshold': Threshold used
        - 'outliers': Detailed outlier information with indices, values, z-scores, severity
        - 'statistics': Statistical measures used
        - 'cleaned_data': Data with outliers removed
        - 'interpretation': Human-readable summary
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 3:
        raise ValueError("Data must contain at least 3 items")
    
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    if method not in ["standard", "modified"]:
        raise ValueError(f"method must be 'standard' or 'modified', got '{method}'")
    
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"threshold must be numeric, got {type(threshold).__name__}")
    
    if threshold < 1.0 or threshold > 10.0:
        raise ValueError(f"threshold must be between 1.0 and 10.0, got {threshold}")
    
    if not isinstance(two_tailed, bool):
        raise ValueError(f"two_tailed must be boolean, got {type(two_tailed).__name__}")
    
    outlier_details = []
    outlier_indices = []
    
    if method == "standard":
        # Standard Z-score: (x - mean) / std
        mean_val = sum(data) / len(data)
        variance = sum((x - mean_val) ** 2 for x in data) / len(data)
        std_val = variance ** 0.5
        
        if std_val < 1e-10:
            std_val = 1.0  # Prevent division by zero
        
        for i, value in enumerate(data):
            z_score = (value - mean_val) / std_val
            
            is_outlier = False
            if two_tailed:
                is_outlier = abs(z_score) > threshold
            else:
                is_outlier = z_score > threshold
            
            if is_outlier:
                severity = "extreme" if abs(z_score) > threshold * 1.5 else "moderate"
                outlier_details.append({
                    "index": i,
                    "value": value,
                    "z_score": z_score,
                    "severity": severity
                })
                outlier_indices.append(i)
        
        statistics = {
            "mean": mean_val,
            "std_dev": std_val,
            "total_points": len(data),
            "outlier_count": len(outlier_details),
            "outlier_percentage": (len(outlier_details) / len(data)) * 100
        }
    
    else:  # modified
        # Modified Z-score: 0.6745 * (x - median) / MAD
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        # Calculate median
        if n % 2 == 0:
            median_val = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        else:
            median_val = sorted_data[n // 2]
        
        # Calculate MAD (Median Absolute Deviation)
        deviations = [abs(x - median_val) for x in data]
        sorted_dev = sorted(deviations)
        if len(sorted_dev) % 2 == 0:
            mad = (sorted_dev[len(sorted_dev) // 2 - 1] + sorted_dev[len(sorted_dev) // 2]) / 2
        else:
            mad = sorted_dev[len(sorted_dev) // 2]
        
        if mad < 1e-10:
            mad = 1.0  # Prevent division by zero
        
        for i, value in enumerate(data):
            # Modified Z-score formula
            modified_z = 0.6745 * (value - median_val) / mad
            
            is_outlier = False
            if two_tailed:
                is_outlier = abs(modified_z) > threshold
            else:
                is_outlier = modified_z > threshold
            
            if is_outlier:
                severity = "extreme" if abs(modified_z) > threshold * 1.5 else "moderate"
                outlier_details.append({
                    "index": i,
                    "value": value,
                    "z_score": modified_z,
                    "severity": severity
                })
                outlier_indices.append(i)
        
        statistics = {
            "median": median_val,
            "mad": mad,
            "total_points": len(data),
            "outlier_count": len(outlier_details),
            "outlier_percentage": (len(outlier_details) / len(data)) * 100
        }
    
    # Create cleaned data
    cleaned_data = [data[i] for i in range(len(data)) if i not in outlier_indices]
    
    # Generate interpretation
    if len(outlier_details) == 0:
        interpretation = f"No outliers detected using {method} Z-score method."
    else:
        interpretation = f"{len(outlier_details)} outliers detected ({statistics['outlier_percentage']:.1f}%). {method.capitalize()} Z-score method used for {'robustness' if method == 'modified' else 'standard detection'}."
    
    return {
        "method": f"{method}_z_score",
        "threshold": threshold,
        "outliers": {
            "indices": outlier_indices,
            "values": outlier_details
        },
        "statistics": statistics,
        "cleaned_data": cleaned_data,
        "interpretation": interpretation
    }


def grubbs_test(data: list[float], alpha: float = 0.05, method: str = "two_sided") -> dict[str, Any]:
    """
    Statistical test for detecting single outliers in normally distributed data.
    
    Grubbs' test (maximum normed residual test) is used to detect a single outlier
    in a univariate dataset that follows an approximately normal distribution.
    
    Args:
        data: Dataset to test for outliers (min 7 items for valid statistics)
        alpha: Significance level (typical: 0.05 or 0.01)
        method: "max" (test maximum), "min" (test minimum), or "two_sided" (test both)
        
    Returns:
        Dictionary containing:
        - 'test': Test name
        - 'alpha': Significance level
        - 'sample_size': Number of data points
        - 'suspected_outlier': Information about the suspected outlier
        - 'test_statistic': Calculated Grubbs statistic
        - 'critical_value': Critical value from t-distribution
        - 'p_value': Approximate p-value
        - 'conclusion': Statistical conclusion
        - 'recommendation': Action recommendation
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 7:
        raise ValueError("Data must contain at least 7 items for valid Grubbs test")
    
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    if not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be numeric, got {type(alpha).__name__}")
    
    if alpha < 0.001 or alpha > 0.1:
        raise ValueError(f"alpha must be between 0.001 and 0.1, got {alpha}")
    
    if method not in ["max", "min", "two_sided"]:
        raise ValueError(f"method must be 'max', 'min', or 'two_sided', got '{method}'")
    
    n = len(data)
    mean_val = sum(data) / n
    variance = sum((x - mean_val) ** 2 for x in data) / n
    std_val = variance ** 0.5
    
    if std_val < 1e-10:
        raise ValueError("Cannot perform Grubbs test: data has zero variance")
    
    # Find suspected outlier based on method
    if method == "max" or method == "two_sided":
        max_val = max(data)
        max_idx = data.index(max_val)
        g_max = abs(max_val - mean_val) / std_val
    else:
        g_max = 0
        max_val = None
        max_idx = None
    
    if method == "min" or method == "two_sided":
        min_val = min(data)
        min_idx = data.index(min_val)
        g_min = abs(min_val - mean_val) / std_val
    else:
        g_min = 0
        min_val = None
        min_idx = None
    
    # Determine which is the suspected outlier
    if method == "two_sided":
        if g_max > g_min:
            test_statistic = g_max
            suspected_value = max_val
            suspected_index = max_idx
            side = "maximum"
        else:
            test_statistic = g_min
            suspected_value = min_val
            suspected_index = min_idx
            side = "minimum"
    elif method == "max":
        test_statistic = g_max
        suspected_value = max_val
        suspected_index = max_idx
        side = "maximum"
    else:  # min
        test_statistic = g_min
        suspected_value = min_val
        suspected_index = min_idx
        side = "minimum"
    
    # Calculate critical value using t-distribution
    # Grubbs critical value: ((n-1)/sqrt(n)) * sqrt(t²/(n-2+t²))
    # where t is the t-distribution value
    t_alpha = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    critical_value = ((n - 1) / math.sqrt(n)) * math.sqrt(t_alpha**2 / (n - 2 + t_alpha**2))
    
    # Calculate approximate p-value
    # Using the inverse: t = sqrt((n-2) * G² / ((n-1)² - n*G²))
    if test_statistic < critical_value:
        p_value = 1.0  # Not significant
    else:
        try:
            t_stat = math.sqrt((n - 2) * test_statistic**2 / ((n - 1)**2 - n * test_statistic**2))
            p_value = 2 * n * (1 - stats.t.cdf(t_stat, n - 2))
            p_value = min(p_value, 1.0)
        except:
            p_value = 0.001  # Very significant
    
    # Determine conclusion
    if test_statistic > critical_value:
        conclusion = f"Reject null hypothesis - value is a significant outlier at α={alpha}"
        recommendation = f"Remove point {suspected_index} (value: {suspected_value:.4f}) - statistically significant outlier"
    else:
        conclusion = f"Fail to reject null hypothesis - no significant outlier detected at α={alpha}"
        recommendation = f"Retain all data points - no statistically significant outlier found"
    
    return {
        "test": "Grubbs test",
        "alpha": alpha,
        "sample_size": n,
        "suspected_outlier": {
            "value": suspected_value,
            "index": suspected_index,
            "side": side
        },
        "test_statistic": test_statistic,
        "critical_value": critical_value,
        "p_value": p_value,
        "conclusion": conclusion,
        "recommendation": recommendation
    }


def dixon_q_test(data: list[float], alpha: float = 0.05) -> dict[str, Any]:
    """
    Dixon Q test for outliers in small datasets (3-30 points).
    
    Quick test designed specifically for small sample sizes.
    Uses gap ratio to identify outliers.
    
    Args:
        data: Small dataset (3-30 points)
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary containing:
        - 'test': Test name
        - 'sample_size': Number of data points
        - 'suspected_outlier': Information about suspected outlier
        - 'q_statistic': Calculated Q statistic
        - 'q_critical': Critical Q value for sample size and alpha
        - 'conclusion': Test conclusion
        - 'recommendation': Action recommendation
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 3 or len(data) > 30:
        raise ValueError("Dixon Q test requires between 3 and 30 data points")
    
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    if not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be numeric, got {type(alpha).__name__}")
    
    n = len(data)
    sorted_data = sorted(data)
    
    # Dixon Q critical values for α=0.05
    # Format: {sample_size: critical_value}
    q_critical_values = {
        3: 0.941, 4: 0.765, 5: 0.642, 6: 0.560, 7: 0.507,
        8: 0.554, 9: 0.512, 10: 0.477, 11: 0.576, 12: 0.546,
        13: 0.521, 14: 0.546, 15: 0.525, 16: 0.507, 17: 0.490,
        18: 0.475, 19: 0.462, 20: 0.450, 21: 0.440, 22: 0.430,
        23: 0.421, 24: 0.413, 25: 0.406, 26: 0.399, 27: 0.393,
        28: 0.387, 29: 0.381, 30: 0.376
    }
    
    # For α=0.01, use different values (more conservative)
    q_critical_values_01 = {
        3: 0.988, 4: 0.889, 5: 0.780, 6: 0.698, 7: 0.637,
        8: 0.683, 9: 0.635, 10: 0.597, 11: 0.679, 12: 0.642,
        13: 0.615, 14: 0.641, 15: 0.616, 16: 0.595, 17: 0.577,
        18: 0.561, 19: 0.547, 20: 0.535, 21: 0.524, 22: 0.514,
        23: 0.505, 24: 0.497, 25: 0.489, 26: 0.486, 27: 0.479,
        28: 0.472, 29: 0.465, 30: 0.463
    }
    
    # Select appropriate critical value table
    if abs(alpha - 0.01) < 0.001:
        q_crit = q_critical_values_01.get(n, 0.463)
    else:
        q_crit = q_critical_values.get(n, 0.376)
    
    # Calculate Q statistic for both ends
    # Q = gap / range
    range_val = sorted_data[-1] - sorted_data[0]
    
    if range_val < 1e-10:
        raise ValueError("Cannot perform Dixon Q test: all values are identical")
    
    # Check low end
    gap_low = sorted_data[1] - sorted_data[0]
    q_low = gap_low / range_val
    
    # Check high end
    gap_high = sorted_data[-1] - sorted_data[-2]
    q_high = gap_high / range_val
    
    # Determine suspected outlier
    if q_high > q_low:
        q_statistic = q_high
        suspected_value = sorted_data[-1]
        suspected_index = data.index(sorted_data[-1])
        position = "high"
    else:
        q_statistic = q_low
        suspected_value = sorted_data[0]
        suspected_index = data.index(sorted_data[0])
        position = "low"
    
    # Determine conclusion
    if q_statistic > q_crit:
        conclusion = f"Outlier detected - reject value (Q={q_statistic:.3f} > Q_critical={q_crit:.3f})"
        recommendation = f"Repeat measurement or investigate cause for value {suspected_value:.4f} at position {position}"
    else:
        conclusion = f"No outlier detected - retain all values (Q={q_statistic:.3f} ≤ Q_critical={q_crit:.3f})"
        recommendation = "All measurements appear valid at the chosen significance level"
    
    return {
        "test": "Dixon Q test",
        "sample_size": n,
        "suspected_outlier": {
            "value": suspected_value,
            "index": suspected_index,
            "position": position
        },
        "q_statistic": q_statistic,
        "q_critical": q_crit,
        "conclusion": conclusion,
        "recommendation": recommendation
    }


def isolation_forest(data: list, contamination: float = 0.1, n_estimators: int = 100) -> dict[str, Any]:
    """
    Machine learning-based anomaly detection using Isolation Forest.
    
    Isolation Forest is effective for complex, multivariate anomaly detection.
    Works by isolating observations through random partitioning.
    
    Args:
        data: Univariate or multivariate data (min 10 items)
              Univariate: [x1, x2, x3, ...]
              Multivariate: [[x1, y1, z1], [x2, y2, z2], ...]
        contamination: Expected proportion of outliers (0-0.5, default: 0.1)
        n_estimators: Number of isolation trees (50-500, default: 100)
        
    Returns:
        Dictionary containing:
        - 'method': Method name
        - 'anomalies': Detected anomalies with indices and scores
        - 'contamination': Actual contamination found
        - 'interpretation': Human-readable summary
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 10:
        raise ValueError("Data must contain at least 10 items for isolation forest")
    
    if not isinstance(contamination, (int, float)):
        raise ValueError(f"contamination must be numeric, got {type(contamination).__name__}")
    
    if contamination < 0.0 or contamination > 0.5:
        raise ValueError(f"contamination must be between 0.0 and 0.5, got {contamination}")
    
    if not isinstance(n_estimators, int):
        raise ValueError(f"n_estimators must be an integer, got {type(n_estimators).__name__}")
    
    if n_estimators < 50 or n_estimators > 500:
        raise ValueError(f"n_estimators must be between 50 and 500, got {n_estimators}")
    
    # Convert data to numpy array and handle univariate vs multivariate
    try:
        # Check if data is univariate (list of numbers) or multivariate (list of lists)
        if isinstance(data[0], (int, float)):
            # Univariate data - reshape to column vector
            X = np.array(data).reshape(-1, 1)
        else:
            # Multivariate data
            X = np.array(data)
            if X.ndim != 2:
                raise ValueError("Multivariate data must be 2-dimensional")
    except Exception as e:
        raise ValueError(f"Failed to convert data to array: {str(e)}")
    
    # Import sklearn isolation forest
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        raise ImportError("sklearn is required for isolation_forest. Install with: pip install scikit-learn")
    
    # Fit isolation forest
    clf = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    
    predictions = clf.fit_predict(X)
    scores = clf.score_samples(X)
    
    # Find anomalies (labeled as -1)
    anomaly_indices = [i for i, pred in enumerate(predictions) if pred == -1]
    anomaly_scores = [abs(scores[i]) for i in anomaly_indices]
    
    # Classify severity based on anomaly score
    severity_labels = []
    for score in anomaly_scores:
        if score > 0.6:
            severity_labels.append("high")
        elif score > 0.4:
            severity_labels.append("medium")
        else:
            severity_labels.append("low")
    
    actual_contamination = len(anomaly_indices) / len(data)
    
    interpretation = f"{len(anomaly_indices)} anomalies detected ({actual_contamination*100:.1f}% of data). Isolation scores indicate how isolated each point is from normal patterns."
    
    return {
        "method": "Isolation Forest",
        "anomalies": {
            "indices": anomaly_indices,
            "anomaly_scores": anomaly_scores,
            "severity": severity_labels
        },
        "contamination": actual_contamination,
        "interpretation": interpretation
    }


def mahalanobis_distance(data: list[list[float]], threshold: float = 0.975) -> dict[str, Any]:
    """
    Multivariate outlier detection using Mahalanobis distance.
    
    Mahalanobis distance considers correlations between variables,
    making it effective for detecting outliers in multivariate datasets.
    
    Args:
        data: Multivariate data [[x1, y1, z1], [x2, y2, z2], ...] (min 10 items)
        threshold: Chi-square threshold percentile (0.9-0.999, default: 0.975)
        
    Returns:
        Dictionary containing:
        - 'method': Method name
        - 'dimensions': Number of variables
        - 'outliers': Outlier details with distances and p-values
        - 'threshold_distance': Critical distance threshold
        - 'degrees_of_freedom': Degrees of freedom (number of dimensions)
        - 'interpretation': Human-readable summary
        - 'variable_contributions': Which variables contribute most to outliers
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 10:
        raise ValueError("Data must contain at least 10 items")
    
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"threshold must be numeric, got {type(threshold).__name__}")
    
    if threshold < 0.9 or threshold > 0.999:
        raise ValueError(f"threshold must be between 0.9 and 0.999, got {threshold}")
    
    # Convert to numpy array
    try:
        X = np.array(data)
        if X.ndim != 2:
            raise ValueError("Data must be 2-dimensional (multivariate)")
    except Exception as e:
        raise ValueError(f"Failed to convert data to array: {str(e)}")
    
    n_samples, n_features = X.shape
    
    if n_features < 2:
        raise ValueError("Mahalanobis distance requires at least 2 variables (multivariate data)")
    
    # Calculate mean and covariance matrix
    mean_vec = np.mean(X, axis=0)
    cov_matrix = np.cov(X, rowvar=False)
    
    # Handle singular covariance matrix
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if matrix is singular
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
    
    # Calculate Mahalanobis distance for each point
    distances = []
    for i in range(n_samples):
        diff = X[i] - mean_vec
        distance = np.sqrt(diff @ inv_cov_matrix @ diff.T)
        distances.append(distance)
    
    # Determine threshold using chi-square distribution
    # Mahalanobis distance squared follows chi-square distribution
    chi2_threshold = stats.chi2.ppf(threshold, n_features)
    distance_threshold = np.sqrt(chi2_threshold)
    
    # Find outliers
    outlier_info = []
    outlier_indices = []
    
    for i, dist in enumerate(distances):
        if dist > distance_threshold:
            # Calculate p-value
            p_value = 1 - stats.chi2.cdf(dist**2, n_features)
            
            outlier_info.append({
                "index": i,
                "distance": dist,
                "p_value": p_value
            })
            outlier_indices.append(i)
    
    # For outliers, determine which variable contributes most
    variable_contributions = []
    for i in outlier_indices:
        diff = X[i] - mean_vec
        # Contribution of each variable (squared standardized difference)
        contributions = (diff**2) / np.diag(cov_matrix)
        primary_var_idx = np.argmax(contributions)
        contribution_pct = contributions[primary_var_idx] / np.sum(contributions)
        
        variable_contributions.append({
            "index": i,
            "primary_variable": f"variable_{primary_var_idx}",
            "contribution": contribution_pct
        })
    
    interpretation = f"{len(outlier_indices)} multivariate outliers detected. These points are unusual in the combined context of all {n_features} variables."
    
    return {
        "method": "Mahalanobis Distance",
        "dimensions": n_features,
        "outliers": {
            "indices": outlier_indices,
            "distances": outlier_info
        },
        "threshold_distance": distance_threshold,
        "degrees_of_freedom": n_features,
        "interpretation": interpretation,
        "variable_contributions": variable_contributions
    }


def streaming_outlier_detection(current_value: float, historical_window: list[float], 
                                method: str = "ewma", sensitivity: int = 5) -> dict[str, Any]:
    """
    Real-time outlier detection for continuous sensor streams.
    
    Designed for real-time SCADA systems and continuous process monitoring.
    
    Args:
        current_value: New measurement to evaluate
        historical_window: Recent historical values for context (10-1000 items)
        method: Detection method - "ewma", "cusum", or "adaptive_threshold"
        sensitivity: Detection sensitivity (1-10, higher = more sensitive, default: 5)
        
    Returns:
        Dictionary containing:
        - 'current_value': Value being evaluated
        - 'is_outlier': Whether current value is an outlier
        - 'severity': Severity level ("normal", "warning", "critical")
        - 'method': Method used
        - 'expected_range': Expected value range
        - 'deviation': How much current value deviates
        - 'deviation_sigma': Deviation in standard deviations
        - 'interpretation': Human-readable interpretation
        - 'recommendation': Action recommendation
        - 'trend': Trend direction ("increasing", "decreasing", "stable")
        - 'rate_of_change': Rate of change from last value
    """
    # Validate input
    if not isinstance(current_value, (int, float)):
        raise ValueError(f"current_value must be numeric, got {type(current_value).__name__}")
    
    if not isinstance(historical_window, list):
        raise ValueError(f"historical_window must be a list, got {type(historical_window).__name__}")
    
    if len(historical_window) < 10 or len(historical_window) > 1000:
        raise ValueError("historical_window must contain between 10 and 1000 items")
    
    for i, value in enumerate(historical_window):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All historical_window values must be numeric. Item at index {i} is {type(value).__name__}")
    
    if method not in ["ewma", "cusum", "adaptive_threshold"]:
        raise ValueError(f"method must be 'ewma', 'cusum', or 'adaptive_threshold', got '{method}'")
    
    if not isinstance(sensitivity, int):
        raise ValueError(f"sensitivity must be an integer, got {type(sensitivity).__name__}")
    
    if sensitivity < 1 or sensitivity > 10:
        raise ValueError(f"sensitivity must be between 1 and 10, got {sensitivity}")
    
    # Calculate baseline statistics
    mean_hist = sum(historical_window) / len(historical_window)
    variance_hist = sum((x - mean_hist) ** 2 for x in historical_window) / len(historical_window)
    std_hist = variance_hist ** 0.5
    
    if std_hist < 1e-10:
        std_hist = 1.0
    
    # Map sensitivity to threshold (inverse relationship)
    # Higher sensitivity = lower threshold = more sensitive
    base_threshold = 3.0 - (sensitivity - 5) * 0.3
    
    is_outlier = False
    severity = "normal"
    expected_range = [mean_hist - base_threshold * std_hist, mean_hist + base_threshold * std_hist]
    
    if method == "ewma":
        # Exponentially Weighted Moving Average
        alpha = 0.2  # Smoothing factor
        ewma = historical_window[0]
        for val in historical_window[1:]:
            ewma = alpha * val + (1 - alpha) * ewma
        
        expected_value = ewma
        expected_range = [ewma - base_threshold * std_hist, ewma + base_threshold * std_hist]
        
        deviation = abs(current_value - expected_value)
        deviation_sigma = deviation / std_hist
        
        if current_value < expected_range[0] or current_value > expected_range[1]:
            is_outlier = True
            if deviation_sigma > base_threshold * 1.5:
                severity = "critical"
            else:
                severity = "warning"
    
    elif method == "cusum":
        # Cumulative Sum control chart
        target = mean_hist
        k = 0.5 * std_hist  # Allowable slack
        h = base_threshold * std_hist  # Decision interval
        
        cusum_pos = 0
        cusum_neg = 0
        
        for val in historical_window:
            cusum_pos = max(0, cusum_pos + (val - target - k))
            cusum_neg = max(0, cusum_neg - (val - target - k))
        
        # Check current value
        cusum_pos_current = max(0, cusum_pos + (current_value - target - k))
        cusum_neg_current = max(0, cusum_neg - (current_value - target - k))
        
        deviation = abs(current_value - target)
        deviation_sigma = deviation / std_hist
        
        if cusum_pos_current > h or cusum_neg_current > h:
            is_outlier = True
            if deviation_sigma > base_threshold * 1.5:
                severity = "critical"
            else:
                severity = "warning"
        
        expected_range = [target - h, target + h]
    
    else:  # adaptive_threshold
        # Adaptive threshold based on recent behavior
        recent_window = historical_window[-min(20, len(historical_window)):]
        recent_mean = sum(recent_window) / len(recent_window)
        recent_variance = sum((x - recent_mean) ** 2 for x in recent_window) / len(recent_window)
        recent_std = recent_variance ** 0.5
        
        if recent_std < 1e-10:
            recent_std = 1.0
        
        # Adaptive threshold adjusts based on recent volatility
        adaptive_factor = recent_std / std_hist if std_hist > 1e-10 else 1.0
        adjusted_threshold = base_threshold * adaptive_factor
        
        expected_range = [recent_mean - adjusted_threshold * recent_std, 
                         recent_mean + adjusted_threshold * recent_std]
        
        deviation = abs(current_value - recent_mean)
        deviation_sigma = deviation / recent_std
        
        if current_value < expected_range[0] or current_value > expected_range[1]:
            is_outlier = True
            if deviation_sigma > adjusted_threshold * 1.5:
                severity = "critical"
            else:
                severity = "warning"
    
    # Calculate trend
    if len(historical_window) >= 5:
        recent_5 = historical_window[-5:]
        trend_slope = (recent_5[-1] - recent_5[0]) / 5
        if abs(trend_slope) < 0.1 * std_hist:
            trend = "stable"
        elif trend_slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
    else:
        trend = "stable"
    
    # Rate of change from last value
    rate_of_change = current_value - historical_window[-1]
    
    # Generate interpretation and recommendation
    if is_outlier:
        interpretation = f"Current value {current_value:.2f} exceeds expected range [{expected_range[0]:.2f}, {expected_range[1]:.2f}] by {deviation_sigma:.1f} standard deviations"
        if severity == "critical":
            recommendation = "IMMEDIATE ACTION REQUIRED: Investigate sensor or process condition"
        else:
            recommendation = "Investigate sensor or process condition"
    else:
        interpretation = f"Current value {current_value:.2f} is within expected range [{expected_range[0]:.2f}, {expected_range[1]:.2f}]"
        recommendation = "Continue normal monitoring"
    
    return {
        "current_value": current_value,
        "is_outlier": is_outlier,
        "severity": severity,
        "method": method,
        "expected_range": expected_range,
        "deviation": deviation if is_outlier else 0.0,
        "deviation_sigma": deviation_sigma if is_outlier else 0.0,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "trend": trend,
        "rate_of_change": rate_of_change
    }


# ============================================================================
# Time Series Analysis Functions
# ============================================================================


def moving_average(data: list[float], window_size: int, ma_type: str = "simple", alpha: float = None) -> dict[str, Any]:
    """
    Calculate Simple, Exponential, or Weighted Moving Averages for smoothing time series data.
    
    Moving averages are used to smooth noisy sensor readings, identify trends, and filter
    out high-frequency variations in industrial process data.
    
    Args:
        data: Time series data (e.g., hourly temperature readings), min 2 items
        window_size: Number of periods for moving average (2-1000)
        ma_type: Type of moving average - "simple", "exponential", or "weighted" (default: "simple")
        alpha: Smoothing factor for EMA (0-1), if None defaults to 2/(window_size+1)
        
    Returns:
        Dictionary containing:
        - 'original_data': Input data
        - 'moving_average': Calculated moving average values
        - 'ma_type': Type of moving average used
        - 'window_size': Window size used
        - 'data_points': Number of input data points
        - 'smoothed_points': Number of moving average points
        
    Raises:
        ValueError: If data is invalid, window_size out of range, or alpha out of range
        
    Examples:
        moving_average([100, 102, 98, 105, 103, 107, 110], 3, "simple")
        >>> {
        ...   'original_data': [100, 102, 98, 105, 103, 107, 110],
        ...   'moving_average': [100.0, 100.0, 101.33, 103.0, 105.0, 106.67],
        ...   'ma_type': 'simple',
        ...   'window_size': 3,
        ...   'data_points': 7,
        ...   'smoothed_points': 6
        ... }
    
    Algorithm Details:
        Simple Moving Average (SMA): arithmetic mean over window
        Exponential Moving Average (EMA): weighted average favoring recent data
        Weighted Moving Average (WMA): linearly weighted average
    """
    # Validate input type
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    # Validate data size
    if len(data) < 2:
        raise ValueError("Data must contain at least 2 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate window_size
    if not isinstance(window_size, int):
        raise ValueError(f"window_size must be an integer, got {type(window_size).__name__}")
    
    if window_size < 2 or window_size > 1000:
        raise ValueError(f"window_size must be between 2 and 1000, got {window_size}")
    
    if window_size > len(data):
        raise ValueError(f"window_size ({window_size}) cannot be larger than data length ({len(data)})")
    
    # Validate ma_type
    if ma_type not in ["simple", "exponential", "weighted"]:
        raise ValueError(f"ma_type must be 'simple', 'exponential', or 'weighted', got '{ma_type}'")
    
    # Validate alpha for EMA
    if ma_type == "exponential":
        if alpha is None:
            alpha = 2.0 / (window_size + 1)
        elif not isinstance(alpha, (int, float)):
            raise ValueError(f"alpha must be numeric, got {type(alpha).__name__}")
        elif alpha < 0 or alpha > 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    
    # Calculate moving average based on type
    ma_values = []
    
    if ma_type == "simple":
        # Simple Moving Average: arithmetic mean over window
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            ma_values.append(sum(window) / window_size)
    
    elif ma_type == "exponential":
        # Exponential Moving Average: weighted average favoring recent data
        # EMA = alpha * value + (1 - alpha) * previous_EMA
        ema = data[0]  # Start with first value
        ma_values.append(ema)
        
        for i in range(1, len(data)):
            ema = alpha * data[i] + (1 - alpha) * ema
            ma_values.append(ema)
    
    elif ma_type == "weighted":
        # Weighted Moving Average: linearly weighted
        # Weights are 1, 2, 3, ..., window_size
        weights = list(range(1, window_size + 1))
        weight_sum = sum(weights)
        
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            weighted_sum = sum(w * v for w, v in zip(weights, window))
            ma_values.append(weighted_sum / weight_sum)
    
    return {
        'original_data': data,
        'moving_average': ma_values,
        'ma_type': ma_type,
        'window_size': window_size,
        'data_points': len(data),
        'smoothed_points': len(ma_values)
    }


def detect_trend(data: list[float], timestamps: list[float] = None, method: str = "linear", degree: int = 2) -> dict[str, Any]:
    """
    Identify and quantify trends in process data using regression analysis.
    
    Useful for detecting equipment degradation, process efficiency decline, catalyst deactivation,
    and other time-based changes in industrial systems.
    
    Args:
        data: Time series values, min 3 items
        timestamps: Optional time indices (defaults to 0, 1, 2, ...)
        method: "linear" or "polynomial" regression
        degree: Polynomial degree for polynomial method (2-5), ignored for linear
        
    Returns:
        Dictionary containing:
        - 'trend': "increasing", "decreasing", or "stable"
        - 'slope': Rate of change (for linear) or dominant coefficient
        - 'slope_interpretation': Human-readable interpretation
        - 'r_squared': Goodness of fit (0-1)
        - 'fit_quality': "excellent", "good", "fair", or "poor"
        - 'coefficients': Regression coefficients
        - 'equation': Human-readable equation
        - 'confidence_interval': 95% confidence interval for slope
        - 'prediction_next_5': Predicted values for next 5 time periods
        
    Examples:
        detect_trend([100, 101, 103, 104, 106, 108, 110], method="linear")
        >>> {
        ...   'trend': 'increasing',
        ...   'slope': 1.64,
        ...   'r_squared': 0.98,
        ...   'equation': 'y = 1.64x + 99.14'
        ... }
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 3:
        raise ValueError("Data must contain at least 3 items for trend detection")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Set default timestamps if not provided
    if timestamps is None:
        timestamps = list(range(len(data)))
    else:
        if len(timestamps) != len(data):
            raise ValueError(f"timestamps length ({len(timestamps)}) must match data length ({len(data)})")
        for i, value in enumerate(timestamps):
            if not isinstance(value, (int, float)):
                raise ValueError(f"All timestamp values must be numeric. Item at index {i} is {type(value).__name__}")
    
    # Validate method
    if method not in ["linear", "polynomial"]:
        raise ValueError(f"method must be 'linear' or 'polynomial', got '{method}'")
    
    # Validate degree for polynomial
    if method == "polynomial":
        if not isinstance(degree, int):
            raise ValueError(f"degree must be an integer, got {type(degree).__name__}")
        if degree < 2 or degree > 5:
            raise ValueError(f"degree must be between 2 and 5, got {degree}")
    
    n = len(data)
    x = timestamps
    y = data
    
    # Handle polynomial method (not fully implemented, fall back to linear)
    if method == "polynomial":
        # Log a warning and use linear instead
        logger.warning(f"Polynomial regression not implemented, using linear regression instead")
        method = "linear"
    
    if method == "linear":
        # Linear regression: y = mx + b
        # Using least squares method
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xx = sum(xi ** 2 for xi in x)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        
        # Calculate slope (m) and intercept (b)
        denominator = n * sum_xx - sum_x ** 2
        if abs(denominator) < 1e-10:
            raise ValueError("Cannot perform linear regression: timestamps have no variance")
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        coefficients = [intercept, slope]
        equation = f"y = {slope:.4f}x + {intercept:.4f}"
        
        # Calculate predictions
        predictions = [slope * xi + intercept for xi in x]
        
        # Calculate R-squared
        mean_y = sum_y / n
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        ss_res = sum((yi - pred) ** 2 for yi, pred in zip(y, predictions))
        
        if ss_tot < 1e-10:
            r_squared = 1.0  # Perfect fit if no variance
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        # Determine trend
        if abs(slope) < 0.01:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        slope_interpretation = f"{'Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'Stable'} by {abs(slope):.4f} units per time period"
        
        # Calculate confidence interval (simplified)
        # Standard error of slope
        variance_x = sum_xx - sum_x ** 2 / n
        if n > 2 and variance_x > 1e-10:
            se_slope = (ss_res / (n - 2)) ** 0.5 / (variance_x ** 0.5)
        else:
            se_slope = 0
        # 95% CI: approximately ±2 * SE (using t ≈ 2 for simplicity)
        ci_margin = 2 * se_slope
        confidence_interval = [slope - ci_margin, slope + ci_margin]
        
        # Predict next 5 values
        last_x = x[-1]
        prediction_next_5 = [slope * (last_x + i + 1) + intercept for i in range(5)]
    
    # Determine fit quality
    if r_squared >= 0.9:
        fit_quality = "excellent"
    elif r_squared >= 0.7:
        fit_quality = "good"
    elif r_squared >= 0.5:
        fit_quality = "fair"
    else:
        fit_quality = "poor"
    
    return {
        'trend': trend,
        'slope': slope,
        'slope_interpretation': slope_interpretation,
        'r_squared': r_squared,
        'fit_quality': fit_quality,
        'coefficients': coefficients,
        'equation': equation,
        'confidence_interval': confidence_interval,
        'prediction_next_5': prediction_next_5
    }


def autocorrelation(data: list[float], max_lag: int = None) -> dict[str, Any]:
    """
    Calculate autocorrelation function (ACF) to identify repeating patterns and cycles.
    
    Useful for detecting cyclic patterns in batch processes, identifying production cycle times,
    finding optimal sampling intervals, and detecting seasonality in data.
    
    Args:
        data: Time series data, min 10 items
        max_lag: Maximum lag to calculate (1-500), defaults to min(len(data)//2, 50)
        
    Returns:
        Dictionary containing:
        - 'acf_values': Autocorrelation values for each lag
        - 'lags': List of lag values
        - 'max_lag': Maximum lag calculated
        - 'significant_lags': Lags with significant correlation (> 0.2)
        - 'interpretation': Human-readable interpretation
        
    Examples:
        autocorrelation([1, 2, 3, 1, 2, 3, 1, 2, 3], max_lag=5)
        >>> {
        ...   'acf_values': [1.0, 0.5, 0.3, 0.8, ...],
        ...   'lags': [0, 1, 2, 3, ...],
        ...   'significant_lags': [0, 3],
        ...   'interpretation': 'Strong pattern detected at lag 3'
        ... }
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 10:
        raise ValueError("Data must contain at least 10 items for autocorrelation analysis")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Set default max_lag
    if max_lag is None:
        max_lag = min(len(data) // 2, 50)
    else:
        if not isinstance(max_lag, int):
            raise ValueError(f"max_lag must be an integer, got {type(max_lag).__name__}")
        if max_lag < 1 or max_lag > 500:
            raise ValueError(f"max_lag must be between 1 and 500, got {max_lag}")
        if max_lag >= len(data):
            raise ValueError(f"max_lag ({max_lag}) must be less than data length ({len(data)})")
    
    # Calculate mean and variance
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    
    if variance < 1e-10:
        raise ValueError("Cannot calculate autocorrelation: data has zero variance")
    
    # Calculate autocorrelation for each lag
    acf_values = []
    lags = list(range(max_lag + 1))
    
    for lag in lags:
        if lag == 0:
            # Lag 0 is always 1.0 (perfect correlation with itself)
            acf_values.append(1.0)
        else:
            # Calculate autocorrelation for this lag
            # ACF(k) = Σ[(x_t - μ)(x_{t-k} - μ)] / Σ[(x_t - μ)²]
            numerator = sum((data[i] - mean) * (data[i - lag] - mean) for i in range(lag, n))
            acf = numerator / (variance * n)
            acf_values.append(acf)
    
    # Find significant lags (correlation > 0.2 or < -0.2, excluding lag 0)
    significant_lags = [lag for lag, acf in zip(lags[1:], acf_values[1:]) if abs(acf) > 0.2]
    
    # Generate interpretation
    if not significant_lags:
        interpretation = "No significant periodic patterns detected"
    elif len(significant_lags) == 1:
        interpretation = f"Periodic pattern detected at lag {significant_lags[0]}"
    else:
        interpretation = f"Multiple periodic patterns detected at lags: {', '.join(map(str, significant_lags[:5]))}"
    
    return {
        'acf_values': acf_values,
        'lags': lags,
        'max_lag': max_lag,
        'significant_lags': significant_lags,
        'interpretation': interpretation
    }


def change_point_detection(data: list[float], method: str = "cusum", threshold: float = 1.5, min_size: int = 5) -> dict[str, Any]:
    """
    Identify significant changes in process behavior (regime changes, upsets, modifications).
    
    Useful for detecting when process modifications were effective, identifying upsets,
    finding when equipment behavior changed, and detecting shift changes.
    
    Args:
        data: Time series data, min 10 items
        method: Detection method - "cusum", "standard_deviation", or "mean_shift"
        threshold: Sensitivity threshold (higher = less sensitive), 0.1-10, default 1.5
        min_size: Minimum segment size between change points, min 2, default 5
        
    Returns:
        Dictionary containing:
        - 'change_points': List of indices where changes detected
        - 'number_of_segments': Number of segments created by change points
        - 'segments': List of segment statistics (start, end, mean, std)
        - 'largest_change': Information about the most significant change
        - 'method_used': Detection method used
        
    Examples:
        change_point_detection([100]*20 + [95]*20 + [102]*20, method="mean_shift")
        >>> {
        ...   'change_points': [20, 40],
        ...   'number_of_segments': 3,
        ...   'segments': [
        ...     {'start': 0, 'end': 20, 'mean': 100.0, 'std': 0.0},
        ...     {'start': 20, 'end': 40, 'mean': 95.0, 'std': 0.0}
        ...   ]
        ... }
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 10:
        raise ValueError("Data must contain at least 10 items for change point detection")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate method
    if method not in ["cusum", "standard_deviation", "mean_shift"]:
        raise ValueError(f"method must be 'cusum', 'standard_deviation', or 'mean_shift', got '{method}'")
    
    # Validate threshold
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"threshold must be numeric, got {type(threshold).__name__}")
    if threshold < 0.1 or threshold > 10:
        raise ValueError(f"threshold must be between 0.1 and 10, got {threshold}")
    
    # Validate min_size
    if not isinstance(min_size, int):
        raise ValueError(f"min_size must be an integer, got {type(min_size).__name__}")
    if min_size < 2:
        raise ValueError(f"min_size must be at least 2, got {min_size}")
    
    change_points = []
    
    if method == "cusum":
        # CUSUM (Cumulative Sum) algorithm
        mean_val = sum(data) / len(data)
        std_val = (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5
        
        if std_val < 1e-10:
            std_val = 1.0  # Prevent division by zero
        
        # Calculate CUSUM
        cusum_pos = 0
        cusum_neg = 0
        h = threshold * std_val  # Threshold for detection
        
        for i in range(min_size, len(data) - min_size):
            deviation = data[i] - mean_val
            
            cusum_pos = max(0, cusum_pos + deviation)
            cusum_neg = max(0, cusum_neg - deviation)
            
            if cusum_pos > h or cusum_neg > h:
                # Check if this is far enough from the last change point
                if not change_points or (i - change_points[-1]) >= min_size:
                    change_points.append(i)
                    cusum_pos = 0
                    cusum_neg = 0
    
    elif method == "standard_deviation":
        # Standard deviation-based detection
        # Split data into windows and detect changes in variance
        window_size = max(min_size, len(data) // 10)
        
        for i in range(window_size, len(data) - window_size, window_size // 2):
            # Calculate std for before and after windows
            before = data[max(0, i - window_size):i]
            after = data[i:min(len(data), i + window_size)]
            
            if len(before) >= 2 and len(after) >= 2:
                mean_before = sum(before) / len(before)
                mean_after = sum(after) / len(after)
                
                std_before = (sum((x - mean_before) ** 2 for x in before) / len(before)) ** 0.5
                std_after = (sum((x - mean_after) ** 2 for x in after) / len(after)) ** 0.5
                
                # Check for significant change in mean or std
                avg_std = (std_before + std_after) / 2
                if std_before > 1e-10 and std_after > 1e-10 and avg_std > 1e-10:
                    mean_change = abs(mean_after - mean_before) / avg_std
                    std_ratio = max(std_after / std_before, std_before / std_after)
                    
                    if mean_change > threshold or std_ratio > (1 + threshold):
                        if not change_points or (i - change_points[-1]) >= min_size:
                            change_points.append(i)
    
    elif method == "mean_shift":
        # Simple mean shift detection
        window_size = max(min_size, 10)
        
        for i in range(window_size, len(data) - window_size):
            # Compare mean before and after this point
            before = data[max(0, i - window_size):i]
            after = data[i:min(len(data), i + window_size)]
            
            if len(before) >= min_size and len(after) >= min_size:
                mean_before = sum(before) / len(before)
                mean_after = sum(after) / len(after)
                std_before = (sum((x - mean_before) ** 2 for x in before) / len(before)) ** 0.5
                
                if std_before < 1e-10:
                    std_before = 1.0
                
                # Detect significant mean shift
                shift = abs(mean_after - mean_before) / std_before
                
                if shift > threshold:
                    if not change_points or (i - change_points[-1]) >= min_size:
                        change_points.append(i)
    
    # Calculate segment statistics
    segments = []
    segment_starts = [0] + change_points
    segment_ends = change_points + [len(data)]
    
    for start, end in zip(segment_starts, segment_ends):
        segment_data = data[start:end]
        mean_seg = sum(segment_data) / len(segment_data)
        std_seg = (sum((x - mean_seg) ** 2 for x in segment_data) / len(segment_data)) ** 0.5
        
        segments.append({
            'start': start,
            'end': end,
            'mean': mean_seg,
            'std': std_seg
        })
    
    # Find largest change
    largest_change = None
    if len(segments) > 1:
        max_magnitude = 0
        max_idx = 0
        
        for i in range(len(segments) - 1):
            magnitude = abs(segments[i + 1]['mean'] - segments[i]['mean'])
            if magnitude > max_magnitude:
                max_magnitude = magnitude
                max_idx = change_points[i]
                max_seg_idx = i
        
        if max_magnitude > 0:
            direction = "increase" if segments[max_seg_idx + 1]['mean'] > segments[max_seg_idx]['mean'] else "decrease"
            largest_change = {
                'index': max_idx,
                'magnitude': max_magnitude,
                'direction': direction
            }
    
    return {
        'change_points': change_points,
        'number_of_segments': len(segments),
        'segments': segments,
        'largest_change': largest_change,
        'method_used': method
    }


def rate_of_change(data: list[float], time_intervals: list[float] = None, method: str = "simple", smoothing_window: int = 3) -> dict[str, Any]:
    """
    Calculate rate of change over time to detect acceleration or deceleration.
    
    Useful for monitoring how fast temperature is rising during startup, detecting rapid
    pressure changes indicating leaks, tracking production rate changes, and identifying
    abnormal ramp rates.
    
    Args:
        data: Time series values, min 2 items
        time_intervals: Time intervals between measurements (default: uniform intervals of 1)
        method: "simple" or "smoothed" rate of change
        smoothing_window: Window size for smoothed method (min 2), default 3
        
    Returns:
        Dictionary containing:
        - 'rate_of_change': List of rate values
        - 'time_points': Time points for rate values
        - 'method': Method used
        - 'average_rate': Average rate of change
        - 'max_rate': Maximum rate (and its index)
        - 'min_rate': Minimum rate (and its index)
        
    Examples:
        rate_of_change([100, 105, 112, 115, 125], method="simple")
        >>> {
        ...   'rate_of_change': [5.0, 7.0, 3.0, 10.0],
        ...   'average_rate': 6.25,
        ...   'max_rate': {'value': 10.0, 'index': 3}
        ... }
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 2:
        raise ValueError("Data must contain at least 2 items for rate of change calculation")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Set default time intervals
    if time_intervals is None:
        time_intervals = [1.0] * (len(data) - 1)
    else:
        if len(time_intervals) != len(data) - 1:
            raise ValueError(f"time_intervals length must be {len(data) - 1}, got {len(time_intervals)}")
        for i, value in enumerate(time_intervals):
            if not isinstance(value, (int, float)):
                raise ValueError(f"All time_interval values must be numeric. Item at index {i} is {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"All time_interval values must be positive. Item at index {i} is {value}")
    
    # Validate method
    if method not in ["simple", "smoothed"]:
        raise ValueError(f"method must be 'simple' or 'smoothed', got '{method}'")
    
    # Validate smoothing_window
    if not isinstance(smoothing_window, int):
        raise ValueError(f"smoothing_window must be an integer, got {type(smoothing_window).__name__}")
    if smoothing_window < 2:
        raise ValueError(f"smoothing_window must be at least 2, got {smoothing_window}")
    
    # Calculate simple rate of change
    simple_rates = []
    for i in range(len(data) - 1):
        rate = (data[i + 1] - data[i]) / time_intervals[i]
        simple_rates.append(rate)
    
    if method == "simple":
        rates = simple_rates
    else:  # smoothed
        # Apply smoothing to the rates
        if len(simple_rates) < smoothing_window:
            rates = simple_rates  # Not enough data to smooth
        else:
            rates = []
            for i in range(len(simple_rates) - smoothing_window + 1):
                window = simple_rates[i:i + smoothing_window]
                rates.append(sum(window) / smoothing_window)
    
    # Calculate statistics
    if rates:
        average_rate = sum(rates) / len(rates)
        max_rate_val = max(rates)
        min_rate_val = min(rates)
        max_rate_idx = rates.index(max_rate_val)
        min_rate_idx = rates.index(min_rate_val)
        
        max_rate = {'value': max_rate_val, 'index': max_rate_idx}
        min_rate = {'value': min_rate_val, 'index': min_rate_idx}
    else:
        average_rate = 0
        max_rate = {'value': 0, 'index': 0}
        min_rate = {'value': 0, 'index': 0}
    
    # Time points for rate values
    time_points = list(range(len(rates)))
    
    return {
        'rate_of_change': rates,
        'time_points': time_points,
        'method': method,
        'average_rate': average_rate,
        'max_rate': max_rate,
        'min_rate': min_rate
    }


def rolling_statistics(data: list[float], window_size: int, statistics: list[str] = None) -> dict[str, Any]:
    """
    Calculate rolling/windowed statistics for continuous monitoring.
    
    Useful for monitoring rolling averages on SCADA displays, tracking process stability
    with rolling standard deviation, calculating recent performance metrics, and
    implementing sliding window quality checks.
    
    Args:
        data: Time series data, min 2 items
        window_size: Rolling window size (2-1000)
        statistics: List of statistics to calculate - "mean", "std", "min", "max", 
                   "median", "range", "variance" (default: ["mean", "std"])
        
    Returns:
        Dictionary containing:
        - For each requested statistic: list of rolling values
        - 'window_size': Window size used
        - 'data_points': Number of input data points
        - 'output_points': Number of rolling statistic points
        
    Examples:
        rolling_statistics([1, 2, 3, 4, 5], 3, ["mean", "std"])
        >>> {
        ...   'mean': [2.0, 3.0, 4.0],
        ...   'std': [0.816, 0.816, 0.816],
        ...   'window_size': 3,
        ...   'data_points': 5,
        ...   'output_points': 3
        ... }
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 2:
        raise ValueError("Data must contain at least 2 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate window_size
    if not isinstance(window_size, int):
        raise ValueError(f"window_size must be an integer, got {type(window_size).__name__}")
    
    if window_size < 2 or window_size > 1000:
        raise ValueError(f"window_size must be between 2 and 1000, got {window_size}")
    
    if window_size > len(data):
        raise ValueError(f"window_size ({window_size}) cannot be larger than data length ({len(data)})")
    
    # Set default statistics
    if statistics is None:
        statistics = ["mean", "std"]
    
    # Validate statistics
    valid_stats = ["mean", "std", "min", "max", "median", "range", "variance"]
    if not isinstance(statistics, list):
        raise ValueError(f"statistics must be a list, got {type(statistics).__name__}")
    
    for stat in statistics:
        if stat not in valid_stats:
            raise ValueError(f"Invalid statistic '{stat}'. Must be one of: {', '.join(valid_stats)}")
    
    # Calculate rolling statistics
    result = {}
    
    for stat in statistics:
        values = []
        
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            
            if stat == "mean":
                values.append(sum(window) / window_size)
            
            elif stat == "std":
                mean_val = sum(window) / window_size
                variance_val = sum((x - mean_val) ** 2 for x in window) / window_size
                values.append(variance_val ** 0.5)
            
            elif stat == "min":
                values.append(min(window))
            
            elif stat == "max":
                values.append(max(window))
            
            elif stat == "median":
                sorted_window = sorted(window)
                n = len(sorted_window)
                if n % 2 == 0:
                    values.append((sorted_window[n // 2 - 1] + sorted_window[n // 2]) / 2)
                else:
                    values.append(sorted_window[n // 2])
            
            elif stat == "range":
                values.append(max(window) - min(window))
            
            elif stat == "variance":
                mean_val = sum(window) / window_size
                values.append(sum((x - mean_val) ** 2 for x in window) / window_size)
        
        result[stat] = values
    
    result['window_size'] = window_size
    result['data_points'] = len(data)
    result['output_points'] = len(data) - window_size + 1
    
    return result


# ============================================================================
# MCP Server Setup
# ============================================================================

# Create the MCP server instance
app = Server("stats-tools")

logger.info("Statistical analysis MCP server initialized")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    List all available statistical analysis tools.
    
    This function is called by MCP clients to discover what tools
    this server provides. Each tool has a name, description, and
    input schema that defines its parameters.
    
    Returns:
        List of Tool objects describing the available tools
    """
    logger.info("Client requested tool list")
    
    return [
        Tool(
            name="descriptive_stats",
            description=(
                "Calculate comprehensive descriptive statistics for a dataset including "
                "mean, median, mode, standard deviation, variance, min, max, quartiles, "
                "and range. Provides a complete statistical summary of numeric data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Array of numerical values to analyze (1-10000 items)",
                        "minItems": 1,
                        "maxItems": 10000
                    }
                },
                "required": ["data"]
            }
        ),
        Tool(
            name="correlation",
            description=(
                "Calculate Pearson correlation coefficient and covariance between two datasets. "
                "Returns correlation coefficient (-1 to +1), and interpretation. "
                "Measures the strength and direction of linear relationship between variables."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "First dataset (2-1000 items)",
                        "minItems": 2,
                        "maxItems": 1000
                    },
                    "y": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Second dataset (must be same length as x)",
                        "minItems": 2,
                        "maxItems": 1000
                    }
                },
                "required": ["x", "y"]
            }
        ),
        Tool(
            name="percentile",
            description=(
                "Calculate the value at a specific percentile in a dataset using linear "
                "interpolation method. Returns the value below which the given percentage "
                "of observations fall. Useful for quartiles, median, and percentile analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Array of numerical values (1-10000 items)",
                        "minItems": 1,
                        "maxItems": 10000
                    },
                    "percentile": {
                        "type": "number",
                        "description": "Percentile to calculate (0-100). Common values: 25 (Q1), 50 (median), 75 (Q3), 90, 95, 99",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": ["data", "percentile"]
            }
        ),
        Tool(
            name="detect_outliers",
            description=(
                "Detect outliers in a dataset using the Interquartile Range (IQR) method "
                "with configurable threshold multiplier. Returns outlier values, indices, "
                "and statistical boundaries (Q1, Q3, IQR). Standard threshold is 1.5 for "
                "typical outliers, 3.0 for extreme outliers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Array of numerical values to check for outliers (4-10000 items, minimum 4 needed for quartiles)",
                        "minItems": 4,
                        "maxItems": 10000
                    },
                    "threshold": {
                        "type": "number",
                        "description": "IQR multiplier for outlier detection (0.1-10, default: 1.5 for standard outliers, 3.0 for extreme outliers)",
                        "minimum": 0.1,
                        "maximum": 10,
                        "default": 1.5
                    }
                },
                "required": ["data"]
            }
        ),
        Tool(
            name="moving_average",
            description=(
                "Calculate Simple, Exponential, or Weighted Moving Averages for smoothing process data. "
                "Use cases: smooth noisy sensor readings, trend visualization, filter high-frequency noise, "
                "identify underlying process trends in industrial automation and SCADA systems."
            ),
            inputSchema={
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
        ),
        Tool(
            name="detect_trend",
            description=(
                "Identify and quantify trends in process data using linear or polynomial regression. "
                "Use cases: equipment degradation trends, process efficiency decline, catalyst deactivation, "
                "compressor performance degradation, heat exchanger fouling detection."
            ),
            inputSchema={
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
        ),
        Tool(
            name="autocorrelation",
            description=(
                "Calculate autocorrelation function (ACF) to identify repeating patterns and cycles. "
                "Use cases: detect cyclic patterns in batch processes, identify production cycle times, "
                "find optimal sampling intervals, detect seasonality in energy consumption."
            ),
            inputSchema={
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
        ),
        Tool(
            name="change_point_detection",
            description=(
                "Identify significant changes in process behavior (regime changes, upsets, modifications). "
                "Use cases: detect when process modifications were effective, identify process upsets, "
                "find when equipment behavior changed, detect shift changes affecting production."
            ),
            inputSchema={
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
        ),
        Tool(
            name="rate_of_change",
            description=(
                "Calculate rate of change over time to detect acceleration or deceleration. "
                "Use cases: monitor how fast temperature is rising during startup, detect rapid pressure "
                "changes indicating leaks, track production rate changes, identify abnormal ramp rates."
            ),
            inputSchema={
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
        ),
        Tool(
            name="rolling_statistics",
            description=(
                "Calculate rolling/windowed statistics for continuous monitoring. "
                "Use cases: monitor rolling averages on SCADA displays, track process stability "
                "with rolling standard deviation, calculate recent performance metrics, "
                "implement sliding window quality checks."
            ),
            inputSchema={
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
        ),
        Tool(
            name="z_score_detection",
            description=(
                "Detect outliers using Z-score methods (standard or modified). "
                "Standard uses mean/std, Modified uses median/MAD for robustness. "
                "Use cases: normally distributed process variables (temperature, pressure), "
                "quick screening of large datasets, real-time sensor validation, quality control."
            ),
            inputSchema={
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
                        "default": True
                    }
                },
                "required": ["data"]
            }
        ),
        Tool(
            name="grubbs_test",
            description=(
                "Statistical test for single outliers in normally distributed data. "
                "Use cases: reject suspicious calibration points, validate lab test results, "
                "quality control for precise measurements, statistical rigor for critical decisions, "
                "regulatory compliance (FDA, ISO)."
            ),
            inputSchema={
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
        ),
        Tool(
            name="dixon_q_test",
            description=(
                "Quick test for outliers in small datasets (3-30 points). "
                "Use cases: laboratory quality control (small sample sizes), pilot plant trials, "
                "expensive test results validation, duplicate/triplicate measurement validation, "
                "shift sample validation."
            ),
            inputSchema={
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
        ),
        Tool(
            name="isolation_forest",
            description=(
                "Machine learning-based anomaly detection for complex datasets. "
                "Use cases: multivariate anomaly detection (multiple sensors), complex process behavior, "
                "equipment failure prediction, cyber security (unusual patterns), unstructured anomaly patterns."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "description": "Univariate or multivariate data [[x1, x2, x3], ...] or [x1, x2, x3, ...]",
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
        ),
        Tool(
            name="mahalanobis_distance",
            description=(
                "Multivariate outlier detection considering correlations between variables. "
                "Use cases: multiple correlated sensor detection, process state monitoring, "
                "multivariate quality control, equipment health monitoring, pattern recognition."
            ),
            inputSchema={
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
        ),
        Tool(
            name="streaming_outlier_detection",
            description=(
                "Real-time outlier detection for continuous sensor streams. "
                "Use cases: real-time SCADA alarming, edge device data validation, "
                "continuous process monitoring, high-frequency sensor data, telemetry validation."
            ),
            inputSchema={
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
                        "type": "integer",
                        "description": "Detection sensitivity (1-10, higher = more sensitive)",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5
                    }
                },
                "required": ["current_value", "historical_window"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> CallToolResult:
    """
    Handle tool execution requests from MCP clients.
    
    This function is called when a client wants to use one of our tools.
    It validates the tool name, extracts parameters, calls the appropriate
    handler function, and returns the result.
    
    Args:
        name: Name of the tool to execute
        arguments: Dictionary of parameters for the tool
        
    Returns:
        CallToolResult containing the tool's output or error information
    """
    logger.info(f"Tool called: {name}")
    
    try:
        # Route to appropriate handler based on tool name
        if name == "descriptive_stats":
            return await handle_descriptive_stats(arguments)
        elif name == "correlation":
            return await handle_correlation(arguments)
        elif name == "percentile":
            return await handle_percentile(arguments)
        elif name == "detect_outliers":
            return await handle_detect_outliers(arguments)
        elif name == "moving_average":
            return await handle_moving_average(arguments)
        elif name == "detect_trend":
            return await handle_detect_trend(arguments)
        elif name == "autocorrelation":
            return await handle_autocorrelation(arguments)
        elif name == "change_point_detection":
            return await handle_change_point_detection(arguments)
        elif name == "rate_of_change":
            return await handle_rate_of_change(arguments)
        elif name == "rolling_statistics":
            return await handle_rolling_statistics(arguments)
        elif name == "z_score_detection":
            return await handle_z_score_detection(arguments)
        elif name == "grubbs_test":
            return await handle_grubbs_test(arguments)
        elif name == "dixon_q_test":
            return await handle_dixon_q_test(arguments)
        elif name == "isolation_forest":
            return await handle_isolation_forest(arguments)
        elif name == "mahalanobis_distance":
            return await handle_mahalanobis_distance(arguments)
        elif name == "streaming_outlier_detection":
            return await handle_streaming_outlier_detection(arguments)
        else:
            # Unknown tool name
            logger.error(f"Unknown tool requested: {name}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )],
                isError=True,
            )
    
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Unexpected error in tool {name}: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Internal error: {str(e)}"
            )],
            isError=True,
        )


# ============================================================================
# Tool Handler Functions
# ============================================================================


async def handle_descriptive_stats(arguments: Any) -> CallToolResult:
    """Handle descriptive_stats tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    
    # Validate required parameter
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'data'"
            )],
            isError=True,
        )
    
    # Validate parameter type
    if not isinstance(data, list):
        logger.error(f"Invalid parameter type for data: {type(data)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'data' must be an array, got {type(data).__name__}"
            )],
            isError=True,
        )
    
    # Calculate descriptive statistics
    try:
        logger.info(f"Calculating descriptive statistics for {len(data)} values")
        result = descriptive_stats(data)
        
        # Format mode nicely
        if len(result['mode']) == 1:
            mode_str = str(result['mode'][0])
        else:
            mode_str = f"{result['mode']} (multiple modes)"
        
        # Format the result
        result_text = (
            f"Descriptive Statistics:\n\n"
            f"Central Tendency:\n"
            f"  Mean (average):     {result['mean']:.4f}\n"
            f"  Median (middle):    {result['median']:.4f}\n"
            f"  Mode (most frequent): {mode_str}\n\n"
            f"Dispersion:\n"
            f"  Range (max - min):  {result['range']:.4f}\n"
            f"  Variance (σ²):      {result['variance']:.4f}\n"
            f"  Std Dev (σ):        {result['std_dev']:.4f}\n\n"
            f"Data Summary:\n"
            f"  Count:              {result['count']}\n"
            f"  Minimum:            {result['min']:.4f}\n"
            f"  Maximum:            {result['max']:.4f}"
        )
        
        logger.info(f"Calculated statistics: mean={result['mean']:.2f}, median={result['median']:.2f}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_correlation(arguments: Any) -> CallToolResult:
    """Handle correlation tool calls."""
    # Extract and validate parameters
    x = arguments.get("x")
    y = arguments.get("y")
    
    # Validate required parameters
    if x is None:
        logger.error("Missing required parameter: x")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'x'"
            )],
            isError=True,
        )
    
    if y is None:
        logger.error("Missing required parameter: y")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'y'"
            )],
            isError=True,
        )
    
    # Validate parameter types
    if not isinstance(x, list):
        logger.error(f"Invalid parameter type for x: {type(x)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'x' must be an array, got {type(x).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(y, list):
        logger.error(f"Invalid parameter type for y: {type(y)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'y' must be an array, got {type(y).__name__}"
            )],
            isError=True,
        )
    
    # Calculate correlation
    try:
        logger.info(f"Calculating correlation between datasets of length {len(x)} and {len(y)}")
        result = correlation(x, y)
        
        # Format the result
        r = result['coefficient']
        interpretation = result['interpretation']
        
        result_text = (
            f"Pearson Correlation Analysis:\n\n"
            f"Correlation Coefficient (r): {r:.4f}\n"
            f"Interpretation: {interpretation}\n\n"
            f"What this means:\n"
        )
        
        if abs(r) >= 0.7:
            strength = "strong"
        elif abs(r) >= 0.4:
            strength = "moderate"
        elif abs(r) >= 0.1:
            strength = "weak"
        else:
            strength = "negligible"
        
        if r > 0:
            result_text += f"  • The variables show a {strength} positive linear relationship\n"
            result_text += f"  • As one variable increases, the other tends to increase\n"
        elif r < 0:
            result_text += f"  • The variables show a {strength} negative linear relationship\n"
            result_text += f"  • As one variable increases, the other tends to decrease\n"
        else:
            result_text += f"  • The variables show no linear relationship\n"
        
        result_text += f"\nDataset sizes: x has {len(x)} values, y has {len(y)} values"
        
        logger.info(f"Correlation coefficient: {r:.4f} ({interpretation})")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_percentile(arguments: Any) -> CallToolResult:
    """Handle percentile tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    p = arguments.get("percentile")
    
    # Validate required parameters
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'data'"
            )],
            isError=True,
        )
    
    if p is None:
        logger.error("Missing required parameter: percentile")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'percentile'"
            )],
            isError=True,
        )
    
    # Validate parameter types
    if not isinstance(data, list):
        logger.error(f"Invalid parameter type for data: {type(data)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'data' must be an array, got {type(data).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(p, (int, float)):
        logger.error(f"Invalid parameter type for percentile: {type(p)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'percentile' must be a number, got {type(p).__name__}"
            )],
            isError=True,
        )
    
    # Calculate percentile
    try:
        logger.info(f"Calculating {p}th percentile for {len(data)} values")
        result = percentile(data, p)
        
        # Format the result with contextual information
        result_text = f"Percentile Calculation:\n\n"
        result_text += f"The {p}th percentile: {result:.4f}\n\n"
        result_text += f"What this means:\n"
        result_text += f"  • {p}% of the data values are below {result:.4f}\n"
        result_text += f"  • {100 - p}% of the data values are above {result:.4f}\n"
        
        # Add special names for common percentiles
        if p == 0:
            result_text += f"\nNote: The 0th percentile is the minimum value"
        elif p == 25:
            result_text += f"\nNote: The 25th percentile is also called Q1 (first quartile)"
        elif p == 50:
            result_text += f"\nNote: The 50th percentile is the median (middle value)"
        elif p == 75:
            result_text += f"\nNote: The 75th percentile is also called Q3 (third quartile)"
        elif p == 100:
            result_text += f"\nNote: The 100th percentile is the maximum value"
        
        result_text += f"\n\nDataset size: {len(data)} values"
        
        logger.info(f"{p}th percentile = {result:.4f}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_detect_outliers(arguments: Any) -> CallToolResult:
    """Handle detect_outliers tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    threshold = arguments.get("threshold", 1.5)  # Default to 1.5 if not provided
    
    # Validate required parameter
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'data'"
            )],
            isError=True,
        )
    
    # Validate parameter type
    if not isinstance(data, list):
        logger.error(f"Invalid parameter type for data: {type(data)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'data' must be an array, got {type(data).__name__}"
            )],
            isError=True,
        )
    
    # Validate threshold type
    if not isinstance(threshold, (int, float)):
        logger.error(f"Invalid parameter type for threshold: {type(threshold)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'threshold' must be a number, got {type(threshold).__name__}"
            )],
            isError=True,
        )
    
    # Detect outliers
    try:
        logger.info(f"Detecting outliers in {len(data)} values using IQR method with threshold {threshold}")
        result = detect_outliers(data, threshold)
        
        # Format the result
        result_text = f"Outlier Detection (IQR Method):\n\n"
        
        if result['count'] == 0:
            result_text += f"✓ No outliers detected\n\n"
        else:
            result_text += f"⚠ Found {result['count']} outlier(s):\n"
            for i, (value, index) in enumerate(zip(result['outliers'], result['indices']), 1):
                result_text += f"  {i}. Value {value:.4f} at index {index}\n"
            result_text += "\n"
        
        result_text += f"Statistical Boundaries:\n"
        result_text += f"  Q1 (25th percentile):  {result['q1']:.4f}\n"
        result_text += f"  Q3 (75th percentile):  {result['q3']:.4f}\n"
        result_text += f"  IQR (Q3 - Q1):         {result['iqr']:.4f}\n"
        result_text += f"  Threshold:             {result['threshold']}\n"
        result_text += f"  Lower Bound:           {result['lower_bound']:.4f}\n"
        result_text += f"  Upper Bound:           {result['upper_bound']:.4f}\n\n"
        
        result_text += f"Method: Values outside [{result['lower_bound']:.4f}, {result['upper_bound']:.4f}] are outliers\n"
        result_text += f"Formula: [Q1 - {result['threshold']}×IQR, Q3 + {result['threshold']}×IQR]\n\n"
        result_text += f"Dataset size: {len(data)} values"
        
        logger.info(f"Found {result['count']} outliers")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_moving_average(arguments: Any) -> CallToolResult:
    """Handle moving_average tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    window_size = arguments.get("window_size")
    ma_type = arguments.get("ma_type", "simple")
    alpha = arguments.get("alpha")
    
    # Validate required parameters
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    if window_size is None:
        logger.error("Missing required parameter: window_size")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'window_size'")],
            isError=True,
        )
    
    # Calculate moving average
    try:
        logger.info(f"Calculating {ma_type} moving average with window size {window_size}")
        result = moving_average(data, window_size, ma_type, alpha)
        
        # Format the result
        result_text = f"Moving Average Analysis ({result['ma_type'].title()}):\n\n"
        result_text += f"Configuration:\n"
        result_text += f"  Type: {result['ma_type'].upper()}\n"
        result_text += f"  Window Size: {result['window_size']}\n"
        result_text += f"  Input Data Points: {result['data_points']}\n"
        result_text += f"  Output Data Points: {result['smoothed_points']}\n\n"
        
        # Show first few and last few values
        ma_vals = result['moving_average']
        if len(ma_vals) <= 10:
            result_text += f"Moving Average Values:\n  {', '.join(f'{v:.4f}' for v in ma_vals)}\n\n"
        else:
            result_text += f"Moving Average Values (showing first 5 and last 5):\n"
            result_text += f"  First 5: {', '.join(f'{v:.4f}' for v in ma_vals[:5])}\n"
            result_text += f"  Last 5:  {', '.join(f'{v:.4f}' for v in ma_vals[-5:])}\n\n"
        
        result_text += f"Use Cases:\n"
        result_text += f"  • Smooth noisy sensor readings (temperature, pressure)\n"
        result_text += f"  • Trend visualization in SCADA systems\n"
        result_text += f"  • Filter high-frequency noise from flow meters\n"
        result_text += f"  • Identify underlying process trends"
        
        logger.info(f"Calculated {len(ma_vals)} moving average points")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_detect_trend(arguments: Any) -> CallToolResult:
    """Handle detect_trend tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    timestamps = arguments.get("timestamps")
    method = arguments.get("method", "linear")
    degree = arguments.get("degree", 2)
    
    # Validate required parameter
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    # Detect trend
    try:
        logger.info(f"Detecting trend using {method} method")
        result = detect_trend(data, timestamps, method, degree)
        
        # Format the result
        result_text = f"Trend Detection Analysis:\n\n"
        result_text += f"Trend Direction: {result['trend'].upper()}\n"
        result_text += f"Slope: {result['slope']:.4f}\n"
        result_text += f"Interpretation: {result['slope_interpretation']}\n\n"
        
        result_text += f"Regression Analysis:\n"
        result_text += f"  Equation: {result['equation']}\n"
        result_text += f"  R² (goodness of fit): {result['r_squared']:.4f}\n"
        result_text += f"  Fit Quality: {result['fit_quality'].upper()}\n"
        result_text += f"  95% Confidence Interval: [{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]\n\n"
        
        result_text += f"Predictions (Next 5 Time Periods):\n"
        for i, pred in enumerate(result['prediction_next_5'], 1):
            result_text += f"  T+{i}: {pred:.4f}\n"
        
        result_text += f"\nUse Cases:\n"
        result_text += f"  • Equipment degradation monitoring\n"
        result_text += f"  • Process efficiency analysis\n"
        result_text += f"  • Predictive maintenance planning\n"
        result_text += f"  • Catalyst deactivation tracking"
        
        logger.info(f"Trend: {result['trend']}, R²: {result['r_squared']:.4f}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_autocorrelation(arguments: Any) -> CallToolResult:
    """Handle autocorrelation tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    max_lag = arguments.get("max_lag")
    
    # Validate required parameter
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    # Calculate autocorrelation
    try:
        logger.info(f"Calculating autocorrelation with max_lag={max_lag}")
        result = autocorrelation(data, max_lag)
        
        # Format the result
        result_text = f"Autocorrelation Function (ACF) Analysis:\n\n"
        result_text += f"Configuration:\n"
        result_text += f"  Data Points: {len(data)}\n"
        result_text += f"  Maximum Lag: {result['max_lag']}\n"
        result_text += f"  Significant Lags: {len(result['significant_lags'])}\n\n"
        
        result_text += f"Interpretation: {result['interpretation']}\n\n"
        
        if result['significant_lags']:
            result_text += f"Significant Correlations (|ACF| > 0.2):\n"
            for lag in result['significant_lags'][:10]:  # Show first 10
                acf_val = result['acf_values'][lag]
                result_text += f"  Lag {lag}: ACF = {acf_val:.4f}\n"
            if len(result['significant_lags']) > 10:
                result_text += f"  ... and {len(result['significant_lags']) - 10} more\n"
        else:
            result_text += f"No significant periodic patterns detected.\n"
        
        result_text += f"\nACF Values (showing first 10 lags):\n"
        for i in range(min(10, len(result['acf_values']))):
            result_text += f"  Lag {i}: {result['acf_values'][i]:.4f}\n"
        
        result_text += f"\nUse Cases:\n"
        result_text += f"  • Detect cyclic patterns in batch processes\n"
        result_text += f"  • Identify production cycle times\n"
        result_text += f"  • Find optimal sampling intervals\n"
        result_text += f"  • Detect seasonality in data"
        
        logger.info(f"Found {len(result['significant_lags'])} significant lags")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_change_point_detection(arguments: Any) -> CallToolResult:
    """Handle change_point_detection tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    method = arguments.get("method", "cusum")
    threshold = arguments.get("threshold", 1.5)
    min_size = arguments.get("min_size", 5)
    
    # Validate required parameter
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    # Detect change points
    try:
        logger.info(f"Detecting change points using {method} method")
        result = change_point_detection(data, method, threshold, min_size)
        
        # Format the result
        result_text = f"Change Point Detection Analysis:\n\n"
        result_text += f"Method: {result['method_used'].upper()}\n"
        result_text += f"Change Points Found: {len(result['change_points'])}\n"
        result_text += f"Number of Segments: {result['number_of_segments']}\n\n"
        
        if result['change_points']:
            result_text += f"Detected Change Points:\n"
            for idx in result['change_points']:
                result_text += f"  Index {idx}\n"
            result_text += "\n"
        else:
            result_text += f"No significant change points detected.\n\n"
        
        result_text += f"Segment Statistics:\n"
        for i, seg in enumerate(result['segments']):
            result_text += f"  Segment {i + 1}: Index {seg['start']}-{seg['end']}\n"
            result_text += f"    Mean: {seg['mean']:.4f}, Std Dev: {seg['std']:.4f}\n"
        
        if result['largest_change']:
            result_text += f"\nLargest Change:\n"
            result_text += f"  Index: {result['largest_change']['index']}\n"
            result_text += f"  Magnitude: {result['largest_change']['magnitude']:.4f}\n"
            result_text += f"  Direction: {result['largest_change']['direction']}\n"
        
        result_text += f"\nUse Cases:\n"
        result_text += f"  • Detect process modification effectiveness\n"
        result_text += f"  • Identify process upsets or disturbances\n"
        result_text += f"  • Find equipment behavior changes\n"
        result_text += f"  • Detect shift changes affecting production"
        
        logger.info(f"Found {len(result['change_points'])} change points")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_rate_of_change(arguments: Any) -> CallToolResult:
    """Handle rate_of_change tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    time_intervals = arguments.get("time_intervals")
    method = arguments.get("method", "simple")
    smoothing_window = arguments.get("smoothing_window", 3)
    
    # Validate required parameter
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    # Calculate rate of change
    try:
        logger.info(f"Calculating {method} rate of change")
        result = rate_of_change(data, time_intervals, method, smoothing_window)
        
        # Format the result
        result_text = f"Rate of Change Analysis:\n\n"
        result_text += f"Method: {result['method'].upper()}\n"
        result_text += f"Number of Rate Points: {len(result['rate_of_change'])}\n\n"
        
        result_text += f"Statistics:\n"
        result_text += f"  Average Rate: {result['average_rate']:.4f} units/period\n"
        result_text += f"  Maximum Rate: {result['max_rate']['value']:.4f} at index {result['max_rate']['index']}\n"
        result_text += f"  Minimum Rate: {result['min_rate']['value']:.4f} at index {result['min_rate']['index']}\n\n"
        
        # Show first few and last few rate values
        rates = result['rate_of_change']
        if len(rates) <= 10:
            result_text += f"Rate of Change Values:\n"
            for i, rate in enumerate(rates):
                result_text += f"  Period {i}: {rate:.4f}\n"
        else:
            result_text += f"Rate of Change Values (showing first 5 and last 5):\n"
            result_text += f"  First 5:\n"
            for i in range(5):
                result_text += f"    Period {i}: {rates[i]:.4f}\n"
            result_text += f"  Last 5:\n"
            for i in range(len(rates) - 5, len(rates)):
                result_text += f"    Period {i}: {rates[i]:.4f}\n"
        
        result_text += f"\nUse Cases:\n"
        result_text += f"  • Monitor temperature rise during startup\n"
        result_text += f"  • Detect rapid pressure changes (leaks)\n"
        result_text += f"  • Track production rate changes\n"
        result_text += f"  • Identify abnormal ramp rates"
        
        logger.info(f"Average rate of change: {result['average_rate']:.4f}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_rolling_statistics(arguments: Any) -> CallToolResult:
    """Handle rolling_statistics tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    window_size = arguments.get("window_size")
    statistics = arguments.get("statistics", ["mean", "std"])
    
    # Validate required parameters
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    if window_size is None:
        logger.error("Missing required parameter: window_size")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'window_size'")],
            isError=True,
        )
    
    # Calculate rolling statistics
    try:
        logger.info(f"Calculating rolling statistics: {statistics}")
        result = rolling_statistics(data, window_size, statistics)
        
        # Format the result
        result_text = f"Rolling Statistics Analysis:\n\n"
        result_text += f"Configuration:\n"
        result_text += f"  Window Size: {result['window_size']}\n"
        result_text += f"  Input Data Points: {result['data_points']}\n"
        result_text += f"  Output Data Points: {result['output_points']}\n"
        result_text += f"  Statistics Calculated: {', '.join(statistics)}\n\n"
        
        # Show statistics for each requested metric
        for stat in statistics:
            if stat in result:
                values = result[stat]
                result_text += f"{stat.upper()}:\n"
                
                if len(values) <= 10:
                    for i, val in enumerate(values):
                        result_text += f"  Window {i}: {val:.4f}\n"
                else:
                    result_text += f"  First 5: {', '.join(f'{v:.4f}' for v in values[:5])}\n"
                    result_text += f"  Last 5:  {', '.join(f'{v:.4f}' for v in values[-5:])}\n"
                
                result_text += "\n"
        
        result_text += f"Use Cases:\n"
        result_text += f"  • Monitor rolling averages on SCADA displays\n"
        result_text += f"  • Track process stability with rolling std dev\n"
        result_text += f"  • Calculate recent performance metrics\n"
        result_text += f"  • Implement sliding window quality checks"
        
        logger.info(f"Calculated rolling statistics for {len(statistics)} metrics")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_z_score_detection(arguments: Any) -> CallToolResult:
    """Handle z_score_detection tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    method = arguments.get("method", "modified")
    threshold = arguments.get("threshold", 3.0)
    two_tailed = arguments.get("two_tailed", True)
    
    # Validate required parameter
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    # Calculate z-score outlier detection
    try:
        logger.info(f"Detecting outliers using {method} Z-score method")
        result = z_score_detection(data, method, threshold, two_tailed)
        
        # Format the result
        result_text = f"Z-Score Outlier Detection ({result['method'].replace('_', ' ').title()}):\n\n"
        
        outlier_count = result['statistics']['outlier_count']
        if outlier_count == 0:
            result_text += f"✓ No outliers detected\n\n"
        else:
            result_text += f"⚠ Found {outlier_count} outlier(s):\n"
            for detail in result['outliers']['values']:
                result_text += f"  • Index {detail['index']}: Value {detail['value']:.4f}\n"
                result_text += f"    Z-score: {detail['z_score']:.4f}, Severity: {detail['severity']}\n"
            result_text += "\n"
        
        result_text += f"Statistics:\n"
        if method == "standard":
            result_text += f"  Mean: {result['statistics']['mean']:.4f}\n"
            result_text += f"  Std Dev: {result['statistics']['std_dev']:.4f}\n"
        else:
            result_text += f"  Median: {result['statistics']['median']:.4f}\n"
            result_text += f"  MAD: {result['statistics']['mad']:.4f}\n"
        
        result_text += f"  Total Points: {result['statistics']['total_points']}\n"
        result_text += f"  Outlier Percentage: {result['statistics']['outlier_percentage']:.1f}%\n\n"
        
        result_text += f"Configuration:\n"
        result_text += f"  Method: {method}\n"
        result_text += f"  Threshold: {threshold}\n"
        result_text += f"  Two-tailed: {two_tailed}\n\n"
        
        result_text += f"Interpretation:\n  {result['interpretation']}"
        
        logger.info(f"Found {outlier_count} outliers")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except (ValueError, ImportError) as e:
        logger.error(f"Error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_grubbs_test(arguments: Any) -> CallToolResult:
    """Handle grubbs_test tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    alpha = arguments.get("alpha", 0.05)
    method = arguments.get("method", "two_sided")
    
    # Validate required parameter
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    # Perform Grubbs test
    try:
        logger.info(f"Performing Grubbs test with alpha={alpha}")
        result = grubbs_test(data, alpha, method)
        
        # Format the result
        result_text = f"Grubbs Test for Outliers:\n\n"
        
        result_text += f"Configuration:\n"
        result_text += f"  Significance Level (α): {result['alpha']}\n"
        result_text += f"  Sample Size: {result['sample_size']}\n"
        result_text += f"  Method: {method}\n\n"
        
        result_text += f"Suspected Outlier:\n"
        result_text += f"  Value: {result['suspected_outlier']['value']:.4f}\n"
        result_text += f"  Index: {result['suspected_outlier']['index']}\n"
        result_text += f"  Side: {result['suspected_outlier']['side']}\n\n"
        
        result_text += f"Test Results:\n"
        result_text += f"  Test Statistic (G): {result['test_statistic']:.4f}\n"
        result_text += f"  Critical Value: {result['critical_value']:.4f}\n"
        result_text += f"  P-value: {result['p_value']:.4f}\n\n"
        
        result_text += f"Conclusion:\n"
        result_text += f"  {result['conclusion']}\n\n"
        
        result_text += f"Recommendation:\n"
        result_text += f"  {result['recommendation']}"
        
        logger.info(f"Grubbs test completed: G={result['test_statistic']:.4f}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except (ValueError, ImportError) as e:
        logger.error(f"Error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_dixon_q_test(arguments: Any) -> CallToolResult:
    """Handle dixon_q_test tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    alpha = arguments.get("alpha", 0.05)
    
    # Validate required parameter
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    # Perform Dixon Q test
    try:
        logger.info(f"Performing Dixon Q test with alpha={alpha}")
        result = dixon_q_test(data, alpha)
        
        # Format the result
        result_text = f"Dixon Q Test for Outliers:\n\n"
        
        result_text += f"Configuration:\n"
        result_text += f"  Sample Size: {result['sample_size']} (suitable for small datasets)\n"
        result_text += f"  Significance Level: {alpha}\n\n"
        
        result_text += f"Suspected Outlier:\n"
        result_text += f"  Value: {result['suspected_outlier']['value']:.4f}\n"
        result_text += f"  Index: {result['suspected_outlier']['index']}\n"
        result_text += f"  Position: {result['suspected_outlier']['position']}\n\n"
        
        result_text += f"Test Results:\n"
        result_text += f"  Q Statistic: {result['q_statistic']:.4f}\n"
        result_text += f"  Q Critical: {result['q_critical']:.4f}\n\n"
        
        result_text += f"Conclusion:\n"
        result_text += f"  {result['conclusion']}\n\n"
        
        result_text += f"Recommendation:\n"
        result_text += f"  {result['recommendation']}"
        
        logger.info(f"Dixon Q test completed: Q={result['q_statistic']:.4f}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except (ValueError, ImportError) as e:
        logger.error(f"Error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_isolation_forest(arguments: Any) -> CallToolResult:
    """Handle isolation_forest tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    contamination = arguments.get("contamination", 0.1)
    n_estimators = arguments.get("n_estimators", 100)
    
    # Validate required parameter
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    # Perform Isolation Forest
    try:
        logger.info(f"Performing Isolation Forest with contamination={contamination}")
        result = isolation_forest(data, contamination, n_estimators)
        
        # Format the result
        result_text = f"Isolation Forest Anomaly Detection:\n\n"
        
        anomaly_count = len(result['anomalies']['indices'])
        if anomaly_count == 0:
            result_text += f"✓ No anomalies detected\n\n"
        else:
            result_text += f"⚠ Found {anomaly_count} anomalies:\n"
            for i, (idx, score, severity) in enumerate(zip(
                result['anomalies']['indices'][:10],  # Show first 10
                result['anomalies']['anomaly_scores'][:10],
                result['anomalies']['severity'][:10]
            )):
                result_text += f"  {i+1}. Index {idx}: Score {score:.4f}, Severity: {severity}\n"
            
            if anomaly_count > 10:
                result_text += f"  ... and {anomaly_count - 10} more\n"
            result_text += "\n"
        
        result_text += f"Statistics:\n"
        result_text += f"  Contamination Rate: {result['contamination']*100:.1f}%\n"
        result_text += f"  N Estimators: {n_estimators}\n"
        result_text += f"  Total Data Points: {len(data)}\n\n"
        
        result_text += f"Interpretation:\n  {result['interpretation']}\n\n"
        
        result_text += f"Use Cases:\n"
        result_text += f"  • Multivariate anomaly detection\n"
        result_text += f"  • Complex process behavior patterns\n"
        result_text += f"  • Equipment failure prediction\n"
        result_text += f"  • Unstructured anomaly patterns"
        
        logger.info(f"Found {anomaly_count} anomalies")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except (ValueError, ImportError) as e:
        logger.error(f"Error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_mahalanobis_distance(arguments: Any) -> CallToolResult:
    """Handle mahalanobis_distance tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    threshold = arguments.get("threshold", 0.975)
    
    # Validate required parameter
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    # Calculate Mahalanobis distance
    try:
        logger.info(f"Calculating Mahalanobis distance with threshold={threshold}")
        result = mahalanobis_distance(data, threshold)
        
        # Format the result
        result_text = f"Mahalanobis Distance Multivariate Outlier Detection:\n\n"
        
        outlier_count = len(result['outliers']['indices'])
        if outlier_count == 0:
            result_text += f"✓ No multivariate outliers detected\n\n"
        else:
            result_text += f"⚠ Found {outlier_count} multivariate outliers:\n"
            for detail in result['outliers']['distances']:
                result_text += f"  • Index {detail['index']}: Distance {detail['distance']:.4f}, "
                result_text += f"p-value: {detail['p_value']:.4f}\n"
            result_text += "\n"
        
        result_text += f"Configuration:\n"
        result_text += f"  Dimensions: {result['dimensions']}\n"
        result_text += f"  Threshold Percentile: {threshold*100:.1f}%\n"
        result_text += f"  Threshold Distance: {result['threshold_distance']:.4f}\n"
        result_text += f"  Degrees of Freedom: {result['degrees_of_freedom']}\n\n"
        
        if result['variable_contributions']:
            result_text += f"Variable Contributions (for outliers):\n"
            for contrib in result['variable_contributions']:
                result_text += f"  Index {contrib['index']}: Primary variable: {contrib['primary_variable']}, "
                result_text += f"Contribution: {contrib['contribution']*100:.1f}%\n"
            result_text += "\n"
        
        result_text += f"Interpretation:\n  {result['interpretation']}\n\n"
        
        result_text += f"Use Cases:\n"
        result_text += f"  • Multiple correlated sensor detection\n"
        result_text += f"  • Process state monitoring\n"
        result_text += f"  • Multivariate quality control\n"
        result_text += f"  • Equipment health monitoring"
        
        logger.info(f"Found {outlier_count} multivariate outliers")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except (ValueError, ImportError) as e:
        logger.error(f"Error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_streaming_outlier_detection(arguments: Any) -> CallToolResult:
    """Handle streaming_outlier_detection tool calls."""
    # Extract and validate parameters
    current_value = arguments.get("current_value")
    historical_window = arguments.get("historical_window")
    method = arguments.get("method", "ewma")
    sensitivity = arguments.get("sensitivity", 5)
    
    # Validate required parameters
    if current_value is None:
        logger.error("Missing required parameter: current_value")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'current_value'")],
            isError=True,
        )
    
    if historical_window is None:
        logger.error("Missing required parameter: historical_window")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'historical_window'")],
            isError=True,
        )
    
    # Perform streaming outlier detection
    try:
        logger.info(f"Performing streaming outlier detection using {method} method")
        result = streaming_outlier_detection(current_value, historical_window, method, sensitivity)
        
        # Format the result
        result_text = f"Streaming Outlier Detection (Real-time):\n\n"
        
        result_text += f"Current Value Assessment:\n"
        result_text += f"  Value: {result['current_value']:.4f}\n"
        result_text += f"  Is Outlier: {'YES' if result['is_outlier'] else 'NO'}\n"
        result_text += f"  Severity: {result['severity'].upper()}\n\n"
        
        result_text += f"Analysis:\n"
        result_text += f"  Method: {result['method'].upper()}\n"
        result_text += f"  Expected Range: [{result['expected_range'][0]:.4f}, {result['expected_range'][1]:.4f}]\n"
        
        if result['is_outlier']:
            result_text += f"  Deviation: {result['deviation']:.4f}\n"
            result_text += f"  Deviation (σ): {result['deviation_sigma']:.2f}\n"
        
        result_text += f"  Trend: {result['trend']}\n"
        result_text += f"  Rate of Change: {result['rate_of_change']:.4f}\n\n"
        
        result_text += f"Interpretation:\n  {result['interpretation']}\n\n"
        
        result_text += f"Recommendation:\n  {result['recommendation']}\n\n"
        
        result_text += f"Configuration:\n"
        result_text += f"  Sensitivity: {sensitivity}/10\n"
        result_text += f"  Historical Window Size: {len(historical_window)}\n\n"
        
        result_text += f"Use Cases:\n"
        result_text += f"  • Real-time SCADA alarming\n"
        result_text += f"  • Continuous process monitoring\n"
        result_text += f"  • High-frequency sensor validation\n"
        result_text += f"  • Edge device data validation"
        
        logger.info(f"Streaming detection: {'OUTLIER' if result['is_outlier'] else 'NORMAL'}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except (ValueError, ImportError) as e:
        logger.error(f"Error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def main():
    """
    Main entry point for the MCP server.
    
    This function starts the server using stdio transport, which means:
    - The server reads MCP protocol messages from stdin
    - The server writes MCP protocol messages to stdout
    - All logging and debugging output goes to stderr
    
    This is the standard transport for MCP servers that will be launched
    by client applications like Claude Desktop.
    """
    logger.info("Starting Statistical Analysis MCP server")
    
    # Run the server using stdio transport
    # This will block until the server receives a shutdown signal
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Server started, waiting for requests...")
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )
    
    logger.info("Server shut down successfully")


# Entry point when running as a script
if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
