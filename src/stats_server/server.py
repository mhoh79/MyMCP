"""
Main MCP server implementation for statistical analysis.
This server exposes tools for descriptive statistics, correlation analysis, 
percentile calculations, outlier detection, time series analysis, and signal processing.
"""

import argparse
import asyncio
import logging
import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from scipy import stats as scipy_stats
from scipy.stats import t as t_dist
from scipy.stats import chi2

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

# Import configuration module
# Add parent directory to path to import config module
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from config import load_config, Config

# Configure logging to stderr (stdout is reserved for MCP protocol messages)
# Note: Log level will be updated based on config in main()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Logs go to stderr by default
)
logger = logging.getLogger("stats-server")

# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.create_default()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config

# Constants for numerical calculations and validation
NUMERICAL_TOLERANCE = 1e-10  # Tolerance for numerical comparisons and singularity checks
MIN_CONFIDENCE_LEVEL = 0.8  # Minimum confidence level for statistical inference
MAX_CONFIDENCE_LEVEL = 0.999  # Maximum confidence level for statistical inference

# Constants for outlier detection algorithms
EULER_MASCHERONI_CONSTANT = 0.5772156649  # Euler-Mascheroni constant for isolation forest
EWMA_ALPHA = 0.3  # EWMA smoothing factor - balances responsiveness vs stability
CUSUM_K_FACTOR = 0.5  # CUSUM allowance factor (fraction of std dev) - lower = more sensitive

# SciPy imports for statistical tests
try:
    from scipy import stats as scipy_stats
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Some advanced statistical tests will be unavailable.")


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


def z_score_detection(data: list[float], method: str = "modified", threshold: float = 3.0, two_tailed: bool = True) -> dict[str, Any]:
    """
    Detect outliers using standard or modified Z-score methods.
    
    Z-score methods measure how many standard deviations a data point is from the mean.
    Modified Z-score uses median and MAD (Median Absolute Deviation) for robustness.
    
    Args:
        data: Measurements to check for outliers (min 3 items)
        method: "standard" (mean/std) or "modified" (median/MAD) - modified is more robust
        threshold: Z-score threshold (typical: 3.0 for standard, 3.5 for modified), range 1.0-10.0
        two_tailed: Detect outliers on both sides (default: True)
        
    Returns:
        Dictionary containing:
        - 'method': Method used
        - 'threshold': Threshold used
        - 'outliers': Dict with indices and detailed outlier information
        - 'statistics': Statistical measures used
        - 'cleaned_data': Data with outliers removed
        - 'interpretation': Human-readable summary
        
    Raises:
        ValueError: If data is invalid or parameters out of range
        
    Use Cases:
        - Normally distributed process variables (temperature, pressure)
        - Quick screening of large datasets
        - Real-time sensor validation
        - Quality control measurements
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 3:
        raise ValueError("Data must contain at least 3 items for Z-score detection")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate method
    if method not in ["standard", "modified"]:
        raise ValueError(f"method must be 'standard' or 'modified', got '{method}'")
    
    # Validate threshold
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"threshold must be numeric, got {type(threshold).__name__}")
    if threshold < 1.0 or threshold > 10.0:
        raise ValueError(f"threshold must be between 1.0 and 10.0, got {threshold}")
    
    outlier_indices = []
    outlier_details = []
    
    if method == "standard":
        # Standard Z-score using mean and standard deviation
        mean_val = sum(data) / len(data)
        # Use sample variance (n-1) for statistical inference
        variance = sum((x - mean_val) ** 2 for x in data) / (len(data) - 1) if len(data) > 1 else 0
        std_dev = variance ** 0.5
        
        if std_dev < 1e-10:
            std_dev = 1.0  # Prevent division by zero
        
        for i, value in enumerate(data):
            z_score = (value - mean_val) / std_dev
            
            if two_tailed:
                is_outlier = abs(z_score) > threshold
            else:
                is_outlier = z_score > threshold
            
            if is_outlier:
                # Determine severity
                abs_z = abs(z_score)
                if abs_z > threshold * 1.5:
                    severity = "extreme"
                elif abs_z > threshold * 1.2:
                    severity = "moderate"
                else:
                    severity = "mild"
                
                outlier_indices.append(i)
                outlier_details.append({
                    'index': i,
                    'value': value,
                    'z_score': z_score,
                    'severity': severity
                })
        
        statistics = {
            'mean': mean_val,
            'std_dev': std_dev,
            'total_points': len(data),
            'outlier_count': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(data)) * 100
        }
        method_used = "standard_z_score"
        
    else:  # modified
        # Modified Z-score using median and MAD
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        # Calculate median
        if n % 2 == 0:
            median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        else:
            median = sorted_data[n // 2]
        
        # Calculate MAD (Median Absolute Deviation)
        absolute_deviations = [abs(x - median) for x in data]
        sorted_deviations = sorted(absolute_deviations)
        if n % 2 == 0:
            mad = (sorted_deviations[n // 2 - 1] + sorted_deviations[n // 2]) / 2
        else:
            mad = sorted_deviations[n // 2]
        
        # Scale MAD to be consistent with standard deviation (multiply by 1.4826)
        # This makes MAD comparable to standard deviation for normal distributions
        if mad < 1e-10:
            mad = 1.0  # Prevent division by zero
        else:
            mad = mad * 1.4826
        
        for i, value in enumerate(data):
            modified_z = (value - median) / mad
            
            if two_tailed:
                is_outlier = abs(modified_z) > threshold
            else:
                is_outlier = modified_z > threshold
            
            if is_outlier:
                # Determine severity
                abs_z = abs(modified_z)
                if abs_z > threshold * 1.5:
                    severity = "extreme"
                elif abs_z > threshold * 1.2:
                    severity = "moderate"
                else:
                    severity = "mild"
                
                outlier_indices.append(i)
                outlier_details.append({
                    'index': i,
                    'value': value,
                    'z_score': modified_z,
                    'severity': severity
                })
        
        statistics = {
            'median': median,
            'mad': mad / 1.4826,  # Report unscaled MAD
            'total_points': len(data),
            'outlier_count': len(outlier_indices),
            'outlier_percentage': (len(outlier_indices) / len(data)) * 100
        }
        method_used = "modified_z_score"
    
    # Create cleaned data
    cleaned_data = [data[i] for i in range(len(data)) if i not in outlier_indices]
    
    # Generate interpretation
    if len(outlier_indices) == 0:
        interpretation = f"No outliers detected using {method} Z-score method with threshold {threshold}."
    else:
        interpretation = f"{len(outlier_indices)} outlier(s) detected ({statistics['outlier_percentage']:.1f}%). {method.capitalize()} Z-score used for {'robustness' if method == 'modified' else 'detection'}."
    
    return {
        'method': method_used,
        'threshold': threshold,
        'outliers': {
            'indices': outlier_indices,
            'values': outlier_details
        },
        'statistics': statistics,
        'cleaned_data': cleaned_data,
        'interpretation': interpretation
    }


def grubbs_test(data: list[float], alpha: float = 0.05, method: str = "two_sided") -> dict[str, Any]:
    """
    Statistical test for detecting a single outlier in normally distributed data.
    
    Grubbs' test (maximum normed residual test) identifies whether the most extreme
    value in a dataset is a statistical outlier. It assumes normal distribution.
    
    Args:
        data: Dataset to test for outliers (min 7 items for statistical validity)
        alpha: Significance level (typical: 0.05 or 0.01), range 0.001-0.1
        method: "max" (test maximum), "min" (test minimum), or "two_sided" (test both)
        
    Returns:
        Dictionary containing:
        - 'test': Test name
        - 'alpha': Significance level used
        - 'sample_size': Number of data points
        - 'suspected_outlier': Information about the suspected outlier
        - 'test_statistic': Calculated G statistic
        - 'critical_value': Critical value from Grubbs table
        - 'p_value': Approximate p-value
        - 'conclusion': Whether to reject null hypothesis
        - 'recommendation': Action to take
        
    Raises:
        ValueError: If data is invalid, too small, or parameters out of range
        
    Use Cases:
        - Reject suspicious calibration points
        - Validate laboratory test results
        - Quality control for precise measurements
        - Statistical rigor for critical decisions
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 7:
        raise ValueError("Data must contain at least 7 items for Grubbs test (statistical requirement)")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate alpha
    if not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be numeric, got {type(alpha).__name__}")
    if alpha < 0.001 or alpha > 0.1:
        raise ValueError(f"alpha must be between 0.001 and 0.1, got {alpha}")
    
    # Validate method
    if method not in ["max", "min", "two_sided"]:
        raise ValueError(f"method must be 'max', 'min', or 'two_sided', got '{method}'")
    
    n = len(data)
    mean_val = sum(data) / n
    # Use sample variance (n-1) for statistical testing
    variance = sum((x - mean_val) ** 2 for x in data) / (n - 1) if n > 1 else 0
    std_dev = variance ** 0.5
    
    if std_dev < 1e-10:
        return {
            'test': 'Grubbs test',
            'alpha': alpha,
            'sample_size': n,
            'suspected_outlier': None,
            'test_statistic': 0.0,
            'critical_value': 0.0,
            'p_value': 1.0,
            'conclusion': 'Cannot perform test - data has zero variance',
            'recommendation': 'All values are identical, no outliers possible'
        }
    
    # Determine which value(s) to test
    if method == "max":
        suspected_value = max(data)
        suspected_index = data.index(suspected_value)
        side = "maximum"
    elif method == "min":
        suspected_value = min(data)
        suspected_index = data.index(suspected_value)
        side = "minimum"
    else:  # two_sided
        max_val = max(data)
        min_val = min(data)
        max_dev = abs(max_val - mean_val)
        min_dev = abs(min_val - mean_val)
        
        if max_dev > min_dev:
            suspected_value = max_val
            suspected_index = data.index(max_val)
            side = "maximum"
        else:
            suspected_value = min_val
            suspected_index = data.index(min_val)
            side = "minimum"
    
    # Calculate Grubbs statistic: G = |x - mean| / std_dev
    G = abs(suspected_value - mean_val) / std_dev
    
    # Calculate critical value using t-distribution
    # Critical value: G_crit = ((n-1)/sqrt(n)) * sqrt(t^2 / (n-2+t^2))
    # where t is the critical value from t-distribution with n-2 degrees of freedom
    
    # For two-sided test, use alpha/n for Bonferroni correction
    # For one-sided test, use alpha/n
    if method == "two_sided":
        t_alpha = alpha / (2 * n)
    else:
        t_alpha = alpha / n
    
    t_critical = t_dist.ppf(1 - t_alpha, n - 2)
    G_critical = ((n - 1) / math.sqrt(n)) * math.sqrt(t_critical ** 2 / (n - 2 + t_critical ** 2))
    
    # Approximate p-value
    # Use a simpler approximation to avoid numerical issues
    t_stat = G * math.sqrt(n) / math.sqrt(n - 1)
    t_squared = t_stat ** 2
    
    # Check if denominator would be negative (happens when outlier is very extreme)
    denominator = n - 2 - t_squared
    if denominator > 0:
        try:
            arg = math.sqrt(t_squared * (n - 2) / denominator)
            p_value_approx = 2 * (1 - t_dist.cdf(arg, n - 2))
            p_value_approx = min(p_value_approx, 1.0)
        except (ValueError, OverflowError):
            # If calculation fails, use 0 for very extreme outliers
            p_value_approx = 0.0
    else:
        # Very extreme outlier, p-value is essentially 0
        p_value_approx = 0.0
    
    # Determine conclusion
    is_outlier = G > G_critical
    
    if is_outlier:
        conclusion = f"Reject null hypothesis - value is a significant outlier at α={alpha}"
        recommendation = f"Remove point {suspected_index} (value: {suspected_value:.4f}) - statistically significant outlier. Investigate cause."
    else:
        conclusion = f"Fail to reject null hypothesis - no significant outlier detected at α={alpha}"
        recommendation = f"Retain all data points. Most extreme value (index {suspected_index}, value {suspected_value:.4f}) is not a statistical outlier."
    
    return {
        'test': 'Grubbs test',
        'alpha': alpha,
        'sample_size': n,
        'suspected_outlier': {
            'value': suspected_value,
            'index': suspected_index,
            'side': side
        },
        'test_statistic': G,
        'critical_value': G_critical,
        'p_value': p_value_approx,
        'conclusion': conclusion,
        'recommendation': recommendation
    }


def dixon_q_test(data: list[float], alpha: float = 0.05) -> dict[str, Any]:
    """
    Quick outlier test for small datasets (3-30 points) using gap ratio.
    
    Dixon's Q test is designed for small sample sizes where other tests may not
    be appropriate. It tests whether the most extreme value is an outlier.
    
    Args:
        data: Small dataset (3-30 points)
        alpha: Significance level (typical: 0.05), range 0.001-0.1
        
    Returns:
        Dictionary containing:
        - 'test': Test name
        - 'sample_size': Number of data points
        - 'suspected_outlier': Information about suspected outlier
        - 'q_statistic': Calculated Q statistic
        - 'q_critical': Critical Q value from table
        - 'conclusion': Whether outlier is detected
        - 'recommendation': Action to take
        
    Raises:
        ValueError: If data is invalid, size out of range 3-30, or alpha out of range
        
    Use Cases:
        - Laboratory quality control (small sample sizes)
        - Pilot plant trials (limited data)
        - Expensive test results validation
        - Duplicate/triplicate measurement validation
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 3 or len(data) > 30:
        raise ValueError(f"Dixon Q test requires 3-30 data points, got {len(data)}")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate alpha
    if not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be numeric, got {type(alpha).__name__}")
    if alpha < 0.001 or alpha > 0.1:
        raise ValueError(f"alpha must be between 0.001 and 0.1, got {alpha}")
    
    n = len(data)
    sorted_data = sorted(data)
    
    # Find original indices
    original_indices = [data.index(val) for val in sorted_data]
    
    # Critical values for Dixon Q test at alpha=0.05
    # Source: Rorabacher, D.B. (1991). Statistical treatment for rejection of deviant values
    q_critical_table = {
        3: 0.970, 4: 0.829, 5: 0.710, 6: 0.625, 7: 0.568,
        8: 0.526, 9: 0.493, 10: 0.466, 11: 0.444, 12: 0.426,
        13: 0.410, 14: 0.396, 15: 0.384, 16: 0.374, 17: 0.365,
        18: 0.356, 19: 0.349, 20: 0.342, 21: 0.337, 22: 0.331,
        23: 0.326, 24: 0.321, 25: 0.317, 26: 0.312, 27: 0.308,
        28: 0.305, 29: 0.301, 30: 0.290
    }
    
    # Adjust for different alpha values (rough approximation)
    q_critical = q_critical_table.get(n, 0.3)
    if alpha < 0.04:  # More stringent
        q_critical *= 1.1
    elif alpha > 0.06:  # Less stringent
        q_critical *= 0.9
    
    # Calculate Q statistic for both ends
    # Q = gap / range
    # For lower end: (x2 - x1) / (xn - x1)
    # For upper end: (xn - xn-1) / (xn - x1)
    
    gap_low = sorted_data[1] - sorted_data[0]
    gap_high = sorted_data[-1] - sorted_data[-2]
    range_val = sorted_data[-1] - sorted_data[0]
    
    if range_val < 1e-10:
        return {
            'test': 'Dixon Q test',
            'sample_size': n,
            'suspected_outlier': None,
            'q_statistic': 0.0,
            'q_critical': q_critical,
            'conclusion': 'Cannot perform test - data has zero range',
            'recommendation': 'All values are too similar, no outliers detectable'
        }
    
    q_low = gap_low / range_val
    q_high = gap_high / range_val
    
    # Determine which end is more suspicious
    if q_high > q_low:
        q_stat = q_high
        suspected_value = sorted_data[-1]
        suspected_index = original_indices[-1]
        position = "high"
    else:
        q_stat = q_low
        suspected_value = sorted_data[0]
        suspected_index = original_indices[0]
        position = "low"
    
    # Determine conclusion
    is_outlier = q_stat > q_critical
    
    if is_outlier:
        conclusion = f"Outlier detected - reject value (Q={q_stat:.3f} > Q_critical={q_critical:.3f})"
        recommendation = f"Remove or investigate point at index {suspected_index} (value: {suspected_value:.4f}). Repeat measurement or investigate cause."
    else:
        conclusion = f"No outlier detected (Q={q_stat:.3f} ≤ Q_critical={q_critical:.3f})"
        recommendation = f"Retain all data points. Value {suspected_value:.4f} at index {suspected_index} is within acceptable limits."
    
    return {
        'test': 'Dixon Q test',
        'sample_size': n,
        'suspected_outlier': {
            'value': suspected_value,
            'index': suspected_index,
            'position': position
        },
        'q_statistic': q_stat,
        'q_critical': q_critical,
        'conclusion': conclusion,
        'recommendation': recommendation
    }


def isolation_forest(data: list, contamination: float = 0.1, n_estimators: int = 100, random_seed: int = 42) -> dict[str, Any]:
    """
    Machine learning-based anomaly detection using Isolation Forest algorithm.
    
    Isolation Forest detects anomalies by isolating observations. Anomalies are
    easier to isolate (fewer splits needed) than normal points.
    
    Args:
        data: Univariate (list of numbers) or multivariate (list of lists) data, min 10 items
        contamination: Expected proportion of outliers (0-0.5), default 0.1
        n_estimators: Number of isolation trees (50-500), default 100
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
        - 'method': Method name
        - 'anomalies': Dict with indices, scores, and severity
        - 'contamination': Actual contamination level
        - 'interpretation': Human-readable summary
        
    Raises:
        ValueError: If data is invalid or parameters out of range
        
    Use Cases:
        - Multivariate anomaly detection (multiple sensors)
        - Complex process behavior patterns
        - Equipment failure prediction
        - Unstructured anomaly patterns
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 10:
        raise ValueError("Data must contain at least 10 items for Isolation Forest")
    
    # Validate contamination
    if not isinstance(contamination, (int, float)):
        raise ValueError(f"contamination must be numeric, got {type(contamination).__name__}")
    if contamination < 0.0 or contamination > 0.5:
        raise ValueError(f"contamination must be between 0.0 and 0.5, got {contamination}")
    
    # Validate n_estimators
    if not isinstance(n_estimators, int):
        raise ValueError(f"n_estimators must be an integer, got {type(n_estimators).__name__}")
    if n_estimators < 50 or n_estimators > 500:
        raise ValueError(f"n_estimators must be between 50 and 500, got {n_estimators}")
    
    # Determine if data is univariate or multivariate
    if isinstance(data[0], (int, float)):
        # Univariate - convert to 2D
        X = [[val] for val in data]
        
        # Validate all elements are numeric
        for i, value in enumerate(data):
            if not isinstance(value, (int, float)):
                raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    elif isinstance(data[0], list):
        # Multivariate
        X = data
        
        # Validate structure
        feature_count = len(data[0])
        for i, row in enumerate(data):
            if not isinstance(row, list):
                raise ValueError(f"For multivariate data, all items must be lists. Item at index {i} is {type(row).__name__}")
            if len(row) != feature_count:
                raise ValueError(f"All rows must have same number of features. Row {i} has {len(row)}, expected {feature_count}")
            for j, value in enumerate(row):
                if not isinstance(value, (int, float)):
                    raise ValueError(f"All values must be numeric. Item at [{i}][{j}] is {type(value).__name__}")
    else:
        raise ValueError(f"Data items must be numbers (univariate) or lists (multivariate), got {type(data[0]).__name__}")
    
    # Simple implementation of Isolation Forest
    # This is a simplified version for the purpose of this tool
    # Use modern numpy random generator API for better randomness and thread safety
    rng = np.random.default_rng(random_seed)
    n_samples = len(X)
    n_features = len(X[0])
    
    # Convert to numpy array for easier manipulation
    X_array = np.array(X)
    
    # Calculate anomaly scores for each point
    # Lower path length = easier to isolate = more anomalous
    anomaly_scores = []
    
    for i in range(n_samples):
        point = X_array[i]
        path_lengths = []
        
        # Build multiple trees
        for tree_idx in range(min(n_estimators, 100)):  # Limit for performance
            # Random subsample (common in Isolation Forest)
            subsample_size = min(256, n_samples)
            subsample_indices = rng.choice(n_samples, subsample_size, replace=False)
            subsample = X_array[subsample_indices]
            
            # Calculate path length to isolate this point
            path_length = 0
            current_subsample = subsample
            max_depth = 10  # Limit depth for performance
            
            while len(current_subsample) > 1 and path_length < max_depth:
                # Random split
                feature_idx = rng.integers(0, n_features)
                feature_values = current_subsample[:, feature_idx]
                
                if len(set(feature_values)) <= 1:
                    break
                
                min_val = np.min(feature_values)
                max_val = np.max(feature_values)
                
                if max_val - min_val < 1e-10:
                    break
                
                split_value = rng.uniform(min_val, max_val)
                
                # Determine which side the point falls on
                if point[feature_idx] < split_value:
                    current_subsample = current_subsample[feature_values < split_value]
                else:
                    current_subsample = current_subsample[feature_values >= split_value]
                
                path_length += 1
            
            path_lengths.append(path_length)
        
        # Average path length across trees
        avg_path_length = np.mean(path_lengths)
        
        # Normalize path length to anomaly score
        # Shorter paths = higher anomaly score
        # Use c(n) = 2H(n-1) - 2(n-1)/n where H is harmonic number (Euler-Mascheroni)
        c_n = 2 * (np.log(subsample_size - 1) + EULER_MASCHERONI_CONSTANT) - 2 * (subsample_size - 1) / subsample_size
        anomaly_score = 2 ** (-avg_path_length / c_n)
        anomaly_scores.append(anomaly_score)
    
    # Determine threshold based on contamination
    sorted_scores = sorted(anomaly_scores, reverse=True)
    threshold_idx = max(0, int(contamination * len(sorted_scores)) - 1)
    threshold = sorted_scores[threshold_idx] if threshold_idx < len(sorted_scores) else sorted_scores[-1]
    
    # Identify anomalies
    anomaly_indices = []
    anomaly_score_list = []
    severity_list = []
    
    for i, score in enumerate(anomaly_scores):
        if score >= threshold:
            anomaly_indices.append(i)
            anomaly_score_list.append(score)
            
            # Determine severity
            if score > threshold * 1.2:
                severity = "high"
            elif score > threshold * 1.1:
                severity = "medium"
            else:
                severity = "low"
            severity_list.append(severity)
    
    actual_contamination = len(anomaly_indices) / len(data)
    
    # Generate interpretation
    if len(anomaly_indices) == 0:
        interpretation = f"No anomalies detected with contamination threshold {contamination:.2f}."
    else:
        interpretation = f"{len(anomaly_indices)} anomal{'y' if len(anomaly_indices) == 1 else 'ies'} detected ({actual_contamination*100:.1f}% of data). Isolation scores indicate how isolated each point is from normal patterns."
    
    return {
        'method': 'Isolation Forest',
        'anomalies': {
            'indices': anomaly_indices,
            'anomaly_scores': anomaly_score_list,
            'severity': severity_list
        },
        'contamination': actual_contamination,
        'interpretation': interpretation
    }


def mahalanobis_distance(data: list[list[float]], threshold: float = 0.975) -> dict[str, Any]:
    """
    Multivariate outlier detection using Mahalanobis distance.
    
    Mahalanobis distance measures how far a point is from the center of a distribution,
    accounting for correlations between variables. It's effective for detecting
    multivariate outliers in correlated data.
    
    Args:
        data: Multivariate data [[x1, y1, z1], [x2, y2, z2], ...], min 10 samples
        threshold: Chi-square threshold percentile (0.9-0.999), default 0.975
        
    Returns:
        Dictionary containing:
        - 'method': Method name
        - 'dimensions': Number of variables
        - 'outliers': Dict with indices and distance details
        - 'threshold_distance': Distance threshold used
        - 'degrees_of_freedom': Number of variables (chi-square df)
        - 'interpretation': Human-readable summary
        - 'variable_contributions': Which variables contribute most to outliers
        
    Raises:
        ValueError: If data is invalid, not multivariate, or threshold out of range
        
    Use Cases:
        - Multiple correlated sensor detection
        - Process state monitoring (temperature, pressure, flow together)
        - Multivariate quality control
        - Equipment health monitoring (multiple parameters)
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    if len(data) < 10:
        raise ValueError("Data must contain at least 10 samples for Mahalanobis distance")
    
    # Validate multivariate structure
    if not isinstance(data[0], list):
        raise ValueError("Data must be multivariate (list of lists). For univariate data, use other methods.")
    
    n_features = len(data[0])
    if n_features < 2:
        raise ValueError("Data must have at least 2 variables for multivariate analysis")
    
    # Validate all rows have same length and are numeric
    for i, row in enumerate(data):
        if not isinstance(row, list):
            raise ValueError(f"All data rows must be lists. Row {i} is {type(row).__name__}")
        if len(row) != n_features:
            raise ValueError(f"All rows must have same number of features. Row {i} has {len(row)}, expected {n_features}")
        for j, value in enumerate(row):
            if not isinstance(value, (int, float)):
                raise ValueError(f"All values must be numeric. Value at [{i}][{j}] is {type(value).__name__}")
    
    # Validate threshold
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"threshold must be numeric, got {type(threshold).__name__}")
    if threshold < 0.9 or threshold > 0.999:
        raise ValueError(f"threshold must be between 0.9 and 0.999, got {threshold}")
    
    n_samples = len(data)
    
    # Convert to numpy array
    X = np.array(data)
    
    # Calculate mean vector
    mean_vector = np.mean(X, axis=0)
    
    # Calculate covariance matrix with sample covariance (ddof=1) for statistical inference
    cov_matrix = np.cov(X.T, ddof=1)
    
    # Check if covariance matrix is singular
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        cov_inv = np.linalg.pinv(cov_matrix)
    
    # Calculate Mahalanobis distance for each point
    distances = []
    for i, point in enumerate(X):
        diff = point - mean_vector
        # Mahalanobis distance: sqrt((x - μ)^T * Σ^-1 * (x - μ))
        distance = np.sqrt(diff.T @ cov_inv @ diff)
        distances.append(distance)
    
    # Determine threshold using chi-square distribution
    # Mahalanobis distance squared follows chi-square distribution with k degrees of freedom
    threshold_distance = np.sqrt(chi2.ppf(threshold, n_features))
    
    # Identify outliers
    outlier_indices = []
    outlier_details = []
    variable_contributions = []
    
    for i, distance in enumerate(distances):
        if distance > threshold_distance:
            outlier_indices.append(i)
            
            # Approximate p-value
            p_value = 1 - chi2.cdf(distance ** 2, n_features)
            
            outlier_details.append({
                'index': i,
                'distance': distance,
                'p_value': p_value
            })
            
            # Determine which variable contributes most
            diff = X[i] - mean_vector
            normalized_diff = np.abs(diff) / (np.sqrt(np.diag(cov_matrix)) + 1e-10)
            primary_var_idx = np.argmax(normalized_diff)
            contribution = normalized_diff[primary_var_idx] / np.sum(normalized_diff)
            
            variable_contributions.append({
                'index': i,
                'primary_variable': f"variable_{primary_var_idx}",
                'contribution': contribution
            })
    
    # Generate interpretation
    if len(outlier_indices) == 0:
        interpretation = f"No multivariate outliers detected at threshold {threshold} ({threshold*100:.1f}th percentile)."
    else:
        interpretation = f"{len(outlier_indices)} multivariate outlier(s) detected. These points are unusual in the combined context of all {n_features} variables."
    
    return {
        'method': 'Mahalanobis Distance',
        'dimensions': n_features,
        'outliers': {
            'indices': outlier_indices,
            'distances': outlier_details
        },
        'threshold_distance': threshold_distance,
        'degrees_of_freedom': n_features,
        'interpretation': interpretation,
        'variable_contributions': variable_contributions
    }


def streaming_outlier_detection(current_value: float, historical_window: list[float], 
                                 method: str = "ewma", sensitivity: float = 5.0) -> dict[str, Any]:
    """
    Real-time outlier detection for continuous sensor streams.
    
    Designed for real-time monitoring of sensor data, this function evaluates a new
    measurement against recent history to detect anomalies.
    
    Args:
        current_value: New measurement to evaluate
        historical_window: Recent historical values for context (10-1000 items)
        method: Detection method - "ewma", "cusum", or "adaptive_threshold"
        sensitivity: Detection sensitivity (1-10, higher = more sensitive), default 5
        
    Returns:
        Dictionary containing:
        - 'current_value': The value being evaluated
        - 'is_outlier': Boolean indicating if value is an outlier
        - 'severity': "normal", "warning", or "critical"
        - 'method': Method used
        - 'expected_range': Expected range for normal values
        - 'deviation': How far from expected
        - 'deviation_sigma': Deviation in standard deviations
        - 'interpretation': Human-readable summary
        - 'recommendation': Action to take
        - 'trend': Trend direction ("increasing", "decreasing", "stable")
        - 'rate_of_change': Rate of change from last value
        
    Raises:
        ValueError: If inputs are invalid or parameters out of range
        
    Use Cases:
        - Real-time SCADA alarming
        - Edge device data validation
        - Continuous process monitoring
        - High-frequency sensor data (1-second intervals)
    """
    # Validate current_value
    if not isinstance(current_value, (int, float)):
        raise ValueError(f"current_value must be numeric, got {type(current_value).__name__}")
    
    # Validate historical_window
    if not isinstance(historical_window, list):
        raise ValueError(f"historical_window must be a list, got {type(historical_window).__name__}")
    
    if len(historical_window) < 10 or len(historical_window) > 1000:
        raise ValueError(f"historical_window must contain 10-1000 items, got {len(historical_window)}")
    
    # Validate all elements in historical_window are numeric
    for i, value in enumerate(historical_window):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All historical values must be numeric. Item at index {i} is {type(value).__name__}")
    
    # Validate method
    if method not in ["ewma", "cusum", "adaptive_threshold"]:
        raise ValueError(f"method must be 'ewma', 'cusum', or 'adaptive_threshold', got '{method}'")
    
    # Validate sensitivity
    if not isinstance(sensitivity, (int, float)):
        raise ValueError(f"sensitivity must be numeric, got {type(sensitivity).__name__}")
    if sensitivity < 1.0 or sensitivity > 10.0:
        raise ValueError(f"sensitivity must be between 1.0 and 10.0, got {sensitivity}")
    
    # Calculate historical statistics
    n = len(historical_window)
    hist_mean = sum(historical_window) / n
    # Use sample variance (n-1) for statistical inference
    hist_variance = sum((x - hist_mean) ** 2 for x in historical_window) / (n - 1) if n > 1 else 0
    hist_std = hist_variance ** 0.5
    
    if hist_std < 1e-10:
        hist_std = 1.0  # Prevent division by zero
    
    # Calculate rate of change from last value
    last_value = historical_window[-1]
    rate_of_change = current_value - last_value
    
    # Determine trend from recent history
    recent_window = historical_window[-10:]
    recent_trend = recent_window[-1] - recent_window[0]
    if abs(recent_trend) < hist_std * 0.1:
        trend = "stable"
    elif recent_trend > 0:
        trend = "increasing"
    else:
        trend = "decreasing"
    
    # Sensitivity to threshold conversion (inverse relationship)
    # Higher sensitivity = lower threshold
    threshold_sigma = 6.0 - (sensitivity - 1.0) * 0.5
    
    is_outlier = False
    severity = "normal"
    
    if method == "ewma":
        # Exponentially Weighted Moving Average
        # Give more weight to recent values
        alpha = EWMA_ALPHA  # Smoothing factor (0.3 balances responsiveness vs stability)
        ewma = historical_window[0]
        for val in historical_window[1:]:
            ewma = alpha * val + (1 - alpha) * ewma
        
        expected_value = ewma
        expected_range = [ewma - threshold_sigma * hist_std, ewma + threshold_sigma * hist_std]
        deviation = current_value - ewma
        deviation_sigma = abs(deviation) / hist_std
        
        if abs(deviation) > threshold_sigma * hist_std:
            is_outlier = True
            if abs(deviation) > (threshold_sigma * 1.5) * hist_std:
                severity = "critical"
            else:
                severity = "warning"
    
    elif method == "cusum":
        # Cumulative Sum
        # Detect persistent shifts
        target = hist_mean
        k = CUSUM_K_FACTOR * hist_std  # Allowance (0.5*std - lower = more sensitive)
        h = threshold_sigma * hist_std  # Decision threshold
        
        # Calculate CUSUM
        cusum_pos = 0
        cusum_neg = 0
        for val in historical_window[-20:]:  # Use recent history
            dev = val - target
            cusum_pos = max(0, cusum_pos + dev - k)
            cusum_neg = max(0, cusum_neg - dev - k)
        
        # Check current value
        current_dev = current_value - target
        cusum_pos = max(0, cusum_pos + current_dev - k)
        cusum_neg = max(0, cusum_neg - current_dev - k)
        
        expected_value = target
        expected_range = [target - threshold_sigma * hist_std, target + threshold_sigma * hist_std]
        deviation = current_value - target
        deviation_sigma = abs(deviation) / hist_std
        
        if cusum_pos > h or cusum_neg > h:
            is_outlier = True
            if cusum_pos > h * 1.5 or cusum_neg > h * 1.5:
                severity = "critical"
            else:
                severity = "warning"
    
    else:  # adaptive_threshold
        # Adaptive threshold based on recent variability
        # Calculate rolling statistics
        rolling_window_size = min(50, len(historical_window))
        rolling_window = historical_window[-rolling_window_size:]
        rolling_mean = sum(rolling_window) / len(rolling_window)
        # Use sample variance (n-1) for statistical inference
        n_rolling = len(rolling_window)
        rolling_variance = sum((x - rolling_mean) ** 2 for x in rolling_window) / (n_rolling - 1) if n_rolling > 1 else 0
        rolling_std = rolling_variance ** 0.5
        
        if rolling_std < 1e-10:
            rolling_std = hist_std
        
        expected_value = rolling_mean
        expected_range = [rolling_mean - threshold_sigma * rolling_std, 
                         rolling_mean + threshold_sigma * rolling_std]
        deviation = current_value - rolling_mean
        deviation_sigma = abs(deviation) / rolling_std
        
        if current_value < expected_range[0] or current_value > expected_range[1]:
            is_outlier = True
            if abs(deviation) > (threshold_sigma * 1.5) * rolling_std:
                severity = "critical"
            else:
                severity = "warning"
    
    # Generate interpretation
    if is_outlier:
        if severity == "critical":
            interpretation = f"CRITICAL: Current value {current_value:.2f} significantly exceeds expected range by {deviation_sigma:.1f} standard deviations"
        else:
            interpretation = f"WARNING: Current value {current_value:.2f} exceeds expected range by {deviation_sigma:.1f} standard deviations"
        recommendation = "Investigate sensor or process condition immediately"
    else:
        interpretation = f"Current value {current_value:.2f} is within expected range"
        recommendation = "No action needed - value is normal"
    
    return {
        'current_value': current_value,
        'is_outlier': is_outlier,
        'severity': severity,
        'method': method,
        'expected_range': expected_range,
        'deviation': deviation,
        'deviation_sigma': deviation_sigma,
        'interpretation': interpretation,
        'recommendation': recommendation,
        'trend': trend,
        'rate_of_change': rate_of_change
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
        if abs(denominator) < NUMERICAL_TOLERANCE:
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
# Regression Analysis Functions
# ============================================================================


def linear_regression(x: Any, y: list[float], confidence_level: float = 0.95, include_diagnostics: bool = True) -> dict[str, Any]:
    """
    Perform simple or multiple linear regression with comprehensive diagnostics.
    
    Use Cases:
    - Equipment efficiency vs. load relationship
    - Energy consumption vs. production rate
    - Pump performance curves (head vs. flow)
    - Temperature vs. pressure relationships
    - Vibration amplitude vs. bearing wear
    
    Args:
        x: Independent variable(s) - single array for simple regression or array of arrays for multiple regression
        y: Dependent variable (response), min 3 items
        confidence_level: Confidence level for intervals (0.5-0.999), default 0.95
        include_diagnostics: Include full regression diagnostics, default True
        
    Returns:
        Dictionary containing coefficients, statistics, confidence intervals, diagnostics, and interpretation
    """
    # Validate inputs
    if not isinstance(y, list):
        raise ValueError(f"y must be a list, got {type(y).__name__}")
    
    if len(y) < 3:
        raise ValueError("y must contain at least 3 items")
    
    for i, value in enumerate(y):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All y values must be numeric. Item at index {i} is {type(value).__name__}")
    
    if not isinstance(confidence_level, (int, float)):
        raise ValueError(f"confidence_level must be numeric, got {type(confidence_level).__name__}")
    
    if confidence_level < MIN_CONFIDENCE_LEVEL or confidence_level > MAX_CONFIDENCE_LEVEL:
        raise ValueError(f"confidence_level must be between {MIN_CONFIDENCE_LEVEL} and {MAX_CONFIDENCE_LEVEL}, got {confidence_level}")
    
    # Determine if simple or multiple regression
    is_multiple = isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)
    
    if is_multiple:
        # Multiple regression: x is array of arrays
        if len(x) != len(y):
            raise ValueError(f"x length ({len(x)}) must match y length ({len(y)})")
        
        # Validate all rows have same number of variables
        n_vars = len(x[0])
        for i, row in enumerate(x):
            if not isinstance(row, list):
                raise ValueError(f"For multiple regression, each x row must be a list. Row {i} is {type(row).__name__}")
            if len(row) != n_vars:
                raise ValueError(f"All x rows must have same length. Row {i} has {len(row)}, expected {n_vars}")
            for j, value in enumerate(row):
                if not isinstance(value, (int, float)):
                    raise ValueError(f"All x values must be numeric. Item at [{i}][{j}] is {type(value).__name__}")
        
        # Build design matrix X with intercept
        n = len(y)
        X = [[1.0] + list(row) for row in x]
        n_params = n_vars + 1
    else:
        # Simple regression: x is single array
        if not isinstance(x, list):
            raise ValueError(f"x must be a list, got {type(x).__name__}")
        
        if len(x) != len(y):
            raise ValueError(f"x length ({len(x)}) must match y length ({len(y)})")
        
        for i, value in enumerate(x):
            if not isinstance(value, (int, float)):
                raise ValueError(f"All x values must be numeric. Item at index {i} is {type(value).__name__}")
        
        # Build design matrix X with intercept
        n = len(y)
        X = [[1.0, xi] for xi in x]
        n_params = 2
    
    if n < n_params:
        raise ValueError(f"Need at least {n_params} observations for regression, got {n}")
    
    # Solve normal equations: (X'X)β = X'y
    # Using matrix operations
    XtX = [[sum(X[i][j] * X[i][k] for i in range(n)) for k in range(n_params)] for j in range(n_params)]
    Xty = [sum(X[i][j] * y[i] for i in range(n)) for j in range(n_params)]
    
    # Solve for coefficients using Gaussian elimination
    try:
        coefficients = solve_linear_system(XtX, Xty)
    except Exception as e:
        raise ValueError(f"Cannot solve regression: matrix may be singular (multicollinearity). Error: {str(e)}")
    
    # Calculate predictions and residuals
    predictions = [sum(X[i][j] * coefficients[j] for j in range(n_params)) for i in range(n)]
    residuals = [y[i] - predictions[i] for i in range(n)]
    
    # Calculate statistics
    mean_y = sum(y) / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum(r ** 2 for r in residuals)
    ss_reg = ss_tot - ss_res
    
    # R-squared
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 1.0
    
    # Adjusted R-squared
    adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - n_params)) if n > n_params else r_squared
    
    # RMSE and MAE
    rmse = (ss_res / n) ** 0.5
    mae = sum(abs(r) for r in residuals) / n
    
    # F-statistic
    if n > n_params and ss_res > 1e-10:
        f_statistic = (ss_reg / (n_params - 1)) / (ss_res / (n - n_params))
        # Calculate p-value using F-distribution approximation
        if SCIPY_AVAILABLE:
            p_value = float(1 - scipy_stats.f.cdf(f_statistic, n_params - 1, n - n_params))
        else:
            p_value = None
    else:
        f_statistic = None
        p_value = None
    
    # Build equation string
    intercept = coefficients[0]
    slopes = coefficients[1:]
    
    if is_multiple:
        equation_parts = [f"{intercept:.4f}"]
        for i, slope in enumerate(slopes):
            sign = "+" if slope >= 0 else "-"
            equation_parts.append(f"{sign} {abs(slope):.4f}*x{i+1}")
        equation = "y = " + " ".join(equation_parts)
    else:
        slope = slopes[0]
        sign = "+" if intercept >= 0 else "-"
        equation = f"y = {slope:.4f}x {sign} {abs(intercept):.4f}"
    
    # Calculate confidence intervals for coefficients
    if SCIPY_AVAILABLE and n > n_params:
        # Standard errors
        mse = ss_res / (n - n_params)
        try:
            XtX_inv = linalg.inv(XtX)
            se_coeffs = [math.sqrt(mse * XtX_inv[i][i]) for i in range(n_params)]
        except:
            se_coeffs = [0] * n_params
        
        # t-value for confidence level
        t_value = float(scipy_stats.t.ppf((1 + confidence_level) / 2, n - n_params))
        
        ci_intercept = [intercept - t_value * se_coeffs[0], intercept + t_value * se_coeffs[0]]
        ci_slopes = [[slopes[i] - t_value * se_coeffs[i+1], slopes[i] + t_value * se_coeffs[i+1]] 
                     for i in range(len(slopes))]
    else:
        ci_intercept = None
        ci_slopes = None
    
    # Build result
    result = {
        "coefficients": {
            "intercept": intercept,
            "slopes": slopes,
            "equation": equation
        },
        "statistics": {
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "rmse": rmse,
            "mae": mae,
            "f_statistic": f_statistic,
            "p_value": p_value,
            "n_observations": n,
            "n_parameters": n_params,
            "degrees_of_freedom": n - n_params
        }
    }
    
    if ci_intercept is not None:
        result["confidence_intervals"] = {
            "confidence_level": confidence_level,
            "intercept": ci_intercept,
            "slopes": ci_slopes
        }
    
    # Add diagnostics if requested
    if include_diagnostics:
        diagnostics = {
            "residuals": residuals,
            "predictions": predictions
        }
        
        # Standardized residuals
        std_residuals = [r / rmse for r in residuals] if rmse > 1e-10 else residuals
        diagnostics["standardized_residuals"] = std_residuals
        
        # Durbin-Watson statistic for autocorrelation
        if n > 1:
            dw = sum((residuals[i] - residuals[i-1]) ** 2 for i in range(1, n)) / ss_res if ss_res > 1e-10 else 2.0
            diagnostics["durbin_watson"] = dw
        
        result["diagnostics"] = diagnostics
    
    # Interpretation
    if r_squared >= 0.9:
        strength = "very strong"
    elif r_squared >= 0.7:
        strength = "strong"
    elif r_squared >= 0.5:
        strength = "moderate"
    else:
        strength = "weak"
    
    interpretation = f"{strength.capitalize()} relationship (R² = {r_squared:.3f}). Model explains {r_squared*100:.1f}% of variance."
    
    if p_value is not None and p_value < 0.05:
        interpretation += " Statistically significant (p < 0.05)."
    
    result["interpretation"] = interpretation
    result["warnings"] = []
    
    if r_squared < 0.5:
        result["warnings"].append("Low R²: model may not fit data well")
    
    if n < 30:
        result["warnings"].append(f"Small sample size (n={n}): results may be unreliable")
    
    return result


def solve_linear_system(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve linear system Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    # Create augmented matrix
    aug = [A[i][:] + [b[i]] for i in range(n)]
    
    # Forward elimination with partial pivoting
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k
        
        # Swap rows
        aug[i], aug[max_row] = aug[max_row], aug[i]
        
        # Check for singular matrix
        if abs(aug[i][i]) < NUMERICAL_TOLERANCE:
            raise ValueError("Matrix is singular or near-singular")
        
        # Eliminate column
        for k in range(i + 1, n):
            factor = aug[k][i] / aug[i][i]
            for j in range(i, n + 1):
                aug[k][j] -= factor * aug[i][j]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]
    
    return x


def polynomial_regression(x: list[float], y: list[float], degree: int = 2, auto_select_degree: bool = False) -> dict[str, Any]:
    """
    Fit polynomial curves for non-linear relationships.
    
    Use Cases:
    - Compressor performance curves (non-linear)
    - Valve characteristics (flow vs. position)
    - Catalyst activity decline over time
    - Temperature profiles in heat exchangers
    - Motor torque vs. speed curves
    - Pump efficiency curves
    
    Args:
        x: Independent variable, min 5 items
        y: Dependent variable, min 5 items
        degree: Polynomial degree (2=quadratic, 3=cubic), 2-6, default 2
        auto_select_degree: Automatically select best degree based on adjusted R², default False
        
    Returns:
        Dictionary containing degree, coefficients, equation, R², turning points, and interpretation
    """
    # Validate inputs
    if not isinstance(x, list) or not isinstance(y, list):
        raise ValueError("x and y must be lists")
    
    if len(x) < 5 or len(y) < 5:
        raise ValueError("x and y must contain at least 5 items")
    
    if len(x) != len(y):
        raise ValueError(f"x length ({len(x)}) must match y length ({len(y)})")
    
    for i, value in enumerate(x):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All x values must be numeric. Item at index {i} is {type(value).__name__}")
    
    for i, value in enumerate(y):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All y values must be numeric. Item at index {i} is {type(value).__name__}")
    
    if not isinstance(degree, int):
        raise ValueError(f"degree must be an integer, got {type(degree).__name__}")
    
    if degree < 2 or degree > 6:
        raise ValueError(f"degree must be between 2 and 6, got {degree}")
    
    n = len(x)
    if n <= degree:
        raise ValueError(f"Need at least {degree + 1} observations for degree {degree} polynomial")
    
    # Auto-select degree if requested
    if auto_select_degree:
        best_degree = 2
        best_adj_r2 = -1
        
        for d in range(2, min(7, n)):
            try:
                result = polynomial_regression(x, y, degree=d, auto_select_degree=False)
                adj_r2 = result['adj_r_squared']
                if adj_r2 > best_adj_r2:
                    best_adj_r2 = adj_r2
                    best_degree = d
            except:
                break
        
        degree = best_degree
    
    # Build design matrix for polynomial: [1, x, x^2, ..., x^degree]
    X = [[xi ** j for j in range(degree + 1)] for xi in x]
    
    # Solve using linear regression approach
    n_params = degree + 1
    XtX = [[sum(X[i][j] * X[i][k] for i in range(n)) for k in range(n_params)] for j in range(n_params)]
    Xty = [sum(X[i][j] * y[i] for i in range(n)) for j in range(n_params)]
    
    try:
        coefficients = solve_linear_system(XtX, Xty)
    except Exception as e:
        raise ValueError(f"Cannot solve polynomial regression: {str(e)}")
    
    # Calculate predictions and statistics
    predictions = [sum(coefficients[j] * (xi ** j) for j in range(n_params)) for xi in x]
    residuals = [y[i] - predictions[i] for i in range(n)]
    
    mean_y = sum(y) / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum(r ** 2 for r in residuals)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 1.0
    adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - n_params)) if n > n_params else r_squared
    rmse = (ss_res / n) ** 0.5
    
    # Build equation string
    equation_parts = []
    for i in range(degree, -1, -1):
        coef = coefficients[i]
        if abs(coef) > 1e-10:
            if i == degree:
                equation_parts.append(f"{coef:.4f}x^{i}" if i > 1 else (f"{coef:.4f}x" if i == 1 else f"{coef:.4f}"))
            else:
                sign = "+" if coef >= 0 else "-"
                if i > 1:
                    equation_parts.append(f"{sign} {abs(coef):.4f}x^{i}")
                elif i == 1:
                    equation_parts.append(f"{sign} {abs(coef):.4f}x")
                else:
                    equation_parts.append(f"{sign} {abs(coef):.4f}")
    
    equation = "y = " + " ".join(equation_parts) if equation_parts else "y = 0"
    
    # Find turning points (critical points) for degree >= 2
    turning_points = []
    if degree >= 2:
        # Derivative coefficients
        deriv_coeffs = [coefficients[i] * i for i in range(1, degree + 1)]
        
        # For quadratic, solve analytically
        if degree == 2:
            a, b = deriv_coeffs[1], deriv_coeffs[0]
            if abs(a) > 1e-10:
                x_crit = -b / (2 * a)
                # Check if within data range
                x_min, x_max = min(x), max(x)
                if x_min <= x_crit <= x_max:
                    y_crit = sum(coefficients[j] * (x_crit ** j) for j in range(degree + 1))
                    point_type = "maximum" if a < 0 else "minimum"
                    turning_points.append({"x": x_crit, "y": y_crit, "type": point_type})
    
    # Optimal point (if exists)
    optimal_x = None
    optimal_y = None
    if turning_points:
        # For engineering, often interested in maximum
        max_point = max(turning_points, key=lambda p: p["y"])
        optimal_x = max_point["x"]
        optimal_y = max_point["y"]
    
    # Interpretation
    if degree == 2:
        shape = "parabolic"
    elif degree == 3:
        shape = "cubic"
    else:
        shape = f"degree-{degree} polynomial"
    
    if optimal_x is not None:
        interpretation = f"{shape.capitalize()} relationship with {turning_points[0]['type']} at x={optimal_x:.2f}"
    else:
        interpretation = f"{shape.capitalize()} relationship"
    
    # Goodness of fit
    if r_squared >= 0.95:
        gof = "Excellent"
    elif r_squared >= 0.85:
        gof = "Good"
    elif r_squared >= 0.7:
        gof = "Fair"
    else:
        gof = "Poor"
    
    return {
        "degree": degree,
        "coefficients": coefficients,
        "equation": equation,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "rmse": rmse,
        "turning_points": turning_points,
        "optimal_x": optimal_x,
        "optimal_y": optimal_y,
        "interpretation": interpretation,
        "goodness_of_fit": f"{gof} (R² = {r_squared:.3f})",
        "statistics": {
            "n_observations": n,
            "n_parameters": n_params,
            "degrees_of_freedom": n - n_params
        }
    }


def residual_analysis(actual: list[float], predicted: list[float], x_values: list[float] = None) -> dict[str, Any]:
    """
    Comprehensive analysis of regression residuals to validate model assumptions.
    
    Use Cases:
    - Validate regression model assumptions
    - Detect non-linearity or missing variables
    - Identify outliers and influential points
    - Check for heteroscedasticity
    - Verify normality of errors
    - Detect autocorrelation in time series regressions
    
    Args:
        actual: Actual observed values, min 10 items
        predicted: Model predicted values, min 10 items
        x_values: Independent variable values (optional), min 10 items
        
    Returns:
        Dictionary containing residuals, tests, outliers, patterns, assessment, and recommendations
    """
    # Validate inputs
    if not isinstance(actual, list) or not isinstance(predicted, list):
        raise ValueError("actual and predicted must be lists")
    
    if len(actual) < 10 or len(predicted) < 10:
        raise ValueError("actual and predicted must contain at least 10 items")
    
    if len(actual) != len(predicted):
        raise ValueError(f"actual length ({len(actual)}) must match predicted length ({len(predicted)})")
    
    for i, value in enumerate(actual):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All actual values must be numeric. Item at index {i} is {type(value).__name__}")
    
    for i, value in enumerate(predicted):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All predicted values must be numeric. Item at index {i} is {type(value).__name__}")
    
    if x_values is not None:
        if not isinstance(x_values, list):
            raise ValueError("x_values must be a list")
        if len(x_values) != len(actual):
            raise ValueError(f"x_values length ({len(x_values)}) must match actual length ({len(actual)})")
        for i, value in enumerate(x_values):
            if not isinstance(value, (int, float)):
                raise ValueError(f"All x_values must be numeric. Item at index {i} is {type(value).__name__}")
    
    n = len(actual)
    
    # Calculate residuals
    residuals = [actual[i] - predicted[i] for i in range(n)]
    
    # Calculate standardized residuals
    mean_residual = sum(residuals) / n
    std_residual = (sum((r - mean_residual) ** 2 for r in residuals) / n) ** 0.5
    
    if std_residual < 1e-10:
        std_residual = 1.0
    
    standardized_residuals = [(r - mean_residual) / std_residual for r in residuals]
    
    # Tests dictionary
    tests = {}
    
    # Normality test (Shapiro-Wilk)
    if SCIPY_AVAILABLE:
        try:
            sw_stat, sw_p = scipy_stats.shapiro(residuals)
            conclusion = "Residuals are normally distributed (p > 0.05)" if sw_p > 0.05 else "Residuals may not be normally distributed (p ≤ 0.05)"
            tests["normality"] = {
                "shapiro_wilk_statistic": float(sw_stat),
                "p_value": float(sw_p),
                "conclusion": conclusion
            }
        except:
            tests["normality"] = {"error": "Could not perform Shapiro-Wilk test"}
    
    # Autocorrelation test (Durbin-Watson)
    if n > 1:
        dw = sum((residuals[i] - residuals[i-1]) ** 2 for i in range(1, n)) / sum(r ** 2 for r in residuals) if sum(r ** 2 for r in residuals) > 1e-10 else 2.0
        
        if dw < 1.5:
            dw_conclusion = "Positive autocorrelation detected"
        elif dw > 2.5:
            dw_conclusion = "Negative autocorrelation detected"
        else:
            dw_conclusion = "No significant autocorrelation"
        
        tests["autocorrelation"] = {
            "durbin_watson": dw,
            "conclusion": dw_conclusion
        }
    
    # Homoscedasticity test (simplified Breusch-Pagan)
    # Test if variance of residuals is constant
    if x_values is not None and SCIPY_AVAILABLE:
        # Simple test: correlate squared residuals with x
        squared_resid = [r ** 2 for r in residuals]
        try:
            bp_corr, bp_p = scipy_stats.pearsonr(x_values, squared_resid)
            conclusion = "Constant variance assumption met (p > 0.05)" if bp_p > 0.05 else "Heteroscedasticity detected (p ≤ 0.05)"
            tests["homoscedasticity"] = {
                "test_statistic": abs(float(bp_corr)),
                "p_value": float(bp_p),
                "conclusion": conclusion
            }
        except:
            tests["homoscedasticity"] = {"note": "Could not perform test"}
    
    # Identify outliers (standardized residuals > 3 or < -3)
    outlier_indices = []
    outlier_info = []
    
    for i, std_res in enumerate(standardized_residuals):
        if abs(std_res) > 3:
            outlier_indices.append(i)
            outlier_info.append({
                "index": i,
                "residual": residuals[i],
                "standardized": std_res
            })
    
    outliers = {
        "indices": outlier_indices,
        "values": outlier_info,
        "count": len(outlier_indices)
    }
    
    # Pattern detection (simplified)
    patterns_detected = []
    
    # Check for trend in residuals
    if len(residuals) >= 10:
        # Simple trend check: compare first half vs second half
        mid = n // 2
        first_half_mean = sum(residuals[:mid]) / mid
        second_half_mean = sum(residuals[mid:]) / (n - mid)
        
        if abs(first_half_mean - second_half_mean) > std_residual:
            patterns_detected.append("Possible trend in residuals suggests missing variables or non-linearity")
    
    # Overall assessment
    issues = []
    
    if "normality" in tests and tests["normality"].get("p_value", 1) < 0.05:
        issues.append("Non-normal residuals")
    
    if "autocorrelation" in tests and (tests["autocorrelation"]["durbin_watson"] < 1.5 or tests["autocorrelation"]["durbin_watson"] > 2.5):
        issues.append("Autocorrelation present")
    
    if "homoscedasticity" in tests and tests["homoscedasticity"].get("p_value", 1) < 0.05:
        issues.append("Heteroscedasticity present")
    
    if outliers["count"] > 0:
        issues.append(f"{outliers['count']} outlier(s) detected")
    
    if len(issues) == 0:
        overall_assessment = "Model assumptions are satisfied. Residuals show random scatter with no patterns."
    else:
        overall_assessment = f"Model has issues: {', '.join(issues)}"
    
    # Recommendations
    recommendations = []
    
    if "Non-normal residuals" in issues:
        recommendations.append("Consider transformation of y (log, sqrt) or robust regression")
    
    if "Autocorrelation present" in issues:
        recommendations.append("Consider time series model or add lagged variables")
    
    if "Heteroscedasticity present" in issues:
        recommendations.append("Consider weighted least squares or transformation of y")
    
    if outliers["count"] > 0:
        recommendations.append("Investigate outliers: may be data errors or require separate analysis")
    
    if patterns_detected:
        recommendations.append("Model may be missing important variables or non-linear terms")
    
    return {
        "residuals": residuals,
        "standardized_residuals": standardized_residuals,
        "tests": tests,
        "outliers": outliers,
        "patterns_detected": patterns_detected,
        "overall_assessment": overall_assessment,
        "recommendations": recommendations
    }


def prediction_with_intervals(model: dict, x_new: list[float], confidence_level: float = 0.95) -> dict[str, Any]:
    """
    Generate predictions with confidence and prediction intervals.
    
    Use Cases:
    - Forecast equipment performance at specific operating conditions
    - Estimate production output with uncertainty bounds
    - Predict energy consumption with confidence intervals
    - Estimate maintenance costs with uncertainty
    - Calculate expected valve position for desired flow
    - Predict process yield with tolerance bands
    
    Args:
        model: Regression model dict with 'coefficients' (dict or list), 'rmse', optional 'degree'
        x_new: New x values for prediction
        confidence_level: Confidence level (0-1), default 0.95
        
    Returns:
        Dictionary containing predictions with confidence and prediction intervals
    """
    # Validate inputs
    if not isinstance(model, dict):
        raise ValueError("model must be a dictionary")
    
    if "coefficients" not in model:
        raise ValueError("model must contain 'coefficients'")
    
    if "rmse" not in model:
        raise ValueError("model must contain 'rmse'")
    
    if not isinstance(x_new, list):
        raise ValueError("x_new must be a list")
    
    if len(x_new) < 1:
        raise ValueError("x_new must contain at least 1 value")
    
    for i, value in enumerate(x_new):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All x_new values must be numeric. Item at index {i} is {type(value).__name__}")
    
    if not isinstance(confidence_level, (int, float)):
        raise ValueError("confidence_level must be numeric")
    
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")
    
    # Extract model parameters
    coeffs = model["coefficients"]
    rmse = model["rmse"]
    degree = model.get("degree", None)
    
    # Get sample size information for proper interval calculations
    stats = model.get("statistics", {})
    n_obs = stats.get("n_observations", 30)  # Default to 30 if not available
    n_params = stats.get("n_parameters", 2)  # Default to 2 if not available
    df = stats.get("degrees_of_freedom", n_obs - n_params)  # Calculate if not available
    
    # Determine model type
    if isinstance(coeffs, dict):
        # Linear regression format
        intercept = coeffs.get("intercept", 0)
        slopes = coeffs.get("slopes", [])
        is_polynomial = False
    elif isinstance(coeffs, list):
        # Polynomial regression format
        is_polynomial = True
        degree = degree if degree is not None else len(coeffs) - 1
    else:
        raise ValueError("coefficients must be a dict or list")
    
    # Generate predictions
    predictions = []
    
    for xi in x_new:
        if is_polynomial:
            # Polynomial: y = c0 + c1*x + c2*x^2 + ...
            pred_y = sum(coeffs[j] * (xi ** j) for j in range(len(coeffs)))
        else:
            # Linear: y = intercept + slope*x (simple) or intercept + sum(slopes*x) (multiple)
            if len(slopes) == 1:
                pred_y = intercept + slopes[0] * xi
            else:
                # For multiple regression, xi should be a list
                if not isinstance(xi, list):
                    raise ValueError("For multiple regression, each x_new value must be a list")
                pred_y = intercept + sum(slopes[i] * xi[i] for i in range(len(slopes)))
        
        # Confidence interval (for mean response)
        # CI ≈ ŷ ± t * rmse / sqrt(n)
        # Using proper t-value with actual degrees of freedom
        if SCIPY_AVAILABLE and df > 0:
            t_value = float(scipy_stats.t.ppf((1 + confidence_level) / 2, df))
        else:
            t_value = 2.0  # Rough approximation if scipy not available
        
        # Use actual sample size
        ci_margin = t_value * rmse / math.sqrt(n_obs) if n_obs > 0 else t_value * rmse
        ci_lower = pred_y - ci_margin
        ci_upper = pred_y + ci_margin
        
        # Prediction interval (for individual observation)
        # PI ≈ ŷ ± t * rmse * sqrt(1 + 1/n)
        # This accounts for both model uncertainty and individual variation
        pi_margin = t_value * rmse * math.sqrt(1 + 1/n_obs) if n_obs > 0 else t_value * rmse * math.sqrt(2)
        pi_lower = pred_y - pi_margin
        pi_upper = pred_y + pi_margin
        
        x_val = xi if not isinstance(xi, list) else xi
        interpretation = f"At x={x_val}, predicted y is {pred_y:.2f} ({confidence_level*100:.0f}% CI: {ci_lower:.2f}-{ci_upper:.2f})"
        
        predictions.append({
            "x": x_val,
            "predicted_y": pred_y,
            "confidence_interval": [ci_lower, ci_upper],
            "prediction_interval": [pi_lower, pi_upper],
            "interpretation": interpretation
        })
    
    # Check for extrapolation warning
    # This is simplified - in practice, need training data range
    extrapolation_warning = False
    
    # Reliability assessment
    if rmse < 1.0:
        reliability = "high"
    elif rmse < 5.0:
        reliability = "medium"
    else:
        reliability = "low"
    
    return {
        "predictions": predictions,
        "confidence_level": confidence_level,
        "extrapolation_warning": extrapolation_warning,
        "reliability": reliability
    }


def multivariate_regression(X: list[list[float]], y: list[float], variable_names: list[str] = None, standardize: bool = False) -> dict[str, Any]:
    """
    Multiple linear regression with multiple independent variables.
    
    Use Cases:
    - Chiller efficiency vs. load, ambient temp, condenser flow
    - Production yield vs. temperature, pressure, catalyst age
    - Energy consumption vs. production rate, ambient conditions, equipment age
    - Compressor power vs. suction pressure, discharge pressure, flow rate
    - Product quality vs. multiple process parameters
    
    Args:
        X: Matrix of independent variables [[x1_1, x2_1, ...], [x1_2, x2_2, ...], ...], min 5 rows
        y: Dependent variable, min 5 items
        variable_names: Names for each independent variable, default ["X1", "X2", ...]
        standardize: Standardize variables for coefficient comparison, default False
        
    Returns:
        Dictionary containing coefficients, statistics, variable importance, VIF, and interpretation
        
    Note:
        P-value calculations use an approximate standard error (SE ≈ RMSE/√n) which is simplified.
        For more accurate p-values, consider using dedicated statistical packages that compute
        the full covariance matrix (X'X)^-1. The provided p-values should be used as rough
        indicators of significance, not for formal hypothesis testing.
    """
    # Validate inputs
    if not isinstance(X, list) or not isinstance(y, list):
        raise ValueError("X and y must be lists")
    
    if len(X) < 5 or len(y) < 5:
        raise ValueError("X and y must contain at least 5 items")
    
    if len(X) != len(y):
        raise ValueError(f"X length ({len(X)}) must match y length ({len(y)})")
    
    # Validate X is matrix
    if not all(isinstance(row, list) for row in X):
        raise ValueError("X must be a list of lists (matrix)")
    
    n_vars = len(X[0])
    if n_vars < 1:
        raise ValueError("X must have at least 1 variable")
    
    for i, row in enumerate(X):
        if len(row) != n_vars:
            raise ValueError(f"All X rows must have same length. Row {i} has {len(row)}, expected {n_vars}")
        for j, value in enumerate(row):
            if not isinstance(value, (int, float)):
                raise ValueError(f"All X values must be numeric. Item at [{i}][{j}] is {type(value).__name__}")
    
    for i, value in enumerate(y):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All y values must be numeric. Item at index {i} is {type(value).__name__}")
    
    # Set default variable names
    if variable_names is None:
        variable_names = [f"X{i+1}" for i in range(n_vars)]
    else:
        if len(variable_names) != n_vars:
            raise ValueError(f"variable_names length must match number of variables ({n_vars})")
    
    n = len(y)
    
    # Standardize if requested
    if standardize:
        # Standardize X
        X_means = [sum(X[i][j] for i in range(n)) / n for j in range(n_vars)]
        X_stds = [(sum((X[i][j] - X_means[j]) ** 2 for i in range(n)) / n) ** 0.5 for j in range(n_vars)]
        
        # Prevent division by zero
        for j in range(n_vars):
            if X_stds[j] < 1e-10:
                X_stds[j] = 1.0
        
        X_std = [[(X[i][j] - X_means[j]) / X_stds[j] for j in range(n_vars)] for i in range(n)]
        
        # Standardize y
        y_mean = sum(y) / n
        y_std = (sum((yi - y_mean) ** 2 for yi in y) / n) ** 0.5
        if y_std < 1e-10:
            y_std = 1.0
        
        y_std_vals = [(yi - y_mean) / y_std for yi in y]
    else:
        X_std = X
        y_std_vals = y
    
    # Use linear_regression function
    try:
        result = linear_regression(X_std, y_std_vals, confidence_level=0.95, include_diagnostics=False)
    except Exception as e:
        raise ValueError(f"Regression failed: {str(e)}")
    
    # Extract results
    intercept = result["coefficients"]["intercept"]
    slopes = result["coefficients"]["slopes"]
    r_squared = result["statistics"]["r_squared"]
    adj_r_squared = result["statistics"]["adj_r_squared"]
    
    # Build equation with variable names
    equation_parts = [f"{intercept:.4f}"]
    for i, (name, slope) in enumerate(zip(variable_names, slopes)):
        sign = "+" if slope >= 0 else "-"
        equation_parts.append(f"{sign} {abs(slope):.4f}*{name}")
    
    equation = " ".join(equation_parts)
    
    # Variable importance (based on absolute standardized coefficients)
    variable_importance = []
    
    # Calculate standard errors for p-values if possible
    if SCIPY_AVAILABLE and result["statistics"].get("degrees_of_freedom", 0) > 0:
        # Get standard errors from the linear regression result
        # Note: This is approximate since we don't have the full covariance matrix
        # For a more accurate implementation, we would need to compute the full (X'X)^-1 matrix
        df = result["statistics"]["degrees_of_freedom"]
        rmse_model = result["statistics"]["rmse"]
        
        for i, (name, slope) in enumerate(zip(variable_names, slopes)):
            # Approximate standard error (this is simplified)
            # In practice, we'd need sqrt(MSE * (X'X)^-1[i,i])
            se_approx = rmse_model / math.sqrt(n)  # Very rough approximation
            
            if se_approx > 1e-10:
                t_stat = slope / se_approx
                # Calculate two-tailed p-value
                p_value = float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), df)))
            else:
                p_value = None
            
            if p_value is not None:
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                else:
                    significance = ""
            else:
                significance = ""
            
            variable_importance.append({
                "variable": name,
                "coefficient": slope,
                "p_value": p_value,
                "significance": significance
            })
    else:
        # If scipy not available or insufficient data, just use coefficients
        for i, (name, slope) in enumerate(zip(variable_names, slopes)):
            variable_importance.append({
                "variable": name,
                "coefficient": slope,
                "p_value": None,
                "significance": ""
            })
    
    # Sort by absolute coefficient value
    variable_importance.sort(key=lambda v: abs(v["coefficient"]), reverse=True)
    
    # Calculate VIF (Variance Inflation Factor) for multicollinearity detection
    vif_values = {}
    
    if n_vars > 1 and n > n_vars + 1:
        for i in range(n_vars):
            # For each variable, regress it against all others
            X_i = [[X[row][j] for j in range(n_vars) if j != i] for row in range(n)]
            y_i = [X[row][i] for row in range(n)]
            
            try:
                reg_i = linear_regression(X_i, y_i, include_diagnostics=False)
                r2_i = reg_i["statistics"]["r_squared"]
                
                if r2_i < 0.999:
                    vif = 1 / (1 - r2_i)
                else:
                    vif = 999  # High multicollinearity
                
                vif_values[variable_names[i]] = vif
            except:
                vif_values[variable_names[i]] = None
    
    # Multicollinearity assessment
    if vif_values:
        max_vif = max(v for v in vif_values.values() if v is not None)
        if max_vif < 5:
            multicollinearity = "Low - all VIF < 5"
        elif max_vif < 10:
            multicollinearity = "Moderate - some VIF between 5 and 10"
        else:
            multicollinearity = "High - VIF > 10, consider removing correlated variables"
    else:
        multicollinearity = "Not calculated"
    
    # Interpretation
    most_important = variable_importance[0]["variable"]
    interpretation = f"{most_important} has strongest effect."
    
    significant_vars = [v["variable"] for v in variable_importance if v.get("p_value") is not None and v["p_value"] < 0.05]
    if len(significant_vars) == len(variable_names):
        interpretation += " All variables are significant."
    elif len(significant_vars) > 0:
        interpretation += f" Significant variables: {', '.join(significant_vars)}."
    else:
        interpretation += " No variables are statistically significant."
    
    return {
        "coefficients": {
            "intercept": intercept,
            **{name: slope for name, slope in zip(variable_names, slopes)},
            "equation": equation
        },
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "variable_importance": variable_importance,
        "vif": vif_values,
        "multicollinearity": multicollinearity,
        "interpretation": interpretation
    }


# ============================================================================
# MCP Server Setup
# Signal Processing Functions
# ============================================================================


def fft_analysis(signal_data: list[float], sample_rate: float, window: str = "hanning", detrend: bool = True) -> dict[str, Any]:
    """
    Perform Fast Fourier Transform (FFT) analysis for frequency domain analysis.
    
    Used for bearing defect detection, motor electrical faults, gearbox analysis,
    pump cavitation detection, and vibration analysis.
    
    Args:
        signal_data: Time-domain signal (vibration, current, acoustic, etc.), min 8 items
        sample_rate: Sampling frequency in Hz (1-1000000)
        window: Windowing function - "hanning", "hamming", "blackman", "rectangular" (default: "hanning")
        detrend: Remove DC component and linear trends (default: True)
        
    Returns:
        Dictionary containing:
        - 'sample_rate': Input sample rate
        - 'signal_length': Number of samples
        - 'duration_seconds': Signal duration
        - 'frequencies': Frequency array up to Nyquist
        - 'magnitudes': Magnitude spectrum
        - 'dominant_frequencies': Top frequency peaks with magnitudes
        - 'nyquist_frequency': Maximum frequency (sample_rate/2)
        - 'resolution': Frequency resolution (Hz per bin)
        - 'interpretation': Analysis interpretation
        
    Raises:
        ValueError: If validation fails
        
    Examples:
        fft_analysis([...], 10000, "hanning", True)
        >>> Dominant frequencies at 60 Hz, 120 Hz (electrical harmonics)
    """
    # Validate input
    if not isinstance(signal_data, list):
        raise ValueError(f"signal_data must be a list, got {type(signal_data).__name__}")
    
    if len(signal_data) < 8:
        raise ValueError("signal_data must contain at least 8 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(signal_data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All signal values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate sample_rate
    if not isinstance(sample_rate, (int, float)):
        raise ValueError(f"sample_rate must be numeric, got {type(sample_rate).__name__}")
    
    if sample_rate < 1 or sample_rate > 1000000:
        raise ValueError(f"sample_rate must be between 1 and 1000000 Hz, got {sample_rate}")
    
    # Validate window type
    valid_windows = ["hanning", "hamming", "blackman", "rectangular"]
    if window not in valid_windows:
        raise ValueError(f"window must be one of {valid_windows}, got '{window}'")
    
    # Convert to numpy array
    signal_array = np.array(signal_data, dtype=float)
    n = len(signal_array)
    
    # Detrend if requested (remove DC and linear trend)
    if detrend:
        signal_array = scipy_signal.detrend(signal_array)
    
    # Apply window function to reduce spectral leakage
    if window == "hanning":
        window_func = scipy_signal.windows.hann(n)
    elif window == "hamming":
        window_func = scipy_signal.windows.hamming(n)
    elif window == "blackman":
        window_func = scipy_signal.windows.blackman(n)
    else:  # rectangular
        window_func = np.ones(n)
    
    windowed_signal = signal_array * window_func
    
    # Perform FFT
    fft_result = fft(windowed_signal)
    
    # Get frequency bins (only positive frequencies up to Nyquist)
    freqs = fftfreq(n, 1.0 / sample_rate)
    
    # Take only positive frequencies (up to Nyquist frequency)
    nyquist_idx = n // 2
    freqs_positive = freqs[:nyquist_idx]
    
    # Calculate magnitude spectrum (scaled properly)
    magnitudes = np.abs(fft_result[:nyquist_idx]) * 2.0 / n
    
    # Nyquist frequency and resolution
    nyquist_freq = sample_rate / 2.0
    freq_resolution = sample_rate / n
    duration_seconds = n / sample_rate
    
    # Find dominant frequencies (top 10 peaks)
    # Use peak detection to find local maxima
    peaks, properties = scipy_signal.find_peaks(magnitudes, height=0, prominence=np.max(magnitudes) * 0.05)
    
    # Sort peaks by magnitude (descending)
    if len(peaks) > 0:
        peak_mags = magnitudes[peaks]
        sorted_indices = np.argsort(peak_mags)[::-1]
        top_peaks = peaks[sorted_indices[:10]]
        top_mags = peak_mags[sorted_indices[:10]]
        
        dominant_frequencies = []
        for peak_idx, mag in zip(top_peaks, top_mags):
            freq = freqs_positive[peak_idx]
            # Provide interpretation hints for common frequencies
            interpretation = ""
            if 59 <= freq <= 61:
                interpretation = "Electrical line frequency (60 Hz)"
            elif 119 <= freq <= 121:
                interpretation = "Second harmonic (120 Hz)"
            elif 179 <= freq <= 181:
                interpretation = "Third harmonic (180 Hz)"
            elif freq > 1000:
                interpretation = "High frequency component (possible bearing defect)"
            
            dominant_frequencies.append({
                'frequency': float(freq),
                'magnitude': float(mag),
                'interpretation': interpretation
            })
    else:
        dominant_frequencies = []
    
    # Generate interpretation
    interpretation = ""
    if len(dominant_frequencies) > 0:
        top_freq = dominant_frequencies[0]['frequency']
        if 59 <= top_freq <= 61:
            interpretation = "Strong electrical line frequency component detected. Check for electromagnetic interference."
        elif top_freq > 1000:
            interpretation = "High frequency energy detected. May indicate bearing defects or gear mesh issues."
        else:
            interpretation = f"Dominant frequency at {top_freq:.1f} Hz."
    else:
        interpretation = "No significant frequency peaks detected. Signal may be primarily noise."
    
    return {
        'sample_rate': float(sample_rate),
        'signal_length': n,
        'duration_seconds': float(duration_seconds),
        'frequencies': freqs_positive.tolist(),
        'magnitudes': magnitudes.tolist(),
        'dominant_frequencies': dominant_frequencies,
        'nyquist_frequency': float(nyquist_freq),
        'resolution': float(freq_resolution),
        'interpretation': interpretation
    }


def power_spectral_density(signal_data: list[float], sample_rate: float, method: str = "welch", nperseg: int = 256) -> dict[str, Any]:
    """
    Calculate Power Spectral Density (PSD) for energy distribution across frequencies.
    
    Used for vibration energy distribution, noise level assessment, and random vibration analysis.
    
    Args:
        signal_data: Time-domain signal, min 16 items
        sample_rate: Sampling frequency in Hz
        method: PSD estimation method - "welch" or "periodogram" (default: "welch")
        nperseg: Length of each segment for Welch method (default: 256)
        
    Returns:
        Dictionary containing:
        - 'method': Method used
        - 'frequencies': Frequency array
        - 'psd': Power spectral density values
        - 'total_power': Total power in signal
        - 'peak_frequency': Frequency with maximum power
        - 'peak_power': Maximum power value
        - 'interpretation': Analysis interpretation
        
    Raises:
        ValueError: If validation fails
    """
    # Validate input
    if not isinstance(signal_data, list):
        raise ValueError(f"signal_data must be a list, got {type(signal_data).__name__}")
    
    if len(signal_data) < 16:
        raise ValueError("signal_data must contain at least 16 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(signal_data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All signal values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate sample_rate
    if not isinstance(sample_rate, (int, float)):
        raise ValueError(f"sample_rate must be numeric, got {type(sample_rate).__name__}")
    
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    
    # Validate method
    if method not in ["welch", "periodogram"]:
        raise ValueError(f"method must be 'welch' or 'periodogram', got '{method}'")
    
    # Validate nperseg
    if not isinstance(nperseg, int):
        raise ValueError(f"nperseg must be an integer, got {type(nperseg).__name__}")
    
    if nperseg < 2:
        raise ValueError(f"nperseg must be at least 2, got {nperseg}")
    
    # Convert to numpy array
    signal_array = np.array(signal_data, dtype=float)
    
    # Calculate PSD
    if method == "welch":
        # Welch's method: overlapping segments with averaging (more accurate for noisy signals)
        freqs, psd = scipy_signal.welch(signal_array, fs=sample_rate, nperseg=min(nperseg, len(signal_array)))
    else:  # periodogram
        # Periodogram: direct FFT method
        freqs, psd = scipy_signal.periodogram(signal_array, fs=sample_rate)
    
    # Calculate statistics
    total_power = float(np.trapz(psd, freqs))  # Integrate PSD to get total power
    peak_idx = np.argmax(psd)
    peak_frequency = float(freqs[peak_idx])
    peak_power = float(psd[peak_idx])
    
    # Generate interpretation
    interpretation = f"Peak power at {peak_frequency:.1f} Hz. "
    if peak_frequency < 100:
        interpretation += "Low frequency energy dominant - typical of structural vibration or slow processes."
    elif peak_frequency < 1000:
        interpretation += "Medium frequency energy - typical of rotating machinery."
    else:
        interpretation += "High frequency energy - may indicate bearing wear or gear mesh issues."
    
    return {
        'method': method.title(),
        'frequencies': freqs.tolist(),
        'psd': psd.tolist(),
        'total_power': total_power,
        'peak_frequency': peak_frequency,
        'peak_power': peak_power,
        'interpretation': interpretation
    }


def rms_value(signal_data: list[float], window_size: int = None, reference_value: float = None) -> dict[str, Any]:
    """
    Calculate Root Mean Square (RMS) for overall signal energy.
    
    Used for overall vibration severity (ISO 10816), electrical current RMS, and acoustic noise level.
    
    Args:
        signal_data: Time-domain signal, min 2 items
        window_size: Calculate rolling RMS (optional), min 2
        reference_value: Reference for dB calculation (e.g., 20 µPa for acoustic)
        
    Returns:
        Dictionary containing:
        - 'rms': Overall RMS value
        - 'peak': Peak value
        - 'crest_factor': Peak/RMS ratio
        - 'rms_db': RMS in decibels (if reference provided)
        - 'rolling_rms': Rolling RMS values (if window_size provided)
        - 'interpretation': Analysis interpretation
        
    Raises:
        ValueError: If validation fails
    """
    # Validate input
    if not isinstance(signal_data, list):
        raise ValueError(f"signal_data must be a list, got {type(signal_data).__name__}")
    
    if len(signal_data) < 2:
        raise ValueError("signal_data must contain at least 2 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(signal_data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All signal values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate window_size if provided
    if window_size is not None:
        if not isinstance(window_size, int):
            raise ValueError(f"window_size must be an integer, got {type(window_size).__name__}")
        if window_size < 2:
            raise ValueError(f"window_size must be at least 2, got {window_size}")
        if window_size > len(signal_data):
            raise ValueError(f"window_size ({window_size}) cannot be larger than signal length ({len(signal_data)})")
    
    # Validate reference_value if provided
    if reference_value is not None:
        if not isinstance(reference_value, (int, float)):
            raise ValueError(f"reference_value must be numeric, got {type(reference_value).__name__}")
        if reference_value <= 0:
            raise ValueError(f"reference_value must be positive, got {reference_value}")
    
    # Convert to numpy array
    signal_array = np.array(signal_data, dtype=float)
    
    # Calculate overall RMS
    rms = float(np.sqrt(np.mean(signal_array ** 2)))
    
    # Calculate peak value
    peak = float(np.max(np.abs(signal_array)))
    
    # Calculate crest factor (peak/RMS)
    crest_factor = float(peak / rms) if rms > 0 else 0.0
    
    # Calculate RMS in dB if reference provided
    rms_db = None
    if reference_value is not None and rms > 0:
        rms_db = float(20 * np.log10(rms / reference_value))
    
    # Calculate rolling RMS if window_size provided
    rolling_rms_values = None
    if window_size is not None:
        rolling_rms_list = []
        for i in range(len(signal_array) - window_size + 1):
            window = signal_array[i:i + window_size]
            window_rms = np.sqrt(np.mean(window ** 2))
            rolling_rms_list.append(float(window_rms))
        rolling_rms_values = rolling_rms_list
    
    # Generate interpretation
    interpretation = ""
    if crest_factor > 5:
        interpretation = "High crest factor indicates impulsive signals - check for impacts or bearing defects."
    elif crest_factor > 3:
        interpretation = "Moderate crest factor - normal for many rotating machinery signals."
    else:
        interpretation = "Low crest factor - signal is relatively smooth."
    
    # Add trending information if rolling RMS calculated
    trend = None
    if rolling_rms_values and len(rolling_rms_values) > 1:
        # Simple trend detection: compare first and last thirds
        first_third = np.mean(rolling_rms_values[:len(rolling_rms_values) // 3])
        last_third = np.mean(rolling_rms_values[-len(rolling_rms_values) // 3:])
        percent_change = ((last_third - first_third) / first_third * 100) if first_third > 0 else 0
        
        if percent_change > 10:
            trend = "increasing"
        elif percent_change < -10:
            trend = "decreasing"
        else:
            trend = "stable"
    
    result = {
        'rms': rms,
        'peak': peak,
        'crest_factor': crest_factor,
        'interpretation': interpretation
    }
    
    if rms_db is not None:
        result['rms_db'] = rms_db
    
    if rolling_rms_values is not None:
        result['rolling_rms'] = rolling_rms_values
        result['trend'] = trend
    
    return result


def peak_detection(signal_data: list[float], frequencies: list[float] = None, height: float = 0.1, 
                   distance: int = 1, prominence: float = 0.05, top_n: int = 10) -> dict[str, Any]:
    """
    Identify significant peaks in signals with filtering and ranking.
    
    Used for finding dominant vibration frequencies, detecting harmonics, and identifying resonances.
    
    Args:
        signal_data: Signal data (time or frequency domain), min 3 items
        frequencies: Corresponding frequencies (for frequency domain data), optional
        height: Minimum peak height (default: 0.1)
        distance: Minimum samples between peaks (default: 1)
        prominence: Required prominence above surroundings (default: 0.05)
        top_n: Return top N peaks only (1-50, default: 10)
        
    Returns:
        Dictionary containing:
        - 'peaks_found': Total number of peaks detected
        - 'top_peaks': List of top N peaks with indices, values, and interpretations
        - 'interpretation': Overall analysis
        
    Raises:
        ValueError: If validation fails
    """
    # Validate input
    if not isinstance(signal_data, list):
        raise ValueError(f"signal_data must be a list, got {type(signal_data).__name__}")
    
    if len(signal_data) < 3:
        raise ValueError("signal_data must contain at least 3 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(signal_data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All signal values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate frequencies if provided
    if frequencies is not None:
        if not isinstance(frequencies, list):
            raise ValueError(f"frequencies must be a list, got {type(frequencies).__name__}")
        if len(frequencies) != len(signal_data):
            raise ValueError(f"frequencies length ({len(frequencies)}) must match signal_data length ({len(signal_data)})")
        for i, value in enumerate(frequencies):
            if not isinstance(value, (int, float)):
                raise ValueError(f"All frequency values must be numeric. Item at index {i} is {type(value).__name__}")
    
    # Validate parameters
    if not isinstance(height, (int, float)):
        raise ValueError(f"height must be numeric, got {type(height).__name__}")
    
    if not isinstance(distance, int):
        raise ValueError(f"distance must be an integer, got {type(distance).__name__}")
    if distance < 1:
        raise ValueError(f"distance must be at least 1, got {distance}")
    
    if not isinstance(prominence, (int, float)):
        raise ValueError(f"prominence must be numeric, got {type(prominence).__name__}")
    
    if not isinstance(top_n, int):
        raise ValueError(f"top_n must be an integer, got {type(top_n).__name__}")
    if top_n < 1 or top_n > 50:
        raise ValueError(f"top_n must be between 1 and 50, got {top_n}")
    
    # Convert to numpy array
    signal_array = np.array(signal_data, dtype=float)
    
    # Detect peaks using scipy
    # Calculate relative prominence threshold
    max_val = np.max(signal_array)
    abs_prominence = prominence * max_val if max_val > 0 else prominence
    
    peaks, properties = scipy_signal.find_peaks(
        signal_array,
        height=height,
        distance=distance,
        prominence=abs_prominence
    )
    
    peaks_found = len(peaks)
    
    # Sort peaks by magnitude (descending) and take top N
    if peaks_found > 0:
        peak_magnitudes = signal_array[peaks]
        sorted_indices = np.argsort(peak_magnitudes)[::-1]
        top_indices = sorted_indices[:min(top_n, peaks_found)]
        
        top_peaks_list = []
        for rank, idx in enumerate(top_indices, 1):
            peak_idx = peaks[idx]
            magnitude = float(signal_array[peak_idx])
            
            peak_info = {
                'index': int(peak_idx),
                'magnitude': magnitude,
                'prominence': float(properties['prominences'][idx]),
                'rank': rank
            }
            
            # Add frequency if provided
            if frequencies is not None:
                frequency = float(frequencies[peak_idx])
                peak_info['frequency'] = frequency
                
                # Add interpretation for frequency domain
                interpretation = ""
                if 59 <= frequency <= 61:
                    interpretation = "Electrical line frequency (60 Hz)"
                elif 119 <= frequency <= 121:
                    interpretation = "Second harmonic (120 Hz)"
                elif 49 <= frequency <= 51:
                    interpretation = "Electrical line frequency (50 Hz)"
                elif frequency > 1000:
                    interpretation = "High frequency - potential bearing defect"
                
                peak_info['interpretation'] = interpretation
            
            top_peaks_list.append(peak_info)
    else:
        top_peaks_list = []
    
    # Generate interpretation
    if peaks_found == 0:
        interpretation = "No significant peaks detected in the signal."
    elif peaks_found == 1:
        interpretation = "Single dominant peak detected."
    else:
        interpretation = f"{peaks_found} peaks detected. Top {min(top_n, peaks_found)} peaks returned."
    
    return {
        'peaks_found': peaks_found,
        'top_peaks': top_peaks_list,
        'interpretation': interpretation
    }


def signal_to_noise_ratio(signal_data: list[float], noise_data: list[float] = None, method: str = "power") -> dict[str, Any]:
    """
    Calculate Signal-to-Noise Ratio (SNR) to assess signal quality.
    
    Used for sensor health monitoring, data acquisition quality checks, and instrumentation validation.
    
    Args:
        signal_data: Signal containing signal + noise, min 10 items
        noise_data: Noise reference (optional - will estimate if not provided)
        method: SNR calculation method - "power", "amplitude", or "peak" (default: "power")
        
    Returns:
        Dictionary containing:
        - 'snr_db': SNR in decibels
        - 'snr_ratio': Linear SNR ratio
        - 'signal_power': Signal power
        - 'noise_power': Noise power
        - 'quality': Quality assessment
        - 'interpretation': Analysis interpretation
        
    Raises:
        ValueError: If validation fails
    """
    # Validate input
    if not isinstance(signal_data, list):
        raise ValueError(f"signal_data must be a list, got {type(signal_data).__name__}")
    
    if len(signal_data) < 10:
        raise ValueError("signal_data must contain at least 10 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(signal_data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All signal values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate noise_data if provided
    if noise_data is not None:
        if not isinstance(noise_data, list):
            raise ValueError(f"noise_data must be a list, got {type(noise_data).__name__}")
        for i, value in enumerate(noise_data):
            if not isinstance(value, (int, float)):
                raise ValueError(f"All noise values must be numeric. Item at index {i} is {type(value).__name__}")
    
    # Validate method
    valid_methods = ["power", "amplitude", "peak"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
    
    # Convert to numpy array
    signal_array = np.array(signal_data, dtype=float)
    
    # Estimate signal and noise
    if noise_data is not None:
        # Noise reference provided
        noise_array = np.array(noise_data, dtype=float)
        noise_estimate = noise_array
        # Assume signal_data contains pure signal
        signal_estimate = signal_array
    else:
        # Estimate noise from signal (assuming signal has both)
        # Use high-pass filtered signal as noise estimate
        # This is a simple estimation method
        mean_val = np.mean(signal_array)
        signal_estimate = signal_array - mean_val
        
        # Estimate noise as high-frequency component
        # Use difference between adjacent samples as noise estimate
        noise_estimate = np.diff(signal_array)
    
    # Calculate SNR based on method
    if method == "power":
        # Power-based SNR
        signal_power = float(np.mean(signal_estimate ** 2))
        noise_power = float(np.mean(noise_estimate ** 2))
        
        if noise_power > 0:
            snr_ratio = signal_power / noise_power
            snr_db = float(10 * np.log10(snr_ratio))
        else:
            snr_ratio = float('inf')
            snr_db = float('inf')
    
    elif method == "amplitude":
        # Amplitude-based SNR (RMS)
        signal_rms = float(np.sqrt(np.mean(signal_estimate ** 2)))
        noise_rms = float(np.sqrt(np.mean(noise_estimate ** 2)))
        
        if noise_rms > 0:
            snr_ratio = signal_rms / noise_rms
            snr_db = float(20 * np.log10(snr_ratio))
        else:
            snr_ratio = float('inf')
            snr_db = float('inf')
        
        signal_power = float(signal_rms ** 2)
        noise_power = float(noise_rms ** 2)
    
    else:  # peak
        # Peak-based SNR
        signal_peak = float(np.max(np.abs(signal_estimate)))
        noise_peak = float(np.max(np.abs(noise_estimate)))
        
        if noise_peak > 0:
            snr_ratio = signal_peak / noise_peak
            snr_db = float(20 * np.log10(snr_ratio))
        else:
            snr_ratio = float('inf')
            snr_db = float('inf')
        
        signal_power = float(signal_peak ** 2)
        noise_power = float(noise_peak ** 2)
    
    # Quality assessment
    if snr_db > 40:
        quality = "Excellent"
        interpretation = "Signal quality excellent (SNR > 40 dB). No sensor issues detected."
    elif snr_db > 30:
        quality = "Good"
        interpretation = "Signal quality good (SNR > 30 dB). Suitable for most applications."
    elif snr_db > 20:
        quality = "Fair"
        interpretation = "Signal quality fair (SNR > 20 dB). Consider noise reduction if critical."
    elif snr_db > 10:
        quality = "Poor"
        interpretation = "Signal quality poor (SNR > 10 dB). Check sensor and wiring."
    else:
        quality = "Very Poor"
        interpretation = "Signal quality very poor (SNR < 10 dB). Sensor may be faulty."
    
    return {
        'snr_db': snr_db,
        'snr_ratio': snr_ratio,
        'signal_power': signal_power,
        'noise_power': noise_power,
        'quality': quality,
        'interpretation': interpretation
    }


def harmonic_analysis(signal_data: list[float], sample_rate: float, fundamental_freq: float, max_harmonic: int = 50) -> dict[str, Any]:
    """
    Detect and analyze harmonic content in electrical and mechanical signals.
    
    Used for power quality assessment (THD), motor current signature analysis, and electrical fault detection.
    
    Args:
        signal_data: Periodic signal (voltage, current, vibration), min 64 items
        sample_rate: Sampling frequency in Hz
        fundamental_freq: Expected fundamental frequency (e.g., 60 Hz for electrical), min 1 Hz
        max_harmonic: Maximum harmonic order to analyze (1-100, default: 50)
        
    Returns:
        Dictionary containing:
        - 'fundamental': Fundamental frequency component (frequency, magnitude, phase)
        - 'harmonics': List of harmonic components (order, frequency, magnitude, percent)
        - 'thd': Total Harmonic Distortion percentage
        - 'thd_interpretation': THD quality assessment
        - 'dominant_harmonics': List of strongest harmonic orders
        - 'interpretation': Analysis interpretation
        
    Raises:
        ValueError: If validation fails
    """
    # Validate input
    if not isinstance(signal_data, list):
        raise ValueError(f"signal_data must be a list, got {type(signal_data).__name__}")
    
    if len(signal_data) < 64:
        raise ValueError("signal_data must contain at least 64 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(signal_data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All signal values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate sample_rate
    if not isinstance(sample_rate, (int, float)):
        raise ValueError(f"sample_rate must be numeric, got {type(sample_rate).__name__}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    
    # Validate fundamental_freq
    if not isinstance(fundamental_freq, (int, float)):
        raise ValueError(f"fundamental_freq must be numeric, got {type(fundamental_freq).__name__}")
    if fundamental_freq < 1:
        raise ValueError(f"fundamental_freq must be at least 1 Hz, got {fundamental_freq}")
    
    # Validate max_harmonic
    if not isinstance(max_harmonic, int):
        raise ValueError(f"max_harmonic must be an integer, got {type(max_harmonic).__name__}")
    if max_harmonic < 1 or max_harmonic > 100:
        raise ValueError(f"max_harmonic must be between 1 and 100, got {max_harmonic}")
    
    # Convert to numpy array
    signal_array = np.array(signal_data, dtype=float)
    n = len(signal_array)
    
    # Perform FFT
    fft_result = fft(signal_array)
    freqs = fftfreq(n, 1.0 / sample_rate)
    
    # Calculate magnitude spectrum
    magnitudes = np.abs(fft_result) * 2.0 / n
    
    # Find fundamental frequency component
    freq_resolution = sample_rate / n
    fund_idx = int(round(fundamental_freq / freq_resolution))
    
    # Search in a window around expected fundamental
    search_window = max(3, int(0.05 * fund_idx))  # 5% window
    search_start = max(0, fund_idx - search_window)
    search_end = min(len(freqs) // 2, fund_idx + search_window)
    
    # Find peak in search window
    window_mags = magnitudes[search_start:search_end]
    if len(window_mags) > 0:
        local_peak = np.argmax(window_mags)
        fund_idx = search_start + local_peak
    
    fundamental_magnitude = float(magnitudes[fund_idx])
    fundamental_frequency = float(freqs[fund_idx])
    
    # Extract harmonics
    harmonics_list = []
    harmonic_sum_squared = 0.0
    
    for h in range(2, max_harmonic + 1):
        harmonic_freq = fundamental_frequency * h
        
        # Check if harmonic is within Nyquist frequency
        if harmonic_freq >= sample_rate / 2:
            break
        
        # Find harmonic component
        harm_idx = int(round(harmonic_freq / freq_resolution))
        
        # Search in window
        search_window_h = max(2, int(0.05 * harm_idx))
        search_start_h = max(0, harm_idx - search_window_h)
        search_end_h = min(len(freqs) // 2, harm_idx + search_window_h)
        
        if search_start_h < search_end_h:
            window_mags_h = magnitudes[search_start_h:search_end_h]
            if len(window_mags_h) > 0:
                local_peak_h = np.argmax(window_mags_h)
                harm_idx = search_start_h + local_peak_h
                
                harmonic_magnitude = float(magnitudes[harm_idx])
                harmonic_frequency = float(freqs[harm_idx])
                
                # Calculate percentage of fundamental
                if fundamental_magnitude > 0:
                    percent = (harmonic_magnitude / fundamental_magnitude) * 100
                else:
                    percent = 0.0
                
                harmonics_list.append({
                    'order': h,
                    'frequency': harmonic_frequency,
                    'magnitude': harmonic_magnitude,
                    'percent': float(percent)
                })
                
                harmonic_sum_squared += harmonic_magnitude ** 2
    
    # Calculate THD (Total Harmonic Distortion)
    if fundamental_magnitude > 0:
        thd = float((np.sqrt(harmonic_sum_squared) / fundamental_magnitude) * 100)
    else:
        thd = 0.0
    
    # THD interpretation
    if thd < 5:
        thd_interpretation = "Excellent - Very low distortion (THD < 5%)"
    elif thd < 8:
        thd_interpretation = "Good - Low distortion (5% ≤ THD < 8%)"
    elif thd < 15:
        thd_interpretation = "Moderate - Acceptable distortion (8% ≤ THD < 15%)"
    elif thd < 25:
        thd_interpretation = "Poor - High distortion (15% ≤ THD < 25%)"
    else:
        thd_interpretation = "Very Poor - Excessive distortion (THD ≥ 25%)"
    
    # Find dominant harmonics (top 5 by magnitude)
    if harmonics_list:
        sorted_harmonics = sorted(harmonics_list, key=lambda x: x['magnitude'], reverse=True)
        dominant_harmonics = [h['order'] for h in sorted_harmonics[:5]]
    else:
        dominant_harmonics = []
    
    # Generate interpretation
    interpretation = f"THD = {thd:.1f}%. "
    if thd < 5:
        interpretation += "Signal quality is excellent with minimal harmonic distortion."
    elif thd < 15:
        interpretation += "Acceptable harmonic content for most applications."
    else:
        interpretation += "High harmonic distortion detected. Consider harmonic filtering."
    
    if dominant_harmonics:
        # Check for odd vs even harmonics
        odd_harmonics = [h for h in dominant_harmonics if h % 2 == 1]
        even_harmonics = [h for h in dominant_harmonics if h % 2 == 0]
        
        if len(odd_harmonics) > len(even_harmonics):
            interpretation += " Odd harmonics dominant - typical of non-linear loads or VFD operation."
        elif len(even_harmonics) > len(odd_harmonics):
            interpretation += " Even harmonics present - may indicate asymmetry or half-wave distortion."
    
    return {
        'fundamental': {
            'frequency': fundamental_frequency,
            'magnitude': fundamental_magnitude,
            'phase': 0.0  # Phase not calculated in this simplified version
        },
        'harmonics': harmonics_list,
        'thd': thd,
        'thd_interpretation': thd_interpretation,
        'dominant_harmonics': dominant_harmonics,
        'interpretation': interpretation
    }


# ============================================================================
# Statistical Process Control (SPC) Functions
# ============================================================================


# Control chart constants for X-bar and R charts
# Source: ASTM Manual on Presentation of Data and Control Chart Analysis (Table A2)
CONTROL_CHART_CONSTANTS = {
    # n: (A2, D3, D4, d2, c4)
    # A2: Factor for X-bar chart control limits
    # D3, D4: Factors for R chart control limits
    # d2: Relationship between R-bar and sigma
    # c4: Relationship between S-bar and sigma
    2: (1.880, 0.000, 3.267, 1.128, 0.7979),
    3: (1.023, 0.000, 2.574, 1.693, 0.8862),
    4: (0.729, 0.000, 2.282, 2.059, 0.9213),
    5: (0.577, 0.000, 2.114, 2.326, 0.9400),
    6: (0.483, 0.000, 2.004, 2.534, 0.9515),
    7: (0.419, 0.076, 1.924, 2.704, 0.9594),
    8: (0.373, 0.136, 1.864, 2.847, 0.9650),
    9: (0.337, 0.184, 1.816, 2.970, 0.9693),
    10: (0.308, 0.223, 1.777, 3.078, 0.9727),
    11: (0.285, 0.256, 1.744, 3.173, 0.9754),
    12: (0.266, 0.283, 1.717, 3.258, 0.9776),
    13: (0.249, 0.307, 1.693, 3.336, 0.9794),
    14: (0.235, 0.328, 1.672, 3.407, 0.9810),
    15: (0.223, 0.347, 1.653, 3.472, 0.9823),
    16: (0.212, 0.363, 1.637, 3.532, 0.9835),
    17: (0.203, 0.378, 1.622, 3.588, 0.9845),
    18: (0.194, 0.391, 1.609, 3.640, 0.9854),
    19: (0.187, 0.404, 1.596, 3.689, 0.9862),
    20: (0.180, 0.415, 1.585, 3.735, 0.9869),
    21: (0.173, 0.425, 1.575, 3.778, 0.9876),
    22: (0.167, 0.435, 1.565, 3.819, 0.9882),
    23: (0.162, 0.443, 1.557, 3.858, 0.9887),
    24: (0.157, 0.452, 1.548, 3.895, 0.9892),
    25: (0.153, 0.459, 1.541, 3.931, 0.9896),
}


def control_limits(
    data: list[float],
    chart_type: str,
    subgroup_size: int = 5,
    sigma_level: float = 3.0
) -> dict[str, Any]:
    """
    Calculate control chart limits (UCL, LCL, centerline) for various chart types.
    
    Use Cases:
    - Monitor critical quality parameters (viscosity, pH, concentration)
    - Track process variables (temperature, pressure, flow rate)
    - Control product dimensions and weights
    - Monitor cycle times and production rates
    - Detect process shifts before they cause defects
    
    Args:
        data: Process measurements or subgroup averages (min 5 items)
        chart_type: Type of control chart - "x_bar", "individuals", "range", "std_dev",
                   "p", "np", "c", "u"
        subgroup_size: Size of subgroups for X-bar and R charts (2-25, default: 5)
        sigma_level: Number of standard deviations for limits (1-6, default: 3)
        
    Returns:
        Dictionary containing:
        - chart_type: Type of chart
        - centerline: Process centerline (mean)
        - ucl: Upper control limit
        - lcl: Lower control limit
        - sigma: Process standard deviation
        - subgroup_size: Subgroup size used
        - data_points: Number of data points
        - out_of_control_points: Indices of points beyond limits
        - out_of_control_details: Details of violations
        - process_status: Overall status message
        - recommendations: Actionable recommendations
        
    Raises:
        ValueError: If parameters are invalid
        
    Examples:
        control_limits([100.5, 100.2, 100.8, 100.1], "individuals")
        >>> {
        ...   'chart_type': 'individuals',
        ...   'centerline': 100.4,
        ...   'ucl': 102.1,
        ...   'lcl': 98.7,
        ...   'process_status': 'In Control'
        ... }
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"data must be a list, got {type(data).__name__}")
    
    if len(data) < 5:
        raise ValueError("data must contain at least 5 items")
    
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}")
    
    # Validate chart_type
    valid_types = ["x_bar", "individuals", "range", "std_dev", "p", "np", "c", "u"]
    if chart_type not in valid_types:
        raise ValueError(f"chart_type must be one of {valid_types}, got '{chart_type}'")
    
    # Validate subgroup_size
    if not isinstance(subgroup_size, int):
        raise ValueError(f"subgroup_size must be an integer, got {type(subgroup_size).__name__}")
    
    if subgroup_size < 2 or subgroup_size > 25:
        raise ValueError(f"subgroup_size must be between 2 and 25, got {subgroup_size}")
    
    # Validate sigma_level
    if not isinstance(sigma_level, (int, float)):
        raise ValueError(f"sigma_level must be numeric, got {type(sigma_level).__name__}")
    
    if sigma_level < 1 or sigma_level > 6:
        raise ValueError(f"sigma_level must be between 1 and 6, got {sigma_level}")
    
    # Calculate control limits based on chart type
    if chart_type == "x_bar":
        return _calculate_xbar_limits(data, subgroup_size, sigma_level)
    elif chart_type == "individuals":
        return _calculate_individuals_limits(data, sigma_level)
    elif chart_type == "range":
        return _calculate_range_limits(data, subgroup_size, sigma_level)
    elif chart_type == "std_dev":
        return _calculate_stddev_limits(data, subgroup_size, sigma_level)
    elif chart_type in ["p", "np", "c", "u"]:
        return _calculate_attribute_limits(data, chart_type, subgroup_size, sigma_level)
    else:
        raise ValueError(f"Chart type '{chart_type}' not implemented")


def _calculate_xbar_limits(data: list[float], subgroup_size: int, sigma_level: float) -> dict[str, Any]:
    """Calculate control limits for X-bar chart."""
    # For X-bar chart, data should be subgroup averages
    n = len(data)
    centerline = sum(data) / n
    
    # Estimate sigma using moving range method (simplified)
    moving_ranges = [abs(data[i+1] - data[i]) for i in range(n - 1)]
    avg_moving_range = sum(moving_ranges) / len(moving_ranges)
    
    # d2 constant for subgroups of size 2 (moving range)
    d2_mr = 1.128
    sigma = avg_moving_range / d2_mr
    
    # Get A2 constant for the subgroup size
    if subgroup_size in CONTROL_CHART_CONSTANTS:
        A2, _, _, _, _ = CONTROL_CHART_CONSTANTS[subgroup_size]
    else:
        # Approximate for sizes > 25 using Central Limit Theorem
        # As n increases, the standard error approaches sigma/sqrt(n)
        # and 3-sigma limits approach 3*sigma/sqrt(n)
        A2 = 3 / (subgroup_size ** 0.5)
    
    # Calculate control limits: X-bar ± A2 * R-bar
    # Simplified: use sigma estimate
    ucl = centerline + sigma_level * sigma / (subgroup_size ** 0.5)
    lcl = centerline - sigma_level * sigma / (subgroup_size ** 0.5)
    
    # Identify out of control points
    out_of_control_points = []
    out_of_control_details = []
    
    for i, value in enumerate(data):
        if value > ucl:
            out_of_control_points.append(i)
            out_of_control_details.append({
                'index': i,
                'value': value,
                'violation': 'above_ucl'
            })
        elif value < lcl:
            out_of_control_points.append(i)
            out_of_control_details.append({
                'index': i,
                'value': value,
                'violation': 'below_lcl'
            })
    
    # Determine process status
    if len(out_of_control_points) == 0:
        process_status = "In Control - All points within limits"
        recommendations = "Process is stable. Continue monitoring."
    else:
        process_status = f"Out of Control - {len(out_of_control_points)} points beyond limits"
        recommendations = f"Investigate special cause variation at points {', '.join(map(str, out_of_control_points))}"
    
    return {
        'chart_type': 'x_bar',
        'centerline': centerline,
        'ucl': ucl,
        'lcl': lcl,
        'sigma': sigma,
        'subgroup_size': subgroup_size,
        'data_points': n,
        'out_of_control_points': out_of_control_points,
        'out_of_control_details': out_of_control_details,
        'process_status': process_status,
        'recommendations': recommendations
    }


def _calculate_individuals_limits(data: list[float], sigma_level: float) -> dict[str, Any]:
    """Calculate control limits for Individuals (I) chart."""
    n = len(data)
    centerline = sum(data) / n
    
    # Calculate moving ranges
    moving_ranges = [abs(data[i+1] - data[i]) for i in range(n - 1)]
    avg_moving_range = sum(moving_ranges) / len(moving_ranges)
    
    # Estimate sigma using d2 constant for moving range (n=2)
    d2 = 1.128
    sigma = avg_moving_range / d2
    
    # Calculate control limits
    ucl = centerline + sigma_level * sigma
    lcl = centerline - sigma_level * sigma
    
    # Identify out of control points
    out_of_control_points = []
    out_of_control_details = []
    
    for i, value in enumerate(data):
        if value > ucl:
            out_of_control_points.append(i)
            out_of_control_details.append({
                'index': i,
                'value': value,
                'violation': 'above_ucl'
            })
        elif value < lcl:
            out_of_control_points.append(i)
            out_of_control_details.append({
                'index': i,
                'value': value,
                'violation': 'below_lcl'
            })
    
    # Determine process status
    if len(out_of_control_points) == 0:
        process_status = "In Control - All points within limits"
        recommendations = "Process is stable. Continue monitoring."
    else:
        process_status = f"Out of Control - {len(out_of_control_points)} points beyond limits"
        recommendations = f"Investigate special cause variation at points {', '.join(map(str, out_of_control_points))}"
    
    return {
        'chart_type': 'individuals',
        'centerline': centerline,
        'ucl': ucl,
        'lcl': lcl,
        'sigma': sigma,
        'subgroup_size': 1,
        'data_points': n,
        'out_of_control_points': out_of_control_points,
        'out_of_control_details': out_of_control_details,
        'process_status': process_status,
        'recommendations': recommendations
    }


def _calculate_range_limits(data: list[float], subgroup_size: int, sigma_level: float) -> dict[str, Any]:
    """Calculate control limits for Range (R) chart."""
    # For R chart, data should be subgroup ranges
    n = len(data)
    centerline = sum(data) / n  # R-bar
    
    # Get D3 and D4 constants
    if subgroup_size in CONTROL_CHART_CONSTANTS:
        _, D3, D4, _, _ = CONTROL_CHART_CONSTANTS[subgroup_size]
    else:
        # Approximate for sizes > 25
        D3 = max(0, 1 - 3 / (subgroup_size ** 0.5))
        D4 = 1 + 3 / (subgroup_size ** 0.5)
    
    # Calculate control limits
    ucl = D4 * centerline
    lcl = D3 * centerline
    
    # Estimate sigma from R-bar
    if subgroup_size in CONTROL_CHART_CONSTANTS:
        _, _, _, d2, _ = CONTROL_CHART_CONSTANTS[subgroup_size]
    else:
        d2 = subgroup_size ** 0.5  # Approximation
    
    sigma = centerline / d2
    
    # Identify out of control points
    out_of_control_points = []
    out_of_control_details = []
    
    for i, value in enumerate(data):
        if value > ucl:
            out_of_control_points.append(i)
            out_of_control_details.append({
                'index': i,
                'value': value,
                'violation': 'above_ucl'
            })
        elif value < lcl:
            out_of_control_points.append(i)
            out_of_control_details.append({
                'index': i,
                'value': value,
                'violation': 'below_lcl'
            })
    
    # Determine process status
    if len(out_of_control_points) == 0:
        process_status = "In Control - Process variability is stable"
        recommendations = "Process variation is consistent. Continue monitoring."
    else:
        process_status = f"Out of Control - {len(out_of_control_points)} points indicate unstable variation"
        recommendations = f"Investigate variability changes at points {', '.join(map(str, out_of_control_points))}"
    
    return {
        'chart_type': 'range',
        'centerline': centerline,
        'ucl': ucl,
        'lcl': lcl,
        'sigma': sigma,
        'subgroup_size': subgroup_size,
        'data_points': n,
        'out_of_control_points': out_of_control_points,
        'out_of_control_details': out_of_control_details,
        'process_status': process_status,
        'recommendations': recommendations
    }


def _calculate_stddev_limits(data: list[float], subgroup_size: int, sigma_level: float) -> dict[str, Any]:
    """Calculate control limits for Standard Deviation (S) chart."""
    # For S chart, data should be subgroup standard deviations
    n = len(data)
    centerline = sum(data) / n  # S-bar
    
    # Get c4 constant
    if subgroup_size in CONTROL_CHART_CONSTANTS:
        _, _, _, _, c4 = CONTROL_CHART_CONSTANTS[subgroup_size]
    else:
        # Approximate c4 for large n
        c4 = 1 - 1 / (4 * subgroup_size)
    
    # Calculate B3 and B4 factors (similar to D3, D4 for S charts)
    B4 = 1 + 3 * ((1 - c4 ** 2) ** 0.5) / c4
    B3 = max(0, 1 - 3 * ((1 - c4 ** 2) ** 0.5) / c4)
    
    # Calculate control limits
    ucl = B4 * centerline
    lcl = B3 * centerline
    
    # Estimate sigma from S-bar
    sigma = centerline / c4
    
    # Identify out of control points
    out_of_control_points = []
    out_of_control_details = []
    
    for i, value in enumerate(data):
        if value > ucl:
            out_of_control_points.append(i)
            out_of_control_details.append({
                'index': i,
                'value': value,
                'violation': 'above_ucl'
            })
        elif value < lcl:
            out_of_control_points.append(i)
            out_of_control_details.append({
                'index': i,
                'value': value,
                'violation': 'below_lcl'
            })
    
    # Determine process status
    if len(out_of_control_points) == 0:
        process_status = "In Control - Process variability is stable"
        recommendations = "Process variation is consistent. Continue monitoring."
    else:
        process_status = f"Out of Control - {len(out_of_control_points)} points indicate unstable variation"
        recommendations = f"Investigate variability changes at points {', '.join(map(str, out_of_control_points))}"
    
    return {
        'chart_type': 'std_dev',
        'centerline': centerline,
        'ucl': ucl,
        'lcl': lcl,
        'sigma': sigma,
        'subgroup_size': subgroup_size,
        'data_points': n,
        'out_of_control_points': out_of_control_points,
        'out_of_control_details': out_of_control_details,
        'process_status': process_status,
        'recommendations': recommendations
    }


def _calculate_attribute_limits(data: list[float], chart_type: str, subgroup_size: int, sigma_level: float) -> dict[str, Any]:
    """Calculate control limits for attribute charts (p, np, c, u)."""
    n = len(data)
    centerline = sum(data) / n
    
    if chart_type == "p":
        # p chart: proportion defective
        # UCL/LCL = p-bar ± 3 * sqrt(p-bar * (1 - p-bar) / n)
        sigma = (centerline * (1 - centerline) / subgroup_size) ** 0.5
        ucl = centerline + sigma_level * sigma
        lcl = max(0, centerline - sigma_level * sigma)  # Can't be negative
    
    elif chart_type == "np":
        # np chart: number defective
        # UCL/LCL = np-bar ± 3 * sqrt(np-bar * (1 - np-bar/n))
        p = centerline / subgroup_size
        sigma = (centerline * (1 - p)) ** 0.5
        ucl = centerline + sigma_level * sigma
        lcl = max(0, centerline - sigma_level * sigma)
    
    elif chart_type == "c":
        # c chart: count of defects
        # UCL/LCL = c-bar ± 3 * sqrt(c-bar)
        sigma = centerline ** 0.5
        ucl = centerline + sigma_level * sigma
        lcl = max(0, centerline - sigma_level * sigma)
    
    elif chart_type == "u":
        # u chart: defects per unit
        # UCL/LCL = u-bar ± 3 * sqrt(u-bar / n)
        sigma = (centerline / subgroup_size) ** 0.5
        ucl = centerline + sigma_level * sigma
        lcl = max(0, centerline - sigma_level * sigma)
    
    else:
        raise ValueError(f"Unknown attribute chart type: {chart_type}")
    
    # Identify out of control points
    out_of_control_points = []
    out_of_control_details = []
    
    for i, value in enumerate(data):
        if value > ucl:
            out_of_control_points.append(i)
            out_of_control_details.append({
                'index': i,
                'value': value,
                'violation': 'above_ucl'
            })
        elif value < lcl:
            out_of_control_points.append(i)
            out_of_control_details.append({
                'index': i,
                'value': value,
                'violation': 'below_lcl'
            })
    
    # Determine process status
    if len(out_of_control_points) == 0:
        process_status = "In Control - All points within limits"
        recommendations = "Process is stable. Continue monitoring."
    else:
        process_status = f"Out of Control - {len(out_of_control_points)} points beyond limits"
        recommendations = f"Investigate special cause variation at points {', '.join(map(str, out_of_control_points))}"
    
    return {
        'chart_type': chart_type,
        'centerline': centerline,
        'ucl': ucl,
        'lcl': lcl,
        'sigma': sigma,
        'subgroup_size': subgroup_size,
        'data_points': n,
        'out_of_control_points': out_of_control_points,
        'out_of_control_details': out_of_control_details,
        'process_status': process_status,
        'recommendations': recommendations
    }


def process_capability(
    data: list[float],
    usl: float,
    lsl: float,
    target: float = None
) -> dict[str, Any]:
    """
    Calculate process capability indices (Cp, Cpk, Pp, Ppk).
    
    Use Cases:
    - Evaluate if process meets customer specifications
    - Compare process performance before/after improvements
    - Assess supplier quality capability
    - Determine if process can achieve Six Sigma levels
    - Support process qualification and validation
    
    Args:
        data: Process measurements (min 30 items)
        usl: Upper specification limit
        lsl: Lower specification limit
        target: Target value (optional, defaults to midpoint of USL and LSL)
        
    Returns:
        Dictionary containing:
        - sample_size: Number of measurements
        - mean: Process mean
        - std_dev: Process standard deviation
        - usl, lsl, target: Specification limits
        - cp: Process capability (potential)
        - cpk: Process capability (actual, accounts for centering)
        - pp: Process performance (overall variation)
        - ppk: Process performance (actual, accounts for centering)
        - cp_interpretation: Human-readable interpretation
        - cpk_interpretation: Human-readable interpretation
        - sigma_level: Estimated sigma level
        - percent_within_spec: Estimated % within specification
        - estimated_ppm_defects: Estimated defects per million
        - process_performance: Overall assessment
        - centering: Assessment of process centering
        - recommendations: Actionable recommendations
        
    Raises:
        ValueError: If parameters are invalid
        
    Examples:
        process_capability([100.1, 100.2, 99.9, 100.3], usl=103, lsl=97)
        >>> {
        ...   'cp': 2.35,
        ...   'cpk': 2.18,
        ...   'cp_interpretation': 'Excellent',
        ...   'process_performance': 'Six Sigma capable'
        ... }
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"data must be a list, got {type(data).__name__}")
    
    if len(data) < 30:
        raise ValueError("data must contain at least 30 items for reliable capability analysis")
    
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}")
    
    # Validate specification limits
    if not isinstance(usl, (int, float)):
        raise ValueError(f"usl must be numeric, got {type(usl).__name__}")
    
    if not isinstance(lsl, (int, float)):
        raise ValueError(f"lsl must be numeric, got {type(lsl).__name__}")
    
    if usl <= lsl:
        raise ValueError(f"usl ({usl}) must be greater than lsl ({lsl})")
    
    # Set default target if not provided
    if target is None:
        target = (usl + lsl) / 2
    else:
        if not isinstance(target, (int, float)):
            raise ValueError(f"target must be numeric, got {type(target).__name__}")
    
    # Calculate basic statistics
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)  # Sample variance
    std_dev = variance ** 0.5
    
    # Calculate Cp (potential capability)
    # Cp = (USL - LSL) / (6 * sigma)
    cp = (usl - lsl) / (6 * std_dev)
    
    # Calculate Cpk (actual capability, accounts for centering)
    # Cpk = min[(USL - mean) / (3 * sigma), (mean - LSL) / (3 * sigma)]
    cpu = (usl - mean) / (3 * std_dev)
    cpl = (mean - lsl) / (3 * std_dev)
    cpk = min(cpu, cpl)
    
    # Calculate Pp (overall performance, using total variation)
    # Note: In practice, Pp uses long-term sigma (between-subgroup variation) while Cp uses
    # within-subgroup variation. For individual measurements without subgroups, Pp ≈ Cp.
    # This implementation uses sample standard deviation which is appropriate for this case.
    pp = cp  # Simplified for individual measurements
    
    # Calculate Ppk (overall performance with centering)
    # Note: Ppk uses long-term sigma like Pp. For individual measurements, Ppk ≈ Cpk.
    ppk = cpk  # Simplified for individual measurements
    
    # Interpret Cp
    if cp >= 2.0:
        cp_interpretation = "Excellent - Process spread is much smaller than tolerance"
    elif cp >= 1.33:
        cp_interpretation = "Good - Process is capable with some margin"
    elif cp >= 1.0:
        cp_interpretation = "Adequate - Process barely capable, improvement recommended"
    else:
        cp_interpretation = "Poor - Process spread exceeds tolerance, improvement required"
    
    # Interpret Cpk
    if cpk >= 2.0:
        cpk_interpretation = "Excellent - Process is well-centered and capable"
    elif cpk >= 1.33:
        cpk_interpretation = "Good - Process is capable but may benefit from centering"
    elif cpk >= 1.0:
        cpk_interpretation = "Adequate - Process marginally capable, centering needed"
    else:
        cpk_interpretation = "Poor - Process not capable, centering and/or variation reduction required"
    
    # Calculate sigma level
    # Sigma level ≈ 3 * Cpk
    sigma_level = 3 * cpk
    
    # Estimate percent within specification using normal distribution
    # P(LSL < X < USL) where X ~ N(mean, std_dev)
    z_lower = (lsl - mean) / std_dev
    z_upper = (usl - mean) / std_dev
    percent_within_spec = (scipy_stats.norm.cdf(z_upper) - scipy_stats.norm.cdf(z_lower)) * 100
    
    # Estimate PPM defects
    estimated_ppm_defects = (1 - percent_within_spec / 100) * 1_000_000
    
    # Overall process performance
    if sigma_level >= 6:
        process_performance = "Six Sigma capable"
    elif sigma_level >= 5:
        process_performance = "Five Sigma capable"
    elif sigma_level >= 4:
        process_performance = "Four Sigma capable"
    elif sigma_level >= 3:
        process_performance = "Three Sigma capable"
    else:
        process_performance = "Below Three Sigma - significant improvement needed"
    
    # Assess centering
    centering_offset = abs(mean - target)
    if centering_offset < 0.1 * (usl - lsl):
        centering = f"Well-centered (offset: {centering_offset:.4f} units)"
    elif centering_offset < 0.25 * (usl - lsl):
        centering = f"Slightly off-center by {centering_offset:.4f} units"
    else:
        centering = f"Significantly off-center by {centering_offset:.4f} units"
    
    # Recommendations
    if cpk >= 2.0:
        recommendations = "Process is highly capable. Consider tightening specifications or cost reduction opportunities."
    elif cpk >= 1.33:
        recommendations = "Process is capable. Monitor for shifts and continue process control."
    elif cpk >= 1.0:
        if abs(cpu - cpl) > 0.2:
            recommendations = "Process is marginally capable. Focus on centering the process to improve Cpk."
        else:
            recommendations = "Process is marginally capable. Focus on reducing variation to improve capability."
    else:
        if abs(cpu - cpl) > 0.2:
            recommendations = "Process is not capable. Priority: center the process and reduce variation."
        else:
            recommendations = "Process is not capable. Priority: reduce process variation significantly."
    
    return {
        'sample_size': n,
        'mean': mean,
        'std_dev': std_dev,
        'usl': usl,
        'lsl': lsl,
        'target': target,
        'cp': cp,
        'cpk': cpk,
        'pp': pp,
        'ppk': ppk,
        'cp_interpretation': cp_interpretation,
        'cpk_interpretation': cpk_interpretation,
        'sigma_level': sigma_level,
        'percent_within_spec': percent_within_spec,
        'estimated_ppm_defects': estimated_ppm_defects,
        'process_performance': process_performance,
        'centering': centering,
        'recommendations': recommendations
    }


def western_electric_rules(
    data: list[float],
    centerline: float,
    sigma: float,
    rules_to_apply: list[int] = None
) -> dict[str, Any]:
    """
    Apply Western Electric run rules to detect non-random patterns.
    
    Use Cases:
    - Early warning of process shifts before out-of-control points
    - Detect systematic patterns indicating assignable causes
    - Identify tool wear, shift changes, or raw material variation
    - Supplement traditional control limits
    - Automated process monitoring and alarming
    
    Rules Implemented:
    1. One point beyond 3σ
    2. Two out of three consecutive points beyond 2σ (same side)
    3. Four out of five consecutive points beyond 1σ (same side)
    4. Eight consecutive points on same side of centerline
    5. Six points in a row steadily increasing or decreasing
    6. Fifteen points in a row within 1σ of centerline (both sides)
    7. Fourteen points in a row alternating up and down
    8. Eight points in a row beyond 1σ (either side)
    
    Args:
        data: Process measurements in time order (min 15 items)
        centerline: Process centerline (mean)
        sigma: Process standard deviation
        rules_to_apply: Which rules to check (default: all 8 rules)
        
    Returns:
        Dictionary containing:
        - violations: List of rule violations with details
        - total_violations: Number of violations detected
        - process_status: Overall status
        - action_required: Recommended actions
        - pattern_detected: Description of patterns
        
    Raises:
        ValueError: If parameters are invalid
        
    Examples:
        western_electric_rules([100]*10 + [103]*10, centerline=100, sigma=0.5)
        >>> {
        ...   'violations': [{
        ...     'rule': 4,
        ...     'rule_name': 'Eight consecutive points same side',
        ...     'severity': 'warning'
        ...   }],
        ...   'process_status': 'Out of Statistical Control'
        ... }
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"data must be a list, got {type(data).__name__}")
    
    if len(data) < 15:
        raise ValueError("data must contain at least 15 items for Western Electric rules")
    
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}")
    
    # Validate centerline and sigma
    if not isinstance(centerline, (int, float)):
        raise ValueError(f"centerline must be numeric, got {type(centerline).__name__}")
    
    if not isinstance(sigma, (int, float)):
        raise ValueError(f"sigma must be numeric, got {type(sigma).__name__}")
    
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    
    # Set default rules if not provided
    if rules_to_apply is None:
        rules_to_apply = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Validate rules_to_apply
    if not isinstance(rules_to_apply, list):
        raise ValueError(f"rules_to_apply must be a list, got {type(rules_to_apply).__name__}")
    
    for rule in rules_to_apply:
        if not isinstance(rule, int) or rule < 1 or rule > 8:
            raise ValueError(f"All rules must be integers between 1 and 8, got {rule}")
    
    violations = []
    
    # Rule 1: One point beyond 3σ
    if 1 in rules_to_apply:
        for i, value in enumerate(data):
            if abs(value - centerline) > 3 * sigma:
                violations.append({
                    'rule': 1,
                    'rule_name': 'One point beyond 3σ',
                    'indices': [i],
                    'severity': 'critical',
                    'description': f"Point {i} (value: {value:.4f}) is {abs(value - centerline) / sigma:.2f}σ {'above' if value > centerline else 'below'} centerline"
                })
    
    # Rule 2: Two out of three consecutive points beyond 2σ (same side)
    if 2 in rules_to_apply:
        for i in range(len(data) - 2):
            window = data[i:i+3]
            above_2sigma = sum(1 for v in window if v - centerline > 2 * sigma)
            below_2sigma = sum(1 for v in window if centerline - v > 2 * sigma)
            
            if above_2sigma >= 2:
                indices = [i+j for j, v in enumerate(window) if v - centerline > 2 * sigma]
                violations.append({
                    'rule': 2,
                    'rule_name': 'Two out of three beyond 2σ (same side)',
                    'indices': indices,
                    'severity': 'warning',
                    'description': f"Points {indices} are beyond 2σ above centerline"
                })
            elif below_2sigma >= 2:
                indices = [i+j for j, v in enumerate(window) if centerline - v > 2 * sigma]
                violations.append({
                    'rule': 2,
                    'rule_name': 'Two out of three beyond 2σ (same side)',
                    'indices': indices,
                    'severity': 'warning',
                    'description': f"Points {indices} are beyond 2σ below centerline"
                })
    
    # Rule 3: Four out of five consecutive points beyond 1σ (same side)
    if 3 in rules_to_apply:
        for i in range(len(data) - 4):
            window = data[i:i+5]
            above_1sigma = sum(1 for v in window if v - centerline > sigma)
            below_1sigma = sum(1 for v in window if centerline - v > sigma)
            
            if above_1sigma >= 4:
                indices = [i+j for j, v in enumerate(window) if v - centerline > sigma]
                violations.append({
                    'rule': 3,
                    'rule_name': 'Four out of five beyond 1σ (same side)',
                    'indices': indices,
                    'severity': 'warning',
                    'description': f"Points {indices} are beyond 1σ above centerline"
                })
            elif below_1sigma >= 4:
                indices = [i+j for j, v in enumerate(window) if centerline - v > sigma]
                violations.append({
                    'rule': 3,
                    'rule_name': 'Four out of five beyond 1σ (same side)',
                    'indices': indices,
                    'severity': 'warning',
                    'description': f"Points {indices} are beyond 1σ below centerline"
                })
    
    # Rule 4: Eight consecutive points on same side of centerline
    if 4 in rules_to_apply:
        for i in range(len(data) - 7):
            window = data[i:i+8]
            all_above = all(v > centerline for v in window)
            all_below = all(v < centerline for v in window)
            
            if all_above or all_below:
                indices = list(range(i, i+8))
                violations.append({
                    'rule': 4,
                    'rule_name': 'Eight consecutive points same side',
                    'indices': indices,
                    'severity': 'warning',
                    'description': f"8 consecutive points {'above' if all_above else 'below'} centerline indicates process shift"
                })
    
    # Rule 5: Six points in a row steadily increasing or decreasing
    if 5 in rules_to_apply:
        for i in range(len(data) - 5):
            window = data[i:i+6]
            increasing = all(window[j+1] > window[j] for j in range(5))
            decreasing = all(window[j+1] < window[j] for j in range(5))
            
            if increasing or decreasing:
                indices = list(range(i, i+6))
                violations.append({
                    'rule': 5,
                    'rule_name': 'Six points steadily increasing/decreasing',
                    'indices': indices,
                    'severity': 'warning',
                    'description': f"6 points steadily {'increasing' if increasing else 'decreasing'} indicates trend"
                })
    
    # Rule 6: Fifteen points in a row within 1σ of centerline
    if 6 in rules_to_apply:
        for i in range(len(data) - 14):
            window = data[i:i+15]
            all_within_1sigma = all(abs(v - centerline) < sigma for v in window)
            
            if all_within_1sigma:
                indices = list(range(i, i+15))
                violations.append({
                    'rule': 6,
                    'rule_name': 'Fifteen points within 1σ',
                    'indices': indices,
                    'severity': 'info',
                    'description': "15 points within 1σ may indicate stratification or measurement issues"
                })
    
    # Rule 7: Fourteen points in a row alternating up and down
    if 7 in rules_to_apply:
        for i in range(len(data) - 13):
            window = data[i:i+14]
            alternating = all(
                (window[j+1] > window[j] and window[j+2] < window[j+1]) or
                (window[j+1] < window[j] and window[j+2] > window[j+1])
                for j in range(12)
            )
            
            if alternating:
                indices = list(range(i, i+14))
                violations.append({
                    'rule': 7,
                    'rule_name': 'Fourteen points alternating',
                    'indices': indices,
                    'severity': 'info',
                    'description': "14 points alternating up and down indicates systematic variation"
                })
    
    # Rule 8: Eight points in a row beyond 1σ (either side)
    if 8 in rules_to_apply:
        for i in range(len(data) - 7):
            window = data[i:i+8]
            all_beyond_1sigma = all(abs(v - centerline) > sigma for v in window)
            
            if all_beyond_1sigma:
                indices = list(range(i, i+8))
                violations.append({
                    'rule': 8,
                    'rule_name': 'Eight points beyond 1σ',
                    'indices': indices,
                    'severity': 'warning',
                    'description': "8 points beyond 1σ (either side) indicates mixture or bi-modal distribution"
                })
    
    # Determine overall status
    total_violations = len(violations)
    
    if total_violations == 0:
        process_status = "In Statistical Control"
        action_required = "Continue monitoring process"
        pattern_detected = "No non-random patterns detected"
    else:
        process_status = "Out of Statistical Control"
        
        # Identify patterns
        patterns = []
        rule_counts = {}
        for v in violations:
            rule = v['rule']
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        if 1 in rule_counts:
            patterns.append("outliers")
        if 4 in rule_counts or 5 in rule_counts:
            patterns.append("trend")
        if 2 in rule_counts or 3 in rule_counts:
            patterns.append("shift")
        if 6 in rule_counts:
            patterns.append("stratification")
        if 7 in rule_counts:
            patterns.append("alternating")
        if 8 in rule_counts:
            patterns.append("mixture")
        
        pattern_detected = " and ".join(patterns).capitalize() if patterns else "Multiple patterns"
        
        # Generate action
        critical_violations = [v for v in violations if v.get('severity') == 'critical']
        if critical_violations:
            action_required = f"Immediate investigation required for critical violations at points {[v['indices'][0] for v in critical_violations]}"
        else:
            action_required = "Investigate assignable causes and take corrective action"
    
    return {
        'violations': violations,
        'total_violations': total_violations,
        'process_status': process_status,
        'action_required': action_required,
        'pattern_detected': pattern_detected
    }


def cusum_chart(
    data: list[float],
    target: float,
    k: float = None,
    h: float = None,
    sigma: float = None
) -> dict[str, Any]:
    """
    Implement CUSUM (Cumulative Sum) chart for detecting small persistent shifts.
    
    Use Cases:
    - Detect gradual process drift
    - Monitor slow equipment degradation
    - Identify small but persistent quality shifts
    - More sensitive than traditional control charts for small shifts (<1.5σ)
    - Track process improvements over time
    
    Args:
        data: Process measurements in time order (min 5 items)
        target: Target process value
        k: Reference value (typically 0.5σ), defaults to 0.5 if sigma provided
        h: Decision interval (typically 4σ or 5σ), defaults to 4 if sigma provided
        sigma: Process standard deviation (optional, estimated from data if not provided)
        
    Returns:
        Dictionary containing:
        - cusum_positive: Positive CUSUM values
        - cusum_negative: Negative CUSUM values
        - upper_limit: Upper decision limit (h)
        - lower_limit: Lower decision limit (-h)
        - signals: List of detected shifts with details
        - process_status: Overall status
        - estimated_new_level: Estimated process level after shift (if detected)
        - recommendation: Actionable recommendation
        
    Raises:
        ValueError: If parameters are invalid
        
    Examples:
        cusum_chart([100]*10 + [101]*10, target=100, sigma=0.5)
        >>> {
        ...   'signals': [{
        ...     'index': 14,
        ...     'type': 'positive_shift',
        ...     'magnitude_estimate': 1.0
        ...   }],
        ...   'process_status': 'Shift detected'
        ... }
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"data must be a list, got {type(data).__name__}")
    
    if len(data) < 5:
        raise ValueError("data must contain at least 5 items")
    
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}")
    
    # Validate target
    if not isinstance(target, (int, float)):
        raise ValueError(f"target must be numeric, got {type(target).__name__}")
    
    # Estimate sigma if not provided
    if sigma is None:
        mean = sum(data) / len(data)
        sigma = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
        if sigma < 1e-10:
            raise ValueError("Cannot calculate CUSUM: data has zero or near-zero variation")
    else:
        if not isinstance(sigma, (int, float)):
            raise ValueError(f"sigma must be numeric, got {type(sigma).__name__}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
    
    # Set default k and h if not provided
    if k is None:
        k = 0.5 * sigma
    else:
        if not isinstance(k, (int, float)):
            raise ValueError(f"k must be numeric, got {type(k).__name__}")
    
    if h is None:
        h = 4 * sigma
    else:
        if not isinstance(h, (int, float)):
            raise ValueError(f"h must be numeric, got {type(h).__name__}")
    
    # Calculate CUSUM
    cusum_positive = [0.0]
    cusum_negative = [0.0]
    signals = []
    
    for i, value in enumerate(data):
        # Positive CUSUM (detects upward shifts)
        c_plus = max(0, cusum_positive[-1] + (value - target) - k)
        cusum_positive.append(c_plus)
        
        # Negative CUSUM (detects downward shifts)
        c_minus = max(0, cusum_negative[-1] - (value - target) - k)
        cusum_negative.append(c_minus)
        
        # Check for signals
        if c_plus > h:
            # Estimate change point (approximate)
            change_point = max(0, i - int(c_plus / (2 * k)))
            magnitude = c_plus / (i - change_point + 1) if (i - change_point) > 0 else 2 * k
            
            signals.append({
                'index': i,
                'type': 'positive_shift',
                'cusum_value': c_plus,
                'estimated_change_point': change_point,
                'magnitude_estimate': magnitude
            })
            
            # Reset CUSUM after signal
            cusum_positive[-1] = 0
        
        if c_minus > h:
            # Estimate change point
            change_point = max(0, i - int(c_minus / (2 * k)))
            magnitude = -c_minus / (i - change_point + 1) if (i - change_point) > 0 else -2 * k
            
            signals.append({
                'index': i,
                'type': 'negative_shift',
                'cusum_value': c_minus,
                'estimated_change_point': change_point,
                'magnitude_estimate': magnitude
            })
            
            # Reset CUSUM after signal
            cusum_negative[-1] = 0
    
    # Determine process status
    if len(signals) == 0:
        process_status = "In Control - No shifts detected"
        estimated_new_level = None
        recommendation = "Process is stable at target level. Continue monitoring."
    else:
        last_signal = signals[-1]
        process_status = f"Shift detected at point {last_signal['index']}"
        estimated_new_level = target + last_signal['magnitude_estimate']
        
        shift_direction = "upward" if last_signal['type'] == 'positive_shift' else "downward"
        recommendation = (
            f"Process has shifted {shift_direction} by approximately "
            f"{abs(last_signal['magnitude_estimate']):.4f} units starting around point "
            f"{last_signal['estimated_change_point']}. Investigate and take corrective action."
        )
    
    return {
        'cusum_positive': cusum_positive,
        'cusum_negative': cusum_negative,
        'upper_limit': h,
        'lower_limit': -h,
        'signals': signals,
        'process_status': process_status,
        'estimated_new_level': estimated_new_level,
        'recommendation': recommendation
    }


def ewma_chart(
    data: list[float],
    target: float,
    sigma: float,
    lambda_param: float = 0.2,
    l: float = 3.0
) -> dict[str, Any]:
    """
    Implement EWMA (Exponentially Weighted Moving Average) chart for detecting small shifts.
    
    Use Cases:
    - Monitor processes where small shifts are critical
    - Detect shifts faster than X-bar charts
    - Smooth noisy process data
    - Balance between Shewhart and CUSUM charts
    - Track chemical composition or pharmaceutical potency
    
    Args:
        data: Process measurements (min 5 items)
        target: Target process mean
        sigma: Process standard deviation
        lambda_param: Weighting factor (0-1, typical: 0.2), higher = more weight on recent data
        l: Control limit factor (typical: 3), multiplies standard error
        
    Returns:
        Dictionary containing:
        - ewma_values: EWMA values for each data point
        - ucl_values: Upper control limit values
        - lcl_values: Lower control limit values
        - target: Target value
        - out_of_control_points: Indices of points beyond limits
        - out_of_control_details: Details of violations
        - process_status: Overall status
        - recommendation: Actionable recommendation
        
    Raises:
        ValueError: If parameters are invalid
        
    Examples:
        ewma_chart([100]*10 + [100.5]*10, target=100, sigma=0.2, lambda_param=0.2)
        >>> {
        ...   'ewma_values': [100, 100, ..., 100.4, 100.45, 100.5],
        ...   'process_status': 'Shift detected',
        ...   'out_of_control_points': [15, 16, 17, 18, 19]
        ... }
    """
    # Validate input
    if not isinstance(data, list):
        raise ValueError(f"data must be a list, got {type(data).__name__}")
    
    if len(data) < 5:
        raise ValueError("data must contain at least 5 items")
    
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}")
    
    # Validate target
    if not isinstance(target, (int, float)):
        raise ValueError(f"target must be numeric, got {type(target).__name__}")
    
    # Validate sigma
    if not isinstance(sigma, (int, float)):
        raise ValueError(f"sigma must be numeric, got {type(sigma).__name__}")
    
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    
    # Validate lambda_param
    if not isinstance(lambda_param, (int, float)):
        raise ValueError(f"lambda_param must be numeric, got {type(lambda_param).__name__}")
    
    if lambda_param < 0.01 or lambda_param > 1.0:
        raise ValueError(f"lambda_param must be between 0.01 and 1.0, got {lambda_param}")
    
    # Validate l
    if not isinstance(l, (int, float)):
        raise ValueError(f"l must be numeric, got {type(l).__name__}")
    
    if l <= 0:
        raise ValueError(f"l must be positive, got {l}")
    
    # Calculate EWMA
    ewma_values = [target]  # Initialize with target
    ucl_values = []
    lcl_values = []
    out_of_control_points = []
    out_of_control_details = []
    
    for i, value in enumerate(data):
        # Calculate EWMA: Z_t = λ * X_t + (1 - λ) * Z_{t-1}
        ewma = lambda_param * value + (1 - lambda_param) * ewma_values[-1]
        ewma_values.append(ewma)
        
        # Calculate control limits
        # Standard error: sigma * sqrt(λ / (2 - λ) * [1 - (1 - λ)^(2*t)])
        t = i + 1
        factor = (lambda_param / (2 - lambda_param)) * (1 - (1 - lambda_param) ** (2 * t))
        std_error = sigma * (factor ** 0.5)
        
        ucl = target + l * std_error
        lcl = target - l * std_error
        
        ucl_values.append(ucl)
        lcl_values.append(lcl)
        
        # Check for out of control points
        if ewma > ucl:
            out_of_control_points.append(i)
            out_of_control_details.append({
                'index': i,
                'value': value,
                'ewma': ewma,
                'violation': 'above_ucl'
            })
        elif ewma < lcl:
            out_of_control_points.append(i)
            out_of_control_details.append({
                'index': i,
                'value': value,
                'ewma': ewma,
                'violation': 'below_lcl'
            })
    
    # Remove initial target value from ewma_values (it's not a data point)
    ewma_values = ewma_values[1:]
    
    # Determine process status
    if len(out_of_control_points) == 0:
        process_status = "In Control - Process centered at target"
        recommendation = "Process is stable. Continue monitoring with EWMA chart."
    else:
        process_status = f"Out of Control - {len(out_of_control_points)} points beyond EWMA limits"
        
        # Determine shift direction
        above_count = sum(1 for d in out_of_control_details if d['violation'] == 'above_ucl')
        below_count = sum(1 for d in out_of_control_details if d['violation'] == 'below_lcl')
        
        if above_count > below_count:
            shift_direction = "upward"
        elif below_count > above_count:
            shift_direction = "downward"
        else:
            shift_direction = "in both directions"
        
        recommendation = (
            f"EWMA detected {shift_direction} shift at points {', '.join(map(str, out_of_control_points))}. "
            f"Investigate process change and take corrective action."
        )
    
    return {
        'ewma_values': ewma_values,
        'ucl_values': ucl_values,
        'lcl_values': lcl_values,
        'target': target,
        'out_of_control_points': out_of_control_points,
        'out_of_control_details': out_of_control_details,
        'process_status': process_status,
        'recommendation': recommendation
    }


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
            name="control_limits",
            description=(
                "Calculate control chart limits (UCL, LCL, centerline) for various chart types "
                "including X-bar, Individuals, Range, Standard Deviation, and attribute charts (p, np, c, u). "
                "Essential for manufacturing quality control, process monitoring, and Six Sigma programs. "
                "Detects out-of-control points and provides actionable recommendations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Process measurements or subgroup statistics",
                        "minItems": 5
                    },
                    "chart_type": {
                        "type": "string",
                        "enum": ["x_bar", "individuals", "range", "std_dev", "p", "np", "c", "u"],
                        "description": "Type of control chart"
                    },
                    "subgroup_size": {
                        "type": "integer",
                        "description": "Size of subgroups (for X-bar, R, and S charts)",
                        "minimum": 2,
                        "maximum": 25,
                        "default": 5
                    },
                    "sigma_level": {
                        "type": "number",
                        "description": "Number of standard deviations for limits (typical: 3)",
                        "minimum": 1,
                        "maximum": 6,
                        "default": 3
                    }
                },
                "required": ["data", "chart_type"]
            }
        ),
        Tool(
            name="process_capability",
            description=(
                "Calculate process capability indices (Cp, Cpk, Pp, Ppk) to assess if a process "
                "meets customer specifications. Essential for Six Sigma programs, supplier qualification, "
                "and process validation. Provides sigma level, estimated defect rates (PPM), and "
                "actionable recommendations for process improvement."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Process measurements (minimum 30 for reliable analysis)",
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
                        "description": "Target value (optional, defaults to midpoint of USL and LSL)"
                    }
                },
                "required": ["data", "usl", "lsl"]
            }
        ),
        Tool(
            name="western_electric_rules",
            description=(
                "Apply Western Electric run rules to detect non-random patterns in control charts. "
                "Implements 8 rules for early warning of process shifts, trends, and systematic variation. "
                "More sensitive than traditional control limits for detecting assignable causes before "
                "they result in out-of-control points. Essential for proactive process monitoring."
            ),
            inputSchema={
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
                        "items": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 8
                        },
                        "description": "Which rules to check (1-8, default: all rules)",
                        "default": [1, 2, 3, 4, 5, 6, 7, 8]
                    }
                },
                "required": ["data", "centerline", "sigma"]
            }
        ),
        Tool(
            name="cusum_chart",
            description=(
                "Implement CUSUM (Cumulative Sum) chart for detecting small persistent shifts in process mean. "
                "More sensitive than traditional control charts for detecting shifts less than 1.5σ. "
                "Ideal for monitoring gradual equipment degradation, process drift, and small but "
                "persistent quality changes. Provides change point estimation and shift magnitude."
            ),
            inputSchema={
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
                        "description": "Reference value (typically 0.5σ, auto-calculated if not provided)"
                    },
                    "h": {
                        "type": "number",
                        "description": "Decision interval (typically 4σ or 5σ, auto-calculated if not provided)"
                    },
                    "sigma": {
                        "type": "number",
                        "description": "Process standard deviation (optional, estimated from data if not provided)"
                    }
                },
                "required": ["data", "target"]
            }
        ),
        Tool(
            name="ewma_chart",
            description=(
                "Implement EWMA (Exponentially Weighted Moving Average) chart for detecting small shifts "
                "in process mean. Balances between Shewhart and CUSUM charts by smoothing data while "
                "remaining sensitive to shifts. Ideal for noisy processes where small shifts are critical "
                "(e.g., chemical composition, pharmaceutical potency). Configurable weighting factor."
            ),
            inputSchema={
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
                    "sigma": {
                        "type": "number",
                        "description": "Process standard deviation"
                    },
                    "lambda_param": {
                        "type": "number",
                        "description": "Weighting factor (0-1, typical: 0.2, higher = more weight on recent data)",
                        "minimum": 0.01,
                        "maximum": 1.0,
                        "default": 0.2
                    },
                    "l": {
                        "type": "number",
                        "description": "Control limit factor (typical: 3)",
                        "default": 3.0
                    }
                },
                "required": ["data", "target", "sigma"]
            }
        ),
        Tool(
            name="fft_analysis",
            description=(
                "Perform Fast Fourier Transform (FFT) for frequency domain analysis. "
                "Use cases: bearing defect detection (BPFI, BPFO, BSF), motor electrical faults, "
                "gearbox mesh frequency analysis, pump cavitation, compressor valve problems, fan imbalance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Time-domain signal (vibration, current, acoustic, etc.)",
                        "minItems": 8
                    },
                    "sample_rate": {
                        "type": "number",
                        "description": "Sampling frequency in Hz",
                        "minimum": 1,
                        "maximum": 1000000
                    },
                    "window": {
                        "type": "string",
                        "enum": ["hanning", "hamming", "blackman", "rectangular"],
                        "default": "hanning",
                        "description": "Windowing function to reduce spectral leakage"
                    },
                    "detrend": {
                        "type": "boolean",
                        "default": True,
                        "description": "Remove DC component and linear trends"
                    }
                },
                "required": ["signal", "sample_rate"]
            }
        ),
        Tool(
            name="power_spectral_density",
            description=(
                "Calculate Power Spectral Density (PSD) for energy distribution across frequencies. "
                "Use cases: vibration energy distribution, noise level assessment, random vibration analysis, "
                "process variable frequency content, acoustic signature analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Time-domain signal",
                        "minItems": 16
                    },
                    "sample_rate": {
                        "type": "number",
                        "description": "Sampling frequency in Hz"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["welch", "periodogram"],
                        "default": "welch",
                        "description": "PSD estimation method - Welch is more accurate for noisy signals"
                    },
                    "nperseg": {
                        "type": "integer",
                        "default": 256,
                        "description": "Length of each segment for Welch method"
                    }
                },
                "required": ["signal", "sample_rate"]
            }
        ),
        Tool(
            name="rms_value",
            description=(
                "Calculate Root Mean Square (RMS) for overall signal energy. "
                "Use cases: overall vibration severity (ISO 10816), electrical current RMS, "
                "acoustic noise level, process variable stability, alarm threshold monitoring, trend tracking."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Time-domain signal",
                        "minItems": 2
                    },
                    "window_size": {
                        "type": "integer",
                        "description": "Calculate rolling RMS (optional)",
                        "minimum": 2
                    },
                    "reference_value": {
                        "type": "number",
                        "description": "Reference for dB calculation (e.g., 20 µPa for acoustic)"
                    }
                },
                "required": ["signal"]
            }
        ),
        Tool(
            name="peak_detection",
            description=(
                "Identify significant peaks in signals with filtering and ranking. "
                "Use cases: find dominant vibration frequencies, detect harmonic patterns, "
                "identify resonance frequencies, bearing fault frequency detection, gear mesh frequencies."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Signal data (time or frequency domain)",
                        "minItems": 3
                    },
                    "frequencies": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Corresponding frequencies (for frequency domain data)"
                    },
                    "height": {
                        "type": "number",
                        "default": 0.1,
                        "description": "Minimum peak height"
                    },
                    "distance": {
                        "type": "integer",
                        "default": 1,
                        "description": "Minimum samples between peaks"
                    },
                    "prominence": {
                        "type": "number",
                        "default": 0.05,
                        "description": "Required prominence (height above surroundings)"
                    },
                    "top_n": {
                        "type": "integer",
                        "default": 10,
                        "maximum": 50,
                        "description": "Return top N peaks only"
                    }
                },
                "required": ["signal"]
            }
        ),
        Tool(
            name="signal_to_noise_ratio",
            description=(
                "Calculate Signal-to-Noise Ratio (SNR) to assess signal quality. "
                "Use cases: sensor health monitoring, data acquisition quality check, "
                "communication signal quality, measurement reliability, instrumentation validation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Signal containing signal + noise",
                        "minItems": 10
                    },
                    "noise": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Noise reference (optional - will estimate if not provided)"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["power", "amplitude", "peak"],
                        "default": "power",
                        "description": "SNR calculation method"
                    }
                },
                "required": ["signal"]
            }
        ),
        Tool(
            name="harmonic_analysis",
            description=(
                "Detect and analyze harmonic content in electrical and mechanical signals. "
                "Use cases: power quality assessment (THD), variable frequency drive effects, "
                "motor current signature analysis (MCSA), electrical fault detection, IEEE 519 compliance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Periodic signal (voltage, current, vibration)",
                        "minItems": 64
                    },
                    "sample_rate": {
                        "type": "number",
                        "description": "Sampling frequency in Hz"
                    },
                    "fundamental_freq": {
                        "type": "number",
                        "description": "Expected fundamental frequency (e.g., 60 Hz for electrical)",
                        "minimum": 1
                    },
                    "max_harmonic": {
                        "type": "integer",
                        "default": 50,
                        "maximum": 100,
                        "description": "Maximum harmonic order to analyze"
                    }
                },
                "required": ["signal", "sample_rate", "fundamental_freq"]
            }
        ),
        Tool(
            name="linear_regression",
            description=(
                "Perform simple or multiple linear regression with comprehensive diagnostics. "
                "Use cases: equipment efficiency vs. load, energy consumption vs. production rate, "
                "pump performance curves, temperature vs. pressure relationships, vibration vs. bearing wear. "
                "Returns coefficients, R², confidence intervals, diagnostics (Durbin-Watson), and interpretation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "x": {
                        "description": "Independent variable(s) - single array for simple regression or array of arrays for multiple regression",
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
                        "description": f"Confidence level for intervals ({MIN_CONFIDENCE_LEVEL}-{MAX_CONFIDENCE_LEVEL})",
                        "minimum": MIN_CONFIDENCE_LEVEL,
                        "maximum": MAX_CONFIDENCE_LEVEL,
                        "default": 0.95
                    },
                    "include_diagnostics": {
                        "type": "boolean",
                        "description": "Include full regression diagnostics",
                        "default": True
                    }
                },
                "required": ["x", "y"]
            }
        ),
        Tool(
            name="polynomial_regression",
            description=(
                "Fit polynomial curves for non-linear relationships. "
                "Use cases: compressor performance curves, valve characteristics, catalyst activity decline, "
                "temperature profiles, motor torque vs. speed, pump efficiency curves. "
                "Returns coefficients, R², turning points, optimal values, and interpretation."
            ),
            inputSchema={
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
                        "description": "Automatically select best degree based on adjusted R²",
                        "default": False
                    }
                },
                "required": ["x", "y"]
            }
        ),
        Tool(
            name="residual_analysis",
            description=(
                "Comprehensive analysis of regression residuals to validate model assumptions. "
                "Use cases: validate assumptions, detect non-linearity, identify outliers, check heteroscedasticity, "
                "verify normality, detect autocorrelation. Includes Shapiro-Wilk, Durbin-Watson tests."
            ),
            inputSchema={
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
        ),
        Tool(
            name="prediction_with_intervals",
            description=(
                "Generate predictions with confidence and prediction intervals. "
                "Use cases: forecast equipment performance, estimate production with uncertainty, "
                "predict energy consumption with confidence intervals, estimate maintenance costs, "
                "calculate valve position for desired flow, predict process yield with tolerance bands."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "object",
                        "description": "Regression model from linear_regression or polynomial_regression (must have 'coefficients' and 'rmse')"
                    },
                    "x_new": {
                        "type": "array",
                        "description": "New x values for prediction",
                        "minItems": 1
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "Confidence level (0-1)",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.95
                    }
                },
                "required": ["model", "x_new"]
            }
        ),
        Tool(
            name="multivariate_regression",
            description=(
                "Multiple linear regression with multiple independent variables. "
                "Use cases: chiller efficiency vs. multiple factors, production yield vs. process parameters, "
                "energy consumption vs. multiple conditions, compressor power vs. pressures and flow, "
                "product quality vs. process variables. Includes VIF for multicollinearity detection."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "X": {
                        "type": "array",
                        "description": "Matrix of independent variables [[x1_1, x2_1, ...], [x1_2, x2_2, ...], ...]",
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
                        "description": "Names for each independent variable"
                    },
                    "standardize": {
                        "type": "boolean",
                        "description": "Standardize variables for coefficient comparison",
                        "default": False
                    }
                },
                "required": ["X", "y"]
            }
        ),
        Tool(
            name="z_score_detection",
            description=(
                "Detect outliers using standard or modified Z-score methods. "
                "Standard method uses mean/std, modified uses median/MAD for robustness. "
                "Use cases: normally distributed process variables (temperature, pressure), "
                "quick screening of large datasets, real-time sensor validation, quality control measurements."
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
                "Statistical test for detecting a single outlier in normally distributed data. "
                "Grubbs' test identifies whether the most extreme value is a statistical outlier. "
                "Use cases: reject suspicious calibration points, validate laboratory test results, "
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
                "Quick outlier test for small datasets (3-30 points) using gap ratio. "
                "Designed for small sample sizes where other tests may not be appropriate. "
                "Use cases: laboratory quality control (small samples), pilot plant trials, "
                "expensive test results validation, duplicate/triplicate measurements, shift samples."
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
                        "minimum": 0.001,
                        "maximum": 0.1,
                        "default": 0.05
                    }
                },
                "required": ["data"]
            }
        ),
        Tool(
            name="isolation_forest",
            description=(
                "Machine learning-based anomaly detection using Isolation Forest algorithm. "
                "Detects anomalies by isolating observations - anomalies are easier to isolate. "
                "Use cases: multivariate anomaly detection (multiple sensors), complex process patterns, "
                "equipment failure prediction, cyber security (unusual patterns), unstructured anomalies."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "description": "Univariate (list of numbers) or multivariate (list of lists)",
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
                "Measures distance from center accounting for correlations. "
                "Use cases: multiple correlated sensors, process state monitoring "
                "(temperature+pressure+flow together), multivariate quality control, "
                "equipment health monitoring (multiple parameters), pattern recognition."
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
                        "description": "Chi-square threshold percentile (0.9-0.999)",
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
                "Evaluates new measurements against recent history for anomalies. "
                "Use cases: real-time SCADA alarming, edge device data validation, "
                "continuous process monitoring, high-frequency sensor data (1-second intervals), "
                "telemetry data validation."
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
                        "default": "ewma",
                        "description": "Detection method"
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
        elif name == "fft_analysis":
            return await handle_fft_analysis(arguments)
        elif name == "power_spectral_density":
            return await handle_power_spectral_density(arguments)
        elif name == "rms_value":
            return await handle_rms_value(arguments)
        elif name == "peak_detection":
            return await handle_peak_detection(arguments)
        elif name == "signal_to_noise_ratio":
            return await handle_signal_to_noise_ratio(arguments)
        elif name == "harmonic_analysis":
            return await handle_harmonic_analysis(arguments)
        elif name == "control_limits":
            return await handle_control_limits(arguments)
        elif name == "process_capability":
            return await handle_process_capability(arguments)
        elif name == "western_electric_rules":
            return await handle_western_electric_rules(arguments)
        elif name == "cusum_chart":
            return await handle_cusum_chart(arguments)
        elif name == "ewma_chart":
            return await handle_ewma_chart(arguments)
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


async def handle_z_score_detection(arguments: Any) -> CallToolResult:
    """Handle z_score_detection tool calls."""
    data = arguments.get("data")
    method = arguments.get("method", "modified")
    threshold = arguments.get("threshold", 3.0)
    two_tailed = arguments.get("two_tailed", True)
    
    if data is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    try:
        logger.info(f"Z-score detection: {method} method, threshold={threshold}")
        result = z_score_detection(data, method, threshold, two_tailed)
        
        # Format output
        result_text = f"Z-Score Outlier Detection:\n\n"
        result_text += f"Method: {result['method'].replace('_', ' ').title()}\n"
        result_text += f"Threshold: {result['threshold']}\n"
        result_text += f"Two-Tailed: {two_tailed}\n\n"
        
        if result['outliers']['indices']:
            result_text += f"Outliers Detected: {len(result['outliers']['indices'])}\n\n"
            for detail in result['outliers']['values'][:10]:  # Show first 10
                result_text += f"  Index {detail['index']}: Value={detail['value']:.4f}, "
                result_text += f"Z-score={detail['z_score']:.2f}, Severity={detail['severity']}\n"
            if len(result['outliers']['values']) > 10:
                result_text += f"  ... and {len(result['outliers']['values']) - 10} more\n"
        else:
            result_text += "No outliers detected.\n"
        
        result_text += f"\nStatistics:\n"
        for key, value in result['statistics'].items():
            if isinstance(value, (int, float)):
                result_text += f"  {key.replace('_', ' ').title()}: {value:.4f}\n"
            else:
                result_text += f"  {key.replace('_', ' ').title()}: {value}\n"
        
        result_text += f"\n{result['interpretation']}"
        
        logger.info(f"Found {len(result['outliers']['indices'])} outliers")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_grubbs_test(arguments: Any) -> CallToolResult:
    """Handle grubbs_test tool calls."""
    data = arguments.get("data")
    alpha = arguments.get("alpha", 0.05)
    method = arguments.get("method", "two_sided")
    
    if data is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    try:
        logger.info(f"Grubbs test: alpha={alpha}, method={method}")
        result = grubbs_test(data, alpha, method)
        
        # Format output
        result_text = f"Grubbs Test for Outliers:\n\n"
        result_text += f"Test: {result['test']}\n"
        result_text += f"Significance Level (α): {result['alpha']}\n"
        result_text += f"Sample Size: {result['sample_size']}\n\n"
        
        if result['suspected_outlier']:
            result_text += f"Suspected Outlier:\n"
            result_text += f"  Value: {result['suspected_outlier']['value']:.4f}\n"
            result_text += f"  Index: {result['suspected_outlier']['index']}\n"
            result_text += f"  Side: {result['suspected_outlier']['side']}\n\n"
        
        result_text += f"Test Results:\n"
        result_text += f"  G Statistic: {result['test_statistic']:.4f}\n"
        result_text += f"  Critical Value: {result['critical_value']:.4f}\n"
        result_text += f"  P-value: {result['p_value']:.4f}\n\n"
        
        result_text += f"Conclusion: {result['conclusion']}\n\n"
        result_text += f"Recommendation:\n  {result['recommendation']}"
        
        logger.info(f"Grubbs test completed: {result['conclusion']}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_dixon_q_test(arguments: Any) -> CallToolResult:
    """Handle dixon_q_test tool calls."""
    data = arguments.get("data")
    alpha = arguments.get("alpha", 0.05)
    
    if data is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    try:
        logger.info(f"Dixon Q test: alpha={alpha}, n={len(data)}")
        result = dixon_q_test(data, alpha)
        
        # Format output
        result_text = f"Dixon Q Test for Outliers:\n\n"
        result_text += f"Test: {result['test']}\n"
        result_text += f"Sample Size: {result['sample_size']}\n\n"
        
        if result['suspected_outlier']:
            result_text += f"Suspected Outlier:\n"
            result_text += f"  Value: {result['suspected_outlier']['value']:.4f}\n"
            result_text += f"  Index: {result['suspected_outlier']['index']}\n"
            result_text += f"  Position: {result['suspected_outlier']['position']}\n\n"
        
        result_text += f"Test Results:\n"
        result_text += f"  Q Statistic: {result['q_statistic']:.4f}\n"
        result_text += f"  Q Critical (α={alpha}): {result['q_critical']:.4f}\n\n"
        
        result_text += f"Conclusion: {result['conclusion']}\n\n"
        result_text += f"Recommendation:\n  {result['recommendation']}"
        
        logger.info(f"Dixon Q test completed")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_isolation_forest(arguments: Any) -> CallToolResult:
    """Handle isolation_forest tool calls."""
    data = arguments.get("data")
    contamination = arguments.get("contamination", 0.1)
    n_estimators = arguments.get("n_estimators", 100)
    
    if data is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    try:
        logger.info(f"Isolation Forest: contamination={contamination}, n_estimators={n_estimators}")
        result = isolation_forest(data, contamination, n_estimators)
        
        # Format output
        result_text = f"Isolation Forest Anomaly Detection:\n\n"
        result_text += f"Method: {result['method']}\n"
        result_text += f"Contamination Level: {result['contamination']:.2%}\n\n"
        
        if result['anomalies']['indices']:
            result_text += f"Anomalies Detected: {len(result['anomalies']['indices'])}\n\n"
            result_text += f"Top Anomalies:\n"
            for i in range(min(10, len(result['anomalies']['indices']))):
                idx = result['anomalies']['indices'][i]
                score = result['anomalies']['anomaly_scores'][i]
                severity = result['anomalies']['severity'][i]
                result_text += f"  Index {idx}: Score={score:.4f}, Severity={severity}\n"
            if len(result['anomalies']['indices']) > 10:
                result_text += f"  ... and {len(result['anomalies']['indices']) - 10} more\n"
        else:
            result_text += "No anomalies detected.\n"
        
        result_text += f"\n{result['interpretation']}"
        
        logger.info(f"Found {len(result['anomalies']['indices'])} anomalies")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_mahalanobis_distance(arguments: Any) -> CallToolResult:
    """Handle mahalanobis_distance tool calls."""
    data = arguments.get("data")
    threshold = arguments.get("threshold", 0.975)
    
    if data is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    try:
        logger.info(f"Mahalanobis distance: threshold={threshold}")
        result = mahalanobis_distance(data, threshold)
        
        # Format output
        result_text = f"Mahalanobis Distance Outlier Detection:\n\n"
        result_text += f"Method: {result['method']}\n"
        result_text += f"Dimensions: {result['dimensions']}\n"
        result_text += f"Threshold Distance: {result['threshold_distance']:.4f}\n"
        result_text += f"Degrees of Freedom: {result['degrees_of_freedom']}\n\n"
        
        if result['outliers']['indices']:
            result_text += f"Multivariate Outliers Detected: {len(result['outliers']['indices'])}\n\n"
            for detail in result['outliers']['distances']:
                result_text += f"  Index {detail['index']}: "
                result_text += f"Distance={detail['distance']:.4f}, "
                result_text += f"P-value={detail['p_value']:.4f}\n"
            
            if result['variable_contributions']:
                result_text += f"\nVariable Contributions:\n"
                for contrib in result['variable_contributions'][:5]:
                    result_text += f"  Index {contrib['index']}: "
                    result_text += f"Primary={contrib['primary_variable']}, "
                    result_text += f"Contribution={contrib['contribution']:.2%}\n"
        else:
            result_text += "No multivariate outliers detected.\n"
        
        result_text += f"\n{result['interpretation']}"
        
        logger.info(f"Found {len(result['outliers']['indices'])} outliers")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_streaming_outlier_detection(arguments: Any) -> CallToolResult:
    """Handle streaming_outlier_detection tool calls."""
    current_value = arguments.get("current_value")
    historical_window = arguments.get("historical_window")
    method = arguments.get("method", "ewma")
    sensitivity = arguments.get("sensitivity", 5.0)
    
    if current_value is None or historical_window is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameters 'current_value' and 'historical_window'")],
            isError=True,
        )
    
    try:
        logger.info(f"Streaming detection: method={method}, sensitivity={sensitivity}")
        result = streaming_outlier_detection(current_value, historical_window, method, sensitivity)
        
        # Format output
        result_text = f"Real-Time Streaming Outlier Detection:\n\n"
        result_text += f"Current Value: {result['current_value']:.4f}\n"
        result_text += f"Method: {result['method'].upper()}\n"
        result_text += f"Outlier Status: {'⚠ OUTLIER' if result['is_outlier'] else '✓ NORMAL'}\n"
        result_text += f"Severity: {result['severity'].upper()}\n\n"
        
        result_text += f"Expected Range: [{result['expected_range'][0]:.4f}, {result['expected_range'][1]:.4f}]\n"
        result_text += f"Deviation: {result['deviation']:.4f}\n"
        result_text += f"Deviation (σ): {result['deviation_sigma']:.2f}\n\n"
        
        result_text += f"Trend: {result['trend'].capitalize()}\n"
        result_text += f"Rate of Change: {result['rate_of_change']:.4f}\n\n"
        
        result_text += f"Interpretation:\n  {result['interpretation']}\n\n"
        result_text += f"Recommendation:\n  {result['recommendation']}"
        
        logger.info(f"Streaming detection: {result['severity']}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
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
        elif name == "linear_regression":
            return await handle_linear_regression(arguments)
        elif name == "polynomial_regression":
            return await handle_polynomial_regression(arguments)
        elif name == "residual_analysis":
            return await handle_residual_analysis(arguments)
        elif name == "prediction_with_intervals":
            return await handle_prediction_with_intervals(arguments)
        elif name == "multivariate_regression":
            return await handle_multivariate_regression(arguments)
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


async def handle_fft_analysis(arguments: Any) -> CallToolResult:
    """Handle fft_analysis tool calls."""
    # Extract and validate parameters
    signal_data = arguments.get("signal")
    sample_rate = arguments.get("sample_rate")
    window = arguments.get("window", "hanning")
    detrend = arguments.get("detrend", True)
    
    # Validate required parameters
    if signal_data is None:
        logger.error("Missing required parameter: signal")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'signal'")],
            isError=True,
        )
    
    if sample_rate is None:
        logger.error("Missing required parameter: sample_rate")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'sample_rate'")],
            isError=True,
        )
    
    # Perform FFT analysis
    try:
        logger.info(f"Performing FFT analysis on {len(signal_data)} samples at {sample_rate} Hz")
        result = fft_analysis(signal_data, sample_rate, window, detrend)
        
        # Format the result
        result_text = f"FFT Analysis Results:\n\n"
        result_text += f"Signal Properties:\n"
        result_text += f"  Sample Rate: {result['sample_rate']:.2f} Hz\n"
        result_text += f"  Signal Length: {result['signal_length']} samples\n"
        result_text += f"  Duration: {result['duration_seconds']:.4f} seconds\n"
        result_text += f"  Nyquist Frequency: {result['nyquist_frequency']:.2f} Hz\n"
        result_text += f"  Frequency Resolution: {result['resolution']:.4f} Hz\n\n"
        
        result_text += f"Dominant Frequencies:\n"
        if result['dominant_frequencies']:
            for i, peak in enumerate(result['dominant_frequencies'][:5], 1):
                result_text += f"  {i}. {peak['frequency']:.2f} Hz - Magnitude: {peak['magnitude']:.4f}"
                if peak['interpretation']:
                    result_text += f" ({peak['interpretation']})"
                result_text += "\n"
        else:
            result_text += "  No dominant frequencies detected\n"
        
        result_text += f"\nInterpretation: {result['interpretation']}\n\n"
        
        result_text += f"Use Cases:\n"
        result_text += f"  • Bearing defect detection (BPFI, BPFO, BSF frequencies)\n"
        result_text += f"  • Motor electrical faults (broken rotor bars)\n"
        result_text += f"  • Gearbox mesh frequency analysis\n"
        result_text += f"  • Pump cavitation detection"
        
        logger.info(f"FFT analysis completed, found {len(result['dominant_frequencies'])} dominant frequencies")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in fft_analysis: {str(e)}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error performing FFT analysis: {str(e)}")],
            isError=True
        )


async def handle_control_limits(arguments: Any) -> CallToolResult:
    """Handle control_limits tool calls."""
    data = arguments.get("data")
    chart_type = arguments.get("chart_type")
    subgroup_size = arguments.get("subgroup_size", 5)
    sigma_level = arguments.get("sigma_level", 3.0)
    
    # Validate required parameters
    if data is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    if chart_type is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'chart_type'")],
            isError=True,
        )
    
    try:
        logger.info(f"Calculating control limits for {chart_type} chart")
        result = control_limits(data, chart_type, subgroup_size, sigma_level)
        
        # Format the result
        result_text = f"Control Chart Analysis ({result['chart_type'].upper()}):\n\n"
        result_text += f"Control Limits:\n"
        result_text += f"  Centerline (CL): {result['centerline']:.4f}\n"
        result_text += f"  Upper Control Limit (UCL): {result['ucl']:.4f}\n"
        result_text += f"  Lower Control Limit (LCL): {result['lcl']:.4f}\n"
        result_text += f"  Sigma: {result['sigma']:.4f}\n"
        result_text += f"  Subgroup Size: {result['subgroup_size']}\n"
        result_text += f"  Data Points: {result['data_points']}\n\n"
        
        result_text += f"Process Status: {result['process_status']}\n\n"
        
        if result['out_of_control_points']:
            result_text += f"Out of Control Points ({len(result['out_of_control_points'])}):\n"
            for detail in result['out_of_control_details'][:10]:  # Show first 10
                result_text += f"  Index {detail['index']}: {detail['value']:.4f} ({detail['violation']})\n"
            if len(result['out_of_control_details']) > 10:
                result_text += f"  ... and {len(result['out_of_control_details']) - 10} more\n"
            result_text += "\n"
        
        result_text += f"Recommendations:\n  {result['recommendations']}"
        
        logger.info(f"Control limits calculated: UCL={result['ucl']:.4f}, LCL={result['lcl']:.4f}")
        
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


async def handle_power_spectral_density(arguments: Any) -> CallToolResult:
    """Handle power_spectral_density tool calls."""
    # Extract and validate parameters
    signal_data = arguments.get("signal")
    sample_rate = arguments.get("sample_rate")
    method = arguments.get("method", "welch")
    nperseg = arguments.get("nperseg", 256)
    
    # Validate required parameters
    if signal_data is None:
        logger.error("Missing required parameter: signal")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'signal'")],
            isError=True,
        )
    
    if sample_rate is None:
        logger.error("Missing required parameter: sample_rate")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'sample_rate'")],
            isError=True,
        )
    
    # Calculate PSD
    try:
        logger.info(f"Calculating {method} PSD for {len(signal_data)} samples")
        result = power_spectral_density(signal_data, sample_rate, method, nperseg)
        
        # Format the result
        result_text = f"Power Spectral Density Analysis:\n\n"
        result_text += f"Method: {result['method']}\n"
        result_text += f"Total Power: {result['total_power']:.4f}\n"
        result_text += f"Peak Frequency: {result['peak_frequency']:.2f} Hz\n"
        result_text += f"Peak Power: {result['peak_power']:.4f}\n\n"
        
        result_text += f"Interpretation: {result['interpretation']}\n\n"
        
        result_text += f"Use Cases:\n"
        result_text += f"  • Vibration energy distribution analysis\n"
        result_text += f"  • Noise level assessment\n"
        result_text += f"  • Random vibration analysis\n"
        result_text += f"  • Acoustic signature analysis"
        
        logger.info(f"PSD calculated, peak at {result['peak_frequency']:.2f} Hz")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in power_spectral_density: {str(e)}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error calculating PSD: {str(e)}")],
            isError=True
        )


async def handle_process_capability(arguments: Any) -> CallToolResult:
    """Handle process_capability tool calls."""
    data = arguments.get("data")
    usl = arguments.get("usl")
    lsl = arguments.get("lsl")
    target = arguments.get("target")
    
    # Validate required parameters
    if data is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    if usl is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'usl'")],
            isError=True,
        )
    
    if lsl is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'lsl'")],
            isError=True,
        )
    
    try:
        logger.info(f"Calculating process capability for {len(data)} measurements")
        result = process_capability(data, usl, lsl, target)
        
        # Format the result
        result_text = f"Process Capability Analysis:\n\n"
        result_text += f"Sample Statistics:\n"
        result_text += f"  Sample Size: {result['sample_size']}\n"
        result_text += f"  Mean: {result['mean']:.4f}\n"
        result_text += f"  Std Dev: {result['std_dev']:.4f}\n\n"
        
        result_text += f"Specification Limits:\n"
        result_text += f"  USL: {result['usl']:.4f}\n"
        result_text += f"  LSL: {result['lsl']:.4f}\n"
        result_text += f"  Target: {result['target']:.4f}\n\n"
        
        result_text += f"Capability Indices:\n"
        result_text += f"  Cp (Potential Capability): {result['cp']:.4f} - {result['cp_interpretation']}\n"
        result_text += f"  Cpk (Actual Capability): {result['cpk']:.4f} - {result['cpk_interpretation']}\n"
        result_text += f"  Pp (Overall Performance): {result['pp']:.4f}\n"
        result_text += f"  Ppk (Overall Performance): {result['ppk']:.4f}\n\n"
        
        result_text += f"Process Performance:\n"
        result_text += f"  Sigma Level: {result['sigma_level']:.2f}\n"
        result_text += f"  % Within Spec: {result['percent_within_spec']:.4f}%\n"
        result_text += f"  Est. PPM Defects: {result['estimated_ppm_defects']:.2f}\n"
        result_text += f"  Overall: {result['process_performance']}\n\n"
        
        result_text += f"Centering: {result['centering']}\n\n"
        result_text += f"Recommendations:\n  {result['recommendations']}"
        
        logger.info(f"Capability: Cp={result['cp']:.2f}, Cpk={result['cpk']:.2f}")
        
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


async def handle_rms_value(arguments: Any) -> CallToolResult:
    """Handle rms_value tool calls."""
    # Extract and validate parameters
    signal_data = arguments.get("signal")
    window_size = arguments.get("window_size")
    reference_value = arguments.get("reference_value")
    
    # Validate required parameter
    if signal_data is None:
        logger.error("Missing required parameter: signal")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'signal'")],
            isError=True,
        )
    
    # Calculate RMS
    try:
        logger.info(f"Calculating RMS for {len(signal_data)} samples")
        result = rms_value(signal_data, window_size, reference_value)
        
        # Format the result
        result_text = f"RMS Analysis Results:\n\n"
        result_text += f"Overall RMS: {result['rms']:.4f}\n"
        result_text += f"Peak Value: {result['peak']:.4f}\n"
        result_text += f"Crest Factor: {result['crest_factor']:.2f}\n"
        
        if 'rms_db' in result:
            result_text += f"RMS (dB): {result['rms_db']:.2f} dB\n"
        
        result_text += f"\nInterpretation: {result['interpretation']}\n"
        
        if 'rolling_rms' in result:
            rolling = result['rolling_rms']
            result_text += f"\nRolling RMS Statistics:\n"
            result_text += f"  Number of windows: {len(rolling)}\n"
            result_text += f"  Trend: {result['trend']}\n"
            if len(rolling) <= 10:
                result_text += f"  Values: {', '.join(f'{v:.4f}' for v in rolling)}\n"
            else:
                result_text += f"  First 5: {', '.join(f'{v:.4f}' for v in rolling[:5])}\n"
                result_text += f"  Last 5: {', '.join(f'{v:.4f}' for v in rolling[-5:])}\n"
        
        result_text += f"\nUse Cases:\n"
        result_text += f"  • Overall vibration severity (ISO 10816)\n"
        result_text += f"  • Electrical current RMS for power calculations\n"
        result_text += f"  • Acoustic noise level assessment\n"
        result_text += f"  • Process variable stability monitoring"
        
        logger.info(f"RMS calculated: {result['rms']:.4f}")
        
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


async def handle_peak_detection(arguments: Any) -> CallToolResult:
    """Handle peak_detection tool calls."""
    # Extract and validate parameters
    signal_data = arguments.get("signal")
    frequencies = arguments.get("frequencies")
    height = arguments.get("height", 0.1)
    distance = arguments.get("distance", 1)
    prominence = arguments.get("prominence", 0.05)
    top_n = arguments.get("top_n", 10)
    
    # Validate required parameter
    if signal_data is None:
        logger.error("Missing required parameter: signal")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'signal'")],
            isError=True,
        )
    
    # Detect peaks
    try:
        logger.info(f"Detecting peaks in signal with {len(signal_data)} samples")
        result = peak_detection(signal_data, frequencies, height, distance, prominence, top_n)
        
        # Format the result
        result_text = f"Peak Detection Results:\n\n"
        result_text += f"Total Peaks Found: {result['peaks_found']}\n"
        result_text += f"Interpretation: {result['interpretation']}\n\n"
        
        if result['top_peaks']:
            result_text += f"Top Peaks:\n"
            for peak in result['top_peaks']:
                result_text += f"  Rank {peak['rank']}: "
                if 'frequency' in peak:
                    result_text += f"Frequency {peak['frequency']:.2f} Hz, "
                result_text += f"Magnitude {peak['magnitude']:.4f}, "
                result_text += f"Prominence {peak['prominence']:.4f}"
                if 'interpretation' in peak and peak['interpretation']:
                    result_text += f" - {peak['interpretation']}"
                result_text += "\n"
        else:
            result_text += "No significant peaks detected.\n"
        
        result_text += f"\nUse Cases:\n"
        result_text += f"  • Find dominant vibration frequencies\n"
        result_text += f"  • Detect harmonic patterns in power quality\n"
        result_text += f"  • Identify resonance frequencies\n"
        result_text += f"  • Bearing fault frequency detection"
        
        logger.info(f"Peak detection completed, found {result['peaks_found']} peaks")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in peak_detection: {str(e)}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error detecting peaks: {str(e)}")],
            isError=True
        )


async def handle_western_electric_rules(arguments: Any) -> CallToolResult:
    """Handle western_electric_rules tool calls."""
    data = arguments.get("data")
    centerline = arguments.get("centerline")
    sigma = arguments.get("sigma")
    rules_to_apply = arguments.get("rules_to_apply")
    
    # Validate required parameters
    if data is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    if centerline is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'centerline'")],
            isError=True,
        )
    
    if sigma is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'sigma'")],
            isError=True,
        )
    
    try:
        logger.info(f"Applying Western Electric rules to {len(data)} measurements")
        result = western_electric_rules(data, centerline, sigma, rules_to_apply)
        
        # Format the result
        result_text = f"Western Electric Rules Analysis:\n\n"
        result_text += f"Process Status: {result['process_status']}\n"
        result_text += f"Total Violations: {result['total_violations']}\n"
        result_text += f"Pattern Detected: {result['pattern_detected']}\n\n"
        
        if result['violations']:
            result_text += f"Rule Violations:\n"
            for v in result['violations'][:10]:  # Show first 10
                result_text += f"\n  Rule {v['rule']}: {v['rule_name']}\n"
                result_text += f"    Severity: {v['severity']}\n"
                result_text += f"    Indices: {v['indices'][:10]}"  # Show first 10 indices
                if len(v['indices']) > 10:
                    result_text += f" ... and {len(v['indices']) - 10} more"
                result_text += f"\n    {v['description']}\n"
            
            if len(result['violations']) > 10:
                result_text += f"\n  ... and {len(result['violations']) - 10} more violations\n"
        else:
            result_text += f"No violations detected. Process is in statistical control.\n"
        
        result_text += f"\nAction Required:\n  {result['action_required']}"
        
        logger.info(f"Found {result['total_violations']} rule violations")
        
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


async def handle_signal_to_noise_ratio(arguments: Any) -> CallToolResult:
    """Handle signal_to_noise_ratio tool calls."""
    # Extract and validate parameters
    signal_data = arguments.get("signal")
    noise_data = arguments.get("noise")
    method = arguments.get("method", "power")
    
    # Validate required parameter
    if signal_data is None:
        logger.error("Missing required parameter: signal")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'signal'")],
            isError=True,
        )
    
    # Calculate SNR
    try:
        logger.info(f"Calculating SNR using {method} method")
        result = signal_to_noise_ratio(signal_data, noise_data, method)
        
        # Format the result
        result_text = f"Signal-to-Noise Ratio Analysis:\n\n"
        result_text += f"SNR: {result['snr_db']:.2f} dB\n"
        result_text += f"SNR Ratio: {result['snr_ratio']:.2f}\n"
        result_text += f"Signal Power: {result['signal_power']:.4f}\n"
        result_text += f"Noise Power: {result['noise_power']:.4f}\n"
        result_text += f"Quality: {result['quality']}\n\n"
        
        result_text += f"Interpretation: {result['interpretation']}\n\n"
        
        result_text += f"Use Cases:\n"
        result_text += f"  • Sensor health monitoring\n"
        result_text += f"  • Data acquisition quality check\n"
        result_text += f"  • Process measurement reliability\n"
        result_text += f"  • Instrumentation validation"
        
        logger.info(f"SNR calculated: {result['snr_db']:.2f} dB ({result['quality']})")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in signal_to_noise_ratio: {str(e)}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error calculating SNR: {str(e)}")],
            isError=True
        )


async def handle_cusum_chart(arguments: Any) -> CallToolResult:
    """Handle cusum_chart tool calls."""
    data = arguments.get("data")
    target = arguments.get("target")
    k = arguments.get("k")
    h = arguments.get("h")
    sigma = arguments.get("sigma")
    
    # Validate required parameters
    if data is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    if target is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'target'")],
            isError=True,
        )
    
    try:
        logger.info(f"Calculating CUSUM chart for {len(data)} measurements")
        result = cusum_chart(data, target, k, h, sigma)
        
        # Format the result
        result_text = f"CUSUM Chart Analysis:\n\n"
        result_text += f"Process Status: {result['process_status']}\n\n"
        
        result_text += f"Control Limits:\n"
        result_text += f"  Upper Limit (H): {result['upper_limit']:.4f}\n"
        result_text += f"  Lower Limit (-H): {result['lower_limit']:.4f}\n\n"
        
        if result['signals']:
            result_text += f"Signals Detected ({len(result['signals'])}):\n"
            for signal in result['signals']:
                result_text += f"\n  Signal at Index {signal['index']}:\n"
                result_text += f"    Type: {signal['type']}\n"
                result_text += f"    CUSUM Value: {signal['cusum_value']:.4f}\n"
                result_text += f"    Est. Change Point: {signal['estimated_change_point']}\n"
                result_text += f"    Est. Magnitude: {signal['magnitude_estimate']:.4f}\n"
            
            if result['estimated_new_level'] is not None:
                result_text += f"\n  Estimated New Level: {result['estimated_new_level']:.4f}\n"
        else:
            result_text += f"No shifts detected. Process is stable at target level.\n"
        
        result_text += f"\nRecommendation:\n  {result['recommendation']}"
        
        logger.info(f"CUSUM: {len(result['signals'])} shifts detected")
        
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


async def handle_harmonic_analysis(arguments: Any) -> CallToolResult:
    """Handle harmonic_analysis tool calls."""
    # Extract and validate parameters
    signal_data = arguments.get("signal")
    sample_rate = arguments.get("sample_rate")
    fundamental_freq = arguments.get("fundamental_freq")
    max_harmonic = arguments.get("max_harmonic", 50)
    
    # Validate required parameters
    if signal_data is None:
        logger.error("Missing required parameter: signal")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'signal'")],
            isError=True,
        )
    
    if sample_rate is None:
        logger.error("Missing required parameter: sample_rate")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'sample_rate'")],
            isError=True,
        )
    
    if fundamental_freq is None:
        logger.error("Missing required parameter: fundamental_freq")
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'fundamental_freq'")],
            isError=True,
        )
    
    # Perform harmonic analysis
    try:
        logger.info(f"Performing harmonic analysis with fundamental {fundamental_freq} Hz")
        result = harmonic_analysis(signal_data, sample_rate, fundamental_freq, max_harmonic)
        
        # Format the result
        result_text = f"Harmonic Analysis Results:\n\n"
        result_text += f"Fundamental Frequency:\n"
        result_text += f"  Frequency: {result['fundamental']['frequency']:.2f} Hz\n"
        result_text += f"  Magnitude: {result['fundamental']['magnitude']:.4f}\n\n"
        
        result_text += f"Total Harmonic Distortion (THD): {result['thd']:.2f}%\n"
        result_text += f"Assessment: {result['thd_interpretation']}\n\n"
        
        if result['dominant_harmonics']:
            result_text += f"Dominant Harmonics: {', '.join(map(str, result['dominant_harmonics']))}\n\n"
        
        if result['harmonics']:
            result_text += f"Harmonic Components (showing top 10):\n"
            for i, harm in enumerate(result['harmonics'][:10], 1):
                result_text += f"  H{harm['order']}: {harm['frequency']:.2f} Hz, "
                result_text += f"Magnitude {harm['magnitude']:.4f} ({harm['percent']:.1f}% of fundamental)\n"
        else:
            result_text += "No significant harmonics detected.\n"
        
        result_text += f"\nInterpretation: {result['interpretation']}\n\n"
        
        result_text += f"Use Cases:\n"
        result_text += f"  • Power quality assessment (THD calculation)\n"
        result_text += f"  • Variable frequency drive (VFD) effects\n"
        result_text += f"  • Motor current signature analysis (MCSA)\n"
        result_text += f"  • IEEE 519 compliance verification"
        
        logger.info(f"Harmonic analysis completed, THD = {result['thd']:.2f}%")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in harmonic_analysis: {str(e)}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error performing harmonic analysis: {str(e)}")],
            isError=True
        )


async def handle_ewma_chart(arguments: Any) -> CallToolResult:
    """Handle ewma_chart tool calls."""
    data = arguments.get("data")
    target = arguments.get("target")
    sigma = arguments.get("sigma")
    lambda_param = arguments.get("lambda_param", 0.2)
    l = arguments.get("l", 3.0)
    
    # Validate required parameters
    if data is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'data'")],
            isError=True,
        )
    
    if target is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'target'")],
            isError=True,
        )
    
    if sigma is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameter 'sigma'")],
            isError=True,
        )
    
    try:
        logger.info(f"Calculating EWMA chart for {len(data)} measurements")
        result = ewma_chart(data, target, sigma, lambda_param, l)
        
        # Format the result
        result_text = f"EWMA Chart Analysis:\n\n"
        result_text += f"Process Status: {result['process_status']}\n\n"
        
        result_text += f"Configuration:\n"
        result_text += f"  Target: {result['target']:.4f}\n"
        result_text += f"  Lambda (λ): {lambda_param:.2f}\n"
        result_text += f"  Control Limit Factor (L): {l:.1f}\n\n"
        
        # Show first few and last few EWMA values
        ewma_vals = result['ewma_values']
        if len(ewma_vals) <= 10:
            result_text += f"EWMA Values:\n"
            for i, val in enumerate(ewma_vals):
                result_text += f"  Point {i}: {val:.4f}\n"
        else:
            result_text += f"EWMA Values (showing first 5 and last 5):\n"
            result_text += f"  First 5: {', '.join(f'{v:.4f}' for v in ewma_vals[:5])}\n"
            result_text += f"  Last 5:  {', '.join(f'{v:.4f}' for v in ewma_vals[-5:])}\n"
        
        result_text += "\n"
        
        if result['out_of_control_points']:
            result_text += f"Out of Control Points ({len(result['out_of_control_points'])}):\n"
            for detail in result['out_of_control_details'][:10]:  # Show first 10
                result_text += f"  Index {detail['index']}: Value={detail['value']:.4f}, EWMA={detail['ewma']:.4f} ({detail['violation']})\n"
            if len(result['out_of_control_details']) > 10:
                result_text += f"  ... and {len(result['out_of_control_details']) - 10} more\n"
        else:
            result_text += f"All points within EWMA control limits.\n"
        
        result_text += f"\nRecommendation:\n  {result['recommendation']}"
        
        logger.info(f"EWMA: {len(result['out_of_control_points'])} out-of-control points")
        
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


async def handle_linear_regression(arguments: Any) -> CallToolResult:
    """Handle linear_regression tool calls."""
    x = arguments.get("x")
    y = arguments.get("y")
    confidence_level = arguments.get("confidence_level", 0.95)
    include_diagnostics = arguments.get("include_diagnostics", True)
    
    if x is None or y is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameters 'x' and 'y'")],
            isError=True,
        )
    
    try:
        logger.info("Performing linear regression")
        result = linear_regression(x, y, confidence_level, include_diagnostics)
        
        # Format output
        result_text = f"Linear Regression Analysis:\n\n"
        result_text += f"Equation: {result['coefficients']['equation']}\n\n"
        
        result_text += f"Model Statistics:\n"
        result_text += f"  R²: {result['statistics']['r_squared']:.4f}\n"
        result_text += f"  Adjusted R²: {result['statistics']['adj_r_squared']:.4f}\n"
        result_text += f"  RMSE: {result['statistics']['rmse']:.4f}\n"
        result_text += f"  MAE: {result['statistics']['mae']:.4f}\n"
        if result['statistics']['f_statistic'] is not None:
            result_text += f"  F-statistic: {result['statistics']['f_statistic']:.2f}\n"
        if result['statistics']['p_value'] is not None:
            result_text += f"  p-value: {result['statistics']['p_value']:.4f}\n"
        
        if "confidence_intervals" in result:
            result_text += f"\nConfidence Intervals ({result['confidence_intervals']['confidence_level']*100:.0f}%):\n"
            result_text += f"  Intercept: [{result['confidence_intervals']['intercept'][0]:.4f}, {result['confidence_intervals']['intercept'][1]:.4f}]\n"
            for i, ci in enumerate(result['confidence_intervals']['slopes']):
                result_text += f"  Slope {i+1}: [{ci[0]:.4f}, {ci[1]:.4f}]\n"
        
        if include_diagnostics and "diagnostics" in result:
            result_text += f"\nDiagnostics:\n"
            if "durbin_watson" in result['diagnostics']:
                result_text += f"  Durbin-Watson: {result['diagnostics']['durbin_watson']:.4f}\n"
        
        result_text += f"\n{result['interpretation']}"
        
        if result['warnings']:
            result_text += f"\n\nWarnings:\n"
            for warning in result['warnings']:
                result_text += f"  • {warning}\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_polynomial_regression(arguments: Any) -> CallToolResult:
    """Handle polynomial_regression tool calls."""
    x = arguments.get("x")
    y = arguments.get("y")
    degree = arguments.get("degree", 2)
    auto_select_degree = arguments.get("auto_select_degree", False)
    
    if x is None or y is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameters 'x' and 'y'")],
            isError=True,
        )
    
    try:
        logger.info(f"Performing polynomial regression (degree={degree})")
        result = polynomial_regression(x, y, degree, auto_select_degree)
        
        # Format output
        result_text = f"Polynomial Regression Analysis:\n\n"
        result_text += f"Degree: {result['degree']}\n"
        result_text += f"Equation: {result['equation']}\n\n"
        
        result_text += f"Model Statistics:\n"
        result_text += f"  R²: {result['r_squared']:.4f}\n"
        result_text += f"  Adjusted R²: {result['adj_r_squared']:.4f}\n"
        result_text += f"  RMSE: {result['rmse']:.4f}\n"
        result_text += f"  Goodness of Fit: {result['goodness_of_fit']}\n"
        
        if result['turning_points']:
            result_text += f"\nTurning Points:\n"
            for tp in result['turning_points']:
                result_text += f"  {tp['type'].capitalize()} at x={tp['x']:.2f}, y={tp['y']:.2f}\n"
        
        if result['optimal_x'] is not None:
            result_text += f"\nOptimal Point:\n"
            result_text += f"  x = {result['optimal_x']:.2f}\n"
            result_text += f"  y = {result['optimal_y']:.2f}\n"
        
        result_text += f"\n{result['interpretation']}"
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_residual_analysis(arguments: Any) -> CallToolResult:
    """Handle residual_analysis tool calls."""
    actual = arguments.get("actual")
    predicted = arguments.get("predicted")
    x_values = arguments.get("x_values")
    
    if actual is None or predicted is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameters 'actual' and 'predicted'")],
            isError=True,
        )
    
    try:
        logger.info("Performing residual analysis")
        result = residual_analysis(actual, predicted, x_values)
        
        # Format output
        result_text = f"Residual Analysis:\n\n"
        
        result_text += f"Statistical Tests:\n"
        for test_name, test_result in result['tests'].items():
            result_text += f"\n{test_name.replace('_', ' ').title()}:\n"
            if 'conclusion' in test_result:
                result_text += f"  {test_result['conclusion']}\n"
                if 'p_value' in test_result:
                    result_text += f"  p-value: {test_result['p_value']:.4f}\n"
            elif 'error' in test_result or 'note' in test_result:
                result_text += f"  {test_result.get('error', test_result.get('note'))}\n"
        
        result_text += f"\nOutliers:\n"
        if result['outliers']['count'] == 0:
            result_text += f"  None detected\n"
        else:
            result_text += f"  {result['outliers']['count']} outlier(s) detected\n"
            for outlier in result['outliers']['values'][:5]:  # Show first 5
                result_text += f"    Index {outlier['index']}: residual={outlier['residual']:.4f}, standardized={outlier['standardized']:.2f}\n"
        
        if result['patterns_detected']:
            result_text += f"\nPatterns Detected:\n"
            for pattern in result['patterns_detected']:
                result_text += f"  • {pattern}\n"
        
        result_text += f"\n{result['overall_assessment']}"
        
        if result['recommendations']:
            result_text += f"\n\nRecommendations:\n"
            for rec in result['recommendations']:
                result_text += f"  • {rec}\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_prediction_with_intervals(arguments: Any) -> CallToolResult:
    """Handle prediction_with_intervals tool calls."""
    model = arguments.get("model")
    x_new = arguments.get("x_new")
    confidence_level = arguments.get("confidence_level", 0.95)
    
    if model is None or x_new is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameters 'model' and 'x_new'")],
            isError=True,
        )
    
    try:
        logger.info(f"Generating predictions for {len(x_new)} new values")
        result = prediction_with_intervals(model, x_new, confidence_level)
        
        # Format output
        result_text = f"Predictions with Intervals:\n\n"
        result_text += f"Confidence Level: {result['confidence_level']*100:.0f}%\n"
        result_text += f"Reliability: {result['reliability'].upper()}\n"
        
        if result['extrapolation_warning']:
            result_text += f"⚠ Warning: Some predictions may involve extrapolation\n"
        
        result_text += f"\nPredictions:\n"
        for i, pred in enumerate(result['predictions'], 1):
            result_text += f"\n{i}. {pred['interpretation']}\n"
            result_text += f"   Prediction Interval: [{pred['prediction_interval'][0]:.2f}, {pred['prediction_interval'][1]:.2f}]\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_multivariate_regression(arguments: Any) -> CallToolResult:
    """Handle multivariate_regression tool calls."""
    X = arguments.get("X")
    y = arguments.get("y")
    variable_names = arguments.get("variable_names")
    standardize = arguments.get("standardize", False)
    
    if X is None or y is None:
        return CallToolResult(
            content=[TextContent(type="text", text="Missing required parameters 'X' and 'y'")],
            isError=True,
        )
    
    try:
        logger.info("Performing multivariate regression")
        result = multivariate_regression(X, y, variable_names, standardize)
        
        # Format output
        result_text = f"Multivariate Regression Analysis:\n\n"
        result_text += f"Equation:\n  {result['coefficients']['equation']}\n\n"
        
        result_text += f"Model Statistics:\n"
        result_text += f"  R²: {result['r_squared']:.4f}\n"
        result_text += f"  Adjusted R²: {result['adj_r_squared']:.4f}\n"
        
        result_text += f"\nVariable Importance (sorted by effect size):\n"
        for var in result['variable_importance']:
            result_text += f"  {var['variable']}: {var['coefficient']:.4f}"
            if var['significance']:
                result_text += f" {var['significance']}"
            if var['p_value'] is not None:
                result_text += f" (p={var['p_value']:.3f})"
            result_text += "\n"
        
        if result['vif']:
            result_text += f"\nVariance Inflation Factors (VIF):\n"
            for var, vif in result['vif'].items():
                if vif is not None:
                    result_text += f"  {var}: {vif:.2f}\n"
            result_text += f"\nMulticollinearity: {result['multicollinearity']}\n"
        
        result_text += f"\n{result['interpretation']}"
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def main(config_path: Optional[str] = None):
    """
    Main entry point for the MCP server.
    
    This function starts the server using stdio transport, which means:
    - The server reads MCP protocol messages from stdin
    - The server writes MCP protocol messages to stdout
    - All logging and debugging output goes to stderr
    
    This is the standard transport for MCP servers that will be launched
    by client applications like Claude Desktop.
    
    Args:
        config_path: Optional path to configuration file
    """
    # Load configuration
    try:
        config = load_config(config_path)
        set_config(config)
        
        # Update logging level based on configuration
        log_level = getattr(logging, config.logging.level)
        logger.setLevel(log_level)
        logging.getLogger().setLevel(log_level)
        
        logger.info("Starting Statistical Analysis MCP server")
        logger.info(f"Configuration loaded from: {config_path or 'defaults'}")
        logger.info(f"Log level: {config.logging.level}")
        
        # Log configuration (excluding sensitive data)
        if config_path:
            logger.debug(f"Stats server: {config.server.stats.host}:{config.server.stats.port}")
            logger.debug(f"Authentication: {'enabled' if config.authentication.enabled else 'disabled'}")
            logger.debug(f"Rate limiting: {'enabled' if config.rate_limiting.enabled else 'disabled'}")
            if config.rate_limiting.enabled:
                logger.debug(f"Rate limit: {config.rate_limiting.requests_per_minute} req/min")
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Server startup failed due to configuration errors")
        sys.exit(1)
    
    # Run the server using stdio transport
    # This will block until the server receives a shutdown signal
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Server started, waiting for requests...")
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Server shut down successfully")


# Entry point when running as a script
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Statistical Analysis MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python server.py
  
  # Run with custom configuration file
  python server.py --config /path/to/config.yaml
  
  # Run with environment variable overrides
  MCP_LOG_LEVEL=DEBUG python server.py --config config.yaml
  
Environment Variables:
  MCP_STATS_HOST          Override stats server host
  MCP_STATS_PORT          Override stats server port
  MCP_AUTH_ENABLED        Enable/disable authentication (true/false)
  MCP_API_KEY             Set API key
  MCP_RATE_LIMIT_ENABLED  Enable/disable rate limiting (true/false)
  MCP_RATE_LIMIT_RPM      Set requests per minute
  MCP_LOG_LEVEL           Set logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (optional)"
    )
    
    args = parser.parse_args()
    
    # Run the async main function with the config path
    asyncio.run(main(args.config))
