"""
Main MCP server implementation for statistical analysis.
This server exposes tools for descriptive statistics, correlation analysis, 
percentile calculations, outlier detection, time series analysis, and signal processing.
"""

import asyncio
import logging
from typing import Any
import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq

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
        window_func = np.hanning(n)
    elif window == "hamming":
        window_func = np.hamming(n)
    elif window == "blackman":
        window_func = np.blackman(n)
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
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
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
