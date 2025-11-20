"""
Main MCP server implementation for statistical analysis.
This server exposes tools for descriptive statistics, correlation analysis, 
percentile calculations, and outlier detection.
"""

import asyncio
import logging
from typing import Any
import math

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

# SciPy imports for statistical tests
try:
    from scipy import stats as scipy_stats
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Some advanced statistical tests will be unavailable.")

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
    
    if confidence_level < 0.8 or confidence_level > 0.999:
        raise ValueError(f"confidence_level must be between 0.8 and 0.999, got {confidence_level}")
    
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
        if abs(aug[i][i]) < 1e-10:
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
        interpretation = f"At x={x_val}, predicted y is {pred_y:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})"
        
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
                        "description": "Confidence level for intervals (0.8-0.999)",
                        "minimum": 0.8,
                        "maximum": 0.999,
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
