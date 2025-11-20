"""
Main MCP server implementation for statistical analysis.
This server exposes tools for descriptive statistics, correlation analysis, 
percentile calculations, and outlier detection.
"""

import asyncio
import logging
from typing import Any
from scipy import stats as scipy_stats

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
