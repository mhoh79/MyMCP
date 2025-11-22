"""
Statistics MCP Server.

Provides 32+ statistical analysis tools including:
- Descriptive statistics
- Correlation analysis
- Percentile calculations
- Outlier detection (multiple methods)
- Time series analysis
- Signal processing (FFT, PSD, harmonics)
- Statistical process control
- Linear and polynomial regression
- Multivariate analysis
"""

from .server import StatsServer

__all__ = ["StatsServer"]
