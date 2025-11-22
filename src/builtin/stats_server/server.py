"""
Statistics MCP Server - Refactored using BaseMCPServer.

This server provides tools for statistical analysis, correlation,
outlier detection, time series analysis, signal processing, and
quality control charts.
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
_parent_dir = Path(__file__).parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from core import BaseMCPServer
from .tools import (
    STATS_TOOLS,
    handle_descriptive_stats,
    handle_correlation,
    handle_percentile,
    handle_detect_outliers,
    handle_z_score_detection,
    handle_grubbs_test,
    handle_dixon_q_test,
    handle_isolation_forest,
    handle_mahalanobis_distance,
    handle_streaming_outlier_detection,
    handle_moving_average,
    handle_detect_trend,
    handle_autocorrelation,
    handle_change_point_detection,
    handle_rate_of_change,
    handle_rolling_statistics,
    handle_fft_analysis,
    handle_control_limits,
    handle_power_spectral_density,
    handle_process_capability,
    handle_rms_value,
    handle_peak_detection,
    handle_western_electric_rules,
    handle_signal_to_noise_ratio,
    handle_cusum_chart,
    handle_harmonic_analysis,
    handle_ewma_chart,
    handle_linear_regression,
    handle_polynomial_regression,
    handle_residual_analysis,
    handle_prediction_with_intervals,
    handle_multivariate_regression,
)


class StatsServer(BaseMCPServer):
    """
    Statistics MCP Server with 32+ statistical analysis tools.
    
    Inherits from BaseMCPServer to leverage:
    - Dual transport support (stdio/HTTP)
    - Automatic endpoint management
    - Built-in health/ready/metrics endpoints
    - Graceful shutdown handling
    - Middleware integration (CORS, auth, rate limiting)
    """
    
    def register_tools(self) -> None:
        """Register all statistical analysis tools."""
        # Tool name to handler mapping
        tool_handlers = {
            "descriptive_stats": handle_descriptive_stats,
            "correlation": handle_correlation,
            "percentile": handle_percentile,
            "detect_outliers": handle_detect_outliers,
            "z_score_detection": handle_z_score_detection,
            "grubbs_test": handle_grubbs_test,
            "dixon_q_test": handle_dixon_q_test,
            "isolation_forest": handle_isolation_forest,
            "mahalanobis_distance": handle_mahalanobis_distance,
            "streaming_outlier_detection": handle_streaming_outlier_detection,
            "moving_average": handle_moving_average,
            "detect_trend": handle_detect_trend,
            "autocorrelation": handle_autocorrelation,
            "change_point_detection": handle_change_point_detection,
            "rate_of_change": handle_rate_of_change,
            "rolling_statistics": handle_rolling_statistics,
            "fft_analysis": handle_fft_analysis,
            "control_limits": handle_control_limits,
            "power_spectral_density": handle_power_spectral_density,
            "process_capability": handle_process_capability,
            "rms_value": handle_rms_value,
            "peak_detection": handle_peak_detection,
            "western_electric_rules": handle_western_electric_rules,
            "signal_to_noise_ratio": handle_signal_to_noise_ratio,
            "cusum_chart": handle_cusum_chart,
            "harmonic_analysis": handle_harmonic_analysis,
            "ewma_chart": handle_ewma_chart,
            "linear_regression": handle_linear_regression,
            "polynomial_regression": handle_polynomial_regression,
            "residual_analysis": handle_residual_analysis,
            "prediction_with_intervals": handle_prediction_with_intervals,
            "multivariate_regression": handle_multivariate_regression,
        }
        
        # Register each tool with its handler
        for tool in STATS_TOOLS:
            handler = tool_handlers.get(tool.name)
            if handler:
                self.tool_registry.register_tool(tool, handler)
            else:
                self.logger.warning(f"No handler found for tool: {tool.name}")
    
    def get_server_name(self) -> str:
        """Return the server name."""
        return "stats-server"
    
    def get_server_version(self) -> str:
        """Return the server version."""
        return "2.0.0"


def main():
    """Entry point for the Statistics MCP Server."""
    # Create argument parser with server description
    parser = BaseMCPServer.create_argument_parser(
        description="Statistics MCP Server - Provides 32+ statistical analysis tools"
    )
    args = parser.parse_args()
    
    # Create and run the server
    server = StatsServer(config_path=args.config)
    server.run(
        transport=args.transport,
        host=args.host,
        port=args.port,
        dev_mode=args.dev
    )


if __name__ == "__main__":
    main()
