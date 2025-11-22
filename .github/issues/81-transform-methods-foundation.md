# Issue #81: Implement Transform Methods Foundation

**Priority**: High (required by Control Systems, Signal Processing)  
**Dependencies**: #79 (Engineering Math), #80 (Complex Analysis)  
**Labels**: enhancement, math-tools, transforms, frequency-analysis  
**Estimated Effort**: 1-2 weeks

## Overview

Implement foundational transform methods for time ↔ frequency domain analysis. These tools are essential for signal processing, control system design, vibration analysis, and system identification.

## Objectives

- Provide comprehensive Fourier analysis capabilities
- Implement convolution and correlation operations
- Enable frequency domain system analysis
- Support both continuous and discrete transforms
- Integrate with existing FFT tools in stats server

## Scope

### Group 6: Transform Methods (4 major tools + enhancements)

#### 1. `fourier_series`

**Continuous Fourier Series Decomposition**:
- Decompose periodic functions into harmonics
- Calculate Fourier coefficients (a₀, aₙ, bₙ)
- Complex exponential form (cₙ)
- Magnitude and phase spectrum

**Features**:
```python
fourier_series(
    function="square_wave",    # or custom function/data
    period=2.0,                # Period T
    num_harmonics=10,          # Number of harmonics to compute
    sample_points=1000         # Points for numerical integration
)
```

**Output**:
- DC component (a₀)
- Cosine coefficients (aₙ)
- Sine coefficients (bₙ)
- Complex coefficients (cₙ = (aₙ - jbₙ)/2)
- Magnitude spectrum |cₙ|
- Phase spectrum ∠cₙ
- Reconstructed signal
- RMS convergence error

**Common Waveforms Supported**:
- Square wave
- Triangle wave
- Sawtooth wave
- Pulse train
- Half-wave rectified sine
- Full-wave rectified sine
- Custom data input

**Applications**:
- Power quality analysis (harmonics)
- Vibration analysis
- Audio synthesis
- Heat transfer (periodic boundary conditions)

#### 2. `dft_analysis`

**Discrete Fourier Transform with Advanced Features**:
- Standard DFT computation
- Multiple windowing functions
- Zero-padding for frequency resolution
- Spectral leakage analysis
- Frequency bin interpretation

**Windowing Functions**:
- Rectangular (no window)
- Hann (Hanning)
- Hamming
- Blackman
- Kaiser (with β parameter)
- Flat-top (for amplitude accuracy)
- Custom windows

**Features**:
```python
dft_analysis(
    signal=[...],              # Time-domain signal
    sample_rate=1000,          # Sampling frequency (Hz)
    window="hann",             # Window function
    zero_padding_factor=2,     # Pad to 2× length
    output_format="magnitude_phase"  # or "real_imag"
)
```

**Output**:
- Frequency bins (Hz)
- Magnitude spectrum
- Phase spectrum  
- Power spectrum (magnitude²)
- Single-sided spectrum (positive frequencies)
- Double-sided spectrum (±frequencies)
- Window compensation factors
- Effective noise bandwidth

**Analysis Features**:
- Peak detection with threshold
- Harmonic identification
- Sidelobe levels
- 3dB bandwidth
- Spectral centroid
- Spectral flatness

**Integration with Existing FFT**:
- Leverage stats_server FFT implementation
- Add windowing and advanced analysis
- Provide educational DFT for comparison

#### 3. `convolution_correlation`

**Convolution Operations**:
- Linear convolution
- Circular convolution
- Fast convolution (FFT-based)
- Valid/same/full output modes
- 2D convolution for images

**Cross-Correlation**:
- Cross-correlation for signal matching
- Time delay estimation
- Pattern detection
- Similarity measurement

**Auto-Correlation**:
- Auto-correlation function
- Periodic pattern detection
- Noise analysis
- Signal energy at lag

**Features**:
```python
convolution_correlation(
    signal1=[...],
    signal2=[...],
    operation="convolve",      # or "cross_correlate", "auto_correlate"
    method="fft",              # or "direct"
    mode="full"                # or "valid", "same"
)
```

**Applications**:
- FIR filter implementation (convolution)
- System impulse response
- Echo detection (auto-correlation)
- Time delay estimation (cross-correlation)
- Pattern matching
- Signal energy calculation

**Output**:
- Convolution/correlation result
- Peak locations and values
- Time delay (for correlation)
- Confidence measure
- Computation method used

#### 4. `laplace_transform` (Symbolic)

**Symbolic Laplace Transform**:
- Transform common functions
- Partial fraction decomposition
- Inverse Laplace transform
- Initial/final value theorems

**Supported Functions**:
- Exponentials: e^(-at)
- Polynomials: t^n
- Sinusoids: sin(ωt), cos(ωt)
- Hyperbolic: sinh(at), cosh(at)
- Unit step: u(t)
- Dirac delta: δ(t)
- Ramp: t·u(t)

**Transfer Functions**:
- First-order: 1/(τs + 1)
- Second-order: ω²/(s² + 2ζωs + ω²)
- Integrator: 1/s
- Differentiator: s
- Time delay: e^(-sT)

**Operations**:
- Linearity (transform of sum)
- Time shifting: f(t-a)u(t-a)
- Frequency shifting: e^(at)f(t)
- Scaling: f(at)
- Differentiation: df/dt
- Integration: ∫f(t)dt
- Convolution theorem

**Features**:
```python
laplace_transform(
    expression="exp(-2*t)*sin(3*t)",
    variable="t",
    transform_variable="s",
    operation="forward"        # or "inverse"
)
```

**Output**:
- Transformed expression
- Region of convergence (ROC)
- Poles and zeros
- Simplified form
- Partial fraction expansion (for inverse)

**Applications**:
- Control system analysis
- Circuit transient analysis
- Differential equation solving
- Transfer function manipulation

## Technical Implementation

### File Structure

Add to `src/builtin/engineering_math_server/tools/transforms.py`:

```python
"""Transform methods for frequency analysis and system identification."""

import numpy as np
from scipy import signal, fft
from scipy.integrate import quad
import sympy as sp
from typing import Union, List, Dict, Tuple, Optional

# Fourier series implementation
def create_fourier_series_tool() -> Tool:
    """Create Fourier series decomposition tool."""
    return Tool(
        name="fourier_series",
        description="""Decompose periodic function into Fourier series.
        
        Computes Fourier coefficients for periodic signals:
        - DC component (a₀)
        - Cosine coefficients (aₙ)  
        - Sine coefficients (bₙ)
        - Complex form (cₙ)
        - Magnitude and phase spectra
        
        Applications:
        - Harmonic analysis
        - Power quality (THD)
        - Vibration analysis
        - Signal reconstruction
        """,
        inputSchema={
            "type": "object",
            "properties": {
                "function_type": {
                    "type": "string",
                    "enum": [
                        "square_wave", "triangle_wave", "sawtooth_wave",
                        "pulse_train", "half_rectified_sine", "full_rectified_sine",
                        "custom_data"
                    ],
                    "description": "Type of periodic function"
                },
                "data": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Custom data points (if function_type='custom_data')"
                },
                "period": {
                    "type": "number",
                    "description": "Period T of the function",
                    "minimum": 0,
                    "exclusiveMinimum": True
                },
                "num_harmonics": {
                    "type": "integer",
                    "description": "Number of harmonics to compute",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100
                },
                "sample_points": {
                    "type": "integer",
                    "description": "Points for numerical integration",
                    "default": 1000,
                    "minimum": 100
                }
            },
            "required": ["function_type", "period"]
        }
    )

async def handle_fourier_series(arguments: dict) -> CallToolResult:
    """Handle Fourier series decomposition."""
    try:
        func_type = arguments["function_type"]
        period = arguments["period"]
        num_harmonics = arguments.get("num_harmonics", 10)
        sample_points = arguments.get("sample_points", 1000)
        
        # Generate or use custom data
        if func_type == "custom_data":
            data = np.array(arguments["data"])
        else:
            data = generate_periodic_function(func_type, period, sample_points)
        
        # Compute Fourier coefficients
        coefficients = compute_fourier_coefficients(
            data, period, num_harmonics
        )
        
        # Reconstruct signal
        reconstructed = reconstruct_from_coefficients(
            coefficients, period, sample_points
        )
        
        # Calculate error
        rms_error = np.sqrt(np.mean((data - reconstructed)**2))
        
        # Format output
        output = format_fourier_series_result(
            coefficients, rms_error, period
        )
        
        return CallToolResult(
            content=[TextContent(type="text", text=output)]
        )
        
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )

def compute_fourier_coefficients(
    data: np.ndarray,
    period: float,
    num_harmonics: int
) -> Dict[str, np.ndarray]:
    """Compute Fourier series coefficients numerically."""
    N = len(data)
    t = np.linspace(0, period, N, endpoint=False)
    omega_0 = 2 * np.pi / period
    
    # DC component
    a0 = np.mean(data)
    
    # Initialize coefficient arrays
    an = np.zeros(num_harmonics)
    bn = np.zeros(num_harmonics)
    
    # Compute coefficients using numerical integration
    for n in range(1, num_harmonics + 1):
        # Cosine coefficient
        integrand_cos = data * np.cos(n * omega_0 * t)
        an[n-1] = (2 / period) * np.trapz(integrand_cos, t)
        
        # Sine coefficient
        integrand_sin = data * np.sin(n * omega_0 * t)
        bn[n-1] = (2 / period) * np.trapz(integrand_sin, t)
    
    # Complex coefficients
    cn = (an - 1j * bn) / 2
    
    # Magnitude and phase
    magnitude = np.abs(cn)
    phase = np.angle(cn)
    
    return {
        'a0': a0,
        'an': an,
        'bn': bn,
        'cn': cn,
        'magnitude': magnitude,
        'phase': phase
    }
```

### Integration with Existing Stats Server FFT

Enhance coordination between tools:

```python
# In documentation and tool descriptions
"""
Note: This tool provides detailed DFT analysis with windowing.
For fast computation of large datasets, use the 'fft_analysis' 
tool from the stats-server. Both tools are complementary:

- dft_analysis: Educational, windowing, detailed analysis
- fft_analysis: Fast, large datasets, basic spectrum
"""
```

## Key Engineering Applications

### 1. Power Quality Analysis
```python
# Analyze voltage waveform for harmonics
fourier_series(
    function_type="custom_data",
    data=voltage_samples,
    period=0.0167,  # 60 Hz = 16.67ms period
    num_harmonics=20
)
# Check THD, identify dominant harmonics
```

### 2. Vibration Analysis
```python
# Bearing fault detection via FFT
dft_analysis(
    signal=accelerometer_data,
    sample_rate=10000,
    window="hann",
    zero_padding_factor=4
)
# Look for peaks at bearing fault frequencies
```

### 3. FIR Filter Design
```python
# Implement FIR filter via convolution
convolution_correlation(
    signal1=input_signal,
    signal2=filter_coefficients,
    operation="convolve",
    method="fft",
    mode="same"
)
```

### 4. Time Delay Estimation
```python
# Find delay between two sensors
convolution_correlation(
    signal1=sensor1_data,
    signal2=sensor2_data,
    operation="cross_correlate"
)
# Peak location indicates time delay
```

### 5. Control System Analysis
```python
# Transform differential equation to transfer function
laplace_transform(
    expression="exp(-2*t)*cos(3*t)",
    variable="t",
    transform_variable="s"
)
# Analyze poles/zeros for stability
```

## Testing Requirements

### Unit Tests
- Fourier series of known waveforms (analytical solutions)
- DFT correctness vs. FFT
- Convolution properties (commutative, associative)
- Correlation peak detection
- Laplace transform of standard functions

### Validation Tests
```python
# Test Parseval's theorem (energy conservation)
time_energy = np.sum(signal**2)
freq_energy = np.sum(np.abs(fft_signal)**2) / N
assert np.isclose(time_energy, freq_energy)

# Test convolution theorem
conv_time = np.convolve(x, h, mode='same')
conv_freq = np.fft.ifft(np.fft.fft(x) * np.fft.fft(h, len(x)))
assert np.allclose(conv_time, conv_freq.real)
```

### Engineering Application Tests
- Square wave Fourier series (Gibbs phenomenon)
- Window sidelobe levels
- Cross-correlation delay estimation accuracy
- Laplace transform partial fractions

## Documentation Requirements

1. **Mathematical Background**
   - Fourier series theory
   - DFT vs. FFT
   - Windowing effects
   - Convolution theorem
   - Laplace transform properties

2. **Application Examples**
   - Harmonic analysis workflow
   - Vibration signature analysis
   - Filter implementation
   - System identification
   - Transfer function analysis

3. **Window Selection Guide**
   - Comparison table (sidelobe, bandwidth)
   - Application recommendations
   - Spectral leakage explanation

## Deliverables

- [ ] Fourier series tool implementation
- [ ] DFT analysis tool with windowing
- [ ] Convolution/correlation tool
- [ ] Laplace transform tool (symbolic)
- [ ] Comprehensive test suite
- [ ] Documentation with examples
- [ ] Integration with stats server FFT
- [ ] Window function comparison guide

## Success Criteria

- ✅ All 4 transform tools functional
- ✅ Numerical accuracy validated
- ✅ Windowing functions correct
- ✅ Integration with existing FFT verified
- ✅ Engineering examples working
- ✅ Performance acceptable (<1s typical)

## Timeline

**Week 1**:
- Days 1-2: Fourier series implementation
- Days 3-4: DFT analysis with windowing
- Day 5: Testing and validation

**Week 2**:
- Days 1-2: Convolution/correlation
- Days 3-4: Laplace transform (symbolic)
- Day 5: Documentation and integration testing

## Related Issues

- Requires: #79 (Engineering Math Server)
- Requires: #80 (Complex Analysis)
- Blocks: #83 (Control Systems Server)
- Blocks: #85 (Signal Processing Server)
- Related: Stats Server FFT tools (enhance coordination)

## References

- Signals and Systems (Oppenheim & Willsky)
- Digital Signal Processing (Proakis & Manolakis)
- Understanding Digital Signal Processing (Lyons)
- The Scientist and Engineer's Guide to DSP (Smith)
- SciPy signal processing documentation
