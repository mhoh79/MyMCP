# Issue #85: Implement Signal Processing Server

**Priority**: High  
**Dependencies**: #79 (Engineering Math), #80 (Complex Analysis), #81 (Transforms)  
**Labels**: enhancement, builtin-server, signal-processing, application-server  
**Estimated Effort**: 2-3 weeks

## Overview

Create a specialized MCP server for advanced signal processing, frequency analysis, and digital filtering. Enhances existing stats server FFT tools with comprehensive signal processing capabilities for vibration monitoring, electrical analysis, audio processing, and communications.

## MCP Server Architecture Mapping

**Server Name**: `signal_processing_server` (Application Server)  
**Role**: Advanced signal analysis, filtering, and time-frequency processing  
**Tools**: 12 signal processing functions  
**Dependencies**: Engineering Math (#79), Complex Analysis (#80), Transforms (#81)  
**Integration**: Complements existing Stats Server FFT tools

### Application Stack Coverage

This server is the **primary implementation** of:

1. **Signal Processing Stack** - Complete implementation (100%)
   - Filter design (FIR/IIR), wavelets, spectral estimation, modulation, adaptive filters
   
2. **Vibration Analysis & Diagnostics Stack** - Signal processing components (60%)
   - Filtering, spectrogram, order tracking, wavelet analysis for transient detection
   - Works with Stats Server (FFT, RMS, statistics) and Structural Server (#86) for natural frequencies
   
3. **Control & Instrumentation Stack** - Sensor filtering (40%)
   - Anti-aliasing filters, noise reduction, sensor signal conditioning
   - Works with Control Systems Server (#83) for control loops
   
4. **Electrical/Power Systems Stack** - Harmonic analysis (30%)
   - FFT for harmonics, filter design for power quality
   - Works with Complex Analysis (#80) for phasor representation

### Tool Reuse from Foundation Server (#79)

- **Polynomials** → IIR filter transfer functions (numerator/denominator coefficients)
- **Linear algebra** → Filter bank matrices, multichannel processing, sensor arrays
- **Optimization** → Optimal filter design (Parks-McClellan), adaptive filter convergence
- **Calculus** → Convolution (filtering), differentiation (edge detection), integration (smoothing)
- **Numerical methods** → Resampling algorithms, interpolation for sample rate conversion

### Tool Reuse from Complex Analysis (#80)

- **Complex operations** → Filter poles/zeros in z-plane, frequency domain representation
- **Complex functions** → Z-transform calculations, frequency response

### Tool Reuse from Transforms (#81)

- **FFT** → Spectral analysis foundation (enhanced in this server)
- **DFT** → Window function analysis, spectral leakage control
- **Wavelets** → Time-frequency analysis (enhanced in this server)
- **Convolution** → FIR filtering implementation

### Integration with Stats Server

**Division of Responsibility**:
- **Stats Server** (existing): Basic FFT, PSD, RMS, peak detection, SNR, harmonic analysis
- **Signal Processing Server** (this): Advanced filters, wavelets, adaptive processing, order tracking

**Complementary Usage**:
```python
# Stats Server: Quick spectral analysis
fft_analysis(signal, sample_rate=10000)

# Signal Processing Server: Detailed filter design
filter_design(type="butterworth", order=8, cutoff=1000)

# Signal Processing Server: Apply advanced filtering
filter_application(signal, coefficients=..., method="zero_phase")

# Stats Server: Analyze filtered result
statistical_summary(filtered_signal)
```

### Cross-Server Workflows

**Example: Bearing Fault Detection**
```python
# 1. Stats Server: Initial analysis
fft_analysis(vibration_data)  # Identify suspicious frequencies

# 2. Signal Processing Server: Bandpass filter
filter_design(type="bandpass", frequencies=[1000, 5000])

# 3. Signal Processing Server: Envelope analysis
modulation_demodulation(operation="demodulate", type="AM")

# 4. Stats Server: FFT of envelope
fft_analysis(envelope_signal)  # Reveal bearing frequencies

# 5. Signal Processing Server: Wavelet for transients
wavelet_transform(wavelet="db4", type="dwt")
```

## Objectives

- Provide complete signal processing workflow
- Enable advanced frequency domain analysis
- Support digital filter design and implementation
- Facilitate time-frequency analysis
- Integrate with existing stats server FFT tools

## Scope

### Signal Processing Stack Tools (10-12 tools)

#### 1. `filter_design`

**Digital Filter Design**:
- FIR filters (windowing method, Parks-McClellan)
- IIR filters (Butterworth, Chebyshev I/II, Elliptic, Bessel)
- Filter specifications (passband, stopband, ripple)
- Filter order estimation

**Features**:
```python
filter_design(
    filter_type="butterworth",
    filter_class="lowpass",    # lowpass, highpass, bandpass, bandstop
    filter_order=4,
    cutoff_frequency=1000,     # Hz (or [f1, f2] for bandpass)
    sample_rate=10000,         # Hz
    design_method="iir"        # or "fir"
)
```

**Output**:
- Filter coefficients (b, a for IIR; h for FIR)
- Frequency response (magnitude, phase)
- Pole-zero plot data
- Group delay
- Filter implementation code

**Filter Types**:
- **Butterworth**: Maximally flat passband
- **Chebyshev Type I**: Passband ripple, sharp transition
- **Chebyshev Type II**: Stopband ripple
- **Elliptic**: Ripple in pass and stop, sharpest transition
- **Bessel**: Linear phase, good transient response

#### 2. `filter_application`

**Apply Filter to Signal**:
- Zero-phase filtering (filtfilt)
- Causal filtering
- Cascaded filtering
- Adaptive filtering

**Features**:
```python
filter_application(
    signal=accelerometer_data,
    filter_coefficients={"b": [...], "a": [...]},
    method="zero_phase",       # or "causal"
    initial_conditions="steady_state"
)
```

**Applications**: Noise removal, signal conditioning, anti-aliasing

#### 3. `wavelet_transform`

**Time-Frequency Analysis**:
- Continuous Wavelet Transform (CWT)
- Discrete Wavelet Transform (DWT)
- Wavelet packet decomposition
- Multiple wavelet families

**Features**:
```python
wavelet_transform(
    signal=vibration_data,
    wavelet="morlet",          # db4, coif3, sym5, etc.
    transform_type="cwt",      # or "dwt"
    scales=np.arange(1, 128),  # for CWT
    sample_rate=10000
)
```

**Wavelet Families**:
- Morlet (frequency analysis)
- Daubechies (dbN)
- Symlets (symN)
- Coiflets (coifN)
- Haar
- Mexican hat

**Applications**:
- Transient detection
- Non-stationary signal analysis
- Denoising
- Feature extraction
- Compression

#### 4. `spectral_estimation`

**Power Spectral Density Methods**:
- Periodogram
- Welch's method (averaged periodograms)
- Multitaper method
- Parametric methods (AR, ARMA)

**Features**:
```python
spectral_estimation(
    signal=sensor_data,
    method="welch",
    window="hann",
    nperseg=1024,              # Segment length
    noverlap=512,              # Overlap
    sample_rate=10000
)
```

**Output**:
- Frequency bins
- Power spectral density (V²/Hz or similar)
- Confidence intervals
- Peak frequencies
- Integrated power in bands

**Applications**: Noise characterization, vibration spectra, power quality

#### 5. `window_functions`

**Windowing for Spectral Analysis**:
- Generate window functions
- Window properties (main lobe width, sidelobe level)
- Window comparison
- Optimal window selection

**Features**:
```python
window_functions(
    window_type="kaiser",
    length=1024,
    parameters={"beta": 8.6},  # Kaiser parameter
    normalize=True
)
```

**Windows Available**:
- Rectangular (boxcar)
- Triangular (Bartlett)
- Hann (Hanning)
- Hamming
- Blackman
- Kaiser (parameterized)
- Flat-top
- Tukey (tapered cosine)
- Gaussian

**Properties Computed**:
- Equivalent noise bandwidth (ENBW)
- Coherent gain
- 3dB bandwidth
- Sidelobe level
- Scalloping loss

#### 6. `resampling`

**Sample Rate Conversion**:
- Upsampling (interpolation)
- Downsampling (decimation)
- Rational resampling (polyphase)
- Anti-aliasing filtering

**Features**:
```python
resampling(
    signal=audio_data,
    original_rate=44100,
    target_rate=48000,
    method="polyphase",        # or "fft", "linear"
    anti_aliasing=True
)
```

**Applications**: Sample rate matching, data reduction, oversampling

#### 7. `modulation_demodulation`

**Amplitude/Frequency Modulation**:
- AM modulation/demodulation
- FM modulation/demodulation
- Envelope detection
- Hilbert transform (analytic signal)
- Instantaneous frequency

**Features**:
```python
modulation_demodulation(
    operation="demodulate",
    modulation_type="AM",      # or "FM"
    carrier_frequency=1000,    # Hz
    signal=modulated_signal,
    sample_rate=10000
)
```

**Applications**: Communications, bearing fault detection, envelope analysis

#### 8. `adaptive_filtering`

**Adaptive Filter Algorithms**:
- LMS (Least Mean Squares)
- NLMS (Normalized LMS)
- RLS (Recursive Least Squares)
- Noise cancellation
- Echo cancellation

**Features**:
```python
adaptive_filtering(
    algorithm="lms",
    primary_input=noisy_signal,
    reference_input=noise_reference,
    filter_order=32,
    step_size=0.01,            # μ for LMS
    adaptation="continuous"     # or "block"
)
```

**Applications**: Active noise control, echo cancellation, channel equalization

#### 9. `cepstral_analysis`

**Cepstrum Analysis**:
- Real cepstrum
- Complex cepstrum
- Quefrency domain analysis
- Homomorphic filtering
- Pitch detection

**Features**:
```python
cepstral_analysis(
    signal=audio_or_vibration,
    cepstrum_type="real",      # or "complex"
    lifter_type="lowpass",     # or "highpass"
    lifter_cutoff=50,          # quefrency samples
    sample_rate=10000
)
```

**Applications**:
- Echo detection and removal
- Pitch tracking
- Gear/bearing diagnostics (sidebands)
- Speech processing

#### 10. `spectrogram_analysis`

**Time-Frequency Representation**:
- Short-Time Fourier Transform (STFT)
- Spectrogram generation
- Time-frequency resolution tradeoff
- Reassignment methods

**Features**:
```python
spectrogram_analysis(
    signal=time_varying_signal,
    window="hann",
    nperseg=256,
    noverlap=128,
    sample_rate=10000,
    output_format="magnitude_db"  # or "power", "magnitude"
)
```

**Output**:
- Time-frequency matrix
- Time axis, frequency axis
- Color scale recommendations
- Ridge extraction
- Dominant frequency vs. time

**Applications**: Transient analysis, chirp detection, startup/shutdown monitoring

#### 11. `coherence_analysis`

**Signal Coherence**:
- Magnitude-squared coherence
- Cross-spectral density
- Transfer function estimation
- Noise contribution analysis

**Features**:
```python
coherence_analysis(
    signal1=input_signal,
    signal2=output_signal,
    method="welch",
    nperseg=1024,
    sample_rate=10000
)
```

**Output**:
- Coherence function (0-1)
- Frequency bins
- Transfer function H(f) = Gxy/Gxx
- Coherent output power
- Noise power

**Applications**: System identification, input-output relationships, noise analysis

#### 12. `order_tracking`

**Rotating Machinery Analysis**:
- Order analysis (speed-dependent frequencies)
- Campbell diagram data
- Order extraction
- Resampling to angular domain

**Features**:
```python
order_tracking(
    vibration_signal=accel_data,
    tachometer_signal=tacho_pulses,
    orders=[1, 2, 3, 4],       # Shaft orders to extract
    sample_rate=10000
)
```

**Output**:
- Order spectra vs. speed
- Order cuts (amplitude vs. speed)
- Waterfall plot data
- Resonance identification

**Applications**: Rotating equipment diagnostics, balancing, resonance avoidance

## Technical Architecture

### Server Structure
```
src/builtin/signal_processing_server/
├── __init__.py
├── __main__.py
├── server.py
├── tools/
│   ├── __init__.py
│   ├── filters.py           # Tools 1-2
│   ├── wavelets.py          # Tool 3
│   ├── spectral_analysis.py # Tools 4-5, 10
│   ├── resampling.py        # Tool 6
│   ├── modulation.py        # Tool 7
│   ├── adaptive_filters.py  # Tool 8
│   ├── cepstrum.py          # Tool 9
│   ├── coherence.py         # Tool 11
│   └── order_tracking.py    # Tool 12
├── utils/
│   ├── window_properties.py
│   └── filter_design_helpers.py
└── README.md
```

### Dependencies
```python
# Additional requirements
pywavelets>=1.4.0      # Wavelet transforms
scipy.signal           # Already available (signal processing)
```

### Integration with Stats Server

**Coordinate with existing tools**:
- Stats server: Basic FFT, PSD, RMS, peak detection, SNR, harmonic analysis
- Signal Processing server: Advanced filters, wavelets, adaptive processing, order tracking

**Documentation Cross-References**:
```python
"""
For basic FFT analysis, see stats-server 'fft_analysis' tool.
For advanced filtering and time-frequency analysis, use this server.
Both servers complement each other for complete signal analysis.
"""
```

## Key Application Examples

### Example 1: Bearing Fault Detection
```python
# 1. Envelope analysis
modulation_demodulation(
    operation="demodulate",
    modulation_type="AM",
    signal=vibration_data
)

# 2. FFT of envelope
# 3. Identify bearing fault frequencies

# 4. Wavelet analysis for transients
wavelet_transform(
    signal=vibration_data,
    wavelet="db4",
    transform_type="dwt"
)
```

### Example 2: Notch Filter for Power Line Noise
```python
# Design 60 Hz notch filter
filter_design(
    filter_type="elliptic",
    filter_class="bandstop",
    cutoff_frequency=[58, 62],  # Hz
    sample_rate=1000,
    filter_order=4
)

# Apply to ECG signal
filter_application(
    signal=ecg_data,
    filter_coefficients=coeffs,
    method="zero_phase"
)
```

### Example 3: Motor Startup Analysis
```python
# Time-frequency spectrogram
spectrogram_analysis(
    signal=motor_current,
    window="hann",
    nperseg=512,
    sample_rate=10000
)

# Track speed-dependent frequencies
order_tracking(
    vibration_signal=vibration,
    tachometer_signal=tacho,
    orders=[1, 2]  # 1X, 2X shaft speed
)
```

### Example 4: Active Noise Cancellation
```python
# Adaptive filter for noise cancellation
adaptive_filtering(
    algorithm="lms",
    primary_input=microphone_signal,
    reference_input=noise_reference,
    filter_order=256,
    step_size=0.001
)
```

## Testing Requirements

### Unit Tests
- Filter design (frequency response verification)
- Wavelet transform (perfect reconstruction)
- Window properties (ENBW, sidelobe levels)
- Resampling (frequency content preservation)

### Validation Tests
- Compare with MATLAB Signal Processing Toolbox
- Verify against textbook examples
- Cross-check with existing stats server FFT

### Integration Tests
- Complete diagnostic workflows
- Multi-stage filtering
- Cascaded signal processing

## Deliverables

- [ ] SignalProcessingServer implementation
- [ ] All 12 signal processing tools functional
- [ ] Comprehensive test suite
- [ ] Documentation with signal processing examples
- [ ] Integration with stats server documented
- [ ] Wrapper script: `start_signal_processing_server.py`
- [ ] Claude Desktop configuration

## Success Criteria

- ✅ All signal processing tools working
- ✅ Filter designs validated
- ✅ Wavelet transforms accurate
- ✅ Integration with stats server clear
- ✅ Example workflows documented

## Timeline

**Week 1**: Filter design, wavelets, spectral estimation  
**Week 2**: Modulation, adaptive filters, cepstrum  
**Week 3**: Spectrogram, coherence, order tracking, testing

## Related Issues

- Requires: #79, #80, #81
- Related: Stats Server (FFT, PSD, harmonic analysis)
- Part of: Signal Processing Stack

## References

- Digital Signal Processing (Proakis & Manolakis)
- Understanding Digital Signal Processing (Lyons)
- Discrete-Time Signal Processing (Oppenheim & Schafer)
- PyWavelets documentation
- SciPy Signal Processing Guide
