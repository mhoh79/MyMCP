# Add Signal Processing Statistics to Stats Server

## Overview
Implement signal processing and frequency analysis tools for vibration monitoring, electrical signal quality, acoustic analysis, and harmonic detection in industrial equipment.

## Motivation
Signal processing is critical for:
- **Vibration Analysis**: Predict bearing/motor failures through frequency analysis
- **Electrical Power Quality**: Detect harmonics, voltage fluctuations, power factor issues
- **Acoustic Monitoring**: Identify abnormal sounds indicating equipment problems
- **Condition Monitoring**: Non-intrusive equipment health assessment
- **Predictive Maintenance**: Early warning of developing faults
- **Energy Management**: Power quality analysis and optimization

## Tools to Implement

### 1. `fft_analysis`
Fast Fourier Transform for frequency domain analysis.

**Use Cases:**
- Bearing defect detection (BPFI, BPFO, BSF frequencies)
- Motor electrical faults (broken rotor bars, eccentricity)
- Gearbox mesh frequency analysis
- Pump cavitation detection
- Compressor valve problems
- Fan blade imbalance

**Input Schema:**
```json
{
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
      "description": "Remove DC component and linear trends",
      "default": true
    }
  },
  "required": ["signal", "sample_rate"]
}
```

**Example Output:**
```json
{
  "sample_rate": 10000,
  "signal_length": 8192,
  "duration_seconds": 0.8192,
  "frequencies": [0, 1.22, 2.44, ..., 5000],
  "magnitudes": [0.1, 2.5, 1.8, ..., 0.3],
  "dominant_frequencies": [
    {"frequency": 60.0, "magnitude": 15.3, "interpretation": "Electrical line frequency"},
    {"frequency": 120.0, "magnitude": 3.2, "interpretation": "Second harmonic"},
    {"frequency": 1785.0, "magnitude": 8.7, "interpretation": "Potential bearing defect frequency"}
  ],
  "frequency_bands": {
    "subsynchronous": {"range": [0, 50], "rms": 1.2},
    "1x_rpm": {"range": [900, 1100], "rms": 5.8},
    "2x_rpm": {"range": [1800, 2200], "rms": 8.7},
    "high_frequency": {"range": [5000, 10000], "rms": 0.8}
  },
  "nyquist_frequency": 5000,
  "resolution": 1.22,
  "interpretation": "Strong peak at bearing defect frequency (1785 Hz). Recommend bearing inspection."
}
```

### 2. `power_spectral_density`
Calculate Power Spectral Density (PSD) for energy distribution across frequencies.

**Use Cases:**
- Vibration energy distribution analysis
- Noise level assessment
- Process variable frequency content
- Random vibration analysis
- Acoustic signature analysis
- Electrical noise characterization

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "signal": {
      "type": "array",
      "items": {"type": "number"},
      "description": "Time-domain signal",
      "minItems": 16
    },
    "sample_rate": {
      "type": "number"},
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
      "description": "Length of each segment for Welch method (default: 256)",
      "default": 256
    }
  },
  "required": ["signal", "sample_rate"]
}
```

**Example Output:**
```json
{
  "method": "Welch",
  "frequencies": [...],
  "psd": [...],
  "units": "g²/Hz",
  "total_power": 12.5,
  "peak_frequency": 1785.0,
  "peak_power": 3.8,
  "frequency_bands_power": {
    "low": {"range": "0-100 Hz", "power": 2.1, "percent": 16.8},
    "medium": {"range": "100-1000 Hz", "power": 5.3, "percent": 42.4},
    "high": {"range": "1000-5000 Hz", "power": 5.1, "percent": 40.8}
  },
  "interpretation": "Energy concentrated in medium-high frequency bands. Typical of rolling element bearing wear."
}
```

### 3. `rms_value`
Calculate Root Mean Square for overall signal energy.

**Use Cases:**
- Overall vibration severity (ISO 10816 standards)
- Electrical current RMS for power calculations
- Acoustic noise level assessment
- Process variable stability indicator
- Alarm threshold monitoring
- Trend tracking for degradation

**Input Schema:**
```json
{
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
```

**Example Output:**
```json
{
  "rms": 4.85,
  "units": "mm/s",
  "peak": 12.3,
  "crest_factor": 2.54,
  "rms_db": 73.7,
  "iso_10816_zone": "B",
  "interpretation": "Vibration acceptable for this machine class (Zone B per ISO 10816)",
  "alarm_status": "Normal",
  "rolling_rms": [4.2, 4.5, 4.8, 4.9, 5.1],
  "trend": "slightly_increasing"
}
```

### 4. `peak_detection`
Identify significant peaks in signals with filtering and ranking.

**Use Cases:**
- Find dominant vibration frequencies
- Detect harmonic patterns in power quality
- Identify resonance frequencies
- Bearing fault frequency detection
- Gear mesh frequency identification
- Blade pass frequency in fans/pumps

**Input Schema:**
```json
{
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
      "description": "Minimum peak height (absolute or relative)",
      "default": 0.1
    },
    "distance": {
      "type": "integer",
      "description": "Minimum samples between peaks",
      "default": 1
    },
    "prominence": {
      "type": "number",
      "description": "Required prominence (height above surroundings)",
      "default": 0.05
    },
    "top_n": {
      "type": "integer",
      "description": "Return top N peaks only",
      "maximum": 50,
      "default": 10
    }
  },
  "required": ["signal"]
}
```

**Example Output:**
```json
{
  "peaks_found": 8,
  "top_peaks": [
    {
      "index": 146,
      "frequency": 1785.0,
      "magnitude": 8.7,
      "prominence": 7.2,
      "rank": 1,
      "interpretation": "Outer race bearing defect (BPFO)"
    },
    {
      "index": 97,
      "frequency": 1186.5,
      "magnitude": 5.3,
      "prominence": 4.1,
      "rank": 2,
      "interpretation": "Inner race bearing defect (BPFI)"
    },
    {
      "index": 30,
      "frequency": 366.5,
      "magnitude": 4.2,
      "prominence": 3.8,
      "rank": 3,
      "interpretation": "Ball spin frequency (BSF)"
    }
  ],
  "harmonics_detected": [
    {"fundamental": 60.0, "harmonics": [120.0, 180.0, 240.0]}
  ],
  "sidebands_detected": [
    {"carrier": 1785.0, "sidebands": [1755.0, 1815.0], "spacing": 30.0}
  ]
}
```

### 5. `signal_to_noise_ratio`
Calculate SNR to assess signal quality.

**Use Cases:**
- Sensor health monitoring
- Data acquisition quality check
- Communication signal quality
- Process measurement reliability
- Instrumentation validation
- Analog input quality assessment

**Input Schema:**
```json
{
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
```

**Example Output:**
```json
{
  "snr_db": 42.5,
  "snr_ratio": 133.4,
  "signal_power": 125.3,
  "noise_power": 0.94,
  "quality": "Excellent",
  "interpretation": "Signal quality excellent (SNR > 40 dB). No sensor issues detected.",
  "effective_bits": 7.1,
  "recommendation": "Signal suitable for control and analysis"
}
```

### 6. `harmonic_analysis`
Detect and analyze harmonic content in electrical and mechanical signals.

**Use Cases:**
- Power quality assessment (THD calculation)
- Variable frequency drive (VFD) effects
- Motor current signature analysis (MCSA)
- Electrical fault detection
- Power factor analysis
- Compliance with IEEE 519 standards

**Input Schema:**
```json
{
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
      "description": "Maximum harmonic order to analyze",
      "default": 50,
      "maximum": 100
    }
  },
  "required": ["signal", "sample_rate", "fundamental_freq"]
}
```

**Example Output:**
```json
{
  "fundamental": {
    "frequency": 60.0,
    "magnitude": 120.5,
    "phase": 0.0
  },
  "harmonics": [
    {"order": 2, "frequency": 120.0, "magnitude": 2.3, "percent": 1.9},
    {"order": 3, "frequency": 180.0, "magnitude": 5.8, "percent": 4.8},
    {"order": 5, "frequency": 300.0, "magnitude": 8.2, "percent": 6.8},
    {"order": 7, "frequency": 420.0, "magnitude": 3.1, "percent": 2.6}
  ],
  "thd": 9.2,
  "thd_interpretation": "Moderate distortion (5-10% THD)",
  "dominant_harmonics": [5, 3, 7],
  "even_odd_ratio": 0.32,
  "interpretation": "Odd harmonics dominant - typical of non-linear loads or VFD operation",
  "ieee_519_compliance": {
    "compliant": true,
    "limit": 15.0,
    "margin": 5.8
  },
  "recommendations": "THD within limits but consider harmonic filters for sensitive loads"
}
```

## Implementation Requirements

### Algorithms
- FFT: Cooley-Tukey algorithm (radix-2 or mixed-radix)
- Windowing: Hanning, Hamming, Blackman functions
- Welch method: overlapping segments with averaging
- Peak detection: local maxima with prominence filtering
- THD: √(Σ harmonics²) / fundamental

### Dependencies
- scipy.signal for FFT and signal processing
- numpy for efficient array operations
- No additional external dependencies

### Performance Targets
- FFT: < 50ms for 8192 points
- PSD: < 200ms for 100k points (Welch method)
- RMS: < 10ms for 10k points
- Peak detection: < 100ms for 10k points
- Harmonic analysis: < 100ms for 10k points

## Industrial Application Examples

### Example 1: Bearing Defect Detection
```
Input: Accelerometer data from motor bearing (10 kHz sampling, 1 second)
Tools Used:
1. fft_analysis - identify frequency peaks
2. peak_detection - find bearing defect frequencies
3. rms_value - track overall vibration level

Output: "Outer race defect detected at 1785 Hz (4.2x BPFO). RMS increased 35% vs. baseline. Schedule bearing replacement within 2 weeks."
```

### Example 2: Power Quality Analysis
```
Input: Current waveform from VFD-controlled motor (50 kHz, 200ms)
Tools Used:
1. harmonic_analysis - calculate THD and harmonics
2. fft_analysis - full spectrum view
3. signal_to_noise_ratio - validate measurement quality

Output: "THD = 8.2% (compliant). 5th harmonic dominant (6.8%). Typical VFD signature. No issues detected."
```

### Example 3: Pump Cavitation Detection
```
Input: Acoustic sensor data from centrifugal pump (20 kHz, 2 seconds)
Tools Used:
1. power_spectral_density - energy distribution
2. peak_detection - identify cavitation frequencies
3. rms_value - overall noise level

Output: "Broadband high-frequency energy (5-15 kHz) elevated 250%. Cavitation detected. Check NPSH and suction conditions."
```

## Acceptance Criteria

- [ ] All 6 signal processing tools implemented
- [ ] Support for various sampling rates (1 Hz to 1 MHz)
- [ ] Windowing functions for spectral leakage reduction
- [ ] Configurable frequency ranges and bands
- [ ] Industrial standards compliance (ISO 10816, IEEE 519)
- [ ] Clear interpretation of results
- [ ] Bearing defect frequency calculations
- [ ] Harmonic pattern recognition
- [ ] Integration tests with real vibration/electrical data

## Testing Requirements

**Test Data:**
1. Pure sine waves (known frequencies)
2. Multi-frequency signals (harmonics)
3. Real bearing vibration data
4. Electrical current with harmonics
5. Noisy signals
6. High sample rate data (>100 kHz)

**Validation:**
- Compare FFT results with scipy.fft
- Verify THD calculations with test waveforms
- Test bearing frequency detection accuracy
- Validate RMS against hand calculations

## Documentation Requirements

- Signal processing fundamentals for engineers
- Bearing defect frequency formulas
- Harmonic analysis interpretation
- ISO 10816 vibration severity zones
- IEEE 519 harmonic limits
- Sample rate selection guidelines
- Common frequency patterns and causes
- Integration with condition monitoring systems

## Labels
`enhancement`, `stats-server`, `signal-processing`, `condition-monitoring`, `predictive-maintenance`, `tier-2-priority`

## Priority
**Tier 2 - Important** - Essential for predictive maintenance and power quality

## Estimated Effort
8-10 hours for implementation and testing
