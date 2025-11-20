# Signal Processing Tools

## Overview

The stats server now includes 6 comprehensive signal processing tools designed for vibration monitoring, electrical signal quality analysis, acoustic monitoring, and harmonic detection in industrial equipment. These tools enable predictive maintenance, power quality assessment, and condition-based monitoring for rotating machinery and electrical systems.

## Tools Summary

| Tool | Purpose | Use Cases |
|------|---------|-----------|
| `fft_analysis` | Frequency domain analysis | Bearing defects, motor faults, gear mesh analysis |
| `power_spectral_density` | Energy distribution | Vibration energy, noise assessment, random vibration |
| `rms_value` | Overall signal energy | Vibration severity, current RMS, noise levels |
| `peak_detection` | Identify significant peaks | Dominant frequencies, harmonics, resonances |
| `signal_to_noise_ratio` | Signal quality assessment | Sensor health, data quality, instrumentation validation |
| `harmonic_analysis` | Harmonic distortion analysis | Power quality, THD, motor current analysis |

## Detailed Tool Documentation

### 1. `fft_analysis`

Perform Fast Fourier Transform (FFT) for frequency domain analysis of vibration, electrical, and acoustic signals.

**Parameters:**
- `signal` (array, required): Time-domain signal (min 8 items)
- `sample_rate` (number, required): Sampling frequency in Hz (1-1000000)
- `window` (string, optional): Windowing function - "hanning", "hamming", "blackman", "rectangular" (default: "hanning")
- `detrend` (boolean, optional): Remove DC component and linear trends (default: true)

**Example Usage:**
```
"Analyze vibration data from bearing accelerometer: [signal data], sampled at 10 kHz"

"Perform FFT on motor current waveform to detect electrical faults"

"Identify frequency components in acoustic data from pump"
```

**Industrial Use Cases:**

#### Bearing Defect Detection
- **BPFO (Ball Pass Frequency Outer)**: Outer race defect ~3-4x shaft speed
- **BPFI (Ball Pass Frequency Inner)**: Inner race defect ~5-6x shaft speed  
- **BSF (Ball Spin Frequency)**: Rolling element defect ~2x shaft speed
- **FTF (Fundamental Train Frequency)**: Cage defect ~0.4x shaft speed

**Example Output:**
```
FFT Analysis Results:

Signal Properties:
  Sample Rate: 10000.00 Hz
  Signal Length: 8192 samples
  Duration: 0.8192 seconds
  Nyquist Frequency: 5000.00 Hz
  Frequency Resolution: 1.22 Hz

Dominant Frequencies:
  1. 60.00 Hz - Magnitude: 15.3000 (Electrical line frequency (60 Hz))
  2. 120.00 Hz - Magnitude: 3.2000 (Second harmonic (120 Hz))
  3. 1785.00 Hz - Magnitude: 8.7000 (High frequency component (possible bearing defect))

Interpretation: High frequency energy detected. May indicate bearing defects or gear mesh issues.
```

**When to Use:**
- Suspected bearing failures
- Motor electrical issues
- Gearbox problems
- Pump cavitation
- Compressor valve faults
- Fan imbalance

---

### 2. `power_spectral_density`

Calculate Power Spectral Density (PSD) to analyze energy distribution across frequencies.

**Parameters:**
- `signal` (array, required): Time-domain signal (min 16 items)
- `sample_rate` (number, required): Sampling frequency in Hz
- `method` (string, optional): "welch" or "periodogram" (default: "welch")
- `nperseg` (integer, optional): Segment length for Welch method (default: 256)

**Example Usage:**
```
"Calculate PSD of vibration data using Welch method"

"Analyze acoustic noise energy distribution in compressor"

"Assess random vibration characteristics in structure"
```

**Industrial Use Cases:**
- **Vibration Energy Distribution**: Identify frequency bands with excessive energy
- **Noise Assessment**: Quantify acoustic power at different frequencies
- **Random Vibration**: Characterize broadband vibration sources
- **Process Monitoring**: Track energy changes over time

**Method Selection:**
- **Welch**: Better for noisy signals, uses overlapping segments with averaging
- **Periodogram**: Simpler, faster, but more sensitive to noise

**Example Output:**
```
Power Spectral Density Analysis:

Method: Welch
Total Power: 12.5000
Peak Frequency: 1785.00 Hz
Peak Power: 3.8000

Interpretation: High frequency energy - may indicate bearing wear or gear mesh issues.
```

---

### 3. `rms_value`

Calculate Root Mean Square (RMS) for overall signal energy assessment.

**Parameters:**
- `signal` (array, required): Time-domain signal (min 2 items)
- `window_size` (integer, optional): Calculate rolling RMS (min 2)
- `reference_value` (number, optional): Reference for dB calculation

**Example Usage:**
```
"Calculate RMS vibration level for ISO 10816 assessment"

"Compute electrical current RMS for power calculations"

"Monitor rolling RMS to track bearing degradation over time"
```

**Industrial Standards:**

#### ISO 10816 - Vibration Severity Zones
| Zone | RMS Range (mm/s) | Condition | Action |
|------|------------------|-----------|--------|
| A | 0 - 2.3 | Good | Normal operation |
| B | 2.3 - 7.1 | Acceptable | Monitor |
| C | 7.1 - 18 | Unsatisfactory | Plan maintenance |
| D | > 18 | Unacceptable | Stop machine |

**Crest Factor Interpretation:**
- **1.0 - 1.5**: Pure sine wave, very smooth
- **1.5 - 3.0**: Normal rotating machinery
- **3.0 - 5.0**: Some impulsive components
- **> 5.0**: High impulsive content - check for impacts, bearing defects

**Example Output:**
```
RMS Analysis Results:

Overall RMS: 4.8500 mm/s
Peak Value: 12.3000 mm/s
Crest Factor: 2.54

Interpretation: Moderate crest factor - normal for many rotating machinery signals.

Rolling RMS Statistics:
  Number of windows: 20
  Trend: slightly_increasing
```

---

### 4. `peak_detection`

Identify significant peaks in time or frequency domain signals.

**Parameters:**
- `signal` (array, required): Signal data (min 3 items)
- `frequencies` (array, optional): Corresponding frequencies for frequency domain
- `height` (number, optional): Minimum peak height (default: 0.1)
- `distance` (integer, optional): Minimum samples between peaks (default: 1)
- `prominence` (number, optional): Required prominence (default: 0.05)
- `top_n` (integer, optional): Return top N peaks (1-50, default: 10)

**Example Usage:**
```
"Find dominant frequencies in FFT spectrum"

"Detect harmonic pattern in electrical signal"

"Identify resonance peaks in vibration spectrum"
```

**Industrial Applications:**
- **Bearing Analysis**: Detect BPFO, BPFI, BSF frequencies
- **Gear Mesh**: Identify gear mesh frequency and sidebands
- **Electrical Harmonics**: Find 2nd, 3rd, 5th harmonics
- **Resonance**: Locate structural resonance frequencies

**Example Output:**
```
Peak Detection Results:

Total Peaks Found: 8
Interpretation: 8 peaks detected. Top 10 peaks returned.

Top Peaks:
  Rank 1: Frequency 1785.00 Hz, Magnitude 8.7000, Prominence 7.2000 - High frequency - potential bearing defect
  Rank 2: Frequency 1186.50 Hz, Magnitude 5.3000, Prominence 4.1000
  Rank 3: Frequency 366.50 Hz, Magnitude 4.2000, Prominence 3.8000
```

---

### 5. `signal_to_noise_ratio`

Calculate Signal-to-Noise Ratio (SNR) to assess signal quality.

**Parameters:**
- `signal` (array, required): Signal containing signal + noise (min 10 items)
- `noise` (array, optional): Noise reference (estimated if not provided)
- `method` (string, optional): "power", "amplitude", or "peak" (default: "power")

**Example Usage:**
```
"Check SNR of accelerometer signal to validate sensor health"

"Assess data acquisition quality for control system"

"Evaluate instrumentation noise level"
```

**SNR Quality Levels:**
| SNR (dB) | Quality | Interpretation |
|----------|---------|----------------|
| > 40 | Excellent | No sensor issues, suitable for all applications |
| 30-40 | Good | Suitable for most applications |
| 20-30 | Fair | Consider noise reduction for critical measurements |
| 10-20 | Poor | Check sensor and wiring |
| < 10 | Very Poor | Sensor may be faulty or improperly installed |

**Method Selection:**
- **Power**: Based on signal/noise power ratio (most common)
- **Amplitude**: Based on RMS amplitude ratio
- **Peak**: Based on peak values (for impulsive signals)

**Example Output:**
```
Signal-to-Noise Ratio Analysis:

SNR: 42.50 dB
SNR Ratio: 133.40
Signal Power: 125.3000
Noise Power: 0.9400
Quality: Excellent

Interpretation: Signal quality excellent (SNR > 40 dB). No sensor issues detected.
```

---

### 6. `harmonic_analysis`

Detect and analyze harmonic content in electrical and mechanical signals.

**Parameters:**
- `signal` (array, required): Periodic signal (min 64 items)
- `sample_rate` (number, required): Sampling frequency in Hz
- `fundamental_freq` (number, required): Expected fundamental frequency (min 1 Hz)
- `max_harmonic` (integer, optional): Maximum harmonic order (1-100, default: 50)

**Example Usage:**
```
"Calculate THD for 60 Hz voltage waveform"

"Analyze motor current harmonics for MCSA fault detection"

"Assess power quality and IEEE 519 compliance"
```

**Industrial Applications:**

#### Power Quality Assessment
- **THD (Total Harmonic Distortion)**: Overall measure of signal purity
- **Individual Harmonics**: 2nd, 3rd, 5th, 7th order analysis
- **Odd vs Even**: Pattern indicates distortion source
- **IEEE 519 Compliance**: Verify harmonic limits

**THD Interpretation:**
| THD % | Quality | Application Suitability |
|-------|---------|-------------------------|
| < 5% | Excellent | All sensitive equipment |
| 5-8% | Good | Most industrial applications |
| 8-15% | Moderate | General industrial use |
| 15-25% | Poor | May affect sensitive equipment |
| > 25% | Very Poor | Requires filtering |

**Harmonic Patterns:**
- **Odd Harmonics (3rd, 5th, 7th)**: Non-linear loads, VFDs, rectifiers
- **Even Harmonics (2nd, 4th, 6th)**: Asymmetry, half-wave rectification
- **Triplen Harmonics (3rd, 9th, 15th)**: Three-phase imbalance

**Example Output:**
```
Harmonic Analysis Results:

Fundamental Frequency:
  Frequency: 60.00 Hz
  Magnitude: 120.5000

Total Harmonic Distortion (THD): 9.20%
Assessment: Moderate - Acceptable distortion (8% ≤ THD < 15%)

Dominant Harmonics: 5, 3, 7

Harmonic Components (showing top 10):
  H2: 120.00 Hz, Magnitude 2.3000 (1.9% of fundamental)
  H3: 180.00 Hz, Magnitude 5.8000 (4.8% of fundamental)
  H5: 300.00 Hz, Magnitude 8.2000 (6.8% of fundamental)
  H7: 420.00 Hz, Magnitude 3.1000 (2.6% of fundamental)

Interpretation: THD = 9.2%. Acceptable harmonic content for most applications. Odd harmonics dominant - typical of non-linear loads or VFD operation.
```

---

## Complete Industrial Workflows

### Workflow 1: Bearing Condition Monitoring

**Objective**: Detect early bearing failures in a motor

**Steps:**
1. **Capture vibration data** from bearing accelerometer (10 kHz sampling)
2. **Calculate RMS** - Check overall vibration severity vs. ISO 10816
3. **Perform FFT** - Identify frequency peaks
4. **Detect peaks** - Find BPFO, BPFI, BSF frequencies
5. **Monitor trend** - Track RMS increase over time

**Interpretation:**
- RMS increasing? → Bearing degrading
- High frequency peaks? → Specific bearing defect type
- Sidebands around bearing frequencies? → Advanced degradation

**Example:**
```
"Calculate RMS of bearing vibration: [data]"
→ RMS = 6.5 mm/s (Zone B - Acceptable)

"Perform FFT on bearing vibration at 10 kHz"
→ Peaks at 1785 Hz (BPFO), 3570 Hz (2x BPFO)

"Detect peaks in FFT spectrum"
→ Bearing outer race defect confirmed

Recommendation: Schedule bearing replacement within 2-4 weeks
```

---

### Workflow 2: Electrical Power Quality Analysis

**Objective**: Assess power quality and harmonic distortion

**Steps:**
1. **Capture voltage/current waveform** (50 kHz sampling, 200 ms)
2. **Calculate RMS** - Check voltage/current levels
3. **Harmonic analysis** - Calculate THD and individual harmonics
4. **Check SNR** - Validate measurement quality
5. **IEEE 519 compliance** - Verify harmonic limits

**Interpretation:**
- THD > 15%? → Harmonic filtering required
- 5th harmonic dominant? → VFD or rectifier source
- Even harmonics present? → Load asymmetry

**Example:**
```
"Calculate RMS of current waveform: [data]"
→ RMS = 45.2 A

"Perform harmonic analysis on 60 Hz current waveform"
→ THD = 8.2%, 5th harmonic = 6.8%

"Check SNR of current measurement"
→ SNR = 38 dB (Good quality)

Assessment: Power quality acceptable, typical VFD signature
```

---

### Workflow 3: Pump Cavitation Detection

**Objective**: Detect cavitation in centrifugal pump

**Steps:**
1. **Capture acoustic data** from pump (20 kHz sampling)
2. **Calculate PSD** - Check energy distribution
3. **Check RMS** - Overall noise level
4. **Detect peaks** - Look for broadband high frequency

**Interpretation:**
- Broadband high-frequency energy (5-15 kHz)? → Cavitation
- Increased RMS? → Process problem
- Blade pass frequency sidebands? → Impeller issues

**Example:**
```
"Calculate PSD of pump acoustic signal"
→ High energy in 8-15 kHz band

"Calculate RMS of acoustic signal"
→ RMS = 85 dB (elevated)

Diagnosis: Cavitation detected. Check NPSH and suction conditions.
```

---

## Best Practices

### Sampling Rate Selection
- **Bearing analysis**: 10-50 kHz (capture up to 10x BPFO)
- **Motor current**: 20-50 kHz (capture up to 50th harmonic)
- **Gear mesh**: 20-100 kHz (depends on tooth count and speed)
- **Acoustic**: 20-100 kHz (audible range and ultrasonics)

### Window Functions
- **Hanning**: General purpose, good frequency resolution
- **Hamming**: Similar to Hanning, slightly better sidelobe suppression
- **Blackman**: Best sidelobe suppression, wider main lobe
- **Rectangular**: No windowing, maximum frequency resolution, high leakage

### Data Collection
- **Duration**: At least 10 shaft rotations for rotating machinery
- **Samples**: Power of 2 (256, 512, 1024, 2048, 4096, 8192) for efficient FFT
- **Averages**: Use multiple measurements and average for stability

### Trending and Baselines
- Establish baseline measurements during normal operation
- Track changes over time (daily, weekly, monthly)
- Set alarm thresholds based on historical data
- Combine multiple indicators (RMS + frequency peaks + crest factor)

---

## Troubleshooting Guide

### Common Issues

**Issue: No peaks detected in FFT**
- Check sample rate (must be at least 2x highest frequency)
- Verify signal amplitude (may need amplification)
- Reduce prominence threshold
- Check for DC offset (use detrend=true)

**Issue: High noise in measurements**
- Check sensor mounting and alignment
- Verify cable shielding and grounding
- Use lower gain settings if signal is clipping
- Calculate SNR to quantify noise level

**Issue: FFT shows only low frequencies**
- Increase sampling rate
- Check for anti-aliasing filter affecting signal
- Verify sensor frequency response

**Issue: THD calculation seems incorrect**
- Ensure fundamental frequency is correct
- Check that signal is periodic and stable
- Verify sample rate is sufficient (at least 10x fundamental)
- Use longer capture duration for better resolution

---

## Integration Examples

### Python Integration
```python
from stats_server.server import fft_analysis, rms_value, harmonic_analysis

# Bearing vibration monitoring
vibration_data = [...]  # Accelerometer readings
rms_result = rms_value(vibration_data)
print(f"Vibration RMS: {rms_result['rms']:.2f} mm/s")

# FFT for frequency analysis
fft_result = fft_analysis(vibration_data, sample_rate=10000)
print(f"Dominant frequency: {fft_result['dominant_frequencies'][0]['frequency']:.2f} Hz")

# Power quality monitoring
current_data = [...]  # Current waveform
harmonic_result = harmonic_analysis(current_data, sample_rate=20000, fundamental_freq=60)
print(f"THD: {harmonic_result['thd']:.1f}%")
```

### SCADA Integration
- Real-time RMS monitoring on displays
- Trend charts for vibration levels
- Alarm generation based on frequency peaks
- Historical data analysis for predictive maintenance

---

## References and Standards

### Industry Standards
- **ISO 10816**: Mechanical vibration - Evaluation of machine vibration
- **ISO 20816**: Condition monitoring of machines
- **IEEE 519**: Harmonic control in electrical power systems
- **NEMA MG 1**: Motors and generators standards
- **API 610**: Centrifugal pumps for petroleum applications

### Additional Resources
- Vibration analysis handbooks
- Bearing manufacturer defect frequency catalogs
- Power quality guidelines
- Condition monitoring best practices

---

**Built with Python 3.12 • SciPy 1.16.3 • NumPy 2.3.5 • Tested on x86_64**
