"""
Tests for Complex Analysis Tools in Engineering Math Server.

These tests verify the 4 complex analysis tools:
1. complex_operations
2. complex_functions
3. roots_of_unity
4. complex_conjugate_operations
"""

import pytest
import numpy as np
import cmath
from src.builtin.engineering_math_server.tools.complex_analysis import (
    parse_complex_number,
    complex_operations,
    complex_functions,
    roots_of_unity,
    complex_conjugate_operations,
    handle_complex_operations,
    handle_complex_functions,
    handle_roots_of_unity,
    handle_complex_conjugate_operations,
)


class TestParseComplexNumber:
    """Test complex number parsing from various formats."""
    
    def test_parse_rectangular_positive(self):
        """Test parsing rectangular form with positive imaginary part."""
        z = parse_complex_number("3+4j")
        assert abs(z.real - 3.0) < 1e-10
        assert abs(z.imag - 4.0) < 1e-10
    
    def test_parse_rectangular_negative(self):
        """Test parsing rectangular form with negative imaginary part."""
        z = parse_complex_number("3-4j")
        assert abs(z.real - 3.0) < 1e-10
        assert abs(z.imag - (-4.0)) < 1e-10
    
    def test_parse_polar_degrees(self):
        """Test parsing polar form with degrees."""
        z = parse_complex_number("5∠53.13°")
        # 5∠53.13° ≈ 3 + 4j
        assert abs(z.real - 3.0) < 0.01
        assert abs(z.imag - 4.0) < 0.01
    
    def test_parse_polar_radians(self):
        """Test parsing polar form with radians."""
        z = parse_complex_number("5∠0.9273rad")
        # 5∠53.13° ≈ 3 + 4j
        assert abs(z.real - 3.0) < 0.01
        assert abs(z.imag - 4.0) < 0.01
    
    def test_parse_real_only(self):
        """Test parsing real number only."""
        z = parse_complex_number("5+0j")
        assert abs(z.real - 5.0) < 1e-10
        assert abs(z.imag - 0.0) < 1e-10


class TestComplexOperations:
    """Test complex_operations tool."""
    
    def test_addition(self):
        """Test complex number addition."""
        result = complex_operations("3+4j", "add", "1-2j")
        assert abs(result["result"].real - 4.0) < 1e-10
        assert abs(result["result"].imag - 2.0) < 1e-10
    
    def test_subtraction(self):
        """Test complex number subtraction."""
        result = complex_operations("3+4j", "subtract", "1-2j")
        assert abs(result["result"].real - 2.0) < 1e-10
        assert abs(result["result"].imag - 6.0) < 1e-10
    
    def test_multiplication(self):
        """Test complex number multiplication."""
        result = complex_operations("3+4j", "multiply", "1-2j")
        # (3+4j)(1-2j) = 3 - 6j + 4j - 8j² = 3 - 2j + 8 = 11 - 2j
        assert abs(result["result"].real - 11.0) < 1e-10
        assert abs(result["result"].imag - (-2.0)) < 1e-10
    
    def test_division(self):
        """Test complex number division."""
        result = complex_operations("6+8j", "divide", "1+1j")
        # (6+8j)/(1+1j) = (6+8j)(1-1j)/2 = (14+2j)/2 = 7+1j
        assert abs(result["result"].real - 7.0) < 1e-10
        assert abs(result["result"].imag - 1.0) < 1e-10
    
    def test_division_by_zero(self):
        """Test division by zero raises error."""
        with pytest.raises(ValueError, match="Division by zero"):
            complex_operations("3+4j", "divide", "0+0j")
    
    def test_power(self):
        """Test complex number power."""
        result = complex_operations("1+1j", "power", "2+0j")
        # (1+1j)² = 1 + 2j - 1 = 2j
        assert abs(result["result"].real - 0.0) < 1e-10
        assert abs(result["result"].imag - 2.0) < 1e-10
    
    def test_conjugate(self):
        """Test complex conjugate."""
        result = complex_operations("3+4j", "conjugate")
        assert abs(result["result"].real - 3.0) < 1e-10
        assert abs(result["result"].imag - (-4.0)) < 1e-10
    
    def test_magnitude(self):
        """Test magnitude calculation."""
        result = complex_operations("3+4j", "magnitude")
        assert abs(result["magnitude"] - 5.0) < 1e-10
    
    def test_phase_degrees(self):
        """Test phase calculation in degrees."""
        result = complex_operations("1+1j", "phase")
        assert abs(result["phase_degrees"] - 45.0) < 1e-10
    
    def test_rect_to_polar(self):
        """Test rectangular to polar conversion."""
        result = complex_operations("3+4j", "rect_to_polar")
        assert abs(result["magnitude"] - 5.0) < 1e-10
        assert abs(result["phase_degrees"] - 53.13010235) < 0.01
    
    def test_polar_to_rect(self):
        """Test polar to rectangular conversion."""
        result = complex_operations("5∠53.13°", "polar_to_rect")
        assert abs(result["real"] - 3.0) < 0.01
        assert abs(result["imaginary"] - 4.0) < 0.01
    
    def test_mixed_formats(self):
        """Test operations with mixed input formats."""
        result = complex_operations("3+4j", "multiply", "2∠45°")
        # Should work without error
        assert "result" in result


class TestComplexFunctions:
    """Test complex_functions tool."""
    
    def test_exp_eulers_identity(self):
        """Test Euler's identity: e^(iπ) + 1 = 0."""
        result = complex_functions(f"{np.pi}j", "exp")
        z = result["result"]
        assert abs(z.real - (-1.0)) < 1e-10
        assert abs(z.imag - 0.0) < 1e-10
    
    def test_exp_real(self):
        """Test exponential of real number."""
        result = complex_functions("1+0j", "exp")
        assert abs(result["result"].real - np.e) < 1e-10
        assert abs(result["result"].imag - 0.0) < 1e-10
    
    def test_log_of_e(self):
        """Test log(e) = 1."""
        result = complex_functions(f"{np.e}+0j", "log")
        assert abs(result["result"].real - 1.0) < 1e-10
        assert abs(result["result"].imag - 0.0) < 1e-10
    
    def test_log_of_negative(self):
        """Test log of negative real number."""
        result = complex_functions("-1+0j", "log")
        # log(-1) = iπ
        assert abs(result["result"].real - 0.0) < 1e-10
        assert abs(result["result"].imag - np.pi) < 1e-10
    
    def test_log_of_zero(self):
        """Test log of zero raises error."""
        with pytest.raises(ValueError, match="Logarithm of zero"):
            complex_functions("0+0j", "log")
    
    def test_sqrt_positive(self):
        """Test square root of positive real."""
        result = complex_functions("4+0j", "sqrt")
        assert abs(result["result"].real - 2.0) < 1e-10
        assert abs(result["result"].imag - 0.0) < 1e-10
    
    def test_sqrt_negative(self):
        """Test square root of negative real."""
        result = complex_functions("-4+0j", "sqrt")
        assert abs(result["result"].real - 0.0) < 1e-10
        assert abs(result["result"].imag - 2.0) < 1e-10
    
    def test_sin_zero(self):
        """Test sin(0) = 0."""
        result = complex_functions("0+0j", "sin")
        assert abs(result["result"].real - 0.0) < 1e-10
        assert abs(result["result"].imag - 0.0) < 1e-10
    
    def test_cos_zero(self):
        """Test cos(0) = 1."""
        result = complex_functions("0+0j", "cos")
        assert abs(result["result"].real - 1.0) < 1e-10
        assert abs(result["result"].imag - 0.0) < 1e-10
    
    def test_tan_45_degrees(self):
        """Test tan(π/4) = 1."""
        result = complex_functions(f"{np.pi/4}+0j", "tan")
        assert abs(result["result"].real - 1.0) < 1e-10
    
    def test_sinh_zero(self):
        """Test sinh(0) = 0."""
        result = complex_functions("0+0j", "sinh")
        assert abs(result["result"].real - 0.0) < 1e-10
        assert abs(result["result"].imag - 0.0) < 1e-10
    
    def test_cosh_zero(self):
        """Test cosh(0) = 1."""
        result = complex_functions("0+0j", "cosh")
        assert abs(result["result"].real - 1.0) < 1e-10
        assert abs(result["result"].imag - 0.0) < 1e-10


class TestRootsOfUnity:
    """Test roots_of_unity tool."""
    
    def test_roots_of_unity_n1(self):
        """Test first root of unity (n=1)."""
        result = roots_of_unity(1)
        assert result["number_of_roots"] == 1
        assert abs(result["roots"][0]["value"] - 1.0) < 1e-10
    
    def test_roots_of_unity_n2(self):
        """Test square roots of unity (n=2)."""
        result = roots_of_unity(2)
        assert result["number_of_roots"] == 2
        
        # Should be 1 and -1
        roots = [r["value"] for r in result["roots"]]
        assert abs(roots[0] - 1.0) < 1e-10
        assert abs(roots[1] - (-1.0)) < 1e-10
    
    def test_roots_of_unity_n3(self):
        """Test cube roots of unity (n=3)."""
        result = roots_of_unity(3)
        assert result["number_of_roots"] == 3
        
        # All roots should have magnitude 1
        for root in result["roots"]:
            assert abs(root["magnitude"] - 1.0) < 1e-10
        
        # Angular spacing should be 120°
        assert abs(result["angular_spacing_degrees"] - 120.0) < 1e-10
    
    def test_roots_of_unity_n4(self):
        """Test fourth roots of unity (n=4)."""
        result = roots_of_unity(4)
        assert result["number_of_roots"] == 4
        
        # Should be 1, i, -1, -i
        roots = [r["value"] for r in result["roots"]]
        expected = [1, 1j, -1, -1j]
        for computed, expected_val in zip(roots, expected):
            assert abs(computed - expected_val) < 1e-10
    
    def test_roots_of_unity_sum_zero(self):
        """Test that sum of roots of unity is 0 (for n > 1)."""
        for n in [2, 3, 4, 5, 8]:
            result = roots_of_unity(n)
            if n > 1:
                root_sum = result["sum_of_roots"]["value"]
                assert abs(root_sum) < 1e-9, f"Sum should be 0 for n={n}"
    
    def test_nth_root_of_16(self):
        """Test fourth roots of 16."""
        result = roots_of_unity(4, "16+0j")
        
        # All roots should have magnitude 2
        for root in result["roots"]:
            assert abs(root["magnitude"] - 2.0) < 1e-10
        
        # First root should be 2
        assert abs(result["roots"][0]["value"].real - 2.0) < 1e-10
    
    def test_nth_root_specific_index(self):
        """Test requesting specific root index."""
        result = roots_of_unity(4, root_index=2)
        assert "requested_root" in result
        assert result["requested_root"]["index"] == 2
    
    def test_invalid_n(self):
        """Test that invalid n raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            roots_of_unity(0)
        
        with pytest.raises(ValueError, match="positive integer"):
            roots_of_unity(-1)
    
    def test_invalid_root_index(self):
        """Test that invalid root_index raises error."""
        with pytest.raises(ValueError, match="root_index must be"):
            roots_of_unity(4, root_index=5)


class TestComplexConjugateOperations:
    """Test complex_conjugate_operations tool."""
    
    def test_conjugate(self):
        """Test conjugate operation."""
        result = complex_conjugate_operations("3+4j", "conjugate")
        assert abs(result["conjugate"].real - 3.0) < 1e-10
        assert abs(result["conjugate"].imag - (-4.0)) < 1e-10
        assert abs(result["magnitude_unchanged"] - 5.0) < 1e-10
    
    def test_real_part_extraction(self):
        """Test real part extraction using (z + z*)/2."""
        result = complex_conjugate_operations("3+4j", "real_part")
        assert abs(result["real_part"] - 3.0) < 1e-10
        assert abs(result["verification"] - 3.0) < 1e-10
    
    def test_imaginary_part_extraction(self):
        """Test imaginary part extraction using (z - z*)/(2i)."""
        result = complex_conjugate_operations("3+4j", "imaginary_part")
        assert abs(result["imaginary_part"] - 4.0) < 1e-10
        assert abs(result["verification"] - 4.0) < 1e-10
    
    def test_magnitude_squared(self):
        """Test magnitude squared using z·z*."""
        result = complex_conjugate_operations("3+4j", "magnitude_squared")
        assert abs(result["magnitude_squared"] - 25.0) < 1e-10
        assert abs(result["magnitude"] - 5.0) < 1e-10
        assert abs(result["verification"] - 25.0) < 1e-10
    
    def test_conjugate_product(self):
        """Test conjugate product z1·z2* (for power calculations)."""
        result = complex_conjugate_operations("120+0j", "conjugate_product", "10∠-30°")
        # S = V·I* for power calculation
        assert "result" in result
        assert "application" in result
    
    def test_conjugate_sum_property(self):
        """Test (z1 + z2)* = z1* + z2*."""
        result = complex_conjugate_operations("3+4j", "conjugate_sum", "1-2j")
        assert result["are_equal"] is True
        assert result["property"] == "(z1 + z2)* = z1* + z2*"
    
    def test_conjugate_product_property(self):
        """Test (z1·z2)* = z1*·z2*."""
        result = complex_conjugate_operations("3+4j", "conjugate_product_property", "1-2j")
        assert result["are_equal"] is True
        assert result["property"] == "(z1 · z2)* = z1* · z2*"


class TestEngineeringApplications:
    """Test engineering application scenarios."""
    
    def test_ac_impedance_calculation(self):
        """Test AC circuit impedance: Z = R + jωL."""
        # R = 100Ω, L = 0.1H, ω = 314.159 rad/s (60Hz)
        # Z = 100 + j(314.159 × 0.1) = 100 + j31.4159Ω
        result = complex_operations("100+0j", "add", "0+31.4159j")
        
        assert abs(result["magnitude"] - 104.82) < 0.1
        assert abs(result["phase_deg"] - 17.44) < 0.1
    
    def test_phasor_power_calculation(self):
        """Test apparent power: S = V·I*."""
        # V = 120∠0° V, I = 10∠-30° A
        # S = 120∠0° × 10∠30° = 1200∠30° VA
        result = complex_conjugate_operations("120+0j", "conjugate_product", "8.66-5j")
        
        # Real power P = Re(S), Reactive power Q = Im(S)
        s = result["result"]
        p = s.real  # Real power
        q = s.imag  # Reactive power
        
        assert abs(result["magnitude"] - 1200.0) < 1.0
    
    def test_transfer_function_evaluation(self):
        """Test transfer function H(jω) = 1/(1 + jω/ω₀)."""
        # Evaluate at ω = 100 rad/s, ω₀ = 10 rad/s
        # H = 1/(1 + j10)
        result = complex_operations("1+0j", "divide", "1+10j")
        
        magnitude = result["magnitude"]
        phase = result["phase_deg"]
        
        # |H| = 1/√101 ≈ 0.0995
        assert abs(magnitude - 0.0995) < 0.001
        # ∠H = -atan(10) ≈ -84.29°
        assert abs(phase - (-84.29)) < 0.1
    
    def test_frequency_response_bode(self):
        """Test Bode plot calculations."""
        # H(jω) = 10/(1 + jω)
        # At ω = 1: H = 10/(1+j)
        result = complex_operations("10+0j", "divide", "1+1j")
        
        magnitude = result["magnitude"]
        phase = result["phase_deg"]
        
        # Magnitude in dB: 20 log₁₀(|H|)
        magnitude_db = 20 * np.log10(magnitude)
        
        # |H| = 10/√2 ≈ 7.071
        assert abs(magnitude - 7.071) < 0.01
        # ∠H = -45°
        assert abs(phase - (-45.0)) < 0.1


class TestHandlerFunctions:
    """Test async handler functions."""
    
    @pytest.mark.asyncio
    async def test_handle_complex_operations(self):
        """Test complex_operations handler."""
        result = await handle_complex_operations({
            "z1": "3+4j",
            "operation": "magnitude"
        })
        
        assert result.isError is False
        assert "5" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_handle_complex_functions(self):
        """Test complex_functions handler."""
        result = await handle_complex_functions({
            "z": "0+0j",
            "function": "cos"
        })
        
        assert result.isError is False
        assert "1" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_handle_roots_of_unity(self):
        """Test roots_of_unity handler."""
        result = await handle_roots_of_unity({
            "n": 4
        })
        
        assert result.isError is False
        assert "4" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_handle_complex_conjugate_operations(self):
        """Test complex_conjugate_operations handler."""
        result = await handle_complex_conjugate_operations({
            "z": "3+4j",
            "operation": "conjugate"
        })
        
        assert result.isError is False
        assert "-4" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_handler_error_handling(self):
        """Test handler error handling."""
        result = await handle_complex_operations({
            "z1": "3+4j",
            "operation": "invalid_op"
        })
        
        assert result.isError is True
        assert "Error" in result.content[0].text


class TestSpecialValues:
    """Test special values and edge cases."""
    
    def test_zero(self):
        """Test operations with zero."""
        result = complex_operations("0+0j", "magnitude")
        assert abs(result["magnitude"] - 0.0) < 1e-10
    
    def test_one(self):
        """Test operations with one."""
        result = complex_operations("1+0j", "multiply", "3+4j")
        assert abs(result["result"] - (3+4j)) < 1e-10
    
    def test_imaginary_unit(self):
        """Test operations with i."""
        result = complex_operations("0+1j", "power", "2+0j")
        # i² = -1
        assert abs(result["result"] - (-1+0j)) < 1e-10
    
    def test_negative_one(self):
        """Test operations with -1."""
        result = complex_operations("-1+0j", "multiply", "3+4j")
        assert abs(result["result"] - (-3-4j)) < 1e-10
    
    def test_negative_i(self):
        """Test operations with -i."""
        result = complex_operations("0-1j", "power", "2+0j")
        # (-i)² = -1
        assert abs(result["result"] - (-1+0j)) < 1e-10


class TestNumericalAccuracy:
    """Test numerical accuracy requirements."""
    
    def test_euler_formula_accuracy(self):
        """Test Euler's formula: e^(iπ) + 1 = 0 with high accuracy."""
        result = complex_functions(f"{np.pi}j", "exp")
        z = result["result"] + 1
        assert abs(z) < 1e-10
    
    def test_demoivre_theorem(self):
        """Test De Moivre's theorem: (cos θ + i sin θ)^n = cos(nθ) + i sin(nθ)."""
        theta = np.pi / 4  # 45 degrees
        n = 3
        
        # Left side: (cos θ + i sin θ)^n
        z = complex(np.cos(theta), np.sin(theta))
        left = z ** n
        
        # Right side: cos(nθ) + i sin(nθ)
        right = complex(np.cos(n * theta), np.sin(n * theta))
        
        assert abs(left - right) < 1e-10
    
    def test_magnitude_phase_reconstruction(self):
        """Test that magnitude and phase can reconstruct the original number."""
        z_str = "3+4j"
        
        # Get magnitude and phase
        mag_result = complex_operations(z_str, "magnitude")
        phase_result = complex_operations(z_str, "phase")
        
        # Reconstruct
        magnitude = mag_result["magnitude"]
        phase_rad = phase_result["phase_radians"]
        
        reconstructed = cmath.rect(magnitude, phase_rad)
        original = parse_complex_number(z_str)
        
        assert abs(reconstructed - original) < 1e-10
