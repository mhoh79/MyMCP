# Add Unit Converter Tool to MCP Server

## Overview
Implement a comprehensive unit conversion tool for common measurement types.

## Tool to Implement

### unit_convert
Convert between different units of measurement.
- **Input**: 
  - `value` (number)
  - `from_unit` (string)
  - `to_unit` (string)
- **Output**: Converted value with formatted result

## Supported Conversions

### Length
- millimeter (mm), centimeter (cm), meter (m), kilometer (km)
- inch (in), foot (ft), yard (yd), mile (mi)

### Weight/Mass
- milligram (mg), gram (g), kilogram (kg), tonne (t)
- ounce (oz), pound (lb), ton

### Temperature
- Celsius (C), Fahrenheit (F), Kelvin (K)

### Volume
- milliliter (ml), liter (l), cubic meter (m3)
- fluid ounce (fl oz), cup, pint (pt), quart (qt), gallon (gal)

### Time
- millisecond (ms), second (s), minute (min), hour (h), day (d), week, year

### Digital Storage
- bit, byte (B), kilobyte (KB), megabyte (MB), gigabyte (GB), terabyte (TB)

### Speed
- meters per second (m/s), kilometers per hour (km/h), miles per hour (mph)

## Implementation Requirements
- Create conversion factor dictionary for each category
- Handle both abbreviated and full unit names (case-insensitive)
- Validate that units are in the same category
- Show formula used for conversion (especially for temperature)
- Round results to appropriate precision
- Include comprehensive docstrings
- Update list_tools() and call_tool() handlers

## Acceptance Criteria
- [ ] All unit categories implemented
- [ ] Accurate conversion factors
- [ ] Both directions of conversion work
- [ ] Full code annotations
- [ ] Input validation with clear error messages
- [ ] Works correctly in Claude Desktop
- [ ] README documentation with examples

## Example Usage
```
User: "Convert 100 kilometers to miles"
Tool: unit_convert with value=100, from_unit="km", to_unit="mi"
Result: "100 km = 62.137 mi"

User: "How many celsius is 75 fahrenheit?"
Tool: unit_convert with value=75, from_unit="F", to_unit="C"
Result: "75°F = 23.89°C (Formula: (F - 32) × 5/9)"

User: "Convert 2 gigabytes to megabytes"
Tool: unit_convert with value=2, from_unit="GB", to_unit="MB"
Result: "2 GB = 2048 MB"
```

## Technical Notes
- Use precise conversion factors (e.g., 1 inch = 2.54 cm exactly)
- Temperature conversions use formulas, not factors
- Digital storage uses binary (1024) not decimal (1000)

## References
- [Unit Conversion](https://en.wikipedia.org/wiki/Conversion_of_units)
- [SI Units](https://en.wikipedia.org/wiki/International_System_of_Units)
- [Imperial Units](https://en.wikipedia.org/wiki/Imperial_units)

## Labels
enhancement, utilities, good first issue
