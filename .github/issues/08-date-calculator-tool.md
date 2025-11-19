# Add Date Calculator Tool to MCP Server

## Overview
Implement date and time calculation utilities.

## Tools to Implement

### 1. date_diff
Calculate difference between two dates.
- **Input**: 
  - `date1` (string, ISO format: YYYY-MM-DD)
  - `date2` (string, ISO format: YYYY-MM-DD)
  - `unit` (optional: days, weeks, months, years, all)
- **Output**: Difference in specified unit(s)
- **Example**: "2025-01-01" to "2025-12-31" = 364 days

### 2. date_add
Add or subtract time from a date.
- **Input**:
  - `date` (string, ISO format)
  - `amount` (integer, can be negative)
  - `unit` (days, weeks, months, years)
- **Output**: New date in ISO format
- **Example**: "2025-01-15" + 30 days = "2025-02-14"

### 3. business_days
Calculate business days between dates (excluding weekends).
- **Input**:
  - `start_date` (string, ISO format)
  - `end_date` (string, ISO format)
  - `exclude_holidays` (optional array of holiday dates)
- **Output**: Number of business days
- **Note**: Optionally exclude custom holidays

### 4. age_calculator
Calculate age from birthdate.
- **Input**: 
  - `birthdate` (string, ISO format)
  - `reference_date` (optional, default=today)
- **Output**: Age in years, months, days
- **Example**: Born 1990-05-15 → "35 years, 6 months, 4 days old"

### 5. day_of_week
Determine day of week for any date.
- **Input**: `date` (string, ISO format)
- **Output**: Day name and additional info (week number, day of year)
- **Example**: "2025-11-19" → "Wednesday, Week 47, Day 323"

## Implementation Requirements
- Use Python's datetime and dateutil libraries
- Handle leap years correctly
- Validate date formats (ISO 8601)
- Handle timezone awareness (UTC default)
- Add comprehensive docstrings
- Update list_tools() and call_tool() handlers
- Include edge case handling (invalid dates, future dates)

## Acceptance Criteria
- [ ] All 5 date calculator tools implemented
- [ ] Correct handling of leap years
- [ ] Full code annotations
- [ ] Input validation with clear error messages
- [ ] Works correctly in Claude Desktop
- [ ] README documentation with examples

## Example Usage
```
User: "How many days until Christmas 2025?"
Tool: date_diff with date1="2025-11-19", date2="2025-12-25"
Result: "36 days (5 weeks, 1 day)"

User: "What date is 90 days from today?"
Tool: date_add with date="2025-11-19", amount=90, unit="days"
Result: "2026-02-17"

User: "How old is someone born on January 1, 2000?"
Tool: age_calculator with birthdate="2000-01-01"
Result: "25 years, 10 months, 18 days old"

User: "What day of the week was January 1, 2000?"
Tool: day_of_week with date="2000-01-01"
Result: "Saturday, Week 1, Day 1"
```

## Additional Features (Optional)
- Convert between timezones
- Calculate duration in human-readable format
- Find next/previous occurrence of a weekday
- Calculate time until/since with hours/minutes

## Dependencies
Add to requirements.txt:
```
python-dateutil>=2.8.0
```

## References
- [Python datetime](https://docs.python.org/3/library/datetime.html)
- [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601)
- [Leap Year Calculation](https://en.wikipedia.org/wiki/Leap_year)

## Labels
enhancement, utilities, datetime
