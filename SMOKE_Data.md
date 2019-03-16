# EPA CEMS SMOKE Data

Hourly Continuous Emissions Monitoring (CEM) data files formatted for use with the Sparse Matrix Operator Kernel Emissions (SMOKE) modeling system. [Pre-packaged data](https://ampd.epa.gov/ampd/)

## SMOKE Data Format

| Position | Name | Type | Description |
|-----|-----|-----|-----|
| A | ORISID | Char (6) | DOE Plant Identification Code (required) |
| B | BLRID | Char (6) | Boiler Identification Code (required) |
| C | YYMMDD | Int | Date of data in YYMMDD format (required) |
| D | HOUR | Integer | Hour value from 0 to 23 |
| E | NOXMASS | Real | Nitrogen oxide emissions (lb/hr) (required) |
| F | SO2MASS | Real | Sulfur dioxide emissions (lb/hr) (required) |
| G | NOXRATE | Real| Nitrogen oxide emissions rate (lb/MMBtu) |
| H | OPTIME | Real | Fraction of hour unit was operating (optional) |
| I | GLOAD | Real | Gross load (MW) (optional) |
| J | SLOAD | Real | Steam load (1000 lbs/hr) (optional) |
| K | HTINPUT | Real | Heat input (mmBtu) (required) |
| L | HTINPUTMEASURE | Character(2) | Code number indicating measured or substituted |
| M | SO2MEASURE | Character(2) | Code number indicating measured or substituted |
| N | NOXMTMEASURE | Character(2) | Code number indicating measured or substituted |
| O | NOXRMEASURE | Character(2) | Code number indicating measured or substituted |
| P | UNITFLOW | Real |Flow rate (ft3/sec) for the Boiler Unit (optional) |

Code Numbers used for HTINPUTMEASURE, SO2MEASURE, NOXMMEASURE, NOXRMEASURE:  
• 01 = 'Measured'  
• 02 = 'Calculated'  
• 03 = 'Substitute'  
• 04 = 'Measured and Substitute'  
• 97 = 'Not Applicable'  
• 98 = 'Undetermined'  
• 99 = 'Unknown Code'  
