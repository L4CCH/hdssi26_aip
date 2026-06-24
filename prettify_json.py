"""
prettify_json.py

From manual observation, the last record in all_json_2024Sep24.json
was cut off. This script discards the broken record, keeping everything
else in the AIP oral histories JSON dataset.
"""

import json

input_file = "all_json_2024Sep24.json"
output_file = "cleaned_output.json"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Find where the broken final object begins (nid is first field in a record)
last_start = text.rfind('{"nid"')
if last_start == -1:  # 
    raise ValueError("Could not find the start of the last object.")

# Keep everything before the broken final object
fixed = text[:last_start].rstrip()

# Remove trailing comma if present
if fixed.endswith(","):
    fixed = fixed[:-1]

# Close the JSON array
fixed += "]"

# Validate recovered JSON
data = json.loads(fixed)

# Save directly to final output
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Recovered {len(data)} records.")
print(f"Cleaned JSON written to {output_file}")