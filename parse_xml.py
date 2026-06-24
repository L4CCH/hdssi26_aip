import json
import re
import os
import xmltodict

output_dir = "HDSSI_AIP_outputs"
os.makedirs(output_dir, exist_ok=True)

input_file = "ohi_transcripts.xml"

output_file = os.path.join(output_dir, "parsed_unique_transcripts.json")
error_output_file = os.path.join(output_dir, "xml_parse_errors.txt")

with open(input_file, "r", encoding="utf-8") as f:
    xml_text = f.read()

# Find each complete <item>...</item> block independently
item_blocks = re.findall(
    r"<item\b[^>]*>.*?</item>",
    xml_text,
    flags=re.DOTALL
)

total_chunks = len(item_blocks)

seen_bibs = set()
unique_items = []

missing_bib_count = 0
duplicate_bib_count = 0
malformed_item_count = 0

with open(error_output_file, "w", encoding="utf-8") as error_file:
    for index, item_xml in enumerate(item_blocks):
        try:
            parsed = xmltodict.parse(item_xml)
            item = parsed["item"]

        except Exception as e:
            malformed_item_count += 1

            error_file.write(f"\nMalformed item #{index}\n")
            error_file.write(f"Error: {e}\n")
            error_file.write(item_xml[:2000])
            error_file.write("\n")

            continue

        bib = item.get("field_bib_number")

        # Skip records that do not have a bib number
        if bib is None:
            missing_bib_count += 1
            continue

        # Skip duplicate bib numbers and keep only the first occurrence
        if bib in seen_bibs:
            duplicate_bib_count += 1
            continue

        seen_bibs.add(bib)
        unique_items.append(item)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(unique_items, f, indent=2, ensure_ascii=False)

print("\n")
print(f"Total item chunks found: {total_chunks}")
print(f"Unique transcripts saved: {len(unique_items)}")
print(f"Records skipped because of missing bib number: {missing_bib_count}")
print(f"Records skipped because of duplicate bib number: {duplicate_bib_count}")
print(f"Records skipped because of malformed XML item: {malformed_item_count}")
print(
    f"Total records skipped: "
    f"{missing_bib_count + duplicate_bib_count + malformed_item_count}"
)
print(f"Saved parsed transcripts to: {output_file}")
print(f"Saved malformed item logs to: {error_output_file}")