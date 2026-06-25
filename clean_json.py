import json
import re

# input_file = “<insert your path>”
# output_file = “<insert your path>”

name_pattern = re.compile(r'>(.*?)</a>$')

with open(input_file, “r”, encoding=”utf-8”) as f:
	transcripts = json.load(f) 			# transcripts is the input file

cleaned_transcripts = [] 				# initialize empty list of dictionaries

for item in transcripts:
title = item["title"]
bib_number = item["field_bib_number"]
body = item["body"]

interviewee = name_pattern.search(title).group(1).strip()	# group(1) is first occurrence

cleaned_transcripts.append({
        "field_bib_number": bib_number,
        "interviewee": interviewee,
        "body": body
})

with open(output_file, "w", encoding="utf-8") as f:
json.dump(cleaned_transcripts, f, indent=2, ensure_ascii=False)


