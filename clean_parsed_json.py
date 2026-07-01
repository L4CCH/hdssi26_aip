import json
import re

# input_file = “<insert your path>”
# output_file = “<insert your path>”

name_pattern = re.compile(r'>(.*?)</a>$')
date_pattern = re.compile(r'>(.*?)</time>$')

with open(input_file, "r", encoding="utf-8") as f:
    transcripts = json.load(f)

cleaned_transcripts = []

for item in transcripts:
    title = item["title"]
    date = item["field_interview_date"]
    multiple_dates = item["field_multiple_dates"]
    bib_number = item["field_bib_number"]
    body = item["body"]

    interviewee = name_pattern.search(title).group(1).strip()

    # Use field_multiple_dates if present; otherwise extract the single date
    if multiple_dates is not None:
        interview_date = multiple_dates
    elif date is not None and date_pattern.search(date):
        interview_date = date_pattern.search(date).group(1).strip()
    else:
        interview_date = None

    cleaned_transcripts.append({
        "field_bib_number": bib_number,
        "interviewee": interviewee,
        "interview_date": interview_date,
        "body": body
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_transcripts, f, indent=2, ensure_ascii=False)

print(f"Saved {len(cleaned_transcripts)} cleaned transcripts to: {output_file}")
