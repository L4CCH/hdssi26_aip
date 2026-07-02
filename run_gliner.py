"""
run_gliner.py

script to run GLiNER wrapper around spaCy NER
- obtains entities from either:
  1. specific bib numbers, or
  2. a sliced batch of transcripts
- outputs 2 JSON files: by transcript and by entities
"""

import json
import spacy
from collections import defaultdict, Counter

# I/O directories
INPUT_FILE = "bib_interviewee_date_body.json"
OUTPUT_BY_TRANSCRIPT = "gliner_by_transcript.json"
OUTPUT_BY_ENTITY = "gliner_by_entity.json"

# Option 1: specific bib numbers
# If this list is non-empty, the script will use these bib numbers
INPUT_BIB_NUMBERS = [] # ["46733", "47293"]

# Option 2: batching by slice
# Used only if INPUT_BIB_NUMBERS = []
START = 0      # inclusive
END = 2       # exclusive

# zero-shot labels 
LABELS = [
    "name",
    "job",
    "calendar date",
    "laboratory",
    "university",
    "organization",
    "award",
    "conference",
    "city",
    "country"
]

# gliner configurations
custom_config = {
    "gliner_model": "urchade/gliner_multi-v2.1",
    "chunk_size": 250,
    "labels": LABELS,
    "style": "ent",
    "threshold": 0.5
}

"""
sentencizer used for context extraction
- detects sentence boundaries
"""

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")
nlp.add_pipe("gliner_spacy", config=custom_config)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    transcripts = json.load(f)

"""
Choose subset:
- If INPUT_BIB_NUMBERS has values, process only those bib numbers.
- If INPUT_BIB_NUMBERS is empty, process transcripts from START to END.
"""

if INPUT_BIB_NUMBERS:
    wanted_bibs = set(str(bib).strip() for bib in INPUT_BIB_NUMBERS)

    subset = [
        transcript for transcript in transcripts
        if str(transcript.get("field_bib_number")).strip() in wanted_bibs
    ]

    print(f"Running on {len(subset)} selected bib numbers.")

else:
    subset = transcripts[START:END]

    print(f"Running on transcript slice {START}:{END}.")
    print(f"Total transcripts in batch: {len(subset)}")

json_by_transcript = []

entity_contexts = defaultdict(lambda: {
    "entity": None,
    "type": None,
    "contexts": defaultdict(list)
})

"""
for loop to go through input file and extract entities,
keeping count of appearance by transcript

also groups by transcript to count entities
"""

for i, transcript in enumerate(subset, start=1):
    bib = str(transcript.get("field_bib_number"))
    interviewee = transcript.get("interviewee")
    interview_date = transcript.get("interview_date")
    body = transcript.get("body", "")

    print(f"Processing {i}/{len(subset)}: bib {bib}")

    doc = nlp(body)

    entity_counter = Counter()

    for ent in doc.ents:
        entity_text = ent.text.strip()
        entity_type = ent.label_
        score = float(ent._.score) if hasattr(ent._, "score") else None

        if not entity_text:
            continue

        entity_counter[(entity_text, entity_type)] += 1

        key = (entity_text.lower(), entity_type)

        entity_contexts[key]["entity"] = entity_text
        entity_contexts[key]["type"] = entity_type
        entity_contexts[key]["contexts"][bib].append({
            "context": ent.sent.text.strip(),
            "score": score
        })

    transcript_entities = []

    for (entity_text, entity_type), count in entity_counter.items():
        transcript_entities.append({
            "entity": entity_text,
            "type": entity_type,
            "count": count
        })

    json_by_transcript.append({
        "bib": bib,
        "interviewee": interviewee,
        "interview_date": interview_date,
        "entities": transcript_entities
    })

json_by_entity = []

for item in entity_contexts.values():
    contexts = dict(item["contexts"])

    counts_by_transcript = {
        bib: len(context_list)
        for bib, context_list in contexts.items()
    }

    total_count = sum(counts_by_transcript.values())

    json_by_entity.append({
        "entity": item["entity"],
        "type": item["type"],
        "total_count": total_count,
        "counts_by_transcript": counts_by_transcript,
        "contexts": contexts
    })

# write out to output files
with open(OUTPUT_BY_TRANSCRIPT, "w", encoding="utf-8") as f:
    json.dump(json_by_transcript, f, indent=2, ensure_ascii=False)

with open(OUTPUT_BY_ENTITY, "w", encoding="utf-8") as f:
    json.dump(json_by_entity, f, indent=2, ensure_ascii=False)

print(f"Saved transcript-level JSON to {OUTPUT_BY_TRANSCRIPT}")
print(f"Saved entity-level JSON to {OUTPUT_BY_ENTITY}")