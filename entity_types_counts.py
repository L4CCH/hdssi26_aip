import duckdb

# read in all batched input files
input_files = "batch_runs/gliner_by_entity_*_*.json"

# count num of unique entities by type
query = f"""
    SELECT count(DISTINCT entity) as distinct_entities, type
    FROM read_json_auto('{input_files}')
    GROUP BY "type"
    ORDER BY distinct_entities DESC
"""

# run query
final_df = duckdb.sql(query).df()

print(final_df)

# Counts as of 7/6/26
#   distinct_entities           type
#              61210           name
#              46111   organization
#              11851            job
#               9353     laboratory
#               8953  calendar date
#               8058     university
#               7997           city
#               7099     conference
#               3246          award
#               1383        country




