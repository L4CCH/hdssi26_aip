import duckdb

# input json file (gliner by entity)
input_file = "gliner_by_entity_0_100.json"

# query the top 100 name entitys that has the highest total count
query = f"""
    SELECT 
        entity, 
        type, 
        total_count 
    FROM read_json_auto('{input_file}')
    WHERE type = 'name'
    ORDER BY total_count DESC
    LIMIT 100
"""
top_100_df = duckdb.sql(query).df()

# write to csv
output_csv = "top_100_names_0_100.csv"
top_100_df.to_csv(output_csv, index=False)
