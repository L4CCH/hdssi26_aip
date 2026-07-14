####
# zeroshot_charts.R
# Create bar charts for visualization of how zeroshot labels performs/labels on full image dataset. 
###
library(ggplot2)
library(forcats)
library(dplyr)
library(jsonlite)


# Input json file from get_concept_labels.py
data <- fromJSON("INPUT_FILE_HERE_HERE.json")

# Aggregate/count the number of images categorized as each label 
data_counts <- data %>%
  count(assigned_label, name = "count") %>%
  arrange(desc(count))

# create bar chart of counts
data_counts %>% 
  mutate(assigned_label = fct_reorder(assigned_label, count, .desc = TRUE)) %>% 
  ggplot(aes(x = assigned_label, y = count, fill = count)) + 
  geom_col() + 
  scale_fill_gradient(low = "mediumpurple1", high = "mediumpurple4") + 
  labs(x = "Zero Shot Label", y = "Counts", fill = "Counts") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  geom_text(aes(label = count), vjust = -0.5, size = 3.5)
