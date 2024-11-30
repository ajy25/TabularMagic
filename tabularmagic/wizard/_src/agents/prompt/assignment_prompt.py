ASSIGNMENT_PROMPT = """I want to classify a task as one of:
1. `eda`: Exploratory Data Analysis
2. `linear`: Linear Regression or Logistic Regression
3. `ml`: Machine Learning Regression or Classification
-------------------
Here is the task:
{task}
-------------------
Respond with either `eda`, `linear`, or `ml` to classify the task, no quotations.
Do not include any other text in your response.
"""
