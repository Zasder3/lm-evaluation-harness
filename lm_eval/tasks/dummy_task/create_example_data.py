import pandas as pd


# Create some example palindrome data
data = [
    {"question": "racecar", "answer": 1},
    {"question": "level", "answer": 1},
    {"question": "hello", "answer": 0},
    {"question": "madam", "answer": 1},
    {"question": "python", "answer": 0},
    {"question": "A man a plan a canal Panama", "answer": 1},
    {"question": "No lemon, no melon", "answer": 1},
    {"question": "Was it a car or a cat I saw?", "answer": 1},
    {"question": "This is not a palindrome", "answer": 0},
    {"question": "12321", "answer": 1},
]

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("lm_eval/tasks/dummy_task/palindrome_data.csv", index=False)

print("Example palindrome data created!")
