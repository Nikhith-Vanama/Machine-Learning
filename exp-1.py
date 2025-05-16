import pandas as pd
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'Normal', 'Normal'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool', 'Warm'],
    'Forecast': ['Same', 'Same', 'Change', 'Same', 'Same'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes', 'Yes']
}
df = pd.DataFrame(data)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
def find_s(X, y):
    specific_h = None
    for i, val in enumerate(y):
        if val == "Yes":
            if specific_h is None:
                specific_h = X.iloc[i].values.copy()
            else:
                for j in range(len(specific_h)):
                    if specific_h[j] != X.iloc[i, j]:
                        specific_h[j] = '?'
    return specific_h
specific_hypothesis = find_s(X, y)
print("Most Specific Hypothesis:", specific_hypothesis)