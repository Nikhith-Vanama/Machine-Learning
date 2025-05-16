import pandas as pd

# Sample training data
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'Normal', 'Normal'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool', 'Warm'],
    'Forecast': ['Same', 'Same', 'Change', 'Same', 'Same'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes', 'Yes']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Candidate-Elimination Algorithm
def candidate_elimination(X, y):
    specific_h = ["0"] * len(X.columns)
    general_h = [["?"] * len(X.columns)]

    for i, val in enumerate(y):
        if val == "Yes":
            # Update S to generalize to include positive example
            for x in range(len(specific_h)):
                if specific_h[x] == "0":
                    specific_h[x] = X.iloc[i, x]
                elif specific_h[x] != X.iloc[i, x]:
                    specific_h[x] = "?"

            # Remove inconsistent hypotheses from G
            general_h = [g for g in general_h if all(g[x] == "?" or g[x] == specific_h[x] for x in range(len(g)))]

        elif val == "No":
            # Specialize G to exclude the negative example
            new_general_h = []
            for g in general_h:
                for x in range(len(g)):
                    if g[x] == "?" and specific_h[x] != X.iloc[i, x]:
                        new_h = g[:]
                        new_h[x] = specific_h[x]
                        new_general_h.append(new_h)
            general_h = new_general_h

        print(f"Step {i + 1}: Specific Hypothesis = {specific_h},\n
        \ General Hypothesis = {general_h}")

    return specific_h, general_h

# Find the version space of hypotheses
specific_hypothesis, general_hypotheses = candidate_elimination(X, y)
print("\nFinal Specific Hypothesis:", specific_hypothesis)
print("Final General Hypothesis:", general_hypotheses)