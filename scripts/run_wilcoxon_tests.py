from scipy.stats import wilcoxon
import pandas as pd

models = {
    "CSP": [78.5, 51.4, 85.8, 49.0, 38.2, 49.0, 74.7, 80.6, 72.9],
    "FBCSP": [77.4, 53.8, 85.1, 49.7, 72.6, 47.6, 87.2, 82.3, 76.7],
    "CSP+SVM": [80.2, 52.4, 84.0, 47.6, 41.7, 49.7, 75.7, 81.6, 73.6],
    "FBCSP+SVM": [78.1, 55.9, 85.4, 52.1, 69.1, 49.7, 83.0, 83.7, 79.2],
}

comparisons = [
    ("CSP", "FBCSP"),
    ("CSP", "CSP+SVM"),
    ("FBCSP", "FBCSP+SVM"),
    ("CSP", "FBCSP+SVM"),
]

results = []

print("\n========================================")
print("Wilcoxon Signed-Rank Test Results")
print("========================================")

for model1, model2 in comparisons:
    stat, p = wilcoxon(models[model1], models[model2])

    results.append({
        "Comparison": f"{model1} vs {model2}",
        "Statistic": stat,
        "p-value": p,
        "Significant (p<0.05)": p < 0.05,
        "Significant (Bonferroni p<0.0125)": p < 0.0125,
    })

df = pd.DataFrame(results)

print(df.to_string(index=False))

df.to_csv("results/wilcoxon_results.csv", index=False)

print("\nSaved:")
print("results/wilcoxon_results.csv")