from scipy.stats import wilcoxon

csp = [78.5, 51.4, 85.8, 49.0, 38.2, 49.0, 74.7, 80.6, 72.9]
tuned_csp = [None, None, None, None, None, None, None, None, 80.2]
fbcsp = [77.4, 53.8, 85.1, 49.7, 72.6, 47.6, 87.2, 82.3, 76.7]
csp_svm = [80.2, 52.4, 84.0, 47.6, 41.7, 49.7, 75.7, 81.6, 73.6]
fbcsp_svm = [78.1, 55.9, 85.4, 52.1, 69.1, 49.7, 83.0, 83.7, 79.2]

comparisons = {
    "CSP vs FBCSP": (csp, fbcsp),
    "CSP vs CSP+SVM": (csp, csp_svm),
    "FBCSP vs FBCSP+SVM": (fbcsp, fbcsp_svm),
    "CSP vs FBCSP+SVM": (csp, fbcsp_svm),
}

print("\nWilcoxon Signed-Rank Tests Across Subjects")
print("=" * 60)

for name, (a, b) in comparisons.items():
    stat, p = wilcoxon(a, b)
    print(f"{name}: statistic={stat:.4f}, p-value={p:.6f}")
