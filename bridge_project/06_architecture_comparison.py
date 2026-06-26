import pandas as pd
import matplotlib.pyplot as plt

results = {
    "Model": ["ANN", "CNN", "BiLSTM"],
    "Accuracy": [82.5, 85.0, 82.9],
    "Training Time": ["5 Epochs", "5 Epochs", "5 Epochs"],
    "Main Strength": [
        "Simple baseline",
        "Captures local phrases",
        "Remembers long context"
    ],
    "Main Limitation": [
        "Loses sequence information",
        "Cannot remember long-distance context",
        "Slower and needs tuning"
    ]
}

comparison = pd.DataFrame(results)

print(comparison)

plt.figure(figsize=(6,5))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.title("Model Accuracy Comparison")

plt.xlabel("Model")

plt.ylabel("Accuracy (%)")

for i, value in enumerate(comparison["Accuracy"]):
    plt.text(i, value + 0.2, str(value))

plt.show()


# Best Model
best = comparison.loc[
    comparison["Accuracy"].idxmax()
]

print()
print("Best Model")
print(best)


# Engineer Observation
print("CNN is best")
print()

print("Project Observations")
print("- ANN provides a simple baseline.")
print("- CNN captured local sentiment patterns and achieved the highest accuracy.")
print("- BiLSTM learned long-term context but required additional tuning.")
print("- A more complex architecture does not always guarantee better performance.")