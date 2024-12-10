import matplotlib.pyplot as plt

models = [
    "Logistic Regression",
    "CNN",
    "Fine-tuned BART-base",
    "Fine-tuned BERT-base",
]
# Macro precision, recall, and F1 scores for each model
precision_scores = [0.63, 0.52, 0.59, 0.58]
recall_scores = [0.16, 0.15, 0.44, 0.45]
macro_f1_scores = [0.26, 0.20, 0.49, 0.49]

fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.2
space = 0.1
index = range(len(models))

bars1 = plt.bar(
    index, precision_scores, bar_width, label="Macro Precision", color="blue"
)
bars2 = plt.bar(
    [i + bar_width for i in index],
    recall_scores,
    bar_width,
    label="Macro Recall",
    color="green",
)
bars3 = plt.bar(
    [i + 2 * bar_width for i in index],
    macro_f1_scores,
    bar_width,
    label="Macro F1",
    color="red",
)

for bar in bars1:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        round(yval, 2),
        ha="center",
        va="bottom",
    )

for bar in bars2:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        round(yval, 2),
        ha="center",
        va="bottom",
    )

for bar in bars3:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        round(yval, 2),
        ha="center",
        va="bottom",
    )

plt.title(
    "Performance of Different Models in terms of Macro Precision, Recall, and F1 Score"
)
plt.xlabel("Models")
plt.ylabel("Scores")

wrapped_labels = ["\n".join(label.split()) for label in models]
plt.xticks([i + bar_width for i in index], wrapped_labels, rotation=0, ha="center")

plt.legend()
plt.savefig("model_performance.png")
plt.show()
