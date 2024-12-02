import matplotlib.pyplot as plt

# Hardcoded data for models and their macro metrics
models = [
    "Logistic Regression",
    "CNN",
    "BART-LARGE-MNLI",
    "BART-LARGE",
    "Fine-tuned BART-LARGE",
    "BERT",
    "Fine-tuned BERT",
]
precision_scores = [0.63, 0.52, 0.14, 0, 0.49, 0, 0]
recall_scores = [0.16, 0.15, 0.65, 0, 0.25, 0, 0]
macro_f1_scores = [0.26, 0.20, 0.19, 0, 0.30, 0, 0]


fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.2
space = 0.1
index = range(len(models))

plt.bar(index, precision_scores, bar_width, label="Precision", color="blue")

plt.bar(
    [i + bar_width for i in index],
    recall_scores,
    bar_width,
    label="Recall",
    color="green",
)

plt.bar(
    [i + 2 * bar_width for i in index],
    macro_f1_scores,
    bar_width,
    label="Macro F1",
    color="red",
)

plt.title(
    "Performance of Different Models in terms of Precision, Recall, and Macro F1 Score"
)
plt.xlabel("Models")
plt.ylabel("Scores")

wrapped_labels = ["\n".join(label.split()) for label in models]
plt.xticks([i + bar_width for i in index], wrapped_labels)

plt.legend()

plt.show()
