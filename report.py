import pandas as pd
from sklearn.metrics import f1_score, classification_report

def save_classification_report(output_file):
    # Read the file and split by lines
    with open("/bert-train-log.txt", 'r') as file:
        lines = file.readlines()

    # Parse the context, true label, and predicted label from each line
    contexts, true_labels, predicted_labels = [], [], []

    for line in lines:
        parts = line.strip().split(" ||| ")
        if len(parts) == 3:
            contexts.append(parts[0])
            true_labels.append(int(parts[1]))
            predicted_labels.append(int(parts[2]))

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame({
        'Context': contexts,
        'True_Label': true_labels,
        'Predicted_Label': predicted_labels
    })

    # Basic Analysis
    result = []
    result.append("Total number of records: {}".format(len(df)))
    result.append("Number of correctly predicted labels: {}".format(sum(df['True_Label'] == df['Predicted_Label'])))

    # For a more detailed error analysis
    errors = df[df['True_Label'] != df['Predicted_Label']]
    result.append("Number of incorrect predictions: {}".format(len(errors)))

    # Calculate the F1 score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    result.append("F1 Score: {}".format(f1))

    # Classification Report for more detailed metrics
    result.append("Classification Report:\n{}".format(classification_report(true_labels, predicted_labels)))

    # Write the result to the output file
    with open(output_file, 'w') as output:
        output.write("\n".join(result))

# Call the function to save the report to a file
save_classification_report("classification_report.txt")
