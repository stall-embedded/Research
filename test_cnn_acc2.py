import torch
from basic_snn import BasicModel
import pandas as pd
from spikingjelly.activation_based import functional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ALPHA = 3.7132080089425044
TAU = 2.180830180029865
def calculate_confusion_matrix(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    return confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())

def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    accuracy = accuracy_score(labels, predicted)
    precision = precision_score(labels, predicted, average='weighted', zero_division=1)
    recall = recall_score(labels, predicted, average='weighted', zero_division=1)
    f1 = f1_score(labels, predicted, average='weighted')
    # precision = precision_score(labels, predicted, average='micro')
    # recall = recall_score(labels, predicted, average='micro')
    # f1 = f1_score(labels, predicted, average='micro')
    return accuracy, precision, recall, f1
    
def my_collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = torch.stack([input.requires_grad_(False) for input in inputs])
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels

def main():
    torch.cuda.empty_cache()
    testsets = []
    epsilons = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    overshoots = [0.01, 0.02, 0.05, 0.1]
    c_values = [1e-4, 1e-3, 1e-2, 1e-1]
    for c_value in c_values:
        testsets.append(torch.load(f'./adversarial_example/snn_label/pgd_adversarial_half_{c_value}.pth'))
    
    model = BasicModel(seq_num=50, num_channels=3, optimizer="Adam", lr=0.001, alpha=ALPHA, tau=TAU).cuda()
    functional.set_step_mode(model, step_mode='m')
    model.load_state_dict(torch.load('best_basic_model_snn_sj0929.pth'))
    model.eval()

    results_df_list = []
    results_df = pd.DataFrame(columns=['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1'])
    with torch.no_grad():
        for c_value, testset in zip(c_values, testsets):
            print(c_value)
            testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=6, collate_fn=my_collate_fn)
            all_accuracy = []
            all_precision = []
            all_recall = []
            all_f1 = []

            all_labels = []
            all_predictions = []

            for inputs, labels in testloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                functional.reset_net(model)

                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                accuracy, precision, recall, f1 = calculate_metrics(outputs, labels)

                all_accuracy.append(accuracy)
                all_precision.append(precision)
                all_recall.append(recall)
                all_f1.append(f1)

            avg_accuracy = sum(all_accuracy) / len(all_accuracy)
            avg_precision = sum(all_precision) / len(all_precision)
            avg_recall = sum(all_recall) / len(all_recall)
            avg_f1 = sum(all_f1) / len(all_f1)

            df = pd.DataFrame({'Dataset': f'cvalue Epsilon {c_value}', 'Accuracy': [avg_accuracy], 'Precision': [avg_precision], 'Recall': [avg_recall], 'F1': [avg_f1]})
            results_df_list.append(df)

            conf_matrix = confusion_matrix(all_labels, all_predictions)
            conf_matrix_df = pd.DataFrame(conf_matrix)
            conf_matrix_df.to_csv(f'pgd_half_confusion_matrix_cvalue_{c_value}.csv', index=False)

        results_df = pd.concat(results_df_list, ignore_index=True)
        results_df.to_csv('model_pgd_half_performance.csv', index=False)

if __name__ == '__main__':
    main()
