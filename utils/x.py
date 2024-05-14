import csv
from eval_ucf101 import UCFclassification

val_jsons = [
    "/home/matthew/Efficient-3DCNNs/results_shufflenet_bs16_lr0.001/val.json",
    "/home/matthew/Efficient-3DCNNs/result_mobilenet_bs32_lr0.1/val.json",
    "/home/matthew/Efficient-3DCNNs/result_mobilenet_bs8_lr0.01/val.json",
    "/home/matthew/Efficient-3DCNNs/results_shufflenet_bs64_lr0.1/val.json",
    "/home/matthew/Efficient-3DCNNs/result_mobilenet_bs32_lr0.01/val.json",
    "/home/matthew/Efficient-3DCNNs/results_shufflenet_bs32_lr0.1/val.json",
    "/home/matthew/Efficient-3DCNNs/result_mobilenet_bs64_lr0.01/val.json",
    "/home/matthew/Efficient-3DCNNs/result_mobilenet_bs16_lr0.1/val.json",
    "/home/matthew/Efficient-3DCNNs/result_mobilenet_bs8_lr0.1/val.json",
    "/home/matthew/Efficient-3DCNNs/results_shufflenet_bs16_lr0.1/val.json",
    "/home/matthew/Efficient-3DCNNs/result_mobilenet_bs16_lr0.001/val.json",
    "/home/matthew/Efficient-3DCNNs/results_shufflenet_bs64_lr0.001/val.json",
    "/home/matthew/Efficient-3DCNNs/result_mobilenet_bs8_lr0.001/val.json",
    "/home/matthew/Efficient-3DCNNs/results_shufflenet_bs8_lr0.01/val.json",
    "/home/matthew/Efficient-3DCNNs/results_shufflenet_bs8_lr0.001/val.json",
    "/home/matthew/Efficient-3DCNNs/result_mobilenet_bs64_lr0.1/val.json",
    "/home/matthew/Efficient-3DCNNs/results_shufflenet_bs64_lr0.01/val.json",
    "/home/matthew/Efficient-3DCNNs/result_mobilenet_bs32_lr0.001/val.json",
    "/home/matthew/Efficient-3DCNNs/results_shufflenet_bs32_lr0.01/val.json",
    "/home/matthew/Efficient-3DCNNs/results_shufflenet_bs32_lr0.001/val.json",
    "/home/matthew/Efficient-3DCNNs/result_mobilenet_bs64_lr0.001/val.json",
    "/home/matthew/Efficient-3DCNNs/results_shufflenet_bs16_lr0.01/val.json",
    "/home/matthew/Efficient-3DCNNs/result_mobilenet_bs16_lr0.01/val.json",
    "/home/matthew/Efficient-3DCNNs/results_shufflenet_bs8_lr0.1/val.json"
]

def extract_lr_bs(filename):
    parts = filename.split('/')
    bs_lr_part = parts[-2]
    bs_lr = bs_lr_part.split('_')[-2:]
    return float(bs_lr[0][2:]), float(bs_lr[1][2:])

def hit_at_K(file_location):
    ucf_classification = UCFclassification('../annotation_FSL105_30/ucf101_01.json',
                                            file_location,
                                            subset='validation', top_k=1)
    ucf_classification.evaluate()
    hit_at_1 = ucf_classification.hit_at_k

    ucf_classification = UCFclassification('../annotation_FSL105_30/ucf101_01.json',
                                            file_location,
                                            subset='validation', top_k=5)
    ucf_classification.evaluate()
    hit_at_5 = ucf_classification.hit_at_k

    # Extract model name, bs, lr from file_location
    parts = file_location.split('/')
    model_name = parts[-2]
    bs, lr = extract_lr_bs(file_location)

    return model_name, bs, lr, hit_at_1, hit_at_5

# Collect data
data_rows = []
for result in val_jsons:
    data_rows.append(hit_at_K(result))

# Write data to CSV
csv_filename = 'model_results.csv'
csv_columns = ['Model Name', 'Batch Size', 'Learning Rate', 'Hit @ 1', 'Hit @ 5']

with open(csv_filename, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_columns)
    writer.writerows(data_rows)

print(f"Data has been written to {csv_filename}.")
