# from eval_ucf101 import UCFclassification

# def read_file(file_path):
#     with open(file_path, 'r') as file:
#         return [line.strip() for line in file.readlines()]

# val_jsons = read_file("src.txt")
# val_jsons_sorted = sorted(val_jsons)
# print(val_jsons_sorted)


# def hit_at_K(file_location):
#     print(file_location)
#     ucf_classification = UCFclassification('../annotation_FSL105_30/ucf101_01.json',
#                                        file_location,
#                                        subset='validation', top_k=1)
#     ucf_classification.evaluate()
#     print("Hit @ 1", ucf_classification.hit_at_k, "{:.2%}".format(ucf_classification.hit_at_k))

#     ucf_classification = UCFclassification('../annotation_FSL105_30/ucf101_01.json',
#                                        file_location,
#                                        subset='validation', top_k=5)
#     ucf_classification.evaluate()
#     print("Hit @ 5", ucf_classification.hit_at_k, "{:.2%}".format(ucf_classification.hit_at_k))
#     print()

# for result in val_jsons_sorted:
#     hit_at_K(result)
import csv
from eval_ucf101 import UCFclassification
import re

def read_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def parse_file_path(file_path):
    match = re.search(r'results?_([^_]*)_bs(\d+)_lr([\d.]+)/val.json', file_path)
    if match:
        match1 = re.search(r'Efficient-3DCNNs_epoch(\d+)/', file_path)
        if match1:
            epoch = match1.group(1)
        else:
            epoch = None
        model_name = match.group(1).capitalize()
        batch_size = match.group(2)
        learning_rate = match.group(3)
        return model_name, epoch, batch_size, learning_rate
    return None, None, None, None

def hit_at_K(file_location):
    model_name, epoch, batch_size, learning_rate = parse_file_path(file_location)
    
    ucf_classification = UCFclassification('./annotation_FSL105_30/ucf101_01.json',
                                           file_location,
                                           subset='validation', top_k=1)
    ucf_classification.evaluate()
    hit1 = ucf_classification.hit_at_k
    
    ucf_classification = UCFclassification('./annotation_FSL105_30/ucf101_01.json',
                                           file_location,
                                           subset='validation', top_k=5)
    ucf_classification.evaluate()
    hit5 = ucf_classification.hit_at_k
    
    return model_name, epoch, batch_size, learning_rate, hit1, hit5

val_jsons = read_file("./utils/src.txt")
val_jsons_sorted = sorted(val_jsons)
# print(val_jsons_sorted)

results = []
for result in val_jsons_sorted:
    model_name, epoch, batch_size, learning_rate, hit1, hit5 = hit_at_K(result)
    results.append([model_name, epoch, batch_size, learning_rate, hit1, hit5])

with open('./utils/results_full.csv', 'w', newline='') as csvfile:
    fieldnames = ['Model', 'Epoch', 'Batch Size', 'Learning Rate', 'Hit@1', 'Hit@5']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for row in results:
        writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5]])
