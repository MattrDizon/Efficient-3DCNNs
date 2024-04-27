from eval_ucf101 import UCFclassification
from eval_kinetics import KINETICSclassification



# ucf_classification = UCFclassification('../annotation_UCF101/ucf101_01.json',
#                                        '../results/val.json',
#                                        subset='validation', top_k=1)
# ucf_classification.evaluate()
# print(ucf_classification.hit_at_k)


# Template
# ucf_classification = UCFclassification('../annotation_FSL105_30/ucf101_01.json',
#                                        '../results_30_mn/val.json',
#                                        subset='validation', top_k=1)
# ucf_classification.evaluate()
# print(ucf_classification.hit_at_k)

# MobileNet
ucf_classification = UCFclassification('../annotation_FSL105_30/ucf101_01.json',
                                       '/home/matthew/Efficient-3DCNNs/result_mobilenet_test/val.json',
                                       subset='validation', top_k=1)
ucf_classification.evaluate()
print(ucf_classification.hit_at_k)
# 60 Frames
# top_k=1; 0.2426470588235294
# top_k=5; 0.75

# 30 Frames
# top_k=1;
# top_k=5;

# ShuffleNet
ucf_classification = UCFclassification('../annotation_FSL105_30/ucf101_01.json',
                                       '/home/matthew/Efficient-3DCNNs/results_shufflenet_test/val.json',
                                       subset='validation', top_k=1)
ucf_classification.evaluate()
print(ucf_classification.hit_at_k)
# 60 Frames
# top_k=1; 0.2647058823529412
# top_k=5; 0.6691176470588235

# 30 Frames
# top_k=1;
# top_k=5;

# Ensemble
ucf_classification = UCFclassification('../annotation_FSL105_30/ucf101_01.json',
                                       '/home/matthew/Efficient-3DCNNs/result_ensemble/val.json',
                                       subset='validation', top_k=1)
ucf_classification.evaluate()
print(ucf_classification.hit_at_k)
# 60 Frames
# top_k=1; 0.029411764705882353
# top_k=5; 0.08088235294117647

# 30 Frames
# top_k=1; 
# top_k=5; 