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
                                       '/home/matthew/Efficient-3DCNNs/test_result_mobilenet/val.json',
                                       subset='validation', top_k=5)
ucf_classification.evaluate()
print(ucf_classification.hit_at_k)
# top_k=1; 0.2426470588235294
# top_k=5; 0.75

# ShuffleNet
ucf_classification = UCFclassification('../annotation_FSL105_30/ucf101_01.json',
                                       '/home/matthew/Efficient-3DCNNs/test_results_shufflenet/val.json',
                                       subset='validation', top_k=5)
ucf_classification.evaluate()
print(ucf_classification.hit_at_k)
# top_k=1; 0.2647058823529412
# top_k=5; 0.6691176470588235




# kinetics_classification = KINETICSclassification('../annotation_Kinetics/kinetics.json',
#                                        '../results/val.json',
#                                        subset='validation',
#                                        top_k=1,
#                                        check_status=False)
# kinetics_classification.evaluate()
# print(kinetics_classification.hit_at_k)
