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


ucf_classification = UCFclassification('../annotation_FSL105_30/ucf101_01.json',
                                       '/home/matthew/Efficient-3DCNNs/results_30_shufflenet_nocrop_fix_fin/val.json',
                                       subset='validation', top_k=1)
ucf_classification.evaluate()
print(ucf_classification.hit_at_k)

# ucf_classification = UCFclassification('../annotation_FSL105_30/ucf101_01.json',
#                                        '../results_30_sn/val.json',
#                                        subset='validation', top_k=1)
# ucf_classification.evaluate()
# print(ucf_classification.hit_at_k)

# ucf_classification = UCFclassification('../annotation_FSL105_30/ucf101_01.json',
#                                        '../results_30_center_mobilenet/val.json',
#                                        subset='validation', top_k=1)
# ucf_classification.evaluate()
# print(ucf_classification.hit_at_k)

# ucf_classification = UCFclassification('../annotation_FSL105_30/ucf101_01.json',
#                                        '../results_30_center_shufflenet/val.json',
#                                        subset='validation', top_k=1)
# ucf_classification.evaluate()
print(ucf_classification.hit_at_k)



# kinetics_classification = KINETICSclassification('../annotation_Kinetics/kinetics.json',
#                                        '../results/val.json',
#                                        subset='validation',
#                                        top_k=1,
#                                        check_status=False)
# kinetics_classification.evaluate()
# print(kinetics_classification.hit_at_k)
