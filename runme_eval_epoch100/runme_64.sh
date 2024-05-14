echo "start bs64"

# "Train MobileNet"
python main.py  --root_path ~/     --video_path ~/Thesis/FSL105_jpg_30     --annotation_path ~/Thesis/FSL105_anno_30/ucf101_01.json       --result_path Efficient-3DCNNs/result_mobilenet_bs64_lr0.1       --dataset ucf101        --n_classes 30  --model mobilenet       --width_mult 0.5        --train_crop center     --learning_rate 0.1    --sample_duration 16    --downsample 2  --batch_size 64         --n_threads 16  --checkpoint 1  --n_val_samples 1 --n_epochs 100 --test --no_train --no_val --crop_position_in_test c --resume_path /home/matthew/Efficient-3DCNNs/result_mobilenet_bs64_lr0.1/ucf101_mobilenet_0.5x_RGB_16_best.pth

# Train ShuffleNet
python main.py  --root_path ~/     --video_path ~/Thesis/FSL105_jpg_30     --annotation_path ~/Thesis/FSL105_anno_30/ucf101_01.json       --result_path Efficient-3DCNNs/results_shufflenet_bs64_lr0.1    --dataset ucf101        --n_classes 30  --model shufflenet       --width_mult 0.5        --groups 3 --train_crop center     --learning_rate 0.1    --sample_duration 16    --downsample 2  --batch_size 64         --n_threads 16  --checkpoint 1  --n_val_samples 1 --n_epochs 100 --test --no_train --no_val --crop_position_in_test c --resume_path /home/matthew/Efficient-3DCNNs/results_shufflenet_bs64_lr0.1/ucf101_shufflenet_0.5x_RGB_16_best.pth

# "Train MobileNet"
python main.py  --root_path ~/     --video_path ~/Thesis/FSL105_jpg_30     --annotation_path ~/Thesis/FSL105_anno_30/ucf101_01.json       --result_path Efficient-3DCNNs/result_mobilenet_bs64_lr0.01       --dataset ucf101        --n_classes 30  --model mobilenet       --width_mult 0.5        --train_crop center     --learning_rate 0.01    --sample_duration 16    --downsample 2  --batch_size 64         --n_threads 16  --checkpoint 1  --n_val_samples 1 --n_epochs 100 --test --no_train --no_val --crop_position_in_test c --resume_path /home/matthew/Efficient-3DCNNs/result_mobilenet_bs64_lr0.01/ucf101_mobilenet_0.5x_RGB_16_best.pth

# Train ShuffleNet
python main.py  --root_path ~/     --video_path ~/Thesis/FSL105_jpg_30     --annotation_path ~/Thesis/FSL105_anno_30/ucf101_01.json       --result_path Efficient-3DCNNs/results_shufflenet_bs64_lr0.01    --dataset ucf101        --n_classes 30  --model shufflenet       --width_mult 0.5        --groups 3 --train_crop center     --learning_rate 0.01    --sample_duration 16    --downsample 2  --batch_size 64         --n_threads 16  --checkpoint 1  --n_val_samples 1 --n_epochs 100 --test --no_train --no_val --crop_position_in_test c --resume_path /home/matthew/Efficient-3DCNNs/results_shufflenet_bs64_lr0.01/ucf101_shufflenet_0.5x_RGB_16_best.pth


# "Train MobileNet"
python main.py  --root_path ~/     --video_path ~/Thesis/FSL105_jpg_30     --annotation_path ~/Thesis/FSL105_anno_30/ucf101_01.json       --result_path Efficient-3DCNNs/result_mobilenet_bs64_lr0.001       --dataset ucf101        --n_classes 30  --model mobilenet       --width_mult 0.5        --train_crop center     --learning_rate 0.001    --sample_duration 16    --downsample 2  --batch_size 64         --n_threads 16  --checkpoint 1  --n_val_samples 1 --n_epochs 100 --test --no_train --no_val --crop_position_in_test c --resume_path /home/matthew/Efficient-3DCNNs/result_mobilenet_bs64_lr0.001/ucf101_mobilenet_0.5x_RGB_16_best.pth

# Train ShuffleNet
python main.py  --root_path ~/     --video_path ~/Thesis/FSL105_jpg_30     --annotation_path ~/Thesis/FSL105_anno_30/ucf101_01.json       --result_path Efficient-3DCNNs/results_shufflenet_bs64_lr0.001    --dataset ucf101        --n_classes 30  --model shufflenet       --width_mult 0.5        --groups 3 --train_crop center     --learning_rate 0.001    --sample_duration 16    --downsample 2  --batch_size 64         --n_threads 16  --checkpoint 1  --n_val_samples 1 --n_epochs 100 --test --no_train --no_val --crop_position_in_test c --resume_path /home/matthew/Efficient-3DCNNs/results_shufflenet_bs64_lr0.001/ucf101_shufflenet_0.5x_RGB_16_best.pth

echo "end bs64"
