echo "start"

python main.py --root_path ~/     --video_path ~/Thesis/FSL105_jpg_30     --annotation_path ~/Thesis/FSL105_anno_30/ucf101_01.json       --result_path Efficient-3DCNNs/lr1_epc1/result_mobilenet_test       --dataset ucf101        --n_classes 30  --model mobilenet       --width_mult 0.5        --train_crop center     --learning_rate 1    --sample_duration 16    --downsample 2  --batch_size 64         --n_threads 16  --checkpoint 1  --n_val_samples 1 --n_epochs 1000 --test --no_train --no_val --crop_position_in_test c --resume_path /home/matthew/Efficient-3DCNNs/lr1_epc1/result_mobilenet/ucf101_mobilenet_0.5x_RGB_16_best.pth


# python main.py --root_path ~/     --video_path ~/Thesis/FSL105_jpg_30     --annotation_path ~/Thesis/FSL105_anno_30/ucf101_01.json       --result_path Efficient-3DCNNs/flow/result_mobilenet       --dataset ucf101        --n_classes 30  --model mobilenet       --width_mult 0.5        --train_crop center     --learning_rate 0.1    --sample_duration 16    --downsample 2  --batch_size 64         --n_threads 16  --checkpoint 1  --n_val_samples 1 --modality flow
            
# python main.py --root_path ~/     --video_path ~/Thesis/FSL105_jpg_30     --annotation_path ~/Thesis/FSL105_anno_30/ucf101_01.json       --result_path Efficient-3DCNNs/flow/results_shufflenet    --dataset ucf101        --n_classes 30  --model shufflenet       --width_mult 0.5        --groups 3 --train_crop center     --learning_rate 0.1    --sample_duration 16    --downsample 2  --batch_size 64         --n_threads 16  --checkpoint 1  --n_val_samples 1 --modality flow

# python main.py --root_path ~/     --video_path ~/Thesis/FSL105_jpg_30     --annotation_path ~/Thesis/FSL105_anno_30/ucf101_01.json       --result_path Efficient-3DCNNs/rgbflow/result_mobilenet       --dataset ucf101        --n_classes 30  --model mobilenet       --width_mult 0.5        --train_crop center     --learning_rate 0.1    --sample_duration 16    --downsample 2  --batch_size 64         --n_threads 16  --checkpoint 1  --n_val_samples 1 --modality rgbflow
            
# python main.py --root_path ~/     --video_path ~/Thesis/FSL105_jpg_30     --annotation_path ~/Thesis/FSL105_anno_30/ucf101_01.json       --result_path Efficient-3DCNNs/rgbflow/results_shufflenet    --dataset ucf101        --n_classes 30  --model shufflenet       --width_mult 0.5        --groups 3 --train_crop center     --learning_rate 0.1    --sample_duration 16    --downsample 2  --batch_size 64         --n_threads 16  --checkpoint 1  --n_val_samples 1 --modality rgbflow

echo "Completed 4 models"
