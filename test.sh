nohup python -u test.py \
    --batch_size 4 --max_epoch 1000 \
    --outf ./exp/CRSDUN_test/ \
    --method CRSDUN\
    --input_mask SSR \
    --input_setting Y \
    --pretrained_model_path checkpoint/CRSDUN_5stg.pkl \
    --learning_rate 0.0004 \
    --gpu_id 0 \
    --name CRSDUN_test_5stg\
    >exp/CRSDUN_test/log/CRSDUN_test_5stg.log 2>&1 &