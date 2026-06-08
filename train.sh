nohup python -u train.py \
    --batch_size 4 --max_epoch 500 \
    --outf ./exp/CRSDUN/ \
    --method CRSDUN\
    --input_mask SSR \
    --input_setting Y \
    --learning_rate 0.0004 \
    --gpu_id 0 \
    --name CRSDUN_3stg\
    >exp/CRSDUN/log/CRSDUN_3stg.log 2>&1 &