nohup python -u train_ddp.py \
    --batch_size 2 --max_epoch 500 \
    --outf ./exp/CRSDUN/ \
    --method CRSDUN\
    --input_mask SSR \
    --input_setting Y \
    --learning_rate 0.0004 \
    --gpu_id "0, 1" \
    --name CRSDUN_5stg\
    >exp/CRSDUN/log/CRSDUN_5stg.log 2>&1 &