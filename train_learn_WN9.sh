CUDA_VISIBLE_DEVICES=1   python src/learn.py    --model MMLorentzKG \
                                                --dataset WN9 \
                                                --rank 100 \
                                                --valid 5 \
                                                --optimizer Adagrad \
                                                --reg 0.05 \
                                                --learning_rate 0.1 \
                                                --max_epochs 200 \
                                                --batch_size 6000 \
                                                --early_stopping 15 \
                                                --fusion_dscp True \
                                                --fusion_img True \
                                                --modality_split True \
                                                --rand_ratio 1.0 