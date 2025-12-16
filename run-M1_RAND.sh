nohup python finetune3d_PEFT.py \
  --peft_method adapter --adapter_type fadapter \
  --train_path ns3d_pdb_M1_rand --test_path ns3d_pdb_M1_rand \
  --resume_path /home/leeshu/wmm/DPOT-main/checkpoints/model_H.pth \
  --peft_dim 32 --power 2.0 --num_bands 4 --min_dim_factor 0.5 --max_dim_factor 2.0 \
  --gpu "2" --ntrain 90 --batch_size 4 --use_spatial_hf_block 0 --lr 1.0e-04 --comment "M1_rand_fuxian_1e-4"
  # lr 5e-1的复现 模型加载地址错误了