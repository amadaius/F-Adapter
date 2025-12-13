nohup python finetune3d_PEFT.py \
  --peft_method adapter --adapter_type fadapter \
  --train_path ns3d_pdb_M1_rand --test_path ns3d_pdb_M1_rand \
  --resume_path /home/leixu/wmm/DPOT-实验/DPOT-main/checkpoints/model_H.pth \
  --peft_dim 32 --power 2.0 --num_bands 4 --min_dim_factor 0.5 --max_dim_factor 2.0 \
  --gpu "0" \
  --batch_size 4