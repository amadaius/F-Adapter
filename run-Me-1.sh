nohup python finetune3d_PEFT.py \
  --peft_method adapter --adapter_type fadapter \
  --train_path ns3d_pdb_M1e-1_rand --test_path ns3d_pdb_M1e-1_rand \
  --resume_path /home/leeshu/wmm/DPOT-main/checkpoints/model_H.pth \
  --peft_dim 32 --power 2.0 --num_bands 4 --min_dim_factor 0.5 --max_dim_factor 2.0 \
  --gpu "1" --ntrain 90 --batch_size 4 --epochs 1000 --lr 2.5e-04 --comment "M1e-1_fuxian_epoch1000_lr2.5e-4"\
  --use_spatial_hf_block 0 
  # --lr 2.083e-05
  # --hf_mode diff --hf_stride 2 --hf_kernel_size 3 \
  # --hf_gate_init 0.2 --hf_upsample trilinear --hf_down_type avg --hf_use_norm 1 --hf_soft_shrink_tau 0.0 \
  # --enable_spectrum --enable_3d_vis
  #ntrain ns3d_pdb_M1_rand数据集的训练参数是90
  