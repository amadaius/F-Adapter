nohup python finetune3d_PEFT.py \
  --peft_method adapter --adapter_type fadapter \
  --train_path ns3d_pdb_M1e-1_rand --test_path ns3d_pdb_M1e-1_rand \
  --resume_path /home/leeshu/wmm/DPOT-main/checkpoints/model_H.pth \
  --peft_dim 32 --power 2.0 --num_bands 4 --min_dim_factor 0.5 --max_dim_factor 2.0 \
  --gpu "2" --ntrain 90 --batch_size 4 --comment "M1e-1_fuxian_gardAccum"\
  --use_spatial_hf_block 0 --grad_accum 6
  # --lr 2.083e-05
  # --hf_mode diff --hf_stride 2 --hf_kernel_size 3 \
  # --hf_gate_init 0.2 --hf_upsample trilinear --hf_down_type avg --hf_use_norm 1 --hf_soft_shrink_tau 0.0 \
  # --enable_spectrum --enable_3d_vis
  #ntrain ns3d_pdb_M1_rand数据集的训练参数是90
  