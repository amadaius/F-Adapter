python /home/leixu/wmm/F-Adapter-main/finetune3d_PEFT.py \
  --model DPOT --dataset ns3d \
  --peft_method adapter --adapter_type fadapter \
  --train_path ns3d_pdb_M1_rand --test_path ns3d_pdb_M1_rand \
  --resume_path /home/leixu/wmm/DPOT-实验/DPOT-main/checkpoints/model_H.pth \
  --peft_dim 32 --power 2.0 --num_bands 4 --min_dim_factor 0.5 --max_dim_factor 2.0 \
  --gpu "1" --ntrain 90 --batch_size 4 \
  --use_spatial_hf_block 1 --hf_mode diff --hf_stride 2 --hf_kernel_size 3 \
  --hf_gate_init 0.05 --hf_upsample trilinear --hf_down_type avg --hf_use_norm 1 --hf_soft_shrink_tau 0.0 \
  --enable_spectrum --enable_3d_vis