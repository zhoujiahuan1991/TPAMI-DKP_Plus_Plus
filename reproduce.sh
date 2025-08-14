CUDA_VISIBLE_DEVICES=0,1 python continual_train.py \
--logs-dir final_reproduce/setting-1 \
--trans True \
--merge_tri_weight 0.75 \
--distill_weight 0.75 \
--AF_weight 1.5 \
--s_r_tri_weight 1.5 \
--uncertainty True \
--batch-size 64 \
--triplet_loss True 



CUDA_VISIBLE_DEVICES=1,3 python continual_train.py \
--logs-dir final_reproduce/setting-2 \
--trans True \
--merge_tri_weight 0.75 \
--distill_weight 0.75 \
--AF_weight 1.5 \
--s_r_tri_weight 1.5 \
--uncertainty True \
--batch-size 64 \
--triplet_loss True \
--setting 2



