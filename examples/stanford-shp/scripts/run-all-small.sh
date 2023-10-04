# sbatch rlft.sh changemyview /projects/tir6/general/sachink/personalized-LM/2023/models/flan-t5-large-rerun-changemyview
# sbatch rlft.sh legaladvice /projects/tir6/general/sachink/personalized-LM/2023/models/flan-t5-large-rerun-legaladvice
# sbatch rlft.sh explainlikeimfive /projects/tir6/general/sachink/personalized-LM/2023/models/flan-t5-large-rerun-explainlikeimfive
# sbatch rlft.sh askphilosophy /projects/tir6/general/sachink/personalized-LM/2023/models/flan-t5-large-rerun-askphilosophy

# sbatch rlft.sh contextual /projects/tir6/general/sachink/personalized-LM/2023/models/reward/contextualized_flan-t5-large
# sbatch rlft.sh plain /projects/tir6/general/sachink/personalized-LM/2023/models/reward/plain_flan-t5-large
# sbatch rlft.sh subredditname /projects/tir6/general/sachink/personalized-LM/2023/models/reward/subredditname_flan-t5-large

# sbatch sft.sh /projects/tir6/general/sachink/personalized-LM/2023/steamshp/data sft_
# sbatch sft.sh /projects/tir6/general/sachink/personalized-LM/2023/steamshp/data sft_plain_
# sbatch sft.sh /projects/tir6/general/sachink/personalized-LM/2023/steamshp/data sft_subredditname_

# sbatch dpoft.sh contextualized 2
# sbatch dpoft.sh plain 2 

sbatch rlft.sh contextualized /projects/tir6/general/sachink/personalized-LM/2023/models/reward/contextualized_deberta-v3-large_steamshp_2e-05_last_checkpoint
sbatch rlft.sh plain /projects/tir6/general/sachink/personalized-LM/2023/models/reward/plain_deberta-v3-large_steamshp_2e-05_last_checkpoint

# sbatch dpoft.sh subredditname 4
sbatch rlft.sh subredditname /projects/tir6/general/sachink/personalized-LM/2023/models/reward/subredditname_deberta-v3-large_steamshp_2e-05_last_checkpoint