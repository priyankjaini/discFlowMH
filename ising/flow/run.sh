CUDA_VISIBLE_DEVICES=$1 python train.py --num_flows 8 --hidden_units [512] --context_size 392 --iter 10000 --name 10k
CUDA_VISIBLE_DEVICES=$1 python train.py --num_flows 8 --hidden_units [512] --context_size 392 --iter 20000 --name 20k
CUDA_VISIBLE_DEVICES=$1 python train.py --num_flows 8 --hidden_units [512] --context_size 392 --iter 50000 --name 50k
CUDA_VISIBLE_DEVICES=$1 python train.py --num_flows 8 --hidden_units [512] --context_size 392 --iter 100000 --name 100k


for EXP in 10k 20k 50k 100k
do
  echo $EXP
  CUDA_VISIBLE_DEVICES=$1 python run_mcmc.py --num_samples 10000 --steps_per_sample 10 --proposal_scale 0.1 --exp $EXP --mcmc s01
  CUDA_VISIBLE_DEVICES=$1 python eval_ess_std.py --exp $EXP --mcmc s01
  CUDA_VISIBLE_DEVICES=$1 python eval_energy.py --exp $EXP --mcmc s01
done
