CUDA_VISIBLE_DEVICES=$1 python run_dmcmc.py --num_samples 10000 --steps_per_sample 10 --proposal_prob 0.005 --burnin_steps 10000 --name 10k
CUDA_VISIBLE_DEVICES=$1 python run_dmcmc.py --num_samples 10000 --steps_per_sample 10 --proposal_prob 0.005 --burnin_steps 20000 --name 20k
CUDA_VISIBLE_DEVICES=$1 python run_dmcmc.py --num_samples 10000 --steps_per_sample 10 --proposal_prob 0.005 --burnin_steps 50000 --name 50k
CUDA_VISIBLE_DEVICES=$1 python run_dmcmc.py --num_samples 10000 --steps_per_sample 10 --proposal_prob 0.005 --burnin_steps 100000 --name 100k


for EXP in 10k 20k 50k 100k
do
  echo $EXP
  CUDA_VISIBLE_DEVICES=$1 python eval_ess_std.py --exp $EXP --verbose True
  CUDA_VISIBLE_DEVICES=$1 python eval_energy.py --exp $EXP
done
