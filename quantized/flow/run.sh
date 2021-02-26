CUDA_VISIBLE_DEVICES=$1 python train.py --num_flows 4 --hidden_units [256] --context_size 64 --iter 10000 --num_bits 4 --target iris_logreg --name iris_4bit_long
CUDA_VISIBLE_DEVICES=$1 python train.py --num_flows 6 --hidden_units [256] --context_size 64 --iter 10000 --num_bits 4 --target wine_logreg --name wine_4bit_long
CUDA_VISIBLE_DEVICES=$1 python train.py --num_flows 8 --hidden_units [256] --context_size 64 --iter 10000 --num_bits 4 --target bcancer_logreg --name bcancer_4bit_long

for P in 1 2 4 6 8
do
  for DSET in iris wine bcancer
  do
    echo ${DSET}_4bit_long s0${P}_long
    CUDA_VISIBLE_DEVICES=$1 python run_mcmc.py --exp ${DSET}_4bit_long --num_samples 10000 --steps_per_sample 10 --proposal_scale 0.${P} --mcmc s0${P}_long
    CUDA_VISIBLE_DEVICES=$1 python eval_ess_std.py --exp ${DSET}_4bit_long --mcmc s0${P}_long
    CUDA_VISIBLE_DEVICES=$1 python eval_energy.py --exp ${DSET}_4bit_long --mcmc s0${P}_long
    CUDA_VISIBLE_DEVICES=$1 python eval_acc_mcmc.py --exp ${DSET}_4bit_long --mcmc s0${P}_long
  done
done
