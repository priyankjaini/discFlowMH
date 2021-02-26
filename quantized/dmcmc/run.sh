for P in 025 05 1 2 4 6
do
  for DSET in iris wine bcancer
  do
    echo ${DSET}_4bit_long_p0${P}
    CUDA_VISIBLE_DEVICES=$1 python run_dmcmc.py --target ${DSET}_logreg --num_bits 4 --burnin_steps 100000 --num_samples 10000 --steps_per_sample 10 --proposal_prob 0.${P} --name ${DSET}_4bit_long_p0${P}
    CUDA_VISIBLE_DEVICES=$1 python eval_ess_std.py --exp ${DSET}_4bit_long_p0${P}
    CUDA_VISIBLE_DEVICES=$1 python eval_energy.py --exp ${DSET}_4bit_long_p0${P}
    CUDA_VISIBLE_DEVICES=$1 python eval_acc.py --exp ${DSET}_4bit_long_p0${P}
  done
done
