for TOY in 100 200 400
do
  echo toy${TOY}_long

  # Flow
  CUDA_VISIBLE_DEVICES=$1 python train.py --hidden_units [128] --context_size 64 --num_flows 8 --permutation shuffle --iter 10000 --target toy${TOY} --name toy${TOY}_long

  for PS in 1 2 4 6 8
  do
    CUDA_VISIBLE_DEVICES=$1 python run_mcmc.py --exp toy${TOY}_long --num_samples 10000 --steps_per_sample 10 --proposal_scale 0.$PS --mcmc s0$PS
    CUDA_VISIBLE_DEVICES=$1 python eval_ess_std.py --exp toy${TOY}_long --mcmc s0$PS
    CUDA_VISIBLE_DEVICES=$1 python eval_energy.py --exp toy${TOY}_long --mcmc s0$PS
  done

done
