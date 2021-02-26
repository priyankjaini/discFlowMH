for TOY in 100 200 400
do
  echo toy${TOY}_long
  CUDA_VISIBLE_DEVICES=$1 python run_gibbs.py --target toy${TOY} --burnin_steps 100000 --num_samples 10000 --steps_per_sample 10 --name toy${TOY}_long
  CUDA_VISIBLE_DEVICES=$1 python eval_ess_std.py --exp toy${TOY}_long
  CUDA_VISIBLE_DEVICES=$1 python eval_energy.py --exp toy${TOY}_long
done
