for SEED in $(seq 0 9)
do
  TARGET=dgmm5_s$SEED

  python plot_target.py --target $TARGET --num_bits 6
  python train.py --target $TARGET --num_bits 6
  python run_mcmc.py --target $TARGET --num_samples 1000 --steps_per_sample 10
  python plot_mcmc.py --target $TARGET
done
