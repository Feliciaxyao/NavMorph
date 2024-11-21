export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag1="--exp_name release_r2r
      --run-type train
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      IL.iters 29000
      IL.lr 1e-5
      IL.log_every 500
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 4000
      IL.load_from_ckpt True
      IL.is_requeue True
      IL.waypoint_aug  True
	  IL.ckpt_to_load data/checkpoints/ckpt.iter25000.pth
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      MODEL.pretrained_path pretrained/model_step_100000.pt
      "

flag2=" --exp_name release_r2r
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR data/checkpoints/ckpt.pth
	  MODEL.pretrained_path pretrained/model_step_100000.pt
      IL.back_algo control
      "

flag3="--exp_name release_r2r
      --run-type inference
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 1
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH data/checkpoints/ckpt.pth
      INFERENCE.PREDICTIONS_FILE preds.json
	  MODEL.pretrained_path pretrained/model_step_100000.pt
      IL.back_algo control
      "

mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      CUDA_VISIBLE_DEVICES='7' python run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      #CUDA_VISIBLE_DEVICES='5' python -m pdb run.py $flag2
      CUDA_VISIBLE_DEVICES='0' python run.py $flag2
      ;;
      infer)
      echo "###### infer mode ######"
      CUDA_VISIBLE_DEVICES='7' python run.py $flag3
      ;;
esac