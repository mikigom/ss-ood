for i in `seq 1 10`;
do
  for j in `seq 0 9`;
  do
    CUDA_VISIBLE_DEVICES=1 python3 train_others.py --temperature 500 --dataset cifar10 --in_class $j --result_dir "results/cifar10_temperature500/${j}/trial${i}"
  done
done
