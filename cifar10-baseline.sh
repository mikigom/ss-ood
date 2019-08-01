for i in `seq 1 10`;
do
  for j in `seq 0 9`;
  do
    CUDA_VISIBLE_DEVICES=0 python3 train_others.py --dataset cifar10 --in_class $j --result_dir "results/cifar10/${j}/trial${i}"
  done
done
