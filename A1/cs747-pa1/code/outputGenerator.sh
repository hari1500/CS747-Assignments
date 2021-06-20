#!/bin/sh

instances=("../instances/i-1.txt" "../instances/i-2.txt" "../instances/i-3.txt")
algorithms1=("epsilon-greedy" "ucb" "kl-ucb" "thompson-sampling")
algorithms2=("thompson-sampling" "thompson-sampling-with-hint")
epsilons=(0.001 0.005 0.01 0.05 0.1)
horizons=(100 400 1600 6400 25600 102400)

# for instance in ${instances[@]};
# do
#     for algorithm in ${algorithms1[@]};
#     do
#         for seed in `seq 0 10`;
#         do
#             # echo "${instance} ${algorithm} ${seed}"
#             # python bandit.py --instance $instance --algorithm $algorithm --randomSeed $seed --epsilon 0.02 --horizon 102400 -br a > outputDataT1.txt
#         done
#     done
# done

# for instance in ${instances[@]};
# do
#     for seed in `seq 0 49`;
#     do
#         echo "${instance} ${seed}"
#         python bandit.py --instance $instance --algorithm thompson-sampling-with-hint --randomSeed $seed --epsilon 0.02 --horizon 102400 -br a >> outputTSWH.txt
#     done
# done

# for instance in ${instances[@]};
# do
#     for seed in `seq 0 49`;
#     do
#         for epsilon in ${epsilons[@]};
#         do
#             echo "${instance} ${seed} ${epsilon}"
#             python bandit.py --instance $instance --algorithm epsilon-greedy --randomSeed $seed --epsilon $epsilon --horizon 102400 >> outputDataT3.txt
#         done
#     done
# done

for instance in ${instances[@]};
do
    for seed in `seq 0 49`;
    do
        for horizon in ${horizons[@]};
        do
            echo "${instance} ${seed} ${horizon}"
            python bandit.py --instance $instance --algorithm thompson-sampling-with-hint --randomSeed $seed --epsilon 0.02 --horizon $horizon >> outputTSWH.txt
        done
    done
done
