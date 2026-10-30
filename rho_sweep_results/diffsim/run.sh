#!/bin/bash
# DiffSim 학습 테스트: (d) rho 0.05 delta 0.2 -> (d) delta 0 -> (c) cimpc hard forward. 순차.
cd /home/andyjeong/workspace/DynaFlow
OUT=rho_sweep_results/diffsim
COMMON="--n-sub 2 --steps 2000 --batch 256 --horizon 16 --seed 0 --log-every 100 --save-every 1000 --forward cimpc"
run() { name=$1; shift; echo "=== $name: $* ==="; python -u train_cimpc.py $COMMON --name $name "$@" > $OUT/$name.log 2>&1; echo "exit=$? $(date)"; }
run diffsim_rho0.05_delta0.2 --backend diffsim --rho 0.05 --delta 0.2
run diffsim_rho0.05_delta0   --backend diffsim --rho 0.05 --delta 0.0
run cimpc_rho1e-4_fwdhard    --backend cimpc   --rho 1e-4
echo ALL_DONE
