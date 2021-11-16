name=VLNBERT-train-Prevalent-match-pgloss-new-nodyt

flag="--vlnbert prevalent

      --aug data/prevalent/prevalent_aug.json
      --test_only 0
      --load snap/VLNBERT-train-Prevalent-match-pgloss-new-nodyt/state_dict/latest_dict

      --train auglistener

      --features places365
      --maxAction 15
      --batchSize 16
      --feedback sample
      --lr 1e-5
      --iters 72000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5
      --object
      --match_type max
      --st_gumbel
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=3 python r2r_src/train.py $flag --name $name
