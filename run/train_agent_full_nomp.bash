name=cvpr-full-nomp

flag="--vlnbert prevalent

      --aug data/prevalent/prevalent_aug.json
      --test_only 0

      --train auglistener

      --features places365
      --maxAction 15
      --batchSize 8
      --feedback sample
      --lr 1e-5
      --iters 100000
      --optim adamW
      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5

      --object
      --match_type max

      --lr_adjust_type cosine
      --pgWeight 1.0
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=3 python r2r_src/train.py $flag --name $name
