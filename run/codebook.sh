name=VLNBERT-train-Prevalent-baseline-xsdyt

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
      --dropout 0.5"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=6 python r2r_src/train.py $flag --name $name

######## =========== ###########

name=VLNBERT-train-Prevalent-mpadd-matchmax-COSWULR-xsdyt-pgloss-new

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

      --max_pool_feature img_features/ResNet-152-places365-maxpool.pkl
      --mix_type alpha

      --object
      --match_type max
      --st_gumbel

      --lr_adjust_type cosine
      --pgWeight 1.0
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=4 python r2r_src/train.py $flag --name $name

######## =========== ###########

name=VLNBERT-train-Prevalent-baseline-mp-nodyt

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
      --max_pool_feature img_features/ResNet-152-places365-maxpool.pkl
      --mix_type alpha
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=2 python r2r_src/train.py $flag --name $name

######## =========== ###########

name=VLNBERT-train-Prevalent-obj-stmax-add-coslr

flag="--vlnbert prevalent

      --aug data/prevalent/prevalent_aug.json
      --test_only 0

      --train auglistener

      --features places365
      --maxAction 15
      --batchSize 16
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
      --st_gumbel
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=5 python r2r_src/train.py $flag --name $name

######## =========== ###########

name=VLNBERT-train-Prevalent-baseline-mp-xsdyt

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
      --max_pool_feature img_features/ResNet-152-places365-maxpool.pkl
      --mix_type alpha
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=5 python r2r_src/train.py $flag --name $name

######## =========== ###########

name=VLNBERT-train-Prevalent-match-pgloss-new-nodyt

flag="--vlnbert prevalent

      --aug data/prevalent/prevalent_aug.json
      --test_only 0

      --train auglistener

      --features places365
      --maxAction 15
      --batchSize 16
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
      --st_gumbel
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=5 python r2r_src/train.py $flag --name $name
