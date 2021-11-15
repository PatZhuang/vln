name=VLNBERT-test-Prevalent

flag="--vlnbert prevalent

      --submit 1
      --test_only 0

      --train validlistener
      --load snap/VLNBERT-train-Prevalent-mpadd-matchmax-COSWULR-dyt-selfvis-and-cross/state_dict/best_val_unseen

      --features places365
      --maxAction 15
      --batchSize 8
      --feedback sample
      --lr 1e-5
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
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=4 python r2r_src/train.py $flag --name $name
