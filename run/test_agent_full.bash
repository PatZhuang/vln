name=VLNBERT-test-cvpr-full

flag="--vlnbert prevalent

      --submit 1
      --test_only 0

      --train validlistener
      --load snap/cvpr-full/state_dict/best_val_unseen

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

      --lr_adjust_type cosine
      --pgWeight 1.0

      --visualize
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=6 python r2r_src/train.py $flag --name $name
