name=cvpr-full-finetune

flag="--vlnbert prevalent

      --aug data/prevalent/prevalent_aug.json
      --test_only 0
      --load snap/cvpr-full/state_dict/best_val_unseen

      --train auglistener

      --features places365
      --maxAction 15
      --batchSize 8
      --feedback sample
      --lr 1e-6
      --iters 20000
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

      --finetune
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=6 python r2r_src/train.py $flag --name $name
