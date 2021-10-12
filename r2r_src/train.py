import torch

import os
import sys
import time
import json
import random
import numpy as np
from collections import defaultdict

from utils import read_vocab, write_vocab, build_vocab, padding_idx, timeSince, read_img_features, print_progress, read_img
import utils
from env import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args
import pickle as pkl
from rich import print

import warnings
warnings.filterwarnings("ignore")
from tensorboardX import SummaryWriter

from vlnbert.vlnbert_init import get_tokenizer

log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'
RAW_IMAGES = '/root/dev/Matterdata/v1/rgb'

OBJECT_INFO_STORE       = 'img_features/objects/object_info.pickle'
OBJECT_VOCAB            = 'img_features/objects/object_vocab.txt'

if args.features == 'imagenet':
    features = IMAGENET_FEATURES
elif args.features == 'places365':
    features = PLACE365_FEATURES
elif args.features == 'raw':
    features = RAW_IMAGES



feedback_method = args.feedback  # teacher or sample

if args.patchVis:
    sys.path.append('../deit')
    from deit.datasets import build_transform
    from deit.main import get_args_parser
    from deit.cait_models import cait_XS24

print(args); print('')


''' train the listener '''
def train(train_env, tok, n_iters, log_every=2000, val_envs={}, aug_env=None, vit_model=None, vit_args=None):
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction, vit_model, vit_args)

    record_file = open('./logs/' + args.name + '.txt', 'a')
    record_file.write(str(args) + '\n\n')
    record_file.close()

    start_iter = 0
    if args.load is not None:
        if args.aug is None:
            start_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))
        else:
            load_iter = listner.load(os.path.join(args.load))
            print("\nLOAD the model from {}, iteration ".format(args.load, load_iter))

    start = time.time()
    print('\nListener training starts, start iteration: %s' % str(start_iter))

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":"", 'update':False}}

    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=feedback_method)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                args.ml_weight = 0.2
                listner.train(1, feedback=feedback_method)

                # Train with Augmented data
                listner.env = aug_env
                args.ml_weight = 0.2
                listner.train(1, feedback=feedback_method)

                print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total
        RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
        IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
        entropy = sum(listner.logs['entropy']) / total
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/RL_loss", RL_loss, idx)
        writer.add_scalar("loss/IL_loss", IL_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        # print("total_actions", total, ", max_length", length)

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += "{:<11s} | ".format(env_name)
            for metric, val in score_summary.items():
                if metric in ['spl']:
                    writer.add_scalar("spl/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['spl']:
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                        elif (val == best_val[env_name]['spl']) and (score_summary['success_rate'] > best_val[env_name]['sr']):
                            best_val[env_name]['spl'] = val
                            best_val[env_name]['update'] = True
                loss_str += ', %s: %.4f' % (metric, val)
            loss_str += '\n'

        record_file = open('./logs/' + args.name + '.txt', 'a')
        record_file.write(loss_str + '\n')
        record_file.close()

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))
            else:
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "latest_dict"))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

                record_file = open('./logs/' + args.name + '.txt', 'a')
                record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n')
                record_file.close()

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))


def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(0)
    np.random.seed(0)

def train_val(test_only=False, vit_model=None, vit_args=None, img_process=None):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    setup()
    tok = get_tokenizer(args)

    if args.patchVis:
        feat_dict = read_img(features)
    else:
        feat_dict = read_img_features(features, test_only=test_only)

    if test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    with open(OBJECT_INFO_STORE, 'rb') as f:
        obj_store = pkl.load(f)

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok, obj_store=obj_store)
    from collections import OrderedDict

    if args.submit:
        val_env_names.append('test')
    else:
        pass

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok, obj_store=obj_store),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.patchVis:
        train_env.vit_model = vit_model
        train_env.img_process = img_process
        # TODO: image process in validation should be checked
        val_envs.vit_model = vit_model
        val_envs.img_process = img_process

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs, vit_args=vit_args, vit_model=vit_model)
    elif args.train == 'validlistener':
        valid(train_env, tok, val_envs=val_envs)
    else:
        assert False

def train_val_augment(test_only=False, vit_model=None, vit_args=None, img_process=None):
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    tok_bert = get_tokenizer(args)

    # Load the env img features
    if args.patchVis:
        feat_dict = read_img(features)
    else:
        feat_dict = read_img_features(features, test_only=test_only)

    if test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    with open(OBJECT_INFO_STORE, 'rb') as f:
        obj_store = pkl.load(f)

    # Load the augmentation data
    aug_path = args.aug
    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok_bert, obj_store=obj_store)
    aug_env   = R2RBatch(feat_dict, batch_size=args.batchSize, splits=[aug_path], tokenizer=tok_bert, name='aug', obj_store=obj_store)

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok_bert, obj_store=obj_store),
                Evaluation([split], featurized_scans, tok_bert))
                for split in val_env_names}

    if args.patchVis:
        train_env.vit_model = vit_model
        train_env.img_process = img_process
        aug_env.vit_model = vit_model
        aug_env.img_process = img_process
        # TODO: image process in validation should be checked
        val_envs.vit_model = vit_model
        val_envs.img_process = img_process

    # Start training
    train(train_env, tok_bert, args.iters, val_envs=val_envs, aug_env=aug_env, vit_model=vit_model, vit_args=vit_args)


if __name__ == "__main__":
    if args.patchVis:
        # cait finetune args
        vit_args = get_args_parser().parse_known_args()[0]
        vit_args.lr            = 5e-6
        vit_args.weight_decay  = 0.05
        vit_args.decay_epochs  = 10
        vit_args.input_size    = 384   # for XS24-384
        vit_args.repeated_aug  = True
        vit_args.smoothing     = 0.1
        vit_args.warmup_epochs = 5
        vit_args.drop_path     = 0.3
        vit_args.mixup         = 0.8
        vit_args.cutmix        = 1.0

        img_process = build_transform(True, vit_args)

        vit_model = cait_XS24(pretrained=True)

        vit_model.norm = torch.nn.Identity()
        vit_model.head = torch.nn.Linear(vit_model.embed_dim, args.feature_size)

    else:
        vit_model = None
        vit_args = None
        img_process = None

    if args.train in ['listener', 'validlistener']:
        train_val(test_only=args.test_only, vit_model=vit_model, vit_args=vit_args, img_process=img_process)
    elif args.train == 'auglistener':
        train_val_augment(test_only=args.test_only, vit_model=vit_model, vit_args=vit_args, img_process=img_process)
    else:
        assert False
