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
from torch.utils.tensorboard import SummaryWriter

from vlnbert.vlnbert_init import get_tokenizer

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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

print(args); print('')


''' train the listener '''
def train(train_env, tok, n_iters, log_every=args.log_every, val_envs={}, aug_env=None):
    writer = SummaryWriter(log_dir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

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
        if not args.finetune:
            listner.adjust_lr()

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total
        RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
        IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
        PG_loss = sum(listner.logs['PG_loss']) / max(len(listner.logs['PG_loss']), 1)
        Start_loss = sum(listner.logs['Start_loss']) / max(len(listner.logs['Start_loss']), 1)
        Len_loss = sum(listner.logs['Len_loss']) / max(len(listner.logs['Len_loss']), 1)
        Attn_loss = sum(listner.logs['Attn_loss']) / max(len(listner.logs['Attn_loss']), 1)
        # AP_loss = sum(listner.logs['AP_loss']) / max(len(listner.logs['AP_loss']), 1)
        entropy = sum(listner.logs['entropy']) / total
        if not args.finetune:
            lr = listner.logs['loss/lr'][-1]
            writer.add_scalar('loss/lr', lr, idx)
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/RL_loss", RL_loss, idx)
        writer.add_scalar("loss/IL_loss", IL_loss, idx)
        writer.add_scalar("loss/PG_loss", PG_loss, idx)
        writer.add_scalar("loss/Start_loss", Start_loss, idx)
        writer.add_scalar("loss/Attn_loss", Start_loss, idx)
        writer.add_scalar("loss/Len_loss", Len_loss, idx)
        # writer.add_scalar("loss/AP_loss", AP_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)

        # Run validation
        loss_str = "iter {}\n".format(iter)
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            with torch.no_grad():
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

            if args.visualize:
                attn_pos_cnt = np.zeros((args.maxAction, args.maxInput - 1))
                attn_pos_heatmap = np.zeros((args.maxAction, args.maxInput - 1))
                vis_log = listner.visualization_log
                traj_dict = {r['instr_id']: r['trajectory'] for r in result}
                for instr_id in traj_dict.keys():
                    for i, prob in enumerate(vis_log[instr_id]['language_attn_prob']):
                        if i >= args.maxAction:
                            print('path length exceeds max action: %s, %s' % (env_name, instr_id))
                            break
                        attn_pos_heatmap[i] += prob
                        attn_pos_cnt[i] += (np.arange(args.maxInput - 1) < (vis_log[instr_id]['seq_length'] - 1))
                attn_pos_heatmap /= attn_pos_cnt
                fig = plt.figure()
                heatmap = sns.heatmap(attn_pos_heatmap,
                                      square=True,
                                      cbar=False,
                                      cmap=sns.color_palette("light:#5A9", as_cmap=True),
                                      yticklabels=2
                                      )
                writer.add_figure(env_name, fig, idx, close=True)

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

        if iter % 2000 == 0:
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
            with open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w') as f:
                json.dump(
                    result,
                    f,
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
    if args.visualize:
        with open('snap/%s/visualization.pkl' % args.name, 'wb') as f:
            pkl.dump(agent.visualization_log, f)

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(0)
    np.random.seed(0)

def train_val(test_only=False):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    setup()
    tok = get_tokenizer(args)

    if args.render_image:
        feat_dict = None
    else:
        feat_dict = read_img_features(features, test_only=test_only)

    if args.max_pool_feature:
        with open(str(args.max_pool_feature), 'rb') as f:
            mp_feat_dict = pkl.load(f)
    else:
        mp_feat_dict = None

    if args.look_back_feature:
        with open(str(args.look_back_feature), 'rb') as f:
            lb_feat_dict = pkl.load(f)
    else:
        lb_feat_dict = None

    if test_only:
        featurized_scans = None
        val_env_names = ['val_train_seen']
    else:
        if feat_dict is not None:
            featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        else:
            featurized_scans = None
        # val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']
        val_env_names = ['val_seen', 'val_unseen']

    with open(OBJECT_INFO_STORE, 'rb') as f:
        obj_store = pkl.load(f)

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok, obj_store=obj_store,
                         mp_feature_store=mp_feat_dict, lb_feature_store=lb_feat_dict)
    from collections import OrderedDict

    if args.submit:
        val_env_names.append('test')
    else:
        pass

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok, obj_store=obj_store, mp_feature_store=mp_feat_dict),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        valid(train_env, tok, val_envs=val_envs)
    else:
        assert False

def train_val_augment(test_only=False):
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    tok_bert = get_tokenizer(args)

    # Load the env img features
    feat_dict = read_img_features(features, test_only=test_only)

    if args.max_pool_feature:
        with open(args.max_pool_feature, 'rb') as f:
            mp_feat_dict = pkl.load(f)
    else:
        mp_feat_dict = None

    if args.look_back_feature:
        with open(str(args.look_back_feature), 'rb') as f:
            lb_feat_dict = pkl.load(f)
    else:
        lb_feat_dict = None

    if test_only:
        featurized_scans = None
        val_env_names = []
    else:
        if feat_dict is not None:
            featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
        else:
            featurized_scans = None
        # val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']
        val_env_names = ['val_seen', 'val_unseen']

    with open(OBJECT_INFO_STORE, 'rb') as f:
        obj_store = pkl.load(f)

    # Load the augmentation data
    aug_path = args.aug
    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok_bert, obj_store=obj_store,
                         mp_feature_store=mp_feat_dict, lb_feature_store=lb_feat_dict)
    aug_env   = R2RBatch(feat_dict, batch_size=args.batchSize, splits=[aug_path], tokenizer=tok_bert, name='aug', obj_store=obj_store,
                         mp_feature_store=mp_feat_dict, lb_feature_store=lb_feat_dict)

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok_bert, obj_store=obj_store,
                                 mp_feature_store=mp_feat_dict, lb_feature_store=lb_feat_dict),
                Evaluation([split], featurized_scans, tok_bert))
                for split in val_env_names}

    # Start training
    train(train_env, tok_bert, args.iters, val_envs=val_envs, aug_env=aug_env)


if __name__ == "__main__":
    if args.name == 'debug':
        torch.autograd.set_detect_anomaly(True)
    if args.train in ['listener', 'validlistener']:
        train_val(test_only=args.test_only)
    elif args.train == 'auglistener':
        train_val_augment(test_only=args.test_only)
    else:
        assert False
