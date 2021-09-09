import argparse, os, torch
import random
import numpy as np
from utils.data import build_data
from trainer.trainer import Trainer
from models.model import REModel


def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print('Unparsed args: {}'.format(unparsed))
    return args, unparsed


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    data_arg = add_argument_group('Data')
    data_arg.add_argument('--dataset_name', type=str, default='WebNLG')
    data_arg.add_argument('--data_dir', type=str, default='dataset/WebNLG-E/data')
    data_arg.add_argument('--bert_dir', type=str, default='./Bert/bert_base_cased/')
    data_arg.add_argument('--log_dir', type=str, default='output/logs')
    data_arg.add_argument('--ckpt_dir', type=str, default='output/ckpt')
    data_arg.add_argument('--generated_data_directory', type=str, default='./data/generated_data/')
    data_arg.add_argument('--generated_param_directory', type=str, default='./data/generated_data/model_param/')

    train_arg = add_argument_group('Training')
    train_arg.add_argument('--num_epoch', type=int, default=50)
    train_arg.add_argument('--batch_size', type=int, default=8)
    train_arg.add_argument('--lr', type=float, default=2e-5)
    train_arg.add_argument('--decoder_lr', type=float, default=2e-5)
    train_arg.add_argument('--encoder_lr', type=float, default=1e-5)
    train_arg.add_argument('--lr_decay', type=float, default=0.01)
    train_arg.add_argument('--weight_decay', type=float, default=1e-5)
    train_arg.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])
    train_arg.add_argument('--max_grad_norm', type=float, default=0)
    train_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    train_arg.add_argument('--dropout', type=float, default=0.4, help='Input and RNN dropout rate.')  #

    model_arg = add_argument_group('Model')
    model_arg.add_argument('--model_name', type=str, default='Set-Prediction-Networks')
    model_arg.add_argument('--tokens_emb_dim', type=int, default=768, help='bert tokens embedding dimension.')
    model_arg.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding dimension.')
    model_arg.add_argument('--position_emb_dim', type=int, default=20, help='Position embedding dimension.')
    model_arg.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
    model_arg.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')

    evaluation_arg = add_argument_group('Evaluation')
    evaluation_arg.add_argument('--n_best_size', type=int, default=100)
    evaluation_arg.add_argument('--max_span_length', type=int, default=12) #NYT webNLG 10

    misc_arg = add_argument_group('MISC')
    misc_arg.add_argument('--refresh', action='store_true')
    misc_arg.add_argument('--cuda', type=bool, default=True)
    misc_arg.add_argument('--gpu', type=int, default=1)
    misc_arg.add_argument('--seed', type=int, default=1)

    args, unparsed = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    for arg in vars(args):
        print(arg, ':',  getattr(args, arg))
    set_seed(args.random_seed)
    data = build_data(args)
    model = REModel(args, data.relational_alphabet.size())
    trainer = Trainer(model, data, args)
    trainer.train_model()
