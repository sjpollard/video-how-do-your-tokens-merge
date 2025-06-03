import os
import subprocess
import datetime

import tqdm

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
from slowfast.models.build import MODEL_REGISTRY
from slowfast.utils.checkpoint import load_test_checkpoint
from slowfast.datasets import Kinetics, Ssv2, Epickitchens
from slowfast.datasets import loader
from slowfast.datasets import loader

import torch
from torch.nn.functional import softmax
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

import pandas as pd
import json
import ast

import numpy as np

from operator import add


class RandomKinetics(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, patch_depth, num_to_replace):
        self.cfg = cfg
        self.kinetics = Kinetics(cfg, mode)
        self.patch_depth = patch_depth
        self.num_to_replace = num_to_replace
        self.rng = np.random.default_rng(0)


    def __getitem__(self, index):
        clip = self.kinetics.__getitem__(index)
        frames_to_insert = sorted(list(map(add, self.patch_depth * (self.patch_depth * self.rng.choice((self.cfg.DATA.NUM_FRAMES // self.patch_depth) - 1, self.num_to_replace, replace=False)).tolist(), sorted(self.num_to_replace * list(range(self.patch_depth))))))
        index_to_insert = self.rng.choice(self.__len__(), 1)[0]
        clip_to_insert = self.kinetics.__getitem__(index_to_insert)
        for frame in frames_to_insert:
            clip[0][0][:, frame, :, :] = clip_to_insert[0][0][:, frame, :, :]
        return clip


    def __len__(self):
        return self.kinetics.__len__()
    

class RandomSsv2(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, patch_depth, num_to_replace):
        self.cfg = cfg
        self.ssv2 = Ssv2(cfg, mode)
        self.patch_depth = patch_depth
        self.num_to_replace = num_to_replace
        self.rng = np.random.default_rng(0)


    def __getitem__(self, index):
        clip = self.ssv2.__getitem__(index)
        frames_to_insert = sorted(list(map(add, self.patch_depth * (self.patch_depth * self.rng.choice((self.cfg.DATA.NUM_FRAMES // self.patch_depth) - 1, self.num_to_replace, replace=False)).tolist(), sorted(self.num_to_replace * list(range(self.patch_depth))))))
        index_to_insert = self.rng.choice(self.__len__(), 1)[0]
        clip_to_insert = self.ssv2.__getitem__(index_to_insert)
        for frame in frames_to_insert:
            clip[0][0][:, frame, :, :] = clip_to_insert[0][0][:, frame, :, :]
        return clip


    def __len__(self):
        return self.ssv2.__len__()


class RandomEpickitchens(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, patch_depth, num_to_replace):
        self.cfg = cfg
        self.epickitchens = Epickitchens(cfg, mode)
        self.patch_depth = patch_depth
        self.num_to_replace = num_to_replace
        self.rng = np.random.default_rng(0)


    def __getitem__(self, index):
        clip = self.epickitchens.__getitem__(index)
        frames_to_insert = sorted(list(map(add, self.patch_depth * (self.patch_depth * self.rng.choice((self.cfg.DATA.NUM_FRAMES // self.patch_depth) - 1, self.num_to_replace, replace=False)).tolist(), sorted(self.num_to_replace * list(range(self.patch_depth))))))
        index_to_insert = self.rng.choice(self.__len__(), 1)[0]
        clip_to_insert = self.epickitchens.__getitem__(index_to_insert)
        for frame in frames_to_insert:
            clip[0][0][:, frame, :, :] = clip_to_insert[0][0][:, frame, :, :]
        return clip


    def __len__(self):
        return self.epickitchens.__len__()


class SameClassKinetics(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, patch_depth, num_to_replace):
        self.cfg = cfg
        self.kinetics = Kinetics(cfg, mode)
        self.patch_depth = patch_depth
        self.num_to_replace = num_to_replace
        self.rng = np.random.default_rng(0)


    def __getitem__(self, index):
        clip = self.kinetics.__getitem__(index)
        frames_to_insert = sorted(list(map(add, self.patch_depth * (self.patch_depth * self.rng.choice((self.cfg.DATA.NUM_FRAMES // self.patch_depth) - 1, self.num_to_replace, replace=False)).tolist(), sorted(self.num_to_replace * list(range(self.patch_depth))))))
        indices_with_same_class = [i for i, x in enumerate(self.kinetics._labels) if x == clip[1]]
        index_to_insert = self.rng.choice(indices_with_same_class, 1)[0]
        clip_to_insert = self.kinetics.__getitem__(index_to_insert)
        for frame in frames_to_insert:
            clip[0][0][:, frame, :, :] = clip_to_insert[0][0][:, frame, :, :]
        return clip


    def __len__(self):
        return self.kinetics.__len__()
    

class SameClassSsv2(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, patch_depth, num_to_replace):
        self.cfg = cfg
        self.ssv2 = Ssv2(cfg, mode)
        self.patch_depth = patch_depth
        self.num_to_replace = num_to_replace
        self.rng = np.random.default_rng(0)


    def __getitem__(self, index):
        clip = self.ssv2.__getitem__(index)
        frames_to_insert = sorted(list(map(add, self.patch_depth * (self.patch_depth * self.rng.choice((self.cfg.DATA.NUM_FRAMES // self.patch_depth) - 1, self.num_to_replace, replace=False)).tolist(), sorted(self.num_to_replace * list(range(self.patch_depth))))))
        indices_with_same_class = [i for i, x in enumerate(self.ssv2._labels) if x == clip[1]]
        index_to_insert = self.rng.choice(indices_with_same_class, 1)[0]
        clip_to_insert = self.ssv2.__getitem__(index_to_insert)
        for frame in frames_to_insert:
            clip[0][0][:, frame, :, :] = clip_to_insert[0][0][:, frame, :, :]
        return clip


    def __len__(self):
        return self.ssv2.__len__()


class SameClassEpickitchens(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, patch_depth, match_type, num_to_replace):
        self.cfg = cfg
        self.epickitchens = Epickitchens(cfg, mode)
        self.patch_depth = patch_depth
        self.match_type = match_type
        self.num_to_replace = num_to_replace
        self.rng = np.random.default_rng(0)


    def __getitem__(self, index):
        clip = self.epickitchens.__getitem__(index)
        frames_to_insert = sorted(list(map(add, self.patch_depth * (self.patch_depth * self.rng.choice((self.cfg.DATA.NUM_FRAMES // self.patch_depth) - 1, self.num_to_replace, replace=False)).tolist(), sorted(self.num_to_replace * list(range(self.patch_depth))))))
        if self.match_type == 'verb':
            indices_with_same_class = [i for i, x in enumerate(self.epickitchens._verb_labels) if x == clip[1]['verb']]
        elif self.match_type == 'noun':
            indices_with_same_class = [i for i, x in enumerate(self.epickitchens._noun_labels) if x == clip[1]['noun']]
        index_to_insert = self.rng.choice(indices_with_same_class, 1)[0]
        clip_to_insert = self.epickitchens.__getitem__(index_to_insert)
        for frame in frames_to_insert:
            clip[0][0][:, frame, :, :] = clip_to_insert[0][0][:, frame, :, :]
        return clip


    def __len__(self):
        return self.epickitchens.__len__()


class KLSimilarityEpickitchens(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, patch_depth, match_type, num_to_replace):
        self.cfg = cfg
        self.epickitchens = Epickitchens(cfg, mode)
        self.patch_depth = patch_depth
        self.match_type = match_type
        self.num_to_replace = num_to_replace
        self.rng = np.random.default_rng(0)
        self.kl_divergences = torch.load(f'{cfg.DATASET_UTILS.DISTRIBUTION_PATH}/{cfg.TEST.CHECKPOINT_FILE_PATH.split("/")[-1].split(".")[0]}_kl_divergences.pt')


    def __getitem__(self, index):
        clip = self.epickitchens.__getitem__(index)
        labels = clip[1]
        frames_to_insert = sorted(list(map(add, self.patch_depth * (self.patch_depth * self.rng.choice((self.cfg.DATA.NUM_FRAMES // self.patch_depth) - 1, self.num_to_replace, replace=False)).tolist(), sorted(self.num_to_replace * list(range(self.patch_depth))))))
        if self.match_type == 'verb':
            verb_labels = torch.tensor(self.epickitchens._verb_labels, dtype=torch.uint8)
            indices_to_ignore = (verb_labels != labels['verb']).nonzero().squeeze()
            self.kl_divergences[0][index, indices_to_ignore] = torch.inf
            index_to_insert = self.kl_divergences[0].argmin(dim=1)[index].item()
        elif self.match_type == 'noun':
            noun_labels = torch.tensor(self.epickitchens._noun_labels, dtype=torch.uint8)
            indices_to_ignore = (noun_labels != labels['noun']).nonzero().squeeze()
            self.kl_divergences[1][index, indices_to_ignore] = torch.inf
            index_to_insert = self.kl_divergences[1].argmin(dim=1)[index].item()
        clip_to_insert = self.epickitchens.__getitem__(index_to_insert)
        for frame in frames_to_insert:
            clip[0][0][:, frame, :, :] = clip_to_insert[0][0][:, frame, :, :]
        return clip


    def __len__(self):
        return self.epickitchens.__len__()


def create_mini_datasets(cfg):
    if cfg.TEST.DATASET == 'kinetics':
        df = pd.read_csv(f'{cfg.DATA.PATH_TO_DATA_DIR}/test.csv', names=['video', 'label'], delim_whitespace=True)
        df = df.groupby('label').apply(lambda x: x.iloc[0])
        path_list = cfg.DATA.PATH_TO_DATA_DIR.split("/")
        path_list[-1] = f'Mini-{path_list[-1]}'
        output_path = '/'.join(path_list)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        df.to_csv(f'{output_path}/test.csv', sep=' ', header=False, index=False)
    elif cfg.TEST.DATASET == 'ssv2':
        with open(f'{cfg.DATA.PATH_TO_DATA_DIR}/something-something-v2-validation.json') as f:
            dict = json.load(f)
        with open(f'{cfg.DATA.PATH_TO_DATA_DIR}/something-something-v2-labels.json') as f:
            labels = json.load(f)
        df = pd.DataFrame.from_records(dict)
        templates = []
        df.apply(lambda x: templates.append(int(labels[x['template'].replace('[', '').replace(']', '')])), axis='columns')
        df = df.assign(numeric_label=templates)
        df = df.groupby('numeric_label').apply(lambda x: x.iloc[0])
        json_list = []
        df.drop('numeric_label', axis='columns').apply(lambda x: json_list.append(ast.literal_eval(x.to_json())), axis='columns')
        frame_df = pd.read_csv(f'{cfg.DATA.PATH_TO_DATA_DIR}/val.csv', sep=' ')
        indices = frame_df['original_vido_id'].isin(list(map(int, df['id'].values)))
        frame_df = frame_df[indices]
        path_list = cfg.DATA.PATH_TO_DATA_DIR.split("/")
        path_list[-1] = f'Mini-{path_list[-1]}'
        output_path = '/'.join(path_list)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        with open(f'{output_path}/something-something-v2-validation.json', 'w') as f:
            json.dump(json_list, f)
        frame_df.to_csv(f'{output_path}/val.csv', sep=' ', index=False, na_rep='""', quotechar="'")
    elif cfg.TEST.DATASET == 'epickitchens':
        df = pd.read_pickle(f'{cfg.EPICKITCHENS.ANNOTATIONS_DIR}/EPIC_100_validation.pkl').reset_index()
        df_verb = df.groupby('verb_class').apply(lambda x: x.iloc[0])
        df_noun = df.groupby('noun_class').apply(lambda x: x.iloc[0])
        df = pd.concat([df_verb, df_noun]).set_index('narration_id')
        path_list = cfg.EPICKITCHENS.ANNOTATIONS_DIR.split("/")
        path_list[-1] = f'Mini-{path_list[-1]}'
        output_path = '/'.join(path_list)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        df.to_pickle(f'{output_path}/EPIC_100_validation.pkl')


def cache_model_probs(cfg):
    cfg.NUM_GPUS = 1

    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg).to('cuda')
    load_test_checkpoint(cfg, model)

    test_loader = loader.construct_loader(cfg, "test")

    verb_distribution_list = []
    noun_distribution_list = []

    with torch.inference_mode():
        for (inputs, labels, video_idx, time, meta) in test_loader:
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
            output = model(inputs)
            if cfg.TEST.DATASET == 'epickitchens':
                verb_dist = softmax(output[0], dim=1)
                noun_dist = softmax(output[1], dim=1)

                verb_distribution_list.append(verb_dist.to('cpu'))
                noun_distribution_list.append(noun_dist.to('cpu'))

    verb_distributions = torch.cat(verb_distribution_list, dim=0)
    noun_distributions = torch.cat(noun_distribution_list, dim=0)
    distributions = (verb_distributions, noun_distributions)

    if not os.path.exists(cfg.DATASET_UTILS.DISTRIBUTION_PATH):
        os.mkdir(cfg.DATASET_UTILS.DISTRIBUTION_PATH)

    torch.save(distributions, f'{cfg.DATASET_UTILS.DISTRIBUTION_PATH}/{cfg.TEST.CHECKPOINT_FILE_PATH.split("/")[-1].split(".")[0]}_distributions.pt')


def cache_kl_divergences(cfg):
    distributions = torch.load(f'{cfg.DATASET_UTILS.DISTRIBUTION_PATH}/{cfg.TEST.CHECKPOINT_FILE_PATH.split("/")[-1].split(".")[0]}_distributions.pt')
    verb_distributions = distributions[0]
    noun_distributions = distributions[1]

    verb_kl_divergence_list = []
    noun_kl_divergence_list = []

    for i in range(verb_distributions.size(0)):
        verb_kl_divergence_list.append(kl_divergence(Categorical(probs=verb_distributions), Categorical(probs=verb_distributions.roll(-1 * i, dims=0))))

    for i in range(noun_distributions.size(0)):
        noun_kl_divergence_list.append(kl_divergence(Categorical(probs=noun_distributions), Categorical(probs=noun_distributions.roll(-1 * i, dims=0))))

    verb_kl_divergences = torch.dstack(verb_kl_divergence_list).squeeze()
    noun_kl_divergences = torch.dstack(noun_kl_divergence_list).squeeze()

    for i in range(verb_kl_divergences.size(0)):
        verb_kl_divergences[i] = verb_kl_divergences[i].roll(i)

    for i in range(noun_kl_divergences.size(0)):
        noun_kl_divergences[i] = noun_kl_divergences[i].roll(i)

    verb_kl_divergences.fill_diagonal_(torch.inf)
    noun_kl_divergences.fill_diagonal_(torch.inf)

    kl_divergences = (verb_kl_divergences, noun_kl_divergences)

    torch.save(kl_divergences, f'{cfg.DATASET_UTILS.DISTRIBUTION_PATH}/{cfg.TEST.CHECKPOINT_FILE_PATH.split("/")[-1].split(".")[0]}_kl_divergences.pt')


def save_lengths(cfg):
    if cfg.TRAIN.DATASET == 'kinetics' and cfg.TEST.DATASET == 'kinetics':
        kinetics_df = pd.read_csv(f'{cfg.DATA.PATH_TO_DATA_DIR}/test.csv', names=['video', 'class'], sep='\\s+')
        videos = []
        lengths = []
        for video in tqdm.tqdm(kinetics_df['video']):
            output = float(str(subprocess.check_output(f'ffprobe -v quiet -of csv=p=0 -show_entries format=duration {cfg.DATA.PATH_PREFIX}{video}', shell=True, text=True)).strip())
            videos.append(video)
            lengths.append(output)
        length_df = pd.DataFrame(data=list(zip(videos, lengths)), columns=['video', 'length'])
        length_df.to_csv(f'{cfg.DATA.PATH_TO_DATA_DIR}/kinetics_lengths.csv', index=False)
    elif cfg.TRAIN.DATASET == 'ssv2' and cfg.TEST.DATASET == 'ssv2':
        ssv2_df = pd.read_csv(f'{cfg.DATA.PATH_TO_DATA_DIR}/val.csv', sep='\\s+')
        videos = []
        lengths = []
        for video in tqdm.tqdm(ssv2_df['original_vido_id'].unique()):
            output = float(str(subprocess.check_output(f'ffprobe -v quiet -of csv=p=0 -show_entries format=duration {cfg.DATA.PATH_TO_DATA_DIR}/videos/{video}.webm', shell=True, text=True)).strip())
            videos.append(video)
            lengths.append(output)
        length_df = pd.DataFrame(data=list(zip(videos, lengths)), columns=['video', 'length'])
        length_df.to_csv(f'{cfg.DATA.PATH_TO_DATA_DIR}/ssv2_lengths.csv', index=False)
    elif cfg.TRAIN.DATASET == 'epickitchens' and cfg.TEST.DATASET == 'epickitchens':
        epickitchens_df = pd.read_pickle(f'{cfg.EPICKITCHENS.ANNOTATIONS_DIR}/EPIC_100_validation.pkl')
        videos = []
        lengths = []
        epickitchens_df.apply(lambda x: videos.append(x.name), axis='columns')
        epickitchens_df.apply(lambda x: lengths.append(((datetime.datetime.strptime(x['stop_timestamp'], '%H:%M:%S.%f') - datetime.datetime.strptime(x['start_timestamp'], '%H:%M:%S.%f')).total_seconds())), axis='columns')
        length_df = pd.DataFrame(data=list(zip(videos, lengths)), columns=['video', 'length'])
        length_df.to_csv(f'{cfg.EPICKITCHENS.ANNOTATIONS_DIR}/epickitchens_lengths.csv', index=False)


def load_lengths(cfg):
    if cfg.TRAIN.DATASET == 'kinetics' and cfg.TEST.DATASET == 'kinetics':
        length_df = pd.read_csv(f'{cfg.DATA.PATH_TO_DATA_DIR}/kinetics_lengths.csv')
    elif cfg.TRAIN.DATASET == 'ssv2' and cfg.TEST.DATASET == 'ssv2':
        length_df = pd.read_csv(f'{cfg.DATA.PATH_TO_DATA_DIR}/ssv2_lengths.csv')
    elif cfg.TRAIN.DATASET == 'epickitchens' and cfg.TEST.DATASET == 'epickitchens':
        length_df = pd.read_csv(f'{cfg.EPICKITCHENS.ANNOTATIONS_DIR}/epickitchens_lengths.csv')
    return length_df


def main():
    """
    Create subsamples of datasets and cache information
    """
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)
    
    if cfg.DATASET_UTILS.CREATE_MINI_DATASETS:
        create_mini_datasets(cfg)
    if cfg.DATASET_UTILS.CACHE_MODEL_PROBS:
        cache_model_probs(cfg)
    if cfg.DATASET_UTILS.CACHE_KL_DIVERGENCES:
        cache_kl_divergences(cfg)
    if cfg.DATASET_UTILS.SAVE_LENGTHS:
        save_lengths(cfg)


if __name__ == '__main__':
    main()