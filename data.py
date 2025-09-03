import logging
import os
import random
from dataclasses import dataclass

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import csv
from tqdm import tqdm

import sys
import ast
import json
import math
import pickle
import braceexpand
import webdataset as wds
from multiprocessing import Value
from dataclasses import dataclass
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    url_opener,
    tar_file_expander,
    valid_sample,
)
from torch.utils.data import (
    Dataset,
    DataLoader,
    IterableDataset,
    get_worker_info,
)


class CsvDatasetAugCap(Dataset):
    def __init__(
        self,
        input_filename,
        transforms,
        tokenizer=None,
        root=None,
        augmented_caption_filelist=None,
    ):
        logging.debug(f"Loading csv data from {input_filename}.")
        self.images = []
        self.captions = []
        self.root = root
        assert input_filename.endswith(".csv")
        assert (
            augmented_caption_filelist is not None
        ), "augmented_caption_filelist is None, use csvdataset instead"

        num_augcap = len(augmented_caption_filelist)
        augmented_captions = []
        file_length = []
        for f in augmented_caption_filelist:
            with open(f, "r") as file:
                cur_captions = file.readlines()
                file_length.append(len(cur_captions))
                augmented_captions.append(cur_captions)
        assert (
            len(augmented_captions) == num_augcap
        ), "number of augmented captions is not equal to num_augcap"

        for i in range(num_augcap):
            assert (
                file_length[i] == file_length[0]
            ), "number of captions in each file is not the same"
        num_samples = file_length[0]

        with open(input_filename, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            row_index = 0
            for row in tqdm(csv_reader):
                image = row[0]
                prompt = row[1]
                if image.endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(self.root, image)
                    self.images.append(image_path)

                    if row_index < num_samples:
                        self.captions.append([prompt])
                        for augcap_idx in range(num_augcap):
                            self.captions[row_index].append(
                                augmented_captions[augcap_idx][row_index].replace(
                                    "\n", ""
                                )
                            )
                        assert (
                            len(self.captions[row_index]) == num_augcap + 1
                        ), "number of captions is not equal to num_augcap + 1"
                    row_index += 1
            assert (
                row_index % num_samples == 0
            ), "number of samples in csv is not equal to num_samples in new caption"

        self.num_samples = num_samples
        self.transforms = transforms
        logging.debug("Done loading data.")

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        caption_list = self.captions[idx % self.num_samples]
        caption = random.choice(caption_list)
        if len(caption.split(" ")) < 2:
            caption = caption_list[0]
        texts = caption
        texts = self.tokenizer(str(texts))
        return images, texts


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, tokenizer=None, root=None):
        logging.debug(f"Loading csv data from {input_filename}.")
        self.images = []
        self.captions = []
        self.root = root
        assert input_filename.endswith(".csv")
        with open(input_filename, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in tqdm(csv_reader):
                image = row[0]
                prompt = row[1]
                if image.endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(self.root, image)
                    self.images.append(image_path)
                    self.captions.append(prompt)
        self.transforms = transforms
        logging.debug("Done loading data.")

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenizer(str(self.captions[idx]))
        return images, texts


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None

    def set_epoch(self, epoch):
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_csv_dataset(args, preprocess_fn, is_train, tokenizer=None, aug_text=False):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    if args.aug_text:
        augmented_caption_filelist = args.augmented_caption_filelist
        dataset = CsvDatasetAugCap(
            input_filename,
            preprocess_fn,
            root=args.root,
            tokenizer=tokenizer,
            augmented_caption_filelist=augmented_caption_filelist,
        )

    else:
        dataset = CsvDataset(
            input_filename, preprocess_fn, root=args.root, tokenizer=tokenizer
        )

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


# ================================= STUFF FOR WEBDATASET ===================================

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def with_additional_augmentations(
    preprocess_img, augment, tokenizer, rephrased_captions
):
    def map_fn(sample):
        out = {}

        key = f"{sample['__key__']}.jpg"

        image = sample["image"]
        out["image1"] = preprocess_img(image)
        out["image2"] = augment(image)

        text = sample["text"]
        random_text = random.sample(rephrased_captions[key]["paraphrases"] + [text], 1)[0]

        out["text1"] = tokenizer(text)
        out["text2"] = tokenizer(random_text)

        return out

    return map_fn


def get_wds_dataset(
    args, preprocess_img, augment, is_train, epoch=0, floor=False, tokenizer=None
):
    # =============== Needed for TeMo =======================
    _REPHRASED_CAPTIONS_PATH = (
        "/ptmp/dduka/databases/cc12m/paraphrases/cc12m_all_paraphrased.pkl"
    )

    # Load _REPHRASED_CAPTIONS_PATH
    with open(_REPHRASED_CAPTIONS_PATH, "rb") as f:
        REPHRASED_CAPTIONS = pickle.load(f)
    # =============== Needed for TeMo =======================

    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    "Currently, the number of dataset samples must be specified for the training dataset. "
                    "Please specify it via `--train-num-samples` if no dataset length info is present."
                )
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0

    shared_epoch = SharedEpoch(
        epoch=epoch
    )  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert (
            resampled
        ), "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."

    if resampled:
        pipeline = [
            ResampledShards2(
                input_shards,
                weights=args.train_data_upsampling_factors,
                deterministic=True,
                epoch=shared_epoch,
            )
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    detshuffle2(
                        bufsize=_SHARD_SHUFFLE_SIZE,
                        initial=_SHARD_SHUFFLE_INITIAL,
                        seed=args.seed,
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        pipeline.extend(
            [
                # at this point, we have an iterator over the shards assigned to each worker at each node
                tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(handler=log_and_continue),
            ]
        )
    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            # wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
            wds.map(
                with_additional_augmentations(
                    preprocess_img,
                    augment,
                    tokenizer,
                    REPHRASED_CAPTIONS,
                )
            ),
            # wds.to_tuple("image", "text"),
            wds.to_tuple("image1", "text1", "image2", "text2"),
            wds.batched(args.batch_size, partial=not is_train),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert (
                num_shards >= args.workers * args.world_size
            ), "number of shards must be >= total workers"
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(
            num_batches / num_workers
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(
            num_worker_batches
        )  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split("::")
        assert len(weights) == len(
            urllist
        ), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(
                self.weights
            ), f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(
                    url=self.rng.choices(self.urls, weights=self.weights, k=1)[0]
                )


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler, eof_value=None)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def get_data(args, preprocess_fns, tokenizer=None):
    preprocess_train, _ = preprocess_fns
    data = {
        "train": get_wds_dataset(
            args=args,
            preprocess_img=preprocess_train,
            augment=preprocess_train,
            is_train=True,
            tokenizer=tokenizer,
        )
        # "train": get_csv_dataset(
        #     args, preprocess_train, is_train=True, tokenizer=tokenizer
        # )
    }

    return data
