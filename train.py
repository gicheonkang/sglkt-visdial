"""
This code is modified from batra-mlp-lab's repository.
https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
"""
import os
import argparse
import itertools
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from bisect import bisect
import datetime
import random

from visdialch.data.dataset import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.metrics import SparseGTMetrics, NDCG
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import CheckpointManager, load_checkpoint
from visdialch.utils.logging import Logger
from visdialch.utils.scheduler import get_optim, adjust_lr

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml",
    default="configs/sglkt.yml",
    help="Path to a config file listing reader, model and solver parameters.",
)
parser.add_argument(
    "--train-json",
    default="data/visdial_1.0_train.json",
    help="Path to json file containing VisDial v1.0 training data.",
)
parser.add_argument(
    "--val-json",
    default="data/visdial_1.0_val.json",
    help="Path to json file containing VisDial v1.0 validation data.",
)
parser.add_argument(
    "--train-structure-json",
    default="data/visdial_1.0_train_coref_structure.json"
)
parser.add_argument(
    "--val-structure-json",
    default="data/visdial_1.0_val_coref_structure.json"
)
parser.add_argument(
    "--train-neural-dense-json",
    default="data/visdial_1.0_train_dense_labels.json"
)
parser.add_argument(
    "--val-dense-json",
    default="data/visdial_1.0_val_dense_annotations.json",
    help="Path to json file containing VisDial v1.0 validation dense ground "
    "truth annotations.",
)

parser.add_argument_group(
    "Arguments independent of experiment reproducibility"
)
parser.add_argument(
    "--gpu-ids",
    nargs="+",
    type=int,
    default=[0, 1],
    help="List of ids of GPUs to use.",
)
parser.add_argument(
    "--cpu-workers",
    type=int,
    default=8,
    help="Number of CPU workers for dataloader.",
)
parser.add_argument(
    "--overfit",
    default=False,
    help="Overfit model on 5 examples, meant for debugging.",
)
parser.add_argument(
    "--validate",
    default=True,
    help="Whether to validate on val split after every epoch.",
)
parser.add_argument(
    "--in-memory",
    default=False,
    help="Load the whole dataset and pre-extracted image features in memory. "
    "Use only in presence of large RAM, atleast few tens of GBs.",
)

parser.add_argument_group("Checkpointing related arguments")
parser.add_argument(
    "--save-dirpath",
    default="checkpoints/",
    help="Path of directory to create checkpoint directory and save "
    "checkpoints.",
)
parser.add_argument(
    "--load-pthpath",
    default="",
    help="To continue training, path to .pth file of saved checkpoint.",
)

# =============================================================================
#   RANDOM SEED
# =============================================================================
seed = random.randint(0, 99999999)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

# =============================================================================
#   INPUT ARGUMENTS AND CONFIG
# =============================================================================
args = parser.parse_args()

# keys: {"dataset", "model", "solver"}
config = yaml.safe_load(open(args.config_yml))

if isinstance(args.gpu_ids, int):
    args.gpu_ids = [args.gpu_ids]
device = (
    torch.device("cuda", args.gpu_ids[0])
    if args.gpu_ids[0] >= 0
    else torch.device("cpu")
)

# Print config and args.
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))

# =============================================================================
#   SETUP DATASET, DATALOADER, MODEL, CRITERION, OPTIMIZER, SCHEDULER
# =============================================================================
train_dataset = VisDialDataset(
    config=config["dataset"],
    dialogs_jsonpath=args.train_json,
    coref_dependencies_jsonpath=args.train_structure_json,
    answer_plausibility_jsonpath=args.train_neural_dense_json,
    overfit=args.overfit,
    in_memory=args.in_memory,
    return_options=True if config["model"]["decoder"] == "disc" else False,
    add_boundary_toks=False if config["model"]["decoder"] == "disc" else True,
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["solver"]["batch_size"],
    num_workers=args.cpu_workers,
    shuffle=True,
    collate_fn=train_dataset.collate_fn
)

val_dataset = VisDialDataset(
    config=config["dataset"],
    dialogs_jsonpath=args.val_json,
    coref_dependencies_jsonpath=args.val_structure_json,
    dense_annotations_jsonpath=args.val_dense_json,
    overfit=args.overfit,
    in_memory=args.in_memory,
    return_options=True,
    add_boundary_toks=False if config["model"]["decoder"] == "disc" else True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config["solver"]["batch_size"]
    if config["model"]["decoder"] == "disc"
    else 5,
    num_workers=args.cpu_workers,
    collate_fn=val_dataset.collate_fn
)

# Pass vocabulary to construct Embedding layer.
encoder = Encoder(config["model"], train_dataset.vocabulary)
decoder = Decoder(config["model"], train_dataset.vocabulary)
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

# New: Initializing word_embed using GloVe
if config["dataset"]["glove_npy"] != '':
    encoder.word_embed.weight.data = torch.from_numpy(np.load(config["dataset"]["glove_npy"]))
    print("Loaded glove vectors from {}".format(config["dataset"]["glove_npy"]))
    
# Share word embedding between encoder and decoder.
decoder.word_embed = encoder.word_embed

# Wrap encoder and decoder in a model.
model = EncoderDecoderModel(encoder, decoder).to(device)
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)

# Loss function.
if config["model"]["decoder"] == "disc":
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion3 = nn.MSELoss()
elif config["model"]["decoder"] == "gen":
    criterion1 = nn.CrossEntropyLoss(
        ignore_index=train_dataset.vocabulary.PAD_INDEX
    )
    criterion2 = nn.MSELoss()
else:
    raise NotImplementedError

if config["solver"]["training_splits"] == "trainval":
    iterations = (len(train_dataset) + len(val_dataset)) // config["solver"][
        "batch_size"
    ] + 1
else:
    iterations = len(train_dataset) // config["solver"]["batch_size"] + 1

# =============================================================================
#   SETUP BEFORE TRAINING LOOP
# =============================================================================
start_time = datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')
if args.save_dirpath == 'checkpoints/':
    args.save_dirpath += '%s' % start_time

os.makedirs(args.save_dirpath, exist_ok=True)    
logger = Logger(os.path.join(args.save_dirpath, 'log.txt'))
logger.write("{}".format(seed))

sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

# If loading from checkpoint, adjust start epoch and load parameters.
if args.load_pthpath == "":
    start_epoch = 1
    optim = get_optim(config, model, len(train_dataset))

else:
    # "path/to/checkpoint_xx.pth" -> xx
    start_epoch = int(args.load_pthpath.split("_")[-1][:-4]) + 1

    model_state_dict, optimizer_state_dict = load_checkpoint(args.load_pthpath)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    optim = get_optim(config, model, len(train_dataset))
    optim._step = iterations * (start_epoch - 1)
    optim.optimizer.load_state_dict(optimizer_state_dict)
    print("Loaded model from {}".format(args.load_pthpath))

checkpoint_manager = CheckpointManager(
    model, optim.optimizer, args.save_dirpath, last_epoch=start_epoch-1, config=config
)

# =============================================================================
#   TRAINING LOOP
# =============================================================================

running_loss = 0.0 
train_begin = datetime.datetime.utcnow() 
for epoch in range(start_epoch, config["solver"]["num_epochs"]+1):
    # -------------------------------------------------------------------------
    #   ADJUST LEARNING RATE
    # -------------------------------------------------------------------------
    if epoch in config["solver"]["lr_decay_list"]:
        adjust_lr(optim, config["solver"]["lr_decay_rate"])
      
    # -------------------------------------------------------------------------
    #   ON EPOCH START
    # -------------------------------------------------------------------------
    combined_dataloader = itertools.chain(train_dataloader)

    print(f"\nTraining for epoch {epoch}:")
    for i, batch in enumerate(combined_dataloader):
        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = batch[key].cuda()

        optim.zero_grad()
        output, structures = model(batch)       
        target = (
            batch["ans_ind"]
            if config["model"]["decoder"] == "disc"
            else batch["ans_out"]
        )
        if epoch < 5:
            batch_loss = criterion2(output.view(-1, output.size(-1)), batch["ans_ind"].view(-1))
        else: 
            batch_loss = criterion1(output, batch["teacher_scores"])
        batch_loss += criterion3(structures, batch["structures"])

        batch_loss.backward()
        optim.step()

        # --------------------------------------------------------------------
        # update running loss and decay learning rates
        # --------------------------------------------------------------------
        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * batch_loss.item()
        else:
            running_loss = batch_loss.item()
 
        torch.cuda.empty_cache()
        if i % 100 == 0:
            # print current time, running average, learning rate, iteration, epoch
            logger.write("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
                datetime.datetime.utcnow() - train_begin, epoch,
                    (epoch - 1) * iterations + i, running_loss,
                    optim.optimizer.param_groups[0]['lr']))

    # -------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # -------------------------------------------------------------------------
    checkpoint_manager.step()

    # Validate and report automatic metrics.
    if args.validate:
        # Switch dropout, batchnorm etc to the correct mode.
        model.eval()
        logger.write("\nValidation after epoch {}:".format(epoch))
        total_hist_usage = 0
        for i, batch in enumerate(tqdm(val_dataloader)):
            for key in batch:
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                output, structures = model(batch)
                total_hist_usage += torch.sum(structures)
            sparse_metrics.observe(output, batch["ans_ind"])
            if "gt_relevance" in batch:
                output = output[
                    torch.arange(output.size(0)), batch["round_id"] - 1, :
                ]
                ndcg.observe(output, batch["gt_relevance"])

        all_metrics = {}
        all_metrics.update(sparse_metrics.retrieve(reset=True))
        all_metrics.update(ndcg.retrieve(reset=True))
        for metric_name, metric_value in all_metrics.items():
            logger.write("{}: {:4f}".format(metric_name, metric_value))

        total_connct = config["dataset"]["total_connection_val"]
        ratio = total_hist_usage / total_connct
        logger.write("sparsity: {:4f}\n".format(1-torch.sum(ratio).item()))
        model.train()
        torch.cuda.empty_cache()
