"""
This code is modified from batra-mlp-lab's repository.
https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch

--

A Reader simply reads data from disk and returns it almost as is, based on a "primary key", which
for the case of VisDial v1.0 dataset, is the ``image_id``. Readers should be utilized by 
torch ``Dataset``s. Any type of data pre-processing is not recommended in the reader, such as
tokenizing words to integers, embedding tokens, or passing an image through a pre-trained CNN.

Each reader must atleast implement three methods:
    - ``__len__`` to return the length of data this Reader can read.
    - ``__getitem__`` to return data based on ``image_id`` in VisDial v1.0 dataset.
    - ``keys`` to return a list of possible ``image_id``s this Reader can provide data of. 
"""

import copy
import json
from typing import Dict, List, Union, Optional

import h5py
# A bit slow, and just splits sentences to list of words, can be doable in `DialogsReader`.
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import numpy as np

class DialogsReader(object):
    """
    A simple reader for VisDial v1.0 dialog data. The json file must have the same structure as
    mentioned on ``https://visualdialog.org/data``.

    Parameters
    ----------
    dialogs_jsonpath : str
        Path to a json file containing VisDial v1.0 train, val or test dialog data.
    """

    def __init__(self, dialogs_jsonpath: str, coref_structure_jsonpath: str, answer_plausibility_jsonpath: Optional[str] = None):
        with open(dialogs_jsonpath, "r") as visdial_file:
            with open(coref_structure_jsonpath, "r") as cd_file:
                visdial_data = json.load(visdial_file)
                self._split = visdial_data["split"]

                self.questions = visdial_data["data"]["questions"]
                self.answers = visdial_data["data"]["answers"]

                # Add empty question, answer at the end, useful for padding dialog rounds for test.
                self.questions.append("")
                self.answers.append("")

                # Image_id serves as key for all four dicts here.
                self.captions = {}
                self.dialogs = {}
                self.num_rounds = {}
                self.coref_dependencies = {}
                self.answer_plausibility = {}

                # Load sentence-level coreference dependency annotations.
                cd_data = json.load(cd_file)
                if 'train' in self._split:
                    ap_data = json.load(open(answer_plausibility_jsonpath, "r"))
                image_ids = [entry["image_id"] for entry in cd_data]

                for dialog_for_image in visdial_data["data"]["dialogs"]:
                    self.captions[dialog_for_image["image_id"]] = dialog_for_image["caption"]
                    self.coref_dependencies[dialog_for_image["image_id"]] = \
                        cd_data[image_ids.index(dialog_for_image["image_id"])]["coref_dependency"]
                    
                    if 'train' in self._split: 
                        self.answer_plausibility[dialog_for_image["image_id"]] = \
                            ap_data[image_ids.index(dialog_for_image["image_id"])]["scores"]
                    
                    # Record original length of dialog, before padding.
                    # 10 for train and val splits, 10 or less for test split.
                    self.num_rounds[dialog_for_image["image_id"]] = len(dialog_for_image["dialog"])

                    # Pad dialog at the end with empty question and answer pairs (for test split).
                    while len(dialog_for_image["dialog"]) < 10:
                        dialog_for_image["dialog"].append({"question": -1, "answer": -1})

                    # Add empty answer /answer options if not provided (for test split).
                    for i in range(len(dialog_for_image["dialog"])):
                        if "answer" not in dialog_for_image["dialog"][i]:
                            dialog_for_image["dialog"][i]["answer"] = -1
                        if "answer_options" not in dialog_for_image["dialog"][i]:
                            dialog_for_image["dialog"][i]["answer_options"] = [-1] * 100

                    self.dialogs[dialog_for_image["image_id"]] = dialog_for_image["dialog"]

                print(f"[{self._split}] Tokenizing questions...")
                for i in tqdm(range(len(self.questions))):
                    self.questions[i] = word_tokenize(self.questions[i] + "?")

                print(f"[{self._split}] Tokenizing answers...")
                for i in tqdm(range(len(self.answers))):
                    self.answers[i] = word_tokenize(self.answers[i])

                print(f"[{self._split}] Tokenizing captions...")
                for image_id, caption in tqdm(self.captions.items()):
                    self.captions[image_id] = word_tokenize(caption)

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, image_id: int) -> Dict[str, Union[int, str, List]]:
        caption_for_image = self.captions[image_id]
        dialog_for_image = copy.deepcopy(self.dialogs[image_id])
        num_rounds = self.num_rounds[image_id]
        coref_dependency = self.coref_dependencies[image_id]
        if 'train' in self._split:
            plausible_answer = self.answer_plausibility[image_id]

        # Replace question and answer indices with actual word tokens.
        for i in range(len(dialog_for_image)):
            dialog_for_image[i]["question"] = self.questions[dialog_for_image[i]["question"]]
            dialog_for_image[i]["answer"] = self.answers[dialog_for_image[i]["answer"]]
            for j, answer_option in enumerate(dialog_for_image[i]["answer_options"]):
                dialog_for_image[i]["answer_options"][j] = self.answers[answer_option]

        if 'train' in self._split:
            return {
                "image_id": image_id,
                "caption": caption_for_image,
                "dialog": dialog_for_image,
                "num_rounds": num_rounds,
                "structures": coref_dependency,
                "teacher_scores": plausible_answer 
            }
        else:
            return {
                "image_id": image_id,
                "caption": caption_for_image,
                "dialog": dialog_for_image,
                "num_rounds": num_rounds,
                "structures": coref_dependency
            }

    def keys(self) -> List[int]:
        return list(self.dialogs.keys())

    @property
    def split(self):
        return self._split


class DenseAnnotationsReader(object):
    """
    A reader for dense annotations for val split. The json file must have the same structure as mentioned
    on ``https://visualdialog.org/data``.

    Parameters
    ----------
    dense_annotations_jsonpath : str
        Path to a json file containing VisDial v1.0 
    """

    def __init__(self, dense_annotations_jsonpath: str):
        with open(dense_annotations_jsonpath, "r") as visdial_file:
            self._visdial_data = json.load(visdial_file)
            self._image_ids = [entry["image_id"] for entry in self._visdial_data]

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id: int) -> Dict[str, Union[int, List]]:
        index = self._image_ids.index(image_id)
        # keys: {"image_id", "round_id", "gt_relevance"}
        return self._visdial_data[index]

    @property
    def split(self):
        # always
        return "val"

class ImageFeaturesHdfReader(object):
    """
    A reader for HDF files containing pre-extracted image features. A typical HDF file is expected
    to have a column named "image_id", and another column named "features".

    Example of an HDF file:
    ```
    visdial_train_faster_rcnn_bottomup_features.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "features" [shape: (num_images, num_proposals, feature_size)]
       +--- .attrs ("split", "train")
    ```
    Refer ``$PROJECT_ROOT/data/extract_bottomup.py`` script for more details about HDF structure.

    Parameters
    ----------
    features_hdfpath : str
        Path to an HDF file containing VisDial v1.0 train, val or test split image features.
    in_memory : bool
        Whether to load the whole HDF file in memory. Beware, these files are sometimes tens of GBs
        in size. Set this to true if you have sufficient RAM - trade-off between speed and memory.
    """

    def __init__(self, features_hdfpath: str, in_memory: bool = False):
        self.features_hdfpath = features_hdfpath
        self._in_memory = in_memory

        with h5py.File(self.features_hdfpath, "r") as features_hdf:
            self._split = features_hdf.attrs["split"]
            self.image_id_list = list(features_hdf["image_id"])
            # "features" is List[np.ndarray] if the dataset is loaded in-memory
            # If not loaded in memory, then list of None.
            self.features = [None] * len(self.image_id_list)


    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, image_id: int):
        index = self.image_id_list.index(image_id)
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it has a slow start.
            if self.features[index] is not None:
                image_id_features = self.features[index]
            else:
                with h5py.File(self.features_hdfpath, "r") as features_hdf:
                    image_id_features = features_hdf["features"][index]
                    self.features[index] = image_id_features
        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.features_hdfpath, "r") as features_hdf:
                image_id_features = features_hdf["features"][index]
                    
        return image_id_features

    def keys(self) -> List[int]:
        return self.image_id_list

    @property
    def split(self):
        return self._split
