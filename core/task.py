import json
import shutil
from pathlib import Path
from typing import Literal, Dict, get_args

from sklearn.model_selection import train_test_split

from core.model import Model
from core.embedding import EmbeddingStoreRegistry

TASK_MULTICLASS_MODES = Literal["empty_valid", "empty_invalid"]


# todo: optimize labels to be a (maybe ordered) dict
class Task:
    def __init__(self, categories, task_path, multiclass_mode: TASK_MULTICLASS_MODES = "empty_valid"):
        self.multiclass_mode: TASK_MULTICLASS_MODES = multiclass_mode
        self.path = Path(task_path)
        self.categories = categories
        if isinstance(self.categories, str):
            self.categories = [self.categories]
        self.labels = []
        self.user_override = {}
        self.models: Dict[str, Model] = {}
        self.description = ""
        self.validation_samples = []

        self._current_model: str = None

    @property
    def is_initialized(self):
        return self._current_model is not None and len(self.models)

    def get_current_model_name(self):
        return self._current_model

    def get_current_model(self):
        if not self.is_initialized:
            return None
        return self.models[self._current_model]

    def label_name(self, value: int):
        if value == -5:
            return "Invalid"
        if value == -1:
            return "Not labeled"

        return self.categories_full[value]

    def label_encode(self, name: str):
        if name == "Invalid":
            return -5
        if name == "Not labeled":
            return -1
        return self.categories_full.index(name)

    @property
    def categories_full(self):
        if self.multiclass_mode == "empty_valid" or len(self.categories) == 1:
            return ["Empty"] + self.categories
        return self.categories

    def labels_dict(self, samples):
        return dict(zip(samples, self.labels))

    def update_labels(self, samples, labelling):
        """
        |  Will be called lazily
        |  Glossary:
        |       not labelled by user    -1
        |       invalid for this task   -5
        |       categories              0-n, if empty_valid, 0 is empty class
        |  Invalid labels are missing labels with multiclass_mode=="empty_invalid" and
        |  not single-class annotations
        :param samples: list of file paths
        :param labelling: session data of form {sample_path: {"tags":[...], ...}}
        """
        self.labels = []
        empty_is_category = len(self.categories) == 1 or self.multiclass_mode == "empty_valid"
        for sample in samples:
            if sample in self.user_override:
                label = self.user_override[sample]
            elif sample in labelling:
                categories = [c in labelling[sample]["tags"] for c in self.categories]
                if sum(categories) > 1:
                    label = -5
                else:
                    if empty_is_category:
                        if any(categories):
                            label = categories.index(True) + 1
                        else:
                            label = 0
                    else:
                        if any(categories):
                            label = categories.index(True)
                        else:
                            label = -5
            else:
                label = -1
            self.labels.append(label)

    def override_label(self, sample: str, label: int):
        if label == -1:
            try:
                self.user_override.pop(sample)
            except KeyError:
                pass
            return
        try:
            self.label_name(label)
        except IndexError:
            raise RuntimeError("Can not override with invalid label")
        self.user_override[sample] = label
        self.save_state()

    def update_description(self, description: str):
        self.description = description
        self.save_state()

    def update_split(self, validation_split: list):
        self.validation_samples = validation_split
        self.save_state()

    def add_model(self, model: Model, model_name: str):
        if model_name in self.models:
            return False
        self.models[model_name] = model
        model.save()
        self.save_state()

    def __dict__(self):
        out = {"models": {},
               "multiclass_mode": self.multiclass_mode,
               "categories": self.categories,
               "user_override": self.user_override,
               "description": self.description,
               "validation_samples": self.validation_samples
               }
        for model_name, model in self.models.items():
            out["models"][model_name] = model.__dict__()
        return out

    def save_state(self):
        out = {"models": {},
               "multiclass_mode": self.multiclass_mode,
               "categories": self.categories,
               "user_override": self.user_override,
               "description": self.description,
               "validation_samples": self.validation_samples
               }
        for model_name in self.models:
            out["models"][model_name] = str(self.models[model_name].save_path)

        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True)
        with open(self.path, "w") as file:
            json.dump(out, file)

    @classmethod
    def load(cls, path):
        with open(path) as file:
            data = json.load(file)
        task = Task(data["categories"], path, data["multiclass_mode"])
        task.user_override = data["user_override"]
        task.description = data.get("description", "")
        task.validation_samples = data.get("validation_samples", [])
        for model in data["models"]:
            task.add_model(Model.load(data["models"][model]), model)
        return task

    def choose_model(self, model_name=None):
        assert model_name in self.models or model_name is None
        self._current_model = model_name
        print("model: ", self._current_model)

    def delete_model(self, model_name=None):
        if model_name is None:
            model_name = self._current_model
        if model_name is None:
            return
        model = self.models[model_name]
        shutil.rmtree(model.save_path)
        self.models.pop(model_name)
        if model_name == self._current_model:
            self._current_model = None
        self.save_state()


class TaskRegistry:
    def __init__(self, dataset, all_samples, embstore_registry: EmbeddingStoreRegistry, save_dir):
        self._dataset = dataset
        self._all_samples = all_samples
        self.save_dir = Path(save_dir)
        self.embstore_registry = embstore_registry

        self.tasks: Dict[str, Task] = {}
        self._current_task: str = None
        self.load_state()

    @property
    def is_initialized(self):
        return self._current_task is not None and len(self.tasks)

    def save_state(self, light=True):
        out = {task_name: str(task.path) for task_name, task in self.tasks.items()}
        with open(self.save_dir / "task_registry.json", "w") as file:
            json.dump(out, file)
        if not light:
            for t in self.tasks.values():
                t.save_state()
        elif self.is_initialized:
            self.tasks[self._current_task].save_state()

    def load_state(self):
        if (self.save_dir / "task_registry.json").exists():
            with open(self.save_dir / "task_registry.json") as file:
                data = json.load(file)
            for task_name in data:
                self.tasks[task_name] = Task.load(data[task_name])

    def get_task_modes(self):
        return get_args(TASK_MULTICLASS_MODES)

    def get_current_task(self):
        if self.is_initialized:
            return self.tasks[self._current_task]
        return None

    def get_current_task_name(self):
        return self._current_task

    def get_current_models(self, emb_filter=True):
        if not self.is_initialized:
            return {}
        models = self.tasks[self._current_task].models
        if emb_filter:
            return {k: v for k, v in models.items() if
                    self.embstore_registry.get_current_store_name() == v.embstore_name}
        return models

    def validate_selection(self):
        if self.is_initialized:
            task = self.get_current_task()
            if task.is_initialized:
                if (task.get_current_model().embstore_name
                        != self.embstore_registry.get_current_store_name()):
                    self.get_current_task().choose_model()

    def add_task(self, tags, mode: TASK_MULTICLASS_MODES = "empty_valid"):
        if not tags or mode not in self.get_task_modes():
            raise RuntimeError(f"Invalid tags {tags} or mode {mode}")
        tags = sorted(tags)
        for tag in tags:  # can't create overlapping tasks for it will be impossible to merge
            for task_name, task in self.tasks.items():
                if tag in task.categories:
                    raise RuntimeError(f"Tag {tag} already used in {task_name}")

        task_name = "_".join(tags)
        if mode == "empty_valid" or len(tags) == 1:
            task_name += "_empty"
        if task_name in self.tasks:
            raise RuntimeError("This task already exists")
        task_path = self._get_task_path(task_name)
        self.tasks[task_name] = Task(tags, task_path, multiclass_mode=mode)
        self.choose_task(task_name)
        self.save_state()
        return task_name

    def _get_model_path(self, task_name: str, model_name: str):
        for ch in " /\\|,.?!@#$%^&*;:'\"+=":
            task_name = task_name.replace(ch, "_")
            model_name = model_name.replace(ch, "_")
        return self.save_dir / ".model_store" / task_name / model_name

    def _get_task_path(self, task_name: str):
        for ch in " /\\|,.?!@#$%^&*;:'\"+=":
            task_name = task_name.replace(ch, "_")
        return self.save_dir / ".task_store" / f"{task_name}.json"

    def add_model(self, model_name: str, template_name: str):
        if not self.is_initialized:
            raise RuntimeError("Can not add model. Task is not selected")
        if not self.embstore_registry.is_initialized:
            raise RuntimeError("Can not add model. Embedding store is not selected")
        model = Model.from_template(template_name,
                                    self.embstore_registry.get_current_store_name(),
                                    self._get_model_path(self._current_task, model_name))
        self.tasks[self._current_task].add_model(model, model_name)
        self.tasks[self._current_task].choose_model(model_name)

    def delete_model(self):
        if not self.is_initialized:
            return
        self.tasks[self._current_task].delete_model()

    def get_current_model(self):
        if not self.is_initialized:
            return None
        model = self.get_current_task().get_current_model()
        return model

    def generate_split_for_task(self, test_size=0.2, stratified=True):
        if isinstance(test_size, str):
            test_size = float(test_size)
            if test_size > 1:
                test_size = int(test_size)
        task = self.get_current_task()
        task.update_labels(self._all_samples, self._dataset)
        zipped = [(s, l) for s, l in task.labels_dict(self._all_samples).items() if l >= 0]
        samples, labels = list(zip(*zipped))
        _, val_split = train_test_split(samples, test_size=test_size, stratify=labels if stratified else None)
        task.update_split(val_split)

    def fit_current_model(self, callback=None):
        """

        :param callback: str -> None for logging purposes
        :return: metrics dict
        """
        if callback is None:
            callback = print
        callback("Starting dataset init...")
        task = self.get_current_task()
        model = self.get_current_model()
        assert model.embstore_name == self.embstore_registry.get_current_store_name()
        dataset = {}
        for sample, label in self.get_current_labels(as_dict=True).items():
            emb = self.embstore_registry.get_current_store().get_image_embedding(sample, load_only=True)
            if emb is not None:
                dataset[sample] = (emb.cpu().numpy(), label)
        callback("Finished dataset init")
        try:
            res = model.fit(dataset, test_split=task.validation_samples, callback=callback)
        except Exception as e:
            callback(str(e))
        return res

    def choose_task(self, task_name=None):
        assert task_name in self.tasks or task_name is None
        self._current_task = task_name
        if self.is_initialized:
            self.tasks[self._current_task].update_labels(self._all_samples, self._dataset)
        print("task: ", self._current_task)

    def update_current_task(self):
        if self._current_task is None:
            return
        self.tasks[self._current_task].update_labels(self._all_samples, self._dataset)

    def get_current_labels(self, as_dict=False):
        if not self.is_initialized:
            return None
        if as_dict:
            return self.get_current_task().labels_dict(self._all_samples)
        return self.get_current_task().labels

    def delete_task(self):
        if self._current_task is None:
            return
        self._get_task_path(self._current_task).unlink()
        shutil.rmtree(self._get_model_path(self._current_task, "dummy").parent)
        self.tasks.pop(self._current_task)
        self._current_task = None
        self.save_state()

    def get_samples(self):
        return self._all_samples
