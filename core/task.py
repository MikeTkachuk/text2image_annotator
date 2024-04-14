import json
import shutil
from pathlib import Path
from typing import Literal, Dict, get_args, Optional

from sklearn.model_selection import train_test_split

from core.model import Model
from core.embedding import EmbeddingStoreRegistry

TASK_MULTICLASS_MODES = Literal["empty_valid", "empty_invalid"]


class Task:
    """Entity representing binary or multi-class classification.
    It controls the annotations, data split, and models"""
    def __init__(self, categories, task_path, multiclass_mode: TASK_MULTICLASS_MODES = "empty_valid"):
        self.multiclass_mode: TASK_MULTICLASS_MODES = multiclass_mode
        self.path = Path(task_path)
        self.categories = categories
        if isinstance(self.categories, str):
            self.categories = [self.categories]
        self.labels = {}
        self.user_override = {}
        self.models: Dict[str, Model] = {}
        self.description = ""
        self.validation_samples = []

        self._current_model: str = None

    @property
    def is_initialized(self):
        """An initialized task should have an active model selected"""
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
        """Expands categories based on task multiclass mode"""
        if self.multiclass_mode == "empty_valid" or len(self.categories) == 1:
            return ["Empty"] + self.categories
        return self.categories

    def update_labels(self, samples, labelling):
        """
        |  Will be called lazily, computes the latest version of task labels.
        |  Glossary:
        |       not labelled by user    -1
        |       invalid for this task   -5
        |       categories              0-n, if empty_valid, 0 is empty class
        |  Invalid labels are missing labels with multiclass_mode=="empty_invalid" and
        |  not single-class annotations
        :param samples: list of file paths
        :param labelling: session data of form {sample_path: {"tags":[...], ...}}
        """
        self.labels = {}
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
            self.labels[sample] = label

    def override_label(self, sample: str, label: int):
        """Annotates the sample with a task-specific class"""
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
        """Sets new validation split"""
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
        """Loads an instance of task and all of its models"""
        with open(path) as file:
            data = json.load(file)
        task = Task(data["categories"], path, data["multiclass_mode"])
        task.user_override = data["user_override"]
        task.description = data.get("description", "")
        task.validation_samples = data.get("validation_samples", [])
        for model in data["models"]:
            task.models[model] = Model.load(data["models"][model])
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
    """Controls validity and serialization of all tasks in a session"""
    def __init__(self, dataset, all_samples, embstore_registry: EmbeddingStoreRegistry, save_dir):
        self._dataset = dataset
        self._all_samples = all_samples
        self.save_dir = Path(save_dir)
        self.embstore_registry = embstore_registry

        self.tasks: Dict[str, Task] = {}
        self.model_templates: Dict[str, dict] = {}
        self._current_task: str = None
        self.load_state()

    @property
    def is_initialized(self):
        """An initialized task registry should have an active task selected"""
        return self._current_task is not None and len(self.tasks)

    def save_state(self, light=True):
        """Saves task locations with an option to recursively save all the tasks"""
        tasks = {task_name: str(task.path) for task_name, task in self.tasks.items()}
        out = {"tasks": tasks, "model_templates": self.model_templates}
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
            tasks = data["tasks"]
            for task_name in tasks:
                self.tasks[task_name] = Task.load(tasks[task_name])
            self.model_templates = data["model_templates"]
            self.model_templates.update(Model.get_default_templates())

    @staticmethod
    def get_task_modes():
        """Shortcut to get possible multiclass modes"""
        return get_args(TASK_MULTICLASS_MODES)

    def get_current_task(self) -> Optional[Task]:
        if self.is_initialized:
            return self.tasks[self._current_task]
        return None

    def get_current_task_name(self) -> Optional[str]:
        return self._current_task

    def get_current_models(self, emb_filter=True) -> Dict[str, Model]:
        """Returns all models of the current task with an option to
         filter the ones trained with current embedding store"""
        if not self.is_initialized:
            return {}
        models = self.tasks[self._current_task].models
        if emb_filter:
            return {k: v for k, v in models.items() if
                    self.embstore_registry.get_current_store_name() == v.embstore_name}
        return models

    def validate_selection(self):
        """Confirms that the selected model of the selected task matches the current embedding store.
        Otherwise, deinitializes the task (no model selected)"""
        if self.is_initialized:
            task = self.get_current_task()
            if task.is_initialized:
                if (task.get_current_model().embstore_name
                        != self.embstore_registry.get_current_store_name()):
                    self.get_current_task().choose_model()

    def get_tags_in_use(self):
        """Returns a set of tags (classes) that are being used by all the tasks"""
        used_tags = set()
        for task in self.tasks.values():
            used_tags.update(task.categories)
        return used_tags

    def add_task(self, tags, mode: TASK_MULTICLASS_MODES = "empty_valid"):
        """Handles task creation, validation, and naming"""
        if not tags or mode not in self.get_task_modes():
            raise RuntimeError(f"Invalid tags {tags} or mode {mode}")
        tags = sorted(tags)
        used_tags = self.get_tags_in_use()
        for tag in tags:  # can't create overlapping tasks for it will be impossible to merge
            if tag in used_tags:
                raise RuntimeError(f"Tag \"{tag}\" already used")

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

    def add_template(self, template_name: str, model_name=None):
        """Adds new template from current model"""
        if not self.is_initialized:
            raise RuntimeError("Can not add template. Task is not selected")
        if model_name is None and not self.get_current_task().is_initialized:
            raise RuntimeError("Can not add template. Model is not selected")
        if model_name is not None:
            model = self.get_current_task().models[model_name]
        else:
            model = self.get_current_model()
        self.model_templates[template_name] = model.to_template()
        self.save_state()

    def add_model(self, model_name: str, template_name: str):
        """Adds model to the current task based on model name and model template (see Model docs)"""
        if not self.is_initialized:
            raise RuntimeError("Can not add model. Task is not selected")
        if not self.embstore_registry.is_initialized:
            raise RuntimeError("Can not add model. Embedding store is not selected")
        if template_name not in self.model_templates:
            raise ValueError(f"Can not add model. Invalid template name: {template_name}")
        model = Model.from_template(self.model_templates[template_name],
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
        """Generates a new random validation split for the current task
        :param test_size - numeric str, int or float. if text_size > 1 will be treated as exact number of test samples
        :param stratified - if True tries to preserve the class balance in test split
        """
        if isinstance(test_size, str):
            test_size = float(test_size)
            if test_size > 1:
                test_size = int(test_size)
        task = self.get_current_task()
        task.update_labels(self._all_samples, self._dataset)
        zipped = [(s, l) for s, l in task.labels.items() if l >= 0]
        samples, labels = list(zip(*zipped))
        _, val_split = train_test_split(samples, test_size=test_size, stratify=labels if stratified else None)
        task.update_split(val_split)

    def fit_current_model(self, callback=None, kfold=None, use_augs=True):
        """
        Runs training of current model based on current task data.
        :param callback: str -> None for logging purposes
        :param kfold: None or int > 2, cross-validation
        :param use_augs: if True adds uses embeddings from augmented images
        :return: metrics dict
        """
        if callback is None:
            callback = print
        callback("Starting dataset init...")
        task = self.get_current_task()
        model = self.get_current_model()
        assert model.embstore_name == self.embstore_registry.get_current_store_name()
        dataset = self.get_current_labels()
        dataset = {k: v for k, v in dataset.items() if self.embstore_registry.get_current_store().embedding_exists(k)}
        callback("Finished dataset init")
        try:
            res = model.fit(dataset, self.embstore_registry.get_current_store(),
                            test_split=task.validation_samples, callback=callback,
                            kfold=kfold, use_augs=use_augs)
            return res
        except Exception as e:
            import traceback
            callback(f"Error: {''.join(traceback.format_exception(e))}")

    def choose_task(self, task_name=None):
        assert task_name in self.tasks or task_name is None
        self._current_task = task_name
        if self.is_initialized:
            self.tasks[self._current_task].update_labels(self._all_samples, self._dataset)
        print("task: ", self._current_task)

    def update_current_task(self):
        """If a task is active, recomputes task labels with the latest data"""
        if self._current_task is None:
            return
        self.tasks[self._current_task].update_labels(self._all_samples, self._dataset)

    def get_current_labels(self):
        if not self.is_initialized:
            return None
        return self.get_current_task().labels

    def delete_task(self):
        if self._current_task is None:
            return
        self._get_task_path(self._current_task).unlink()
        shutil.rmtree(self._get_model_path(self._current_task, "dummy").parent)
        self.tasks.pop(self._current_task)
        self._current_task = None
        self.save_state()

