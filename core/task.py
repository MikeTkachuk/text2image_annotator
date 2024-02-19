import json
from pathlib import Path
from typing import Literal, Dict, get_args

from core.model import Model

TASK_MULTICLASS_MODES = Literal["empty_valid", "empty_invalid"]


class Task:
    def __init__(self, categories, multiclass_mode: TASK_MULTICLASS_MODES = "empty_valid"):
        self.multiclass_mode: TASK_MULTICLASS_MODES = multiclass_mode
        self.categories = categories
        if isinstance(self.categories, str):
            self.categories = [self.categories]
        self.labels = []
        self.models = {}
        self._current_model: str = None

    @property
    def is_initialized(self):
        return self._current_model is not None and len(self.models)

    def get_current_model_name(self):
        return self._current_model

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
            if sample in labelling:
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

    def add_model(self, model: Model, model_name: str):
        if model_name in self.models:
            return False
        self.models[model_name] = {"model": model, "predictions": None}

    def __dict__(self):
        out = {"models": {}, "multiclass_mode": self.multiclass_mode, "categories": self.categories}
        for model_name in self.models:
            model_dict = self.models[model_name]["model"].__dict__()
            out["models"][model_name] = model_dict
        return out

    @classmethod
    def load(cls, data: dict):
        task = Task(data["categories"], data["multiclass_mode"])
        for model in data["models"]:
            task.add_model(Model.load(data["models"][model]), model)
        return task

    def choose_model(self, model_name):
        assert model_name in self.models
        self._current_model = model_name
        print("model: ", self._current_model)


class TaskRegistry:
    def __init__(self, dataset, all_samples, save_dir):
        self._dataset = dataset
        self._all_samples = all_samples
        self.save_dir = Path(save_dir)

        self.tasks: Dict[str, Task] = {}
        self._current_task: str = None
        self.load_state()

    @property
    def is_initialized(self):
        return self._current_task is not None and len(self.tasks)

    def save_state(self):
        out = {task_name: task.__dict__() for task_name, task in self.tasks.items()}
        with open(self.save_dir / "task_registry.json", "w") as file:
            json.dump(out, file)

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

    def get_current_models(self):
        if not self.is_initialized:
            return {}
        return self.tasks[self._current_task].models

    def add_task(self, tags, mode: TASK_MULTICLASS_MODES = "empty_valid"):
        if not tags or mode not in self.get_task_modes():
            raise RuntimeError(f"Invalid tags {tags} or mode {mode}")
        tags = sorted(tags)
        task_name = "_".join(tags)
        if mode == "empty_valid":
            task_name += "_empty"
        if task_name in self.tasks:
            return False
        self.tasks[task_name] = Task(tags, multiclass_mode=mode)
        self.save_state()
        self.choose_task(task_name)
        return task_name

    def get_model_path(self, task_name: str, model_name: str):
        for ch in " /\\|,.?!@#$%^&*;:'\"+=":
            task_name = task_name.replace(ch, "_")
            model_name = model_name.replace(ch, "_")
        return self.save_dir / ".model_store" / task_name / f"{model_name}.pkl"

    def add_model(self, model_name, model, params, framework):
        if not self.is_initialized:
            raise RuntimeError("Can not add model. Task is not selected")
        model_obj = Model(model, params, framework, self.get_model_path(self._current_task, model_name))
        self.tasks[self._current_task].add_model(model_obj, model_name)
        self.save_state()

    def choose_task(self, task_name):
        assert task_name in self.tasks
        self._current_task = task_name
        self.tasks[self._current_task].update_labels(self._all_samples, self._dataset)
        print("task: ", self._current_task)

    def get_samples(self):
        return self._all_samples
