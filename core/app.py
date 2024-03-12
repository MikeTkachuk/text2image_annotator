import json
import random
import time
from collections import namedtuple
from functools import partial
from threading import Thread

from pathlib import Path
from tqdm import tqdm

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from core.embedding import EmbeddingStoreRegistry
from core.task import TaskRegistry
from core.clustering import Clustering
from views import *
from views.utils import Frame, task_creation_popup, BindTk, model_creation_popup
from config import *


def sort_with_ranks(seq, ranks, reverse=True, return_rank=True):
    assert len(seq) == len(ranks)
    cat = list(zip(seq, ranks))
    sorted_cat = sorted(cat, key=lambda x: x[1], reverse=reverse)
    if return_rank:
        return sorted_cat
    return [cat_el[0] for cat_el in sorted_cat]


# todo session info, import, export, help, manage models (models dependent on tasks and embstores),
#

class App:
    def __init__(self, master: BindTk):
        self.sessions_config_path = Path(WORK_DIR) / "sessions.json"
        if not self.sessions_config_path.exists():
            self.sessions_config_path.touch()
            self.full_session_config = {"sessions": {}, "meta": {}}
        else:
            with open(self.sessions_config_path) as session_config_file:
                self.full_session_config = json.load(session_config_file)
        self.session_config = None
        self.task_registry: TaskRegistry = None
        self.embstore_registry: EmbeddingStoreRegistry = None
        self._image_paths = []  # List to store image paths
        self._current_index = 0  # Index of the currently displayed image
        self.master = master

        # processors
        self.clustering: Clustering = None

        # config values
        self.sort_mode = "alphabetic"

        # views
        self.views = namedtuple("Views", [
            "main",
            "clustering",
            "training"
        ])(
            main=MainFrame(self),
            clustering=ClusteringFrame(self),
            training=TrainingFrame(self)
        )
        self.current_view: ViewBase = self.views.main
        self.current_view.render()

        # load
        if self.full_session_config["meta"].get("last_session"):
            try:
                self.select_folder(self.full_session_config["meta"].get("last_session"))
            except Exception as e:
                print("Failed to load previous session: ", e)

    def select_folder(self, image_dir=None):
        if image_dir is None:
            image_dir = filedialog.askdirectory(title="Select Folder")
        image_dir = Path(image_dir)
        self._image_paths = [str(f) for f in image_dir.rglob("*")
                             if f.suffix in [".jpeg", ".jpg", ".png"] and f.stat().st_size]
        self._current_index = 0
        self.create_session_config(image_dir)
        self.current_view.render()

    def switch_to_view(self, view: ViewBase):
        self.master.unbind_all_user()
        self.current_view.frame_master.destroy()
        self.current_view = view
        self.current_view.render()

    def set_project_toolbar(self, master: tk.Menu):
        project = tk.Menu(master, tearoff=0)
        project.add_command(label="Select Folder", command=self.select_folder)
        export = tk.Menu(project, tearoff=0)
        export.add_command(label="Raw json")
        project.add_cascade(menu=export, label="Export As")
        project.add_separator()

        project.add_command(label="Session info", command=self.session_info_popup)
        project.add_command(label="Manage tasks", command=self.manage_tasks_popup)
        project.add_command(label="Manage embedder", command=self.manage_embedder_popup)
        project.add_command(label="Manage models", command=self.manage_models_popup)
        project.add_separator()

        return project

    def set_navigation_toolbar(self, master: tk.Menu):
        navigate = tk.Menu(master, tearoff=0)
        for view_name in self.views._asdict():
            navigate.add_command(state="normal" if self.is_initialized else "disabled", label=view_name,
                                 command=partial(self.switch_to_view, getattr(self.views, view_name)))
        return navigate

    @property
    def is_initialized(self):
        return self.session_config is not None and self._image_paths

    @property
    def sort_mode(self):
        return self._sort_mode

    @sort_mode.setter
    def sort_mode(self, value):
        assert value in self.sort_modes
        self._sort_mode = value

    @property
    def sort_model_names(self):
        if self.embstore_registry is None:
            return []
        return list(self.embstore_registry.stores)

    @property
    def sort_modes(self):
        return ["alphabetic", "popularity"] + self.sort_model_names

    def create_session_config(self, session_dir: Path):
        session_file_name = self.full_session_config["sessions"].get(str(session_dir))

        if session_file_name is None or not Path(session_file_name).exists():
            session_name = session_dir.name
            new_session_config = {
                "target": str(session_dir),
                "name": session_name,
                "total_files": len(self._image_paths),
                "tags": [],
                "tags_structure": {},
                "data": {}
            }
            self.session_config = new_session_config
            self.full_session_config["sessions"][str(session_dir)] = str(self._get_session_dir() / "data.json")
        else:
            with open(session_file_name) as session_file:
                self.session_config = json.load(session_file)

            missing_keys = set(self.session_config["data"]).difference(set(self._image_paths))
            if missing_keys:
                print("Missing: ", missing_keys)
                ans = messagebox.askyesno(
                    message=f"Tried to resume session {session_dir} but missing " \
                            f"{len(missing_keys)} in the folder. See logs for details. "
                            f"Total files mismatch: {self.session_config['total_files'] - len(self._image_paths)} files. "
                            f"Would you like to continue?")
                if not ans:
                    self.session_config = None
                    self._image_paths = []
                    return

        self.embstore_registry = EmbeddingStoreRegistry(self._get_session_dir(), self.get_data_folder())
        for model_name in ["openai/clip-vit-base-patch32",
                           "openai/clip-vit-large-patch14"]:
            self.embstore_registry.add_store(model_name)
        self.task_registry = TaskRegistry(self.session_config["data"], self._image_paths, self.embstore_registry,
                                          save_dir=self._get_session_dir())
        self.clustering = Clustering(self.embstore_registry, self.task_registry)
        self.full_session_config["meta"]["last_session"] = str(session_dir)
        self._load_session_metadata()
        self.save_state()

    def _load_session_metadata(self):
        try:
            self.task_registry.choose_task(self.session_config["metadata"]["last_task"])
        except Exception as e:
            print("While loading task: ", e)
        try:
            self.embstore_registry.choose_store(self.session_config["metadata"]["last_embstore"])
        except Exception as e:
            print("While loading store: ", e)
        try:
            self.sort_mode = self.session_config["metadata"]["last_sort_mode"]
        except Exception as e:
            print("While loading sort_mode: ", e)
        try:
            self._current_index = self.session_config["metadata"]["last_id"]
            self._current_index = min(len(self._image_paths) - 1, max(0, self._current_index))
        except Exception as e:
            print("While loading id: ", e)

    def _get_session_dir(self):
        session_name = self.session_config["name"]
        return Path(WORK_DIR) / f"sessions/{session_name}"

    def get_current_meta(self):
        if not self.is_initialized:
            return {}
        return self.session_config["data"].get(self._image_paths[self._current_index], {})

    def get_data_folder(self):
        if not self.is_initialized:
            return None
        return self.session_config["target"]

    def get_current_image_path(self):
        return self._image_paths[self._current_index]

    def get_active_samples(self, tasks=True):
        samples = set(self.session_config["data"].keys())
        if tasks:
            for task in self.task_registry.tasks.values():
                samples.update(task.user_override.keys())
        return list(samples)

    def get_stats(self):
        return f"Progress: {len(self.session_config['data'])}/{len(self._image_paths)}\n" \
               f"Id: {self._current_index}"

    def save_state(self):
        if not self.sessions_config_path.parent.exists():
            self.sessions_config_path.parent.mkdir(parents=True)
        with open(self.sessions_config_path, "w") as sessions_config_file:
            json.dump(self.full_session_config, sessions_config_file)

        self.capture_session_metadata()
        session_path = Path(self.full_session_config["sessions"][self.session_config["target"]])
        if not session_path.parent.exists():
            session_path.parent.mkdir(parents=True)
        with open(session_path, "w") as session_data:
            json.dump(self.session_config, session_data)

    def make_record(self, prompt, tags):
        if not self.is_initialized:
            return
        self.session_config["data"][self._image_paths[self._current_index]] = {
            "prompt": prompt,
            "tags": tags
        }
        self.save_state()

    def capture_session_metadata(self):
        if not self.is_initialized:
            return
        self.session_config["metadata"] = {
            "last_task": self.task_registry.get_current_task_name(),
            "last_embstore": self.embstore_registry.get_current_store_name(),
            "last_sort_mode": self.sort_mode,
            "last_id": self._current_index
        }

    def show_next_image(self):
        if self.is_initialized:
            self._current_index = (self._current_index + 1) % len(self._image_paths)

    def show_previous_image(self):
        if self.is_initialized:
            self._current_index = (self._current_index - 1) % len(self._image_paths)

    def go_to_next_unlabeled_image(self):
        if not self.is_initialized:
            return
        search_cursor = self._current_index + 1
        while search_cursor < (len(self._image_paths)):
            if self._image_paths[search_cursor] not in self.session_config["data"]:
                self._current_index = search_cursor
                return
            search_cursor += 1

    def go_to_image(self, id_):
        if self.is_initialized:
            if not isinstance(id_, int):
                id_ = self._image_paths.index(id_)
            self._current_index = min(len(self._image_paths) - 1, max(0, id_))

    def update_tag_structure(self, tag: str, new_path: str):
        if self.is_initialized and tag in self.session_config["tags"]:
            new_path = '/' + new_path.strip('/')
            self.session_config["tags_structure"][tag] = new_path
            self.save_state()

    def add_tag(self, tag_name: str, tag_path=""):
        if not self.is_initialized:
            return
        if tag_name not in self.session_config["tags"]:
            self.session_config["tags"].append(tag_name)
        tag_path = '/' + tag_path.strip('/')
        self.session_config["tags_structure"][tag_name] = tag_path
        self.save_state()

    def delete_tag(self, tag_name):
        if self.is_initialized and tag_name in self.session_config["tags"]:
            self.session_config["tags"].remove(tag_name)
            self.session_config["tags_structure"].pop(tag_name)
            self.save_state()

    def search_tags(self, search_key):
        if not self.is_initialized:
            return []
        tags_pool = self.sort_tags()

        tags = []
        paths = []
        scores = []
        for t in tags_pool:
            score = None
            if isinstance(t, tuple):
                t, score = t
            if search_key in t or search_key in self.session_config["tags_structure"].get(t, '/'):
                tags.append(t)
                paths.append(self.session_config["tags_structure"].get(t, '/'))
                scores.append(score)
        return tags, paths, scores

    def sort_tags(self):
        if self.sort_mode == "alphabetic":
            return sorted(self.session_config["tags"])
        if self.sort_mode == "popularity":
            raise NotImplementedError

        ranks = self.emb_store.get_tag_ranks(self.session_config["tags"],
                                             self._image_paths[self._current_index])
        return sort_with_ranks(self.session_config["tags"], ranks)

    @property
    def emb_store(self):
        if not self.is_initialized:
            return None
        return self.embstore_registry.stores.get(self.sort_mode)

    def recompute_recommendation(self):
        if self.emb_store is None:
            return

        for img_path in tqdm(self.session_config["data"], desc="Loading image embeddings"):
            self.emb_store.get_image_embedding(img_path)

        for tag in tqdm(self.session_config["tags"], desc="Loading tag embeddings"):
            self.emb_store.get_tag_embedding(tag)

        # perform training
        # =====

    # popups
    def session_info_popup(self):
        from PIL import Image
        window = tk.Toplevel(self.master)
        window.title("Session status")

        err_paths_frame = Frame(window, name="err_paths_frame", pack=True)
        err_paths_frame.pack()
        err_label = tk.Label(err_paths_frame, text="#Invalid paths: ")
        err_label.pack(side="left")

        def err_compute():
            count = 0
            for path in self._image_paths:
                try:
                    Image.open(path)
                except Exception as e:
                    print(e, path)
                    count += 1
            err_label.config(text=f"#Invalid paths: {count}. See logs for full path")

        ttk.Button(err_paths_frame, text="Compute", command=err_compute).pack(side="left")

    def manage_tasks_popup(self):
        def reload_comboboxes():
            registry = self.task_registry
            registry.validate_selection()
            task_names = list(registry.tasks)
            task_selection.config(values=task_names)
            model_names = list(registry.get_current_models())
            model_selection.config(values=model_names)
            if registry.is_initialized:
                task_selection.current(task_names.index(registry.get_current_task_name()))
                task = registry.get_current_task()
                if task.is_initialized:
                    model_selection.current(
                        model_names.index(task.get_current_model_name()))

        window = tk.Toplevel(self.master, name="manage_tasks")
        window.title("Task settings")
        task_selection_frame = Frame(master=window, name="task_selection_frame")
        task_selection_frame.grid(row=0, column=0, padx=10, pady=10)

        task_selection_var = tk.StringVar()
        task_selection = ttk.Combobox(task_selection_frame, textvariable=task_selection_var,
                                      values=list(self.task_registry.tasks),
                                      )
        task_selection.var = task_selection_var

        task_selection.bind("<<ComboboxSelected>>",
                            lambda x: (self.task_registry.choose_task(task_selection.get()),
                                       reload_comboboxes())
                            )

        task_selection.grid(row=0, column=0)
        add_task_button = ttk.Button(task_selection_frame,
                                     text="Add task",
                                     command=lambda: task_creation_popup(self, reload_comboboxes))
        add_task_button.grid(row=0, column=1)

        def confirm_delete(delete_func):
            if messagebox.askokcancel(title="Confirm deletion", message="Delete?"):
                delete_func()
                reload_comboboxes()

        delete_task_button = ttk.Button(task_selection_frame,
                                        text="Delete task",
                                        command=lambda: confirm_delete(self.task_registry.delete_task))
        delete_task_button.grid(row=0, column=2)

        model_selection_var = tk.StringVar()
        model_selection = ttk.Combobox(task_selection_frame, textvariable=model_selection_var,
                                       values=list(self.task_registry.get_current_models()))
        model_selection.var = model_selection_var
        model_selection.bind("<<ComboboxSelected>>",
                             lambda x: self.task_registry.get_current_task()
                             .choose_model(model_selection_var.get()))
        model_selection.grid(row=1, column=0)
        add_model_button = ttk.Button(task_selection_frame, text="Add model",
                                      command=model_creation_popup(self, reload_comboboxes))
        add_model_button.grid(row=1, column=1)
        delete_model_button = ttk.Button(task_selection_frame,
                                         text="Delete model",
                                         command=lambda: confirm_delete(self.task_registry.delete_model))
        delete_model_button.grid(row=1, column=2)
        reload_comboboxes()

    def manage_embedder_popup(self):
        window = tk.Toplevel(self.master, name="manage_embedder")
        window.title("Embedder settings")

        emb_info_frame = Frame(window, name="emb_info_frame", pack=True)
        emb_info_frame.pack(padx=20)

        embedder_names = list(self.embstore_registry.stores)
        embedder_selection_var = tk.StringVar()
        embedder_selection = ttk.Combobox(emb_info_frame, textvariable=embedder_selection_var,
                                          values=embedder_names)
        embedder_selection.bind("<<ComboboxSelected>>",
                                lambda x: (self.embstore_registry.choose_store(
                                    embedder_selection_var.get()),
                                           reload_status()))
        embedder_selection.current(embedder_names.index(self.embstore_registry.get_current_store_name()))
        embedder_selection.pack(side="left")

        def _add_by_name():
            try:
                self.embstore_registry.add_store(
                    simpledialog.askstring(prompt="Huggingface model name: ", title="Add model")
                )
                error_label.config(text="")
                error_label.update()
                reload_status()
            except RuntimeError as e:
                error_label.config(text="This model already exists")
                error_label.update()

        add_by_name = ttk.Button(emb_info_frame, text="Add model",
                                 command=_add_by_name
                                 )
        add_by_name.pack(side="left")

        def _add_by_path():
            try:
                self.embstore_registry.add_store(
                    filedialog.askdirectory(title="Select path to model"),
                    store_name=simpledialog.askstring(prompt="Store name: ", title="")
                )
                error_label.config(text="")
                error_label.update()
                reload_status()
            except RuntimeError as e:
                error_label.config(text="This task already exists")
                error_label.update()

        add_by_path = ttk.Button(emb_info_frame, text="Browse",
                                 command=_add_by_path
                                 )
        add_by_path.pack(side="left")

        precompute_frame = Frame(window, name="precompute_frame", pack=True)
        precompute_frame.pack(pady=20, padx=20)
        info_label = tk.Label(precompute_frame, text="")
        info_label.pack()
        precompute_count_var = tk.StringVar()
        tk.Label(precompute_frame, text="#Img to precompute").pack(side="left")
        precompute_count = ttk.Entry(precompute_frame, textvariable=precompute_count_var, width=10)
        precompute_count.pack(side="left")
        do_shuffle_var = tk.BooleanVar()
        tk.Label(precompute_frame, text="Shuffle:").pack(side="left")
        do_shuffle = ttk.Checkbutton(precompute_frame, variable=do_shuffle_var)
        do_shuffle.pack(side="left")

        batch_size_frame = Frame(window, name="batch_size_frame", pack=True)
        batch_size_frame.pack()
        tk.Label(batch_size_frame, text="Batch size:").pack(side="left")
        batch_size_var = tk.StringVar()
        batch_size_var.set("1")
        batch_size_entry = ttk.Entry(batch_size_frame, textvariable=batch_size_var)
        batch_size_entry.pack(side="left")

        def get_precomputed_status():
            store = self.embstore_registry.get_current_store()
            statuses = []
            for s in self._image_paths:
                statuses.append(store.embedding_exists(s))
            return statuses

        def reload_status():
            embedder_selection.current(embedder_names.index(self.embstore_registry.get_current_store_name()))
            statuses = get_precomputed_status()
            info_label.config(text=f" Precomputed: {sum(statuses)}/{len(self._image_paths)}")
            info_label.update()

        def run_precomputing(sample_pool=None):
            store = self.embstore_registry.get_current_store()
            if sample_pool is None:
                selected = []
                max_selected = int(precompute_count_var.get())
                samples = list(self._image_paths)
                if do_shuffle_var.get():
                    random.shuffle(samples)
                for sample in samples:
                    if not store.embedding_exists(sample):
                        selected.append(sample)
                    if len(selected) >= max_selected:
                        break
            else:
                selected = sample_pool

            def callback(i):
                try:  # try if not closed
                    message_label.config(text=f"Progress: {i}/{len(selected)}")
                    message_label.update()
                except:
                    pass
                # time.sleep(1)

            def target():
                count = store.precompute(selected, callback, batch_size=int(batch_size_var.get()))
                try:  # try if not closed
                    message_label.config(text=f"#Errors: {len(selected) - count}")
                    message_label.update()
                    reload_status()
                except:
                    pass

            t = Thread(target=target)
            t.start()

        run_button = ttk.Button(batch_size_frame, text="Run", command=run_precomputing)
        run_button.pack(side="left")

        run_active_frame = Frame(window, name="run_active_frame", pack=True)
        run_active_frame.pack(pady=10)
        tk.Label(run_active_frame, text="Include tasks:").pack(side="left")
        include_tasks_var = tk.BooleanVar()
        include_tasks_var.set(True)
        include_tasks = ttk.Checkbutton(run_active_frame, variable=include_tasks_var)
        include_tasks.pack(side="left")
        run_active_button = ttk.Button(run_active_frame, text="Precompute active",
                                       command=lambda: run_precomputing(
                                           self.get_active_samples(include_tasks_var.get())
                                       )
                                       )
        run_active_button.pack(side="left")
        reload_status()

        message_label = tk.Label(window, text="")
        message_label.pack()

        error_label = tk.Label(window, text="")
        error_label.pack()

        def closure(event=None):
            try:
                pass
            except RuntimeError as e:
                return
            window.destroy()

        exit_button = ttk.Button(window, text="Ok", command=closure)
        exit_button.pack(side="bottom", pady=20)

    def manage_models_popup(self):
        def reload():
            registry = self.task_registry
            registry.validate_selection()
            task_names = list(registry.tasks)
            task_selection.config(values=task_names)
            model_names = list(registry.get_current_models(emb_filter=False))

            if registry.is_initialized:
                task_selection.current(task_names.index(registry.get_current_task_name()))
            task = registry.get_current_task()
            for item in model_selection.get_children():
                model_selection.delete(item)

            for model_name in model_names:
                model = task.models[model_name]
                values = (model_name, str(model.model).split(".")[-1].strip(">'"),
                          str(model.params), model.embstore_name, model.framework)
                model_selection.insert("", "end", values[0], text=values[0],
                                       values=values[1:])

        window = tk.Toplevel(self.master, name="manage_tasks")
        window.title("Task settings")
        task_selection_frame = Frame(master=window, name="task_selection_frame")
        task_selection_frame.grid(row=0, column=0, padx=10, pady=10)

        task_selection_var = tk.StringVar()
        task_selection = ttk.Combobox(task_selection_frame, textvariable=task_selection_var,
                                      values=list(self.task_registry.tasks),
                                      )
        task_selection.var = task_selection_var

        task_selection.bind("<<ComboboxSelected>>",
                            lambda x: (self.task_registry.choose_task(task_selection.get()),
                                       reload())
                            )

        task_selection.grid(row=0, column=0, sticky="w")

        columns = ["#0", "Type", "Params", "Embstore", "Framework"]
        model_selection = ttk.Treeview(task_selection_frame, columns=columns[1:])
        column_names = ["Name", "Type", "Params", "Embstore", "Framework"]
        widths = [100, 100, 350, 100, 100]
        for c, cn, w in zip(columns, column_names, widths):
            model_selection.column(c, width=w)
            model_selection.heading(c, text=cn)
        model_selection.grid(row=1, column=0, columnspan=2)

        def confirm_delete(delete_func):
            if messagebox.askokcancel(title="Confirm deletion", message="Delete?"):
                delete_func()
                reload()

        def delete_model():
            if not model_selection.selection():
                return
            if self.task_registry.is_initialized:
                self.task_registry.get_current_task().delete_model(
                    model_selection.selection()[0]
                )

        delete_model_button = ttk.Button(task_selection_frame,
                                         text="Delete model",
                                         command=lambda: confirm_delete(delete_model)
                                         )
        delete_model_button.grid(row=0, column=1, sticky="e")
        reload()
