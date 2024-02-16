import json
from collections import namedtuple
from functools import partial

from pathlib import Path
from tqdm import tqdm

import tkinter as tk
from tkinter import filedialog, messagebox

from tag_recommendation import EmbeddingStore
from views import *
from config import *


def sort_with_ranks(seq, ranks, reverse=True, return_rank=True):
    assert len(seq) == len(ranks)
    cat = list(zip(seq, ranks))
    sorted_cat = sorted(cat, key=lambda x: x[1], reverse=reverse)
    if return_rank:
        return sorted_cat
    return [cat_el[0] for cat_el in sorted_cat]


class App:
    def __init__(self, master):
        self.sessions_config_path = Path(WORK_DIR) / "sessions.json"
        if not self.sessions_config_path.exists():
            self.sessions_config_path.touch()
            self.full_session_config = {}
        else:
            with open(self.sessions_config_path) as session_config_file:
                self.full_session_config = json.load(session_config_file)
        self.session_config = None
        self.emb_store: EmbeddingStore = None
        self._image_paths = []  # List to store image paths
        self._current_index = 0  # Index of the currently displayed image
        self.master = master

        # config values
        self.sort_modes = ["alphabetic",
                           "popularity",
                           "openai/clip-vit-base-patch32",
                           "openai/clip-vit-large-patch14"]
        self.sort_models_names = self.sort_modes[2:]
        self.sort_mode = "alphabetic"

        # views
        self.views = namedtuple("Views", [
            "main",
            "clustering",
            "training"
        ])(main=MainFrame(self), clustering=ClusteringFrame(self), training=TrainingFrame(self))
        self.current_view: ViewBase = self.views.main
        self.current_view.render()

    def select_folder(self):
        image_dir = Path(filedialog.askdirectory(title="Select Folder"))
        self._image_paths = [str(f) for f in image_dir.rglob("*") if f.suffix in [".jpeg", ".jpg", ".png"]]
        self.create_session_config(image_dir)
        self._current_index = 0
        self.current_view.render()

    def switch_to_view(self, view: ViewBase):
        self.current_view.frame_master.destroy()
        self.current_view = view
        self.current_view.render()

    def set_project_toolbar(self, master: tk.Menu):
        project = tk.Menu(master, tearoff=0)
        project.add_command(label="Select Folder", command=self.select_folder)
        export = tk.Menu(project, tearoff=0)
        export.add_command(label="Raw json")
        project.add_cascade(menu=export, label="Export As")
        return project

    def set_navigation_toolbar(self, master: tk.Menu):
        navigate = tk.Menu(master, tearoff=0)
        for view_name in self.views._asdict():
            navigate.add_command(label=view_name,
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

    def create_session_config(self, session_dir: Path):
        session_file_name = self.full_session_config.get(str(session_dir))

        if session_file_name is None:
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
            self.full_session_config[str(session_dir)] = str(self._get_session_dir() / "data.json")
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
        self.save_state()

    def _get_session_dir(self):
        session_name = self.session_config["name"]
        return Path(WORK_DIR) / f"sessions/{session_name}"

    def get_current_meta(self):
        if not self.is_initialized:
            return {}
        return self.session_config["data"].get(self._image_paths[self._current_index], {})

    def get_current_image_path(self):
        return self._image_paths[self._current_index]

    def get_stats(self):
        return f"Progress: {len(self.session_config['data'])}/{len(self._image_paths)}\n" \
               f"Id: {self._current_index}"

    def save_state(self):
        if not self.sessions_config_path.parent.exists():
            self.sessions_config_path.parent.mkdir(parents=True)
        with open(self.sessions_config_path, "w") as sessions_config_file:
            json.dump(self.full_session_config, sessions_config_file)

        session_path = Path(self.full_session_config[self.session_config["target"]])
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

        if self.emb_store is None or self.sort_mode != self.emb_store.model_name:
            self.recompute_recommendation()
        ranks = self.emb_store.get_tag_ranks(self.session_config["tags"],
                                             self._image_paths[self._current_index],
                                             self.session_config["target"])
        return sort_with_ranks(self.session_config["tags"], ranks)

    def recompute_recommendation(self):
        if not self.is_initialized:
            return
        if self.sort_mode not in self.sort_models_names:
            return

        if self.emb_store is None or self.sort_mode != self.emb_store.model_name:
            self.emb_store = EmbeddingStore(self._get_session_dir() / ".emb_store",
                                            model_name=self.sort_mode)

        for img_path in tqdm(self.session_config["data"], desc="Loading image embeddings"):
            self.emb_store.get_image_embedding(img_path, self.session_config["target"])

        for tag in tqdm(self.session_config["tags"], desc="Loading tag embeddings"):
            self.emb_store.get_tag_embedding(tag)

        # perform training
        # =====
