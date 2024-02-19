from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.app import App

import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

from views.view_base import ViewBase
from views.utils import Frame, task_creation_popup
from core.clustering import cluster
from config import *


# todo task - tags + models
#  add/choose tag task, save this task separately, choose clustering method,
#  parameters, show result, interactive viewer of samples on the map,

class ClusteringFrame(ViewBase):
    def __init__(self, app: App):
        self.app = app
        self.master = self.app.master

    def render(self):
        self._main_setup()

    def get_tools(self, master):
        tools = tk.Menu(master, tearoff=0)
        return tools

    def _main_setup(self):
        self.app.master.title("Clustering")

        self.frame_master = Frame(master=self.master, padx=4, pady=4, name="frame_master")
        self.frame_master.grid(row=0, column=0)
        self.top_left_frame = Frame(master=self.frame_master, name="top_left_frame")
        self.top_left_frame.grid(row=0, column=0, sticky="wn")

        # Toolbar
        self.toolbar = self.get_navigation_menu(self.master)

        # Task, model, and embedder selection
        self.task_selection_frame = Frame(master=self.top_left_frame, name="task_selection_frame")
        self.task_selection_frame.grid(row=0, column=0)

        self.task_selection_var = tk.StringVar()
        self.task_selection = ttk.Combobox(self.task_selection_frame, textvariable=self.task_selection_var,
                                           values=list(self.app.task_registry.tasks),
                                           )
        self.task_selection.bind("<<ComboboxSelected>>",
                                 lambda x: (self.app.task_registry.choose_task(self.task_selection.get()),
                                            self.reload_comboboxes())
                                 )

        self.task_selection.grid(row=0, column=0)
        self.add_task_button = tk.Button(self.task_selection_frame,
                                         text="Add task",
                                         command=lambda: task_creation_popup(self.app, self.reload_comboboxes))
        self.add_task_button.grid(row=0, column=1)

        self.model_selection_var = tk.StringVar()
        self.model_selection = ttk.Combobox(self.task_selection_frame, textvariable=self.model_selection_var,
                                            values=list(self.app.task_registry.get_task_modes()))
        self.model_selection.bind("<<ComboboxSelected>>",
                                  lambda x: self.app.task_registry.get_current_task()
                                  .choose_model(self.model_selection_var.get()))
        self.model_selection.grid(row=1, column=0)
        self.add_model_button = tk.Button(self.task_selection_frame, text="Add model",
                                          command=lambda: self.app.switch_to_view(self.app.views.training))
        self.add_model_button.grid(row=1, column=1)

        self.embedder_selection_var = tk.StringVar()
        self.embedder_selection = ttk.Combobox(self.task_selection_frame, textvariable=self.embedder_selection_var,
                                               values=list(self.app.embstore_registry.stores))
        self.embedder_selection.bind("<<ComboboxSelected>>",
                                     lambda x: self.app.embstore_registry.choose_store(
                                         self.embedder_selection_var.get()))
        self.embedder_selection.grid(row=0, column=2, rowspan=2)
        self.reload_comboboxes()

        # Clustering scatter
        self.clustering_scatter_frame = Frame(self.top_left_frame)
        self.clustering_scatter_frame.grid(row=1, column=0)

        placeholder_image = ImageTk.PhotoImage(
            Image.new("RGB", (CLUSTERING_IMG_SIZE, CLUSTERING_IMG_SIZE), (255, 255, 255)))
        self.clustering_output = ttk.Label(self.clustering_scatter_frame, image=placeholder_image)
        self.clustering_output.image = placeholder_image
        self.clustering_output.grid(row=0, column=0)

        self.clustering_button = tk.Button(self.clustering_scatter_frame, text="Compute",
                                           command=self.show_clustering)
        self.clustering_button.grid(row=1, column=0, sticky="nw")

        # Clustering preview

    def reload_comboboxes(self):
        registry = self.app.task_registry
        task_names = list(registry.tasks)
        self.task_selection.config(values=task_names)
        model_names = list(registry.get_current_models())
        self.model_selection.config(values=model_names)
        embedder_names = list(self.app.embstore_registry.stores)
        self.embedder_selection.config(values=embedder_names)

        if registry.is_initialized:
            self.task_selection.current(task_names.index(registry.get_current_task_name()))
            if registry.get_current_task().is_initialized:
                self.model_selection.current(model_names.index(registry.get_current_task().get_current_model_name()))
        if self.app.embstore_registry.is_initialized:
            self.embedder_selection.current(embedder_names.index(self.app.embstore_registry.get_current_store_name()))

    def show_clustering(self):
        img = cluster(self.app.embstore_registry,
                          self.app.task_registry)
        self.clustering_output.config(
            image=img),
        self.clustering_output.image = img