from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.app import App

import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import ImageTk, Image

from views.view_base import ViewBase
from views.utils import Frame, task_creation_popup, resize_pad_square
from config import *


# todo: filter by label during NN selection (turn legend into buttons),
#

class ClusteringFrame(ViewBase):
    def __init__(self, app: App):
        self.app = app
        self.master = self.app.master

        self.cursor_size = 0.05
        self._selection_data = None
        self._reload_next_nn = False

    def render(self):
        self._main_setup()

    def get_tools(self, master):
        tools = tk.Menu(master, tearoff=0)

        def ask_cursor_size():
            cursor_size = simpledialog.askfloat(title="Set #neighbors", prompt="Values < 1 are radius based",
                                                     initialvalue=self.cursor_size)
            if cursor_size and cursor_size > 0:
                self.cursor_size = cursor_size
        tools.add_command(label="Set cursor size", command=ask_cursor_size)
        return tools

    def _main_setup(self):
        self.master.title("Clustering")

        self.frame_master = Frame(master=self.master, padx=4, pady=4, name="frame_master")
        self.frame_master.grid(row=0, column=0)
        self.top_left_frame = Frame(master=self.frame_master, name="top_left_frame")
        self.top_left_frame.grid(row=0, column=0, sticky="wn")

        # Toolbar
        self.toolbar = self.get_toolbar(self.master)

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
        self.add_task_button = ttk.Button(self.task_selection_frame,
                                          text="Add task",
                                          command=lambda: task_creation_popup(self.app, self.reload_comboboxes))
        self.add_task_button.grid(row=0, column=1)

        self.model_selection_var = tk.StringVar()
        self.model_selection = ttk.Combobox(self.task_selection_frame, textvariable=self.model_selection_var,
                                            values=list(self.app.task_registry.get_current_models()))
        self.model_selection.bind("<<ComboboxSelected>>",
                                  lambda x: self.app.task_registry.get_current_task()
                                  .choose_model(self.model_selection_var.get()))
        self.model_selection.grid(row=1, column=0)
        self.add_model_button = ttk.Button(self.task_selection_frame, text="Add model",
                                           command=lambda: self.app.switch_to_view(self.app.views.training))
        self.add_model_button.grid(row=1, column=1)

        self.embedder_selection_var = tk.StringVar()
        self.embedder_selection = ttk.Combobox(self.task_selection_frame, textvariable=self.embedder_selection_var,
                                               values=list(self.app.embstore_registry.stores))
        self.embedder_selection.bind("<<ComboboxSelected>>",
                                     lambda x: self.app.embstore_registry.choose_store(
                                         self.embedder_selection_var.get()))
        self.embedder_selection.grid(row=0, column=2, rowspan=2, padx=5)

        # Clustering scatter
        self.clustering_scatter_frame = Frame(self.top_left_frame)
        self.clustering_scatter_frame.grid(row=1, column=0)

        self.clustering_output = ttk.Label(self.clustering_scatter_frame)
        self.clustering_output.grid(row=0, column=0, columnspan=2)
        self.clustering_output.bind("<1>", self.populate_neighbors)

        self.clustering_button = ttk.Button(self.clustering_scatter_frame, text="Compute",
                                            command=self.compute_clustering)
        self.clustering_button.grid(row=1, column=0, sticky="nw")

        self.legend_frame = Frame(self.clustering_scatter_frame, name="legend_frame")
        self.legend_frame.grid(row=1, column=1, stick="e")

        # Clustering preview
        self.right_frame = Frame(self.frame_master, name="right_frame")
        self.right_frame.grid(row=0, column=1, rowspan=2)

        self.neighbor_list_frame = Frame(self.right_frame, name="neighbor_list_frame")
        self.neighbor_list_frame.grid(row=0, column=0, sticky="w")
        self.neighbor_scrollbar = ttk.Scrollbar(self.neighbor_list_frame, orient="vertical")
        self.neighbor_choice = ttk.Treeview(self.neighbor_list_frame, height=25 if not DEBUG else 2,
                                            yscrollcommand=self.neighbor_scrollbar.set, selectmode="browse",
                                            columns=("Class",))
        self.neighbor_choice.column("#0", width=50)
        self.neighbor_choice.heading("#0", text="Name")
        self.neighbor_choice.column("Class", width=100)
        self.neighbor_choice.heading("Class", text="Class")
        self.neighbor_scrollbar.config(command=self.neighbor_choice.yview)
        self.neighbor_scrollbar.grid(row=0, column=1, sticky="nsw")
        self.neighbor_choice.grid(row=0, column=0, sticky="w")
        self.neighbor_choice.bind("<<TreeviewSelect>>",
                                  lambda x: self.preview_neighbor())

        self.item_preview_frame = Frame(self.right_frame, name="item_preview_frame")
        self.item_preview_frame.grid(row=0, column=1)

        def update_label(direction=1):
            if not self.neighbor_choice.selection():
                return
            selection = self.neighbor_choice.selection()[0]
            selection_meta = self._selection_data[int(selection)]
            task = self.app.task_registry.get_current_task()
            all_labels = [task.label_name(-1)] + task.categories_full
            new_id = all_labels.index(selection_meta[2]) + direction
            if not 0 <= new_id < len(all_labels):
                return
            new_label = all_labels[all_labels.index(selection_meta[2]) + direction]
            task.override_label(selection_meta[1], task.label_encode(new_label))
            selection_meta[2] = new_label
            self.item_preview_frame.children["preview_label"].config(text=f"<  {new_label}  >")
            self.neighbor_choice.set(selection, column="Class", value=new_label)
            self._reload_next_nn = True
            self.master.update()

        self.master.bind("<Right>", lambda x: update_label(), user=True)
        self.master.bind("<Left>", lambda x: update_label(-1), user=True)

        # post init
        self.reload_comboboxes()
        self.show_clustering_result()

    def reload_comboboxes(self):
        registry = self.app.task_registry
        registry.validate_selection()
        task_names = list(registry.tasks)
        self.task_selection.config(values=task_names)
        model_names = list(registry.get_current_models())
        self.model_selection.config(values=model_names)
        embedder_names = list(self.app.embstore_registry.stores)
        self.embedder_selection.config(values=embedder_names)

        if registry.is_initialized:
            self.task_selection.current(task_names.index(registry.get_current_task_name()))
            task = registry.get_current_task()
            if task.is_initialized:
                self.model_selection.current(model_names.index(task.get_current_model_name()))
        if self.app.embstore_registry.is_initialized:
            self.embedder_selection.current(embedder_names.index(self.app.embstore_registry.get_current_store_name()))

    def show_clustering_result(self):
        img = self.app.clustering.get_base_plot()
        self.clustering_output.config(
            image=img),
        self.clustering_output.image = img
        for ch in list(self.legend_frame.children.values()):
            ch.destroy()
        for i, (sample, name) in enumerate(self.app.clustering.get_legend(img_size=16)):
            image = ImageTk.PhotoImage(Image.fromarray(sample))
            label = tk.Label(self.legend_frame,
                             image=image,
                             text=name, compound="left")
            label.image = image
            label.grid(row=0, column=i)

    def compute_clustering(self):
        self.app.clustering.cluster()
        self.show_clustering_result()

    def populate_neighbors(self, event):
        if self._reload_next_nn:
            self._reload_next_nn = False
            self.app.task_registry.update_current_task()
            self.app.clustering.update_labels()
        out = self.app.clustering.get_nearest_neighbors(event.x, event.y, self.cursor_size)
        if out is None:
            return
        ids, filenames, class_names, viz = out
        self.clustering_output.config(
            image=viz),
        self.clustering_output.image = viz
        for item in self.neighbor_choice.get_children():
            self.neighbor_choice.delete(item)
        self._selection_data = {}
        for i, (id_, filename, class_name) in enumerate(zip(ids, filenames, class_names)):
            self._selection_data[i] = [id_, filename, class_name]
            self.neighbor_choice.insert("", "end", str(i), text=str(i), values=(class_name,))
        self.neighbor_choice.focus_set()
        self.neighbor_choice.selection_set("0")
        self.neighbor_choice.focus("0")
        self.preview_neighbor()

    def preview_neighbor(self):
        for ch in list(self.item_preview_frame.children.values()):
            ch.destroy()
        if not self.neighbor_choice.selection():
            return
        selected_id = self.neighbor_choice.selection()[0]
        selected_meta = self._selection_data[int(selected_id)]
        img = self.app.clustering.draw_selection([selected_meta[0]])
        self.clustering_output.config(
            image=img),
        self.clustering_output.image = img

        img_label = tk.Label(self.item_preview_frame)
        square_image = resize_pad_square(selected_meta[1], IMG_SIZE)
        photo = ImageTk.PhotoImage(square_image)
        img_label.config(image=photo)
        img_label.image = photo
        img_label.grid(row=0, column=0)

        tk.Label(self.item_preview_frame, text=f"<  {selected_meta[2]}  >", name="preview_label").grid(row=1, column=0)
        ttk.Button(self.item_preview_frame,
                   text="Open in annotation tool",
                   command=lambda: (self.app.go_to_image(selected_meta[1]),
                                    self.app.switch_to_view(self.app.views.main))).grid(row=2, column=0)
