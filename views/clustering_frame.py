from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from core.app import App

from threading import Thread

import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import ImageTk, Image

from views.view_base import ViewBase
from views.utils import Frame, task_creation_popup, resize_pad_square
from config import *


# todo: option to hide filtered from the plot or highlight them
# todo: button agree with all model predictions from the list

@dataclass
class SampleInfo:
    id_: int
    filename: str
    class_name: str
    pred_name: str
    meta: dict
    split: str
    version: str


class ClusteringFrame(ViewBase):
    def __init__(self, app: App):
        self.app = app
        self.master = self.app.master

        self.cursor_size = 0.025
        self._selection_data: Dict[int, SampleInfo] = None
        self._reload_next_nn = False
        self._filter = {}
        self._cluster_params = {}

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

        def set_filter_conditions():
            window = tk.Toplevel(self.master)
            window.title("Filter selection")
            class_filter = ttk.Treeview(window, selectmode="extended", height=5)
            class_filter.heading("#0", text="Class Filter")
            class_filter.pack()
            for _, class_name in self.app.clustering.get_legend():
                class_filter.insert("", "end", class_name, text=class_name)
            if self._filter.get("class_name"):
                class_filter.selection_set(self._filter["class_name"])

            pred_filter = ttk.Treeview(window, selectmode="extended", height=5)
            pred_filter.heading("#0", text="Prediction Filter")
            pred_filter.pack()
            model = self.app.task_registry.get_current_model()
            if model is not None:
                preds = [self.app.task_registry.get_current_task().label_name(p) for p in model.get_classes()]
                for pred in sorted(preds):
                    pred_filter.insert("", "end", pred, text=pred)
            if self._filter.get("prediction"):
                pred_filter.selection_set(self._filter["prediction"])

            special_frame = Frame(window, name="special_frame", pack=True)
            special_frame.pack()
            tk.Label(special_frame, text="Conditions:").pack(side="left")
            special_modes = ["All", "Equal", "Not Equal"]
            special_var = tk.StringVar()
            special = ttk.OptionMenu(special_frame, special_var,
                                     special_modes[special_modes.index(self._filter.get("special", "All"))],
                                     *special_modes)
            special.pack(side="left")

            split_frame = Frame(window, name="split_frame", pack=True)
            split_frame.pack()
            tk.Label(split_frame, text="Split:").pack(side="left")
            split_modes = ["All", "train", "val"]
            split_var = tk.StringVar()
            split_filter = ttk.OptionMenu(split_frame, split_var,
                                          split_modes[split_modes.index(self._filter.get("split", "All"))],
                                          *split_modes)
            split_filter.pack(side="left")

            version_frame = Frame(window, name="version_frame", pack=True)
            version_frame.pack()
            tk.Label(version_frame, text="Version:").pack(side="left")
            version_modes = ["All", "orig", "aug"]
            version_var = tk.StringVar()
            version_filter = ttk.OptionMenu(version_frame, version_var,
                                            version_modes[version_modes.index(self._filter.get("version", "All"))],
                                            *version_modes)
            version_filter.pack(side="left")

            def closure(do_reset=False):
                if do_reset:
                    self._filter = {}
                else:
                    if class_filter.selection():
                        self._filter["class_name"] = class_filter.selection()
                    if pred_filter.selection():
                        self._filter["prediction"] = pred_filter.selection()

                    self._filter["split"] = split_var.get()
                    self._filter["version"] = version_var.get()
                    self._filter["special"] = special_var.get()
                window.destroy()

            reset = ttk.Button(window, text="Reset", command=lambda: closure(do_reset=True))
            reset.pack()
            confirm = ttk.Button(window, text="Confirm", command=closure)
            confirm.pack()

        tools.add_command(label="Filter", command=set_filter_conditions)

        def set_clustering_options():
            window = tk.Toplevel(self.master)
            window.title("Clustering options")
            pca_components_frame = Frame(window, name="pca_components_frame", pack=True)
            pca_components_frame.pack()
            tk.Label(pca_components_frame, text="PCA reduction components:").pack(side="left")
            pca_components_var = tk.StringVar(value=str(self._cluster_params.get("pca_components", "50")))
            ttk.Entry(pca_components_frame, textvariable=pca_components_var).pack(side="left")

            tsne_frame = Frame(window, name="tsne_frame", pack=True)
            tsne_frame.pack()
            tk.Label(tsne_frame, text="Use TSNE:").pack(side="left")
            tsne_var = tk.BooleanVar(value=self._cluster_params.get("tsne", True))
            ttk.Checkbutton(tsne_frame, variable=tsne_var).pack(side="left")

            use_model_frame = Frame(window, name="use_model_frame", pack=True)
            use_model_frame.pack()
            tk.Label(use_model_frame, text="Use model activations:").pack(side="left")
            use_model_var = tk.BooleanVar(value=self._cluster_params.get("use_model_features", True))
            ttk.Checkbutton(use_model_frame, variable=use_model_var).pack(side="left")

            layer_frame = Frame(window, name="layer_frame", pack=True)
            layer_frame.pack()
            tk.Label(layer_frame, text="Model layer index:").pack(side="left")
            layer_var = tk.StringVar(value=str(self._cluster_params.get("layer", "-2")))
            ttk.Entry(layer_frame, textvariable=layer_var).pack(side="left")

            augs_frame = Frame(window, name="augs_frame", pack=True)
            augs_frame.pack()
            tk.Label(augs_frame, text="Include augmented:").pack(side="left")
            augs_var = tk.BooleanVar(value=self._cluster_params.get("augs", False))
            ttk.Checkbutton(augs_frame, variable=augs_var).pack(side="left")

            def closure():
                self._cluster_params = {
                    "pca_components": int(pca_components_var.get()),
                    "tsne": tsne_var.get(),
                    "use_model_features": use_model_var.get(),
                    "layer": int(layer_var.get()),
                    "augs": augs_var.get()
                }
                window.destroy()

            ttk.Button(window, text="Confirm", command=closure).pack()

        tools.add_command(label="Clustering options", command=set_clustering_options)
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
                                  lambda x: (self.app.task_registry.get_current_task()
                                             .choose_model(self.model_selection_var.get()), self.update_labels()))
        self.model_selection.grid(row=1, column=0)
        # self.add_model_button = ttk.Button(self.task_selection_frame, text="Add model",
        #                                    command=lambda: self.app.switch_to_view(self.app.views.training))
        # self.add_model_button.grid(row=1, column=1)
        self.show_predictions_var = tk.BooleanVar()
        self.show_predictions_checkbox = ttk.Checkbutton(self.task_selection_frame, text="Show predictions",
                                                         variable=self.show_predictions_var, command=self.update_labels)
        self.show_predictions_checkbox.grid(row=1, column=1)

        self.embedder_selection_var = tk.StringVar()
        self.embedder_selection = ttk.Combobox(self.task_selection_frame, textvariable=self.embedder_selection_var,
                                               values=list(self.app.embstore_registry.stores))
        self.embedder_selection.bind("<<ComboboxSelected>>",
                                     lambda x: (self.app.embstore_registry.choose_store(
                                         self.embedder_selection_var.get()),
                                                self.reload_comboboxes()))
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
        neighbor_columns = ["#0", "Class", "Pred", "Split", "Version"]
        self.neighbor_choice = ttk.Treeview(self.neighbor_list_frame, height=25 if not DEBUG else 2,
                                            yscrollcommand=self.neighbor_scrollbar.set, selectmode="browse",
                                            columns=neighbor_columns[1:])
        neighbor_column_names = ["Id", "Class", "Pred", "Split", "Version"]
        neighbor_column_widths = [25, 75, 75, 50, 50]
        for c, n, w in zip(neighbor_columns, neighbor_column_names, neighbor_column_widths):
            self.neighbor_choice.column(c, width=w)
            self.neighbor_choice.heading(c, text=n)

        self.neighbor_scrollbar.config(command=self.neighbor_choice.yview)
        self.neighbor_scrollbar.grid(row=0, column=1, sticky="nsw")
        self.neighbor_choice.grid(row=0, column=0, sticky="w")
        self.neighbor_choice.bind("<<TreeviewSelect>>",
                                  lambda x: self.preview_neighbor())

        self.item_preview_frame = Frame(self.right_frame, name="item_preview_frame")
        self.item_preview_frame.grid(row=0, column=1)

        def update_label(direction=1, value=None):
            if not self.neighbor_choice.selection():
                return
            selection = self.neighbor_choice.selection()[0]
            selection_meta = self._selection_data[int(selection)]
            task = self.app.task_registry.get_current_task()
            all_labels = [task.label_name(-1)] + task.categories_full
            if value is None:
                new_id = all_labels.index(selection_meta.class_name) + direction
            else:
                new_id = value + 1  # only 0-n accepted, id_0 reserved for not labeled
            if not 0 <= new_id < len(all_labels):
                return
            new_label = all_labels[new_id]
            task.override_label(selection_meta.filename, task.label_encode(new_label))
            selection_meta.class_name = new_label
            self.item_preview_frame.children["preview_label"].config(text=f"<  {new_label}  >")
            self.neighbor_choice.set(selection, column="Class", value=new_label)
            self._reload_next_nn = True  # not self.show_predictions_var.get()
            self.master.update()
            self.master.update_idletasks()

        self.master.bind("<Right>", lambda x: update_label(), user=True)
        self.master.bind("<Left>", lambda x: update_label(-1), user=True)
        for i in range(10):
            self.master.bind(f"{i}", lambda x, y=i: update_label(value=y), user=True)

        # bind annotation suggestion keys
        self.master.bind("<Return>", lambda x: self.focus_next_annotation_suggestion(), user=True)
        self.master.bind("<Control-Return>", lambda x: self.focus_prev_annotation_suggestion(), user=True)

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
            else:
                self.model_selection_var.set("")
        else:
            self.model_selection_var.set("")
            self.task_selection_var.set("")
        if self.app.embstore_registry.is_initialized:
            self.embedder_selection.current(embedder_names.index(self.app.embstore_registry.get_current_store_name()))
        else:
            self.embedder_selection_var.set("")
        self.update_labels()

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
        def target():
            self.app.clustering.cluster(**self._cluster_params)
            try:
                self.show_clustering_result()
            except tk.TclError:
                pass

        t = Thread(target=target)
        t.start()

    def focus_on_sample(self, sample: str):
        dummy_event_loc = self.app.clustering.get_location_of_sample(sample) * CLUSTERING_IMG_SIZE
        if dummy_event_loc is not None:
            self.populate_neighbors(None, x=dummy_event_loc[0], y=dummy_event_loc[1])

    def focus_next_annotation_suggestion(self):
        model = self.app.task_registry.get_current_model()
        if model is None:
            return
        task = self.app.task_registry.get_current_task()
        while True:
            sample = model.next_annotation_suggestion()
            if sample is None:
                return
            if task.labels.get(sample, -1) < 0 and task.user_override.get(sample, -1) < 0:
                # check both as they are synchronized rarely
                self.focus_on_sample(sample)
                return

    def focus_prev_annotation_suggestion(self):
        model = self.app.task_registry.get_current_model()
        if model is None:
            return
        sample = model.prev_annotation_suggestion()
        if sample is None:
            return
        self.focus_on_sample(sample)

    def update_labels(self, idle=False):
        self.app.task_registry.update_current_task()
        self.app.clustering.update_labels(self.show_predictions_var.get())
        if not idle:
            self.show_clustering_result()

    def populate_neighbors(self, event, x=None, y=None):
        if self._reload_next_nn:
            self._reload_next_nn = False
            self.update_labels(idle=True)
        if x is None or y is None:
            x = event.x
            y = event.y
        out = self.app.clustering.get_nearest_neighbors(x, y, self.cursor_size)
        if out is None:
            return
        viz = out[-1]
        self.clustering_output.config(
            image=viz),
        self.clustering_output.image = viz
        for item in self.neighbor_choice.get_children():
            self.neighbor_choice.delete(item)
        self._selection_data = {}

        for i, (id_, filename, class_name, pred_name, meta, split, version) in enumerate(zip(*out[:-1])):
            self._selection_data[i] = SampleInfo(id_, filename, class_name, pred_name, meta, split, version)
            if self.filter_condition(class_name=class_name, prediction=pred_name,
                                     split=split, version=version):
                self.neighbor_choice.insert("", "end", str(i), text=str(i),
                                            values=(class_name, pred_name, split, version))
        if len(self.neighbor_choice.get_children()):
            self.neighbor_choice.focus_set()
            first_child = self.neighbor_choice.get_children()[0]
            self.neighbor_choice.selection_set(first_child)
            self.neighbor_choice.focus(first_child)

    def filter_condition(self, class_name, prediction, split, version):
        key = True
        if self._filter.get("class_name"):
            key = key and class_name in self._filter.get("class_name")
        if self._filter.get("prediction"):
            key = key and prediction in self._filter.get("prediction")
        if self._filter.get("split"):
            if self._filter.get("split") != "All":
                key = key and split == self._filter.get("split")
        if self._filter.get("version"):
            if self._filter.get("version") != "All":
                key = key and self._filter.get("version") in version  # orig/aug_0/...
        special_cond = self._filter.get("special")
        if special_cond:
            if special_cond == "Equal":
                key = key and class_name == prediction
            if special_cond == "Not Equal":
                key = key and class_name != prediction
        return key

    def preview_neighbor(self):
        for ch in list(self.item_preview_frame.children.values()):
            ch.destroy()
        if not self.neighbor_choice.selection():
            return
        selected_id = self.neighbor_choice.selection()[0]
        selected_meta = self._selection_data[int(selected_id)]
        img = self.app.clustering.draw_selection([selected_meta.id_])
        self.clustering_output.config(
            image=img),
        self.clustering_output.image = img

        img_label = tk.Label(self.item_preview_frame)
        square_image = resize_pad_square(selected_meta.filename, IMG_SIZE)
        photo = ImageTk.PhotoImage(square_image)
        img_label.config(image=photo)
        img_label.image = photo
        img_label.grid(row=0, column=0)

        tk.Label(self.item_preview_frame, text=f"<  {selected_meta.class_name}  >", name="preview_label").grid(row=1, column=0)
        if self.show_predictions_var.get():
            tk.Label(self.item_preview_frame,
                     text=f"Prediction: {selected_meta.pred_name} \nOther: {selected_meta.meta} ", name="preview_prediction_label").grid(row=2, column=0)
        ttk.Button(self.item_preview_frame,
                   text="Open in annotation tool",
                   command=lambda: (self.app.go_to_image(selected_meta.filename),
                                    self.app.switch_to_view(self.app.views.main))).grid(row=3, column=0)
