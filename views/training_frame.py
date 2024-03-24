from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.app import App

import time
from threading import Thread

import tkinter as tk
from tkinter import ttk, scrolledtext

from views.view_base import ViewBase
from views.utils import Frame, task_creation_popup, model_creation_popup
from core.utils import thread_killer


# todo: duplicate model/save model template
# todo: display latest metrics
class TrainingFrame(ViewBase):
    def __init__(self, app: App):
        self.app = app
        self.master = self.app.master

        self._parameter_frames = {}
        self._logs = ""
        self._training_params = {}
        self._training_thread = None

    def render(self):
        self._main_setup()

    def get_tools(self, master):
        tools = tk.Menu(master, tearoff=0)

        def set_training_options():
            window = tk.Toplevel(self.master)
            window.title("Training options")
            kfold_frame = Frame(window, name="kfold_frame", pack=True)
            kfold_frame.pack()
            tk.Label(kfold_frame, text="KFold").pack(side="left")
            kfold_var = tk.StringVar(value=str(self._training_params.get("kfold", "None")))
            ttk.Entry(kfold_frame, textvariable=kfold_var).pack(side="left")

            aug_frame = Frame(window, name="aug_frame", pack=True)
            aug_frame.pack()
            tk.Label(aug_frame, text="Use augs").pack(side="left")
            aug_var = tk.BooleanVar(value=self._training_params.get("use_augs", True))
            ttk.Checkbutton(aug_frame, variable=aug_var).pack(side="left")

            def closure():
                self._training_params = {
                    "kfold": self._parse_param(kfold_var.get()),
                    "use_augs": aug_var.get(),
                }
                window.destroy()

            ttk.Button(window, text="Confirm", command=closure).pack()

        tools.add_command(label="Training options", command=set_training_options)
        return tools

    def _main_setup(self):
        self.app.master.title("Training")

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
                                             .choose_model(self.model_selection_var.get()),
                                             self.show_model_info()))
        self.model_selection.grid(row=1, column=0)
        self.add_model_button = ttk.Button(self.task_selection_frame, text="Add model",
                                           command=lambda: model_creation_popup(self.app,
                                                                                self.reload_comboboxes))
        self.add_model_button.grid(row=1, column=1)

        self.embedder_selection_var = tk.StringVar()
        self.embedder_selection = ttk.Combobox(self.task_selection_frame, textvariable=self.embedder_selection_var,
                                               values=list(self.app.embstore_registry.stores))
        self.embedder_selection.bind("<<ComboboxSelected>>",
                                     lambda x: (self.app.embstore_registry.choose_store(
                                         self.embedder_selection_var.get()),
                                                self.reload_comboboxes()))
        self.embedder_selection.grid(row=0, column=2, rowspan=2, padx=5)

        # Model preview
        self.model_preview_frame = Frame(master=self.top_left_frame, name="model_preview_frame")
        self.model_preview_frame.grid(row=1, column=0)

        self.model_info = tk.Label(self.model_preview_frame)
        self.model_info.grid(row=0, column=0, columnspan=2)
        self.parameter_tuning_frame = Frame(master=self.model_preview_frame, pack=True, name="parameter_tuning_frame")
        self.parameter_tuning_frame.grid(row=1, column=0, columnspan=2)

        self.parameter_add_select_var = tk.StringVar()
        self.parameter_add_select = ttk.Combobox(self.model_preview_frame, textvariable=self.parameter_add_select_var)
        self.parameter_add_select.grid(row=2, column=0, sticky="e")
        self.add_parameter_button = ttk.Button(self.model_preview_frame, text="Add parameter",
                                               command=self.add_parameter_widget)
        self.add_parameter_button.grid(row=2, column=1)

        self.fit_button = ttk.Button(self.top_left_frame, text="Fit", command=self.fit_model)
        self.fit_button.grid(row=2, column=0)
        self.cancel_button = ttk.Button(self.top_left_frame, text="Cancel", state="disabled")
        self.cancel_button.grid(row=2, column=1)

        # Other
        self.right_frame = Frame(self.frame_master, name="right_frame")
        self.right_frame.grid(row=0, column=1)

        self.log_frame = Frame(self.right_frame, name="log_frame")
        self.log_frame.grid(row=0, column=0)
        self.log = scrolledtext.ScrolledText(self.log_frame, height=30, state="disabled")
        self.log.grid(row=0, column=0)

        # on load
        self.log_update(self._logs, overwrite=True)
        self.reload_comboboxes()

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

        self.show_model_info()

    def show_model_info(self):
        model = self.app.task_registry.get_current_model()
        for ch in list(self.parameter_tuning_frame.children.values()):
            ch.destroy()
        self._parameter_frames = {}
        if model is None:
            # clear everything
            self.model_info.config(text="")
            self.parameter_add_select.config(values=[])
        else:
            # populate everything
            info = f"Model: {model.model} \nFramework: {model.framework} "
            self.model_info.config(text=info)
            signature = model.get_model_signature()
            values = []
            for name, default in signature.items():
                if name in model.params:
                    self.add_parameter_widget(name, model.params[name])
                values.append(f"{name}={default}")
            self.parameter_add_select.config(values=values)

    def _destroy_param_widget(self, w_id):
        widget = self._parameter_frames.pop(w_id)
        widget[0].destroy()

    def add_parameter_widget(self, parameter_name=None, default=None):
        if self.app.task_registry.get_current_model() is None:
            return
        if parameter_name is None:
            value = self.parameter_add_select_var.get().split("=")
            parameter_name, default = value
        if parameter_name in self._parameter_frames:
            return
        frame = Frame(self.parameter_tuning_frame)
        frame.pack(anchor="w")
        text_var = tk.StringVar(value=str(default))
        tk.Label(frame, text=parameter_name, width=20).grid(row=0, column=0)
        ttk.Entry(frame, textvariable=text_var, ).grid(row=0, column=1)
        ttk.Button(frame, text="X", width=10,
                   command=lambda: self._destroy_param_widget(parameter_name)
                   ).grid(row=0, column=2)
        self._parameter_frames[parameter_name] = (frame, text_var)

    @staticmethod
    def _parse_param(value: str):
        if value == "None":
            return None
        if value == "True":
            return True
        if value == "False":
            return False
        try:
            value = float(value)
            if value == float(int(value)):
                return int(value)
            return value
        except ValueError:
            return value

    def log_update(self, value: str, end="\n", overwrite=False):
        value = value + end

        def func():
            self._logs = value if overwrite else self._logs + value

            try:
                self.log.configure(state='normal')
                if overwrite:
                    self.log.delete(1.0, "end")
                    self.log.insert("end", self._logs)
                else:
                    self.log.insert("end", value)
                self.log.configure(state='disabled')
                # Autoscroll to the bottom
                self.log.yview("end")
            except tk.TclError as e:
                pass
        func()
        # self.master.after(0, func)

    def fit_model(self):
        if self._training_thread is not None and self._training_thread.is_alive():
            thread_killer.set(self._training_thread.ident)
            time.sleep(1)
            self.master.update()
            self.master.update_idletasks()
        self.log_update("", end="", overwrite=True)
        self.master.update()
        self.master.update_idletasks()
        try:
            new_params = {}
            for param_name, (_, var) in self._parameter_frames.items():
                new_params[param_name] = self._parse_param(var.get())

            model = self.app.task_registry.get_current_model()
            if model is None:
                return
            model.params = new_params

            def callback(val, end="\n"):
                self.log_update(val, end=end, overwrite=False)
                print(val, end=end)

            def target():
                self.app.task_registry.fit_current_model(callback=callback, **self._training_params)

            self._training_thread = Thread(target=target)
            self._training_thread.daemon = True
            self._training_thread.start()
            thread_killer.reset(self._training_thread.ident)

            def cancel_func():
                thread_killer.set(self._training_thread.ident)
                self.cancel_button.config(state="disabled")

            self.cancel_button.config(state="normal", command=cancel_func)

        except Exception as e:
            import traceback
            self.log_update(f"Error: {''.join(traceback.format_exception(e))}")
