from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.app import App

from pathlib import Path
from tkinter import ttk, Frame as FrameBase
import tkinter as tk
import PIL
from PIL import Image, ImageDraw
from config import DEBUG
from core.model import Model


class BindTk(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind_dict = {}

    def bind(
            self,
            sequence=...,
            func=...,
            add=...,
            user=False,
    ):
        funcid = super().bind(sequence, func, add)
        if user:
            self.bind_dict[sequence] = funcid

    def unbind_all_user(self):
        self.update()
        self.update_idletasks()
        for seq, func in self.bind_dict.items():
            try:
                self.unbind(seq, func)
            except tk.TclError as e:
                print(e)
        self.bind_dict = {}


class Frame(FrameBase):
    def __init__(self, *args, pack=False, **kwargs):
        if not DEBUG:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs,
                             highlightbackground="black", highlightthickness=2,
                             )
            name = kwargs.get("name")
            if name is not None:
                label = tk.Label(self, text=name, font=('Calibri', 8), padx=0, pady=0)
                if not pack:
                    label.grid(row=10, column=0)
                else:
                    label.pack()


def put_node(tree: ttk.Treeview, path, node: str, score=None):
    def path_to_node_id(p: Path):
        if not len(p.parents):
            return ''
        else:
            return p.as_posix()

    path = Path(path)
    parents = [path] + list(path.parents)
    if len(parents) > 1:
        for i in range(1, len(parents)):
            parent_id = path_to_node_id(parents[-i])
            node_id = path_to_node_id(parents[-i - 1])
            if not tree.exists(node_id):
                tree.insert(parent_id,
                            "end", node_id,
                            text=parents[-i - 1].name, open=True)
    tree.insert(path_to_node_id(parents[0]), 'end', node, text=node, values=(score,))


def resize_pad_square(image_path, size):
    try:
        image = Image.open(image_path)
    except PIL.UnidentifiedImageError:
        square_image = Image.new("RGB", (size, size), (255, 255, 230))
        dr = ImageDraw.Draw(square_image)
        dr.text((size / 2, size / 2), text="Invalid image", align="center", fill=(200, 20, 10))
        return square_image
    width, height = image.size

    # Determine the aspect ratio
    aspect_ratio = width / height

    # Calculate new dimensions while maintaining the aspect ratio
    if width > height:
        new_width = size
        new_height = int(size / aspect_ratio)
    else:
        new_width = int(size * aspect_ratio)
        new_height = size

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a square image with a white background
    square_image = Image.new("RGB", (size, size), (255, 255, 255))
    square_image.paste(resized_image, ((size - new_width) // 2, (size - new_height) // 2))
    return square_image


def task_creation_popup(app: App, callback):
    window = tk.Toplevel(master=app.master, name="task_creation_popup")
    window.geometry("500x500")
    window.title("New task")

    tag_select_frame = Frame(window, name="tag_select_frame")
    tag_select_frame.pack()
    tag_scrollbar = ttk.Scrollbar(tag_select_frame, orient="vertical")
    tag_choice = ttk.Treeview(tag_select_frame, height=15,
                              yscrollcommand=tag_scrollbar.set, selectmode="extended",
                              )
    tag_choice.column("#0", width=210)
    tag_choice.heading("#0", text="Name")
    tag_scrollbar.config(command=tag_choice.yview)
    tag_scrollbar.grid(row=0, column=1, sticky="nsw")
    tag_choice.grid(row=0, column=0, sticky="w")

    task_mode_var = tk.StringVar()
    task_mode_var.set(app.task_registry.get_task_modes()[0])
    task_mode_selection = ttk.OptionMenu(window, task_mode_var, app.task_registry.get_task_modes()[0],
                                         *app.task_registry.get_task_modes())
    task_mode_selection.pack()

    for tag, tag_path, _ in zip(*app.search_tags("")):
        put_node(tag_choice, tag_path, tag)

    error_label = tk.Label(window, text="")
    error_label.pack()

    def closure(event=None):
        try:
            if any([tag_choice.get_children(t) for t in tag_choice.selection()]):
                raise RuntimeError("Selected non-leaf nodes")
            app.task_registry.add_task(tag_choice.selection(), task_mode_var.get())
            callback()
        except RuntimeError as e:
            error_label.config(text=str(e))
            error_label.update()
            return
        window.destroy()

    create_button = ttk.Button(window, text="Confirm", command=closure)
    create_button.pack()


def model_creation_popup(app: App, callback):
    window = tk.Toplevel(master=app.master, name="model_creation_popup")
    window.geometry("500x500")
    window.title("New model")

    model_name_var = tk.StringVar()
    model_name_var.set(f"model_{len(app.task_registry.get_current_models(False))}")
    model_name_entry = ttk.Entry(window, textvariable=model_name_var)
    model_name_entry.selection_range(0, "end")
    model_name_entry.pack()
    model_template_var = tk.StringVar()
    templates = list(Model.get_templates())
    model_template_selection = ttk.OptionMenu(window, model_template_var, templates[0], *templates)
    model_template_selection.pack()

    error_label = tk.Label(window, text="")
    error_label.pack()

    def closure(event=None):
        try:
            app.task_registry.add_model(model_name_var.get(), model_template_var.get())
            callback()
        except RuntimeError as e:
            error_label.config(text=str(e))
            error_label.update()
            return
        window.destroy()

    create_button = ttk.Button(window, text="Confirm", command=closure)
    create_button.pack()
