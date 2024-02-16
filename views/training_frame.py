from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main_frame import App
from views.view_base import ViewBase

import tkinter as tk
from tkinter import ttk


class TrainingFrame(ViewBase):
    def __init__(self, parent: App):
        self.parent = parent
        self._main_setup()

    def render(self):
        self._main_setup()

    def switch_to_main(self):
        self.frame_master.destroy()
        self.parent.render()

    def _main_setup(self):
        self.parent.master.title("Training")

        self.frame_master = ttk.Frame(master=self.parent.master, padding=4)
        self.frame_master.grid(row=0, column=0)

        self.main_view_button = tk.Button(self.frame_master, text="Back",
                                          command=self.switch_to_main)
        self.main_view_button.grid(row=2, column=0, sticky="ws")
