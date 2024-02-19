from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.app import App

import tkinter as tk

from views.view_base import ViewBase
from views.utils import Frame


class TrainingFrame(ViewBase):
    def __init__(self, app: App):
        self.app = app
        self.master = self.app.master

    def render(self):
        self._main_setup()

    def get_tools(self, master):
        tools = tk.Menu(master, tearoff=0)
        return tools

    def _main_setup(self):
        self.app.master.title("Training")

        self.frame_master = Frame(master=self.master, padx=4, pady=4, name="frame_master")
        self.frame_master.grid(row=0, column=0)
        self.top_left_frame = Frame(master=self.frame_master, name="top_left_frame")
        self.top_left_frame.grid(row=0, column=0, sticky="wn")

        # Toolbar
        self.toolbar = self.get_navigation_menu(self.master)
