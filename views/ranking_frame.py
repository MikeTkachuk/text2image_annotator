from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import ImageTk

if TYPE_CHECKING:
    from core.app import App

import time
from threading import Thread

import tkinter as tk
from tkinter import ttk, scrolledtext

from views.view_base import ViewBase
from views.utils import Frame, task_creation_popup, model_creation_popup, documentation_popup, resize_pad_square
from core.utils import thread_killer, tk_plot


class RankingFrame(ViewBase):
    def __init__(self, app: App):
        self.app = app
        self.master = self.app.master

    def render(self):
        self._main_setup()

    def get_tools(self, master):
        tools = tk.Menu(master, tearoff=0)

        info_icon = ImageTk.PhotoImage(file="assets/view_info/info_icon.png")
        tools.add_command(label="Info", image=info_icon, compound="left",
                          command=lambda: documentation_popup(path="assets/view_info/training.html",
                                                              parent=self.master))
        tools.info_icon = info_icon
        return tools

    def _main_setup(self):
        self.app.master.title("Ranking")

        self.frame_master = Frame(master=self.master, padx=4, pady=4, name="frame_master")
        self.frame_master.grid(row=0, column=0)
        self.top_left_frame = Frame(master=self.frame_master, name="top_left_frame")
        self.top_left_frame.grid(row=0, column=0, sticky="wn")

        # Toolbar
        self.toolbar = self.get_toolbar(self.master)

        # Frames
        self.media_frame = Frame(master=self.top_left_frame, name="task_selection_frame")
        self.media_frame.grid(row=0, column=0)

        self.left_preview = ttk.Label(self.media_frame)
        self.left_preview.grid(row=0, column=0)
        self.right_preview = ttk.Label(self.media_frame)
        self.right_preview.grid(row=0, column=1)
        self.meta_label = tk.Label(self.media_frame)
        self.meta_label.grid(row=1, column=0, columnspan=2)

        self.stats_label = tk.Label(self.top_left_frame)
        self.stats_label.grid(row=0, column=1)

        self.master.bind("<Right>", lambda x: self.annotate(1), user=True)
        self.master.bind("<Left>", lambda x: self.annotate(0), user=True)
        self.master.bind("<Return>", lambda x: self.next_pair(), user=True)
        self.master.bind("<Control-Return>", lambda x: self.prev_pair(), user=True)

        self.show_pair()

    def show_pair(self):
        pair = self.app.ranking.get_current_state(decode=True)
        square_image = resize_pad_square(pair.pair[0], 600)
        photo = ImageTk.PhotoImage(square_image)
        self.left_preview.config(image=photo)
        self.left_preview.image = photo
        square_image = resize_pad_square(pair.pair[1], 600)
        photo = ImageTk.PhotoImage(square_image)
        self.right_preview.config(image=photo)
        self.right_preview.image = photo

        self.meta_label.config(text=str(pair.meta))
        self.stats_label.config(text=f"Progress: \n{len(self.app.ranking.state)}/{self.app.ranking.max_num_pairs}")

    def next_pair(self):
        self.app.ranking.next_pair()
        self.show_pair()

    def prev_pair(self):
        self.app.ranking.prev_pair()
        self.show_pair()

    def annotate(self, right_better):
        self.app.ranking.update_meta("right_better", right_better)
        self.next_pair()