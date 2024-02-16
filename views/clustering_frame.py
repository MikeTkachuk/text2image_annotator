from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core_app import App

from views.view_base import ViewBase, Frame
from config import *


class ClusteringFrame(ViewBase):
    def __init__(self, app: App):
        self.app = app
        self.master = self.app.master

    def render(self):
        self._main_setup()

    def _main_setup(self):
        self.app.master.title("Clustering")

        self.frame_master = Frame(master=self.master, padx=4, pady=4, name="frame_master")
        self.frame_master.grid(row=0, column=0)
        self.top_left_frame = Frame(master=self.frame_master, name="top_left_frame")
        self.top_left_frame.grid(row=0, column=0, sticky="wn")

        # Toolbar
        self.toolbar_frame = Frame(master=self.top_left_frame, name="toolbar_frame", pack=True)
        self.toolbar_frame.grid(row=0, column=0, sticky="wn")
        self.toolbar = self.get_navigation_menu(self.toolbar_frame)
        self.toolbar.pack()
