from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core_app import App
from abc import ABC, abstractmethod

import tkinter as tk
from tkinter import Frame as FrameBase
from config import DEBUG


class ViewBase(ABC):
    @abstractmethod
    def __init__(self, app: App):
        self.app: App = app
        self.frame_master = None

    @abstractmethod
    def render(self):
        pass

    def switch_to_view(self, view_name: str):
        self.frame_master.destroy()
        getattr(self.app.views, view_name).render()

    def get_navigation_menu(self, master):
        var = tk.StringVar()
        menu = tk.OptionMenu(master, var, *list(self.app.views._asdict().keys()),
                             command=lambda x: self.switch_to_view(x))
        return menu


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
