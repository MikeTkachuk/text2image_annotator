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

    @abstractmethod
    def get_tools(self, master: tk.Menu) -> tk.Menu:
        pass

    def get_navigation_menu(self, master: tk.Tk):
        menu = tk.Menu(master)
        project = self.app.set_project_toolbar(menu)
        menu.add_cascade(menu=project, label="Project")

        navigate = self.app.set_navigation_toolbar(menu)
        menu.add_cascade(menu=navigate, label="Navigate")

        tools = self.get_tools(menu)
        menu.add_cascade(menu=tools, label="Tools")

        master.config(menu=menu)
        master.update()
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
