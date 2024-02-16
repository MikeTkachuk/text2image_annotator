import tkinter as tk

from core_app import App


if __name__ == "__main__":
    root = tk.Tk()

    app = App(root)
    root.geometry("+100+10")
    root.state('zoomed')

    root.mainloop()
