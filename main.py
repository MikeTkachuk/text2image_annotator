import tkinter as tk

from main_frame import App
from clustering_frame import ClusteringFrame


if __name__ == "__main__":
    root = tk.Tk()

    app = App(root)
    root.geometry("+100+10")
    root.state('zoomed')

    root.mainloop()
