from core import App
from views.utils import BindTk


if __name__ == "__main__":
    root = BindTk()

    app = App(root)
    root.geometry("+100+10")
    root.state('zoomed')

    root.mainloop()
