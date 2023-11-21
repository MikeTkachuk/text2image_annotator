import json

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk, scrolledtext
from PIL import Image, ImageTk


IMG_SIZE = 512
THUMBNAIL_SIZE = 64


def resize_pad_square(image_path, size):
    image = Image.open(image_path)
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

    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Create a square image with a white background
    square_image = Image.new("RGB", (size, size), (255, 255, 255))
    square_image.paste(resized_image, ((size - new_width) // 2, (size - new_height) // 2))
    return square_image


class ImageViewerApp:
    def __init__(self, master):
        self.session_config_path = Path("sessions.json")
        if not self.session_config_path.exists():
            self.session_config_path.touch()
            self.full_session_config = {}
        else:
            with open(self.session_config_path) as session_config_file:
                self.full_session_config = json.load(session_config_file)
        self.session_config = None
        self.image_paths = []  # List to store image paths
        self.current_index = 0  # Index of the currently displayed image

        self.master = master
        self.master.title("Image Viewer")

        self.frame_master = ttk.Frame(master=root, padding=10)
        self.frame_master.grid(row=0, column=0)

        # Media
        self.media_frame = ttk.Frame(master=self.frame_master)
        self.media_frame.grid(row=0, column=0)

        placeholder_image = ImageTk.PhotoImage(Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255)))
        self.image_label = ttk.Label(self.media_frame, image=placeholder_image)
        self.image_label.image = placeholder_image
        self.image_label.grid(row=0, column=0, pady=10, padx=10, sticky="n")

        self._placeholder_preview = ImageTk.PhotoImage(Image.new("RGB", (4 * THUMBNAIL_SIZE, THUMBNAIL_SIZE),
                                                         (255, 255, 255)))
        self.preview_label = ttk.Label(self.media_frame, image=self._placeholder_preview)
        self.preview_label.grid(row=1, column=0, pady=10, padx=10, sticky="n")

        # Text input
        self.inputs_frame = ttk.Frame(self.frame_master)
        self.inputs_frame.grid(row=0, column=1)

        self.text_entry = scrolledtext.ScrolledText(self.inputs_frame, width=30, height=16, )
        self.text_entry.grid(row=0, column=0, pady="0 20")

        self.tags_preview_frame = tk.Frame(self.inputs_frame)
        self.tags_preview_frame.grid(row=1, column=0, pady="0 20")

        self.tag_frame = tk.Frame(self.inputs_frame)
        self.tag_frame.grid(row=2, column=0)
        self.tags_list = []

        self.tag_search = ttk.Entry(self.tag_frame, width=40, )
        self.tag_search.focus_set()
        self.tag_search.grid(row=0, column=0, pady="0 10", sticky="w")
        self.tag_search.bind("<KeyRelease>", self.filter_tag_choice)
        self.tag_search.bind("<Return>", self.add_tag)

        self.add_tag_button = ttk.Button(self.tag_frame, text="Add Tag", command=self.add_tag)
        self.add_tag_button.grid(row=0, column=1, padx=10, pady="0 10")

        self.tag_scrollbar = tk.Scrollbar(self.tag_frame, orient="vertical")
        self.tag_choice = tk.Listbox(self.tag_frame, width=40, height=25, yscrollcommand=self.tag_scrollbar.set)
        self.tag_scrollbar.config(command=self.tag_choice.yview)
        self.tag_scrollbar.grid(row=1, column=1, sticky="nsw")
        self.tag_choice.grid(row=1, column=0, sticky="w")

        self.tag_choice.bind("<Return>", self.select_tag)
        self.tag_choice.bind("<Double-1>", self.select_tag)
        self.tag_choice.bind("<Escape>", lambda x: self.tag_search.focus_set())
        self.tag_search.bind("<Down>", lambda x: self.tag_choice.focus_set())
        self.tag_search.bind("<Up>", lambda x: self.tag_choice.focus_set())

        # Controls
        self.control_frame = ttk.Frame(self.frame_master)
        self.control_frame.grid(row=1, column=0, sticky="w")

        self.select_folder_button = ttk.Button(self.control_frame, text="Select folder",
                                               command=self.select_folder)
        self.select_folder_button.grid(row=0, column=0, padx="0 30", sticky="w")

        self.prev_button = ttk.Button(self.control_frame, text="Previous", command=self.show_previous_image)
        self.prev_button.grid(row=0, column=1)

        self.next_button = ttk.Button(self.control_frame, text="Next", command=self.show_next_image)
        self.next_button.grid(row=0, column=2)

        preview_checkbox_value = tk.IntVar()
        self.preview_checkbox = ttk.Checkbutton(self.control_frame, text="Preview subfolder",
                                                state="deselected", command=self.preview_routine,
                                                variable=preview_checkbox_value)
        self.preview_checkbox.variable = preview_checkbox_value
        self.preview_checkbox.grid(row=0, column=3, padx="20 0")

        self.master.bind("<Control-Right>", lambda x: self.show_next_image())
        self.master.bind("<Control-Left>", lambda x: self.show_previous_image())
        self.master.bind("<Tab>", self.focus_routine)

    def focus_routine(self, event):
        current_widget = event.widget
        next_widget = current_widget.tk_focusNext()
        if next_widget:
            next_widget.focus_set()

    def _get_current_meta(self):
        if self.session_config is None:
            return {}
        return self.session_config["data"].get(self.image_paths[self.current_index], {})

    def create_session_config(self, session_dir: Path):
        session_file_name = self.full_session_config.get(str(session_dir))

        if session_file_name is None:
            session_name = session_dir.name
            assert session_name != "sessions", "Folder name \"sessions\" is reserved. Please rename the folder"
            new_session_config = {
                "target": str(session_dir),
                "name": session_name,
                "total_files": len(self.image_paths),
                "tags": [],
                "data": {}
            }
            self.session_config = new_session_config
            self.full_session_config[str(session_dir)] = f"{session_name}.json"
        else:
            with open(session_file_name) as session_file:
                self.session_config = json.load(session_file)
            assert self.session_config["total_files"] == len(self.image_paths), \
                f"Tried to resume session {session_dir} but missing " \
                f"{self.session_config['total_files'] - len(self.image_paths)}"
        self.save_state()

    def save_state(self):
        with open(self.session_config_path, "w") as session_config_file:
            json.dump(self.full_session_config, session_config_file)
        with open(self.full_session_config[self.session_config["target"]], "w") as session_data:
            json.dump(self.session_config, session_data)

    def select_folder(self):
        image_dir = Path(filedialog.askdirectory(title="Select Folder"))

        self.image_paths = [str(f) for f in image_dir.rglob("*") if f.suffix in [".jpeg", ".jpg", ".png"]]
        self.create_session_config(image_dir)

        self.show_current_image()

    def make_record(self):
        self.session_config["data"][self.image_paths[self.current_index]] = {
            "prompt": self.text_entry.get("1.0", "end").replace("\n", ""),
            "tags": [t.cget("text") for t in self.tags_list]
        }
        self.save_state()

    def show_image_metadata(self):
        current_meta = self._get_current_meta()
        current_prompt = current_meta.get("prompt", "")
        self.text_entry.delete("1.0", "end")
        self.text_entry.insert("1.0", current_prompt)
        self.preview_routine()
        self.filter_tag_choice(None)

        for t in self.tags_list:
            t.destroy()
        self.tags_list = []
        for tag_text in current_meta.get("tags", []):
            tag_button = ttk.Button(master=self.tags_preview_frame, text=tag_text)
            tag_button.config(command=self.make_remove_tag_func(tag_button))
            tag_button.grid(row=0, column=len(self.tags_list), sticky="w")
            self.tags_list.append(tag_button)

    def show_current_image(self):
        if self.image_paths:
            image_path = self.image_paths[self.current_index]
            square_image = resize_pad_square(image_path, IMG_SIZE)
            photo = ImageTk.PhotoImage(square_image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            self.show_image_metadata()

    def preview_routine(self):
        if self.preview_checkbox.variable.get():
            if not self.image_paths:
                return
            subdir = Path(self.image_paths[self.current_index]).parent
            to_preview = [str(f) for f in subdir.glob("*") if f.suffix in [".jpeg", ".jpg", ".png"]]
            thumbnails = [resize_pad_square(f, THUMBNAIL_SIZE) for f in to_preview]
            preview = Image.new("RGB", (len(thumbnails) * THUMBNAIL_SIZE, THUMBNAIL_SIZE), (255, 255, 255))
            for i, img in enumerate(thumbnails):
                preview.paste(img, (i * THUMBNAIL_SIZE, 0))
            preview_tk = ImageTk.PhotoImage(image=preview)
            self.preview_label.config(image=preview_tk)
            self.preview_label.image = preview_tk
        else:
            self.preview_label.config(image=self._placeholder_preview)

    def show_next_image(self):
        if self.image_paths:
            self.make_record()
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.show_current_image()

    def show_previous_image(self):
        if self.image_paths:
            self.make_record()
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.show_current_image()

    def filter_tag_choice(self, event):
        if self.session_config is None:
            return

        search_key = self.tag_search.get()
        tags_pool = self.session_config["tags"]
        filtered = [t for t in tags_pool if search_key in t]
        self.tag_choice.delete(0, 'end')
        for t in filtered:
            self.tag_choice.insert('end', t)

    def add_tag(self, event=None):
        if self.session_config is None:
            return
        if self.tag_search.get() in self.session_config["tags"]:
            return
        self.session_config["tags"].append(self.tag_search.get())
        self.filter_tag_choice(None)
        self.save_state()

    def remove_tag(self, tag_widget):
        self.tags_list.remove(tag_widget)
        tag_widget.destroy()
        self.make_record()
        for i, tag in enumerate(self.tags_list):
            tag.grid(row=0, column=i, sticky="w")

    def make_remove_tag_func(self, tag_widget):
        return lambda: self.remove_tag(tag_widget)

    def select_tag(self, event):
        selected_ids = self.tag_choice.curselection()
        if selected_ids:
            selected_value = self.tag_choice.get(selected_ids)
            if selected_value in self._get_current_meta().get("tags", []):
                return
            tag_button = ttk.Button(master=self.tags_preview_frame, text=selected_value, )
            tag_button.config(command=self.make_remove_tag_func(tag_button))
            tag_button.grid(row=0, column=len(self.tags_list), sticky="w")
            self.tags_list.append(tag_button)
            self.make_record()


if __name__ == "__main__":
    root = tk.Tk()

    app = ImageViewerApp(root)
    root.geometry("+100+10")
    root.mainloop()
