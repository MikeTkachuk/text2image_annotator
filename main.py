import json

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk, scrolledtext
from PIL import Image, ImageTk
from tqdm import tqdm

from tag_recommendation import EmbeddingStore

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


def sort_with_ranks(seq, ranks, reverse=True, return_rank=True):
    assert len(seq) == len(ranks)
    cat = list(zip(seq, ranks))
    sorted_cat = sorted(cat, key=lambda x: x[1], reverse=reverse)
    if return_rank:
        return sorted_cat
    return [cat_el[0] for cat_el in sorted_cat]


def put_node(tree: ttk.Treeview, path, node: str, score=None):
    def path_to_node_id(p: Path):
        if not len(p.parents):
            return ''
        else:
            return p.as_posix()

    path = Path(path)
    parents = [path] + list(path.parents)
    if len(parents) > 1:
        for i in range(1, len(parents)):
            parent_id = path_to_node_id(parents[-i])
            node_id = path_to_node_id(parents[-i - 1])
            if not tree.exists(node_id):
                tree.insert(parent_id,
                            "end", node_id,
                            text=parents[-i - 1].name, open=True)
    tree.insert(path_to_node_id(parents[0]), 'end', node, text=node, values=(score,))


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

        self.frame_master = ttk.Frame(master=root, padding=4)
        self.frame_master.grid(row=0, column=0)

        # Media
        self.media_frame = ttk.Frame(master=self.frame_master)
        self.media_frame.grid(row=0, column=0)

        placeholder_image = ImageTk.PhotoImage(Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255)))
        self.image_label = ttk.Label(self.media_frame, image=placeholder_image)
        self.image_label.image = placeholder_image
        self.image_label.grid(row=0, column=0, pady=5, padx=10, sticky="n")

        self._placeholder_preview = ImageTk.PhotoImage(Image.new("RGB", (4 * THUMBNAIL_SIZE, THUMBNAIL_SIZE),
                                                                 (
                                                                     255, 255,
                                                                     255)))  # used whenever preview is turned off
        self.preview_label = ttk.Label(self.media_frame, image=self._placeholder_preview)
        self.preview_label.grid(row=1, column=0, pady=10, padx=10, sticky="n")

        # Text input
        self.inputs_frame = ttk.Frame(self.frame_master)
        self.inputs_frame.grid(row=0, column=1)

        self.text_entry = scrolledtext.ScrolledText(self.inputs_frame, width=30, height=8, )
        self.text_entry.grid(row=0, column=0, pady="0 5", sticky="w")

        # # Tags preview
        self.tags_preview_frame = tk.Frame(self.inputs_frame, width=350)
        self.tags_preview_frame.grid(row=0, column=1, pady="0 20", sticky="wn")

        self.tag_to_row = {}
        self.tags_list = []

        # # CLIP settings
        self.emb_store = None
        self.tag_rec_frame = tk.Frame(self.inputs_frame)
        self.tag_rec_frame.grid(row=1, column=1, pady="0 20", sticky="ws")
        self.tag_rec_mode = tk.StringVar()
        self.tag_rec_mode.set("alphabetic")
        self.tag_rec_mode_menu = tk.OptionMenu(self.tag_rec_frame, self.tag_rec_mode, self.tag_rec_mode.get(),
                                               *("openai/clip-vit-base-patch32",
                                                 "openai/clip-vit-large-patch14", "popularity"))
        self.tag_rec_mode_menu.grid(row=1, column=0, sticky="ws")

        self.recompute_button = tk.Button(self.tag_rec_frame, text="Recompute",
                                          command=lambda: (self.recompute_recommendation(),
                                                           self.filter_tag_choice))
        self.recompute_button.grid(row=1, column=1, sticky="ws")

        self.blank_entry = scrolledtext.ScrolledText(self.tag_rec_frame, height=16, width=30)
        self.blank_entry.grid(row=0, column=0, columnspan=2)

        # # Tags selection
        self.tag_frame = tk.Frame(self.inputs_frame)
        self.tag_frame.grid(row=1, column=0)
        self.tag_input_frame = tk.Frame(self.tag_frame)
        self.tag_input_frame.grid(row=0, column=0, columnspan=1, sticky="w")
        self.tag_search_label = tk.Label(self.tag_input_frame, text="Search:")
        self.tag_search_label.grid(row=0, column=0, sticky="w")
        self.tag_search = ttk.Entry(self.tag_input_frame, width=30)
        self.tag_search.focus_set()
        self.tag_search.grid(row=0, column=1, pady="5 5", sticky="w")
        self.tag_search.bind("<KeyRelease>", self.filter_tag_choice)
        self.tag_search.bind("<Return>", self.add_tag)
        self.tag_search.bind("<Control-Return>", lambda x: (self.add_tag(),
                                                            self.tag_choice.selection_set(self.tag_search.get()),
                                                            self.select_tag()))
        self.tag_structure_label = tk.Label(self.tag_input_frame, text="Structure:")
        self.tag_structure_label.grid(row=1, column=0, sticky="w")
        self.tag_structure_entry = ttk.Entry(self.tag_input_frame, width=30)
        self.tag_structure_entry.grid(row=1, column=1, sticky="w")
        self.tag_structure_entry.bind("<Return>", self.add_tag)

        self.add_tag_button = ttk.Button(self.tag_input_frame, text="Add Tag", command=self.add_tag)
        self.add_tag_button.grid(row=0, column=2, padx=10, pady="0 0")

        self.delete_tag_button = ttk.Button(self.tag_input_frame, text="Delete Tag", command=self.delete_tag)
        self.delete_tag_button.grid(row=1, column=2, padx=10, pady="0 0")
        self.move_tag_button = ttk.Button(self.tag_input_frame, text="Move", command=self.edit_structure,
                                          width=8)
        self.move_tag_button.grid(row=2, column=0, pady="0 0")
        tree_checkbox_value = tk.IntVar()
        self.tree_checkbox = ttk.Checkbutton(self.tag_input_frame, text="Tree view",
                                             state="deselected", command=self.filter_tag_choice,
                                             variable=tree_checkbox_value)
        self.tree_checkbox.variable = tree_checkbox_value
        self.tree_checkbox.grid(row=2, column=2, padx="0 0")

        self.tag_select_frame = tk.Frame(self.tag_frame)
        self.tag_select_frame.grid(row=2, column=0, sticky="w")
        self.tag_scrollbar = tk.Scrollbar(self.tag_select_frame, orient="vertical")
        self.tag_choice = ttk.Treeview(self.tag_select_frame, height=15,
                                       yscrollcommand=self.tag_scrollbar.set, selectmode="browse",
                                       columns=("Rank",))
        self.tag_choice.column("#0", width=210)
        self.tag_choice.heading("#0", text="Name")
        self.tag_choice.column("Rank", width=35)
        self.tag_choice.heading("Rank", text="Rank")
        self.tag_scrollbar.config(command=self.tag_choice.yview)
        self.tag_scrollbar.grid(row=0, column=1, sticky="nsw")
        self.tag_choice.grid(row=0, column=0, sticky="w")

        self.tag_choice.bind("<Return>", self.select_tag)
        self.tag_choice.bind("<Double-1>", self.select_tag)
        self.master.bind("<Escape>", lambda x: (self.tag_search.focus_set(),
                                                self.tag_search.selection_range(0, 'end')))
        self.tag_choice.bind("<<TreeviewSelect>>", lambda x: (self.tag_structure_entry.delete(0, 'end'),
                                                              self.tag_structure_entry.insert(0,
                                                                                              self._get_current_tag_path()),
                                                              ))

        def _get_focus_to_choice_func(end=False):
            def func(x=None):
                self.tag_choice.focus_set()
                item = self._get_recursive_index(-1 if end else 0)
                self.tag_choice.selection_set(item)
                self.tag_choice.focus(item)
                self.tag_structure_entry.delete(0, 'end'),
                self.tag_structure_entry.insert(0, self._get_current_tag_path())

            return func

        self.tag_search.bind("<Down>", _get_focus_to_choice_func())
        self.tag_search.bind("<Up>", _get_focus_to_choice_func(end=True))

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

        self.next_unlabeled_button = ttk.Button(self.control_frame, text="Next Unlabeled",
                                                command=self.show_next_unlabeled_image)
        self.next_unlabeled_button.grid(row=0, column=3)

        preview_checkbox_value = tk.IntVar()
        self.preview_checkbox = ttk.Checkbutton(self.control_frame, text="Preview subfolder",
                                                state="selected", command=self.preview_routine,
                                                variable=preview_checkbox_value)
        self.preview_checkbox.variable = preview_checkbox_value
        self.preview_checkbox.grid(row=0, column=4, padx="20 0")

        tags_render_value = tk.IntVar()
        self.tags_render_checkbox = ttk.Checkbutton(self.control_frame, text="Edit tags",
                                                    state="deselected", command=self.show_image_metadata,
                                                    variable=tags_render_value)
        self.tags_render_checkbox.variable = tags_render_value
        self.tags_render_checkbox.grid(row=0, column=5, padx="20 0")

        self.progress_info = tk.Label(self.tag_select_frame, text="")
        self.progress_info.grid(row=1, column=0, sticky="w")

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
                "tags_structure": {},
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
        self.current_index = 0
        self.show_current_image()

    def make_record(self):
        prompt = self.text_entry.get("1.0", "end").replace("\n", "")
        tags = [t.cget("text") for t in self.tags_list]

        if not prompt and not tags:
            return

        self.session_config["data"][self.image_paths[self.current_index]] = {
            "prompt": prompt,
            "tags": tags
        }
        self.save_state()

    def _add_tag_widget(self, tag_text):
        if not self.tags_render_checkbox.variable.get():
            tags_label = list(self.tags_preview_frame.children.values())[0]
            tags_label.config(text=tags_label.cget("text") + tag_text + "; ")
            self.tags_list.append(tk.Button(text=tag_text))
            return

        def _create_tag(master_el):
            out = tk.Button(master=master_el, text=tag_text,
                            wraplength=self.tags_preview_frame.cget("width") - 10)
            out.config(command=self.make_remove_tag_func(out))
            return out

        if self.tags_list:
            last_tag = self.tags_list[-1]
            last_row = self.tag_to_row[last_tag]
        else:  # init the first row and exit
            last_row = tk.Frame(master=self.tags_preview_frame)
            last_row.pack(side="top", anchor="nw")
            tag_button = _create_tag(last_row)
            tag_button.pack(side="left", anchor="nw")
            self.tags_list.append(tag_button)
            self.tag_to_row[tag_button] = last_row
            return

        tag_button = _create_tag(last_row)
        tag_button.pack(side="left", anchor="nw")
        max_width = self.tags_preview_frame.cget("width")
        last_row.update_idletasks()
        current_row_width = last_row.winfo_reqwidth()
        if current_row_width > max_width:
            tag_button.destroy()
            last_row = tk.Frame(master=self.tags_preview_frame)
            last_row.pack(side="top", anchor="nw")
            tag_button = _create_tag(last_row)
            tag_button.pack(side="left", anchor="nw")
        self.tags_list.append(tag_button)
        self.tag_to_row[tag_button] = last_row

    def show_image_metadata(self):
        current_meta = self._get_current_meta()
        current_prompt = current_meta.get("prompt", "")
        self.text_entry.delete("1.0", "end")
        self.text_entry.insert("1.0", current_prompt)
        self.preview_routine()
        self.filter_tag_choice()
        self.progress_info.config(text=f"Progress: {len(self.session_config['data'])}/{len(self.image_paths)}\n"
                                       f"Id: {self.current_index}")

        for t in self.tags_list:
            t.destroy()
        for row in list(self.tags_preview_frame.children.values()):
            row.destroy()
        self.tags_list = []
        self.tag_to_row = {}
        should_render = self.tags_render_checkbox.variable.get()
        if not should_render:
            tags_label = tk.Label(master=self.tags_preview_frame, text="", font=("Arial", 12),
                                  wraplength=self.tags_preview_frame.cget("width"))
            tags_label.pack()
        for tag_text in current_meta.get("tags", []):
            self._add_tag_widget(tag_text)

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

    def show_next_unlabeled_image(self):
        if not self.image_paths:
            return

        self.make_record()
        search_cursor = self.current_index + 1
        while search_cursor < (len(self.image_paths)):
            if self.image_paths[search_cursor] not in self.session_config["data"]:
                self.current_index = search_cursor
                self.show_current_image()
                return
            search_cursor += 1

    def edit_structure(self):
        if not self.tag_choice.selection():
            return
        format_structure = self.tag_structure_entry.get()
        format_structure = '/' + format_structure.strip('/')
        self.session_config["tags_structure"][self.tag_choice.selection()[0]] = format_structure
        self.filter_tag_choice()
        self.save_state()

    def add_tag(self, event=None):
        if self.session_config is None:
            return
        if not self.tag_search.get():
            return
        if not self.tag_search.get() in self.session_config["tags"]:
            self.session_config["tags"].append(self.tag_search.get())
        format_structure = self.tag_structure_entry.get()
        format_structure = '/' + format_structure.strip('/')
        self.session_config["tags_structure"][self.tag_search.get()] = format_structure
        self.filter_tag_choice()
        self.save_state()

    def delete_tag(self):
        tag = self.tag_choice.selection()
        if not tag:
            return
        tag = tag[0]
        if not self.tag_choice.get_children(tag):
            self.session_config["tags"].remove(tag)
            self.session_config["tags_structure"].pop(tag)
            self.filter_tag_choice()
            self.save_state()

    def _helper_remove_tag(self, tag_widget):
        tag_id = self.tags_list.index(tag_widget)
        self.tags_list.pop(tag_id)
        self.tag_to_row.pop(tag_widget)
        self.make_record()

        self.show_image_metadata()  # reload metadata after update

    def make_remove_tag_func(self, tag_widget):
        return lambda: self._helper_remove_tag(tag_widget)

    def select_tag(self, event=None):
        selected = self.tag_choice.selection()
        if selected:
            tag = selected[0]
            if self.tag_choice.get_children(tag):  # if not leaf
                return
            if selected in self._get_current_meta().get("tags", []):
                return
            self._add_tag_widget(tag)
            self.make_record()

    def filter_tag_choice(self, event=None):
        if self.session_config is None:
            return
        search_key = self.tag_search.get()
        tags_pool = self.sort_tags()

        filtered = []
        scores = []
        for t in tags_pool:
            score = None
            if isinstance(t, tuple):
                t, score = t
            if search_key in t or search_key in self.session_config["tags_structure"].get(t, '/'):
                filtered.append(t)
                scores.append(score)
        for ch in self.tag_choice.get_children():
            self.tag_choice.delete(ch)
        for t, score in zip(filtered, scores):
            if self.tree_checkbox.variable.get():
                hierarchy = Path(self.session_config["tags_structure"].get(t, '/'))
                put_node(self.tag_choice, hierarchy, t, score=score)
            else:
                self.tag_choice.insert('', 'end', t, text=t, values=(score,))

    def _get_current_tag_path(self):
        if not self.tag_choice.selection():
            return ''
        selection = self.tag_choice.selection()[0]
        if not self.tag_choice.get_children(selection):
            path = self.tag_choice.parent(selection)
        else:
            path = selection
        return '/' + path.strip('/')

    def _get_recursive_index(self, index=0):
        item = ''
        while True:
            if not self.tag_choice.get_children(item):
                return item
            item = self.tag_choice.get_children(item)[index]

    def sort_tags(self):
        if self.tag_rec_mode.get() == "alphabetic":
            return sorted(self.session_config["tags"])
        if self.tag_rec_mode.get() == "popularity":
            raise NotImplementedError

        ranks = self.emb_store.get_tag_ranks(self.session_config["tags"],
                                             self.image_paths[self.current_index],
                                             self.session_config["target"])
        return sort_with_ranks(self.session_config["tags"], ranks)

    def recompute_recommendation(self):
        if self.session_config is None:
            return
        if "clip" not in self.tag_rec_mode.get():
            return

        if self.emb_store is None:
            emb_store_path = "." + Path(self.session_config["target"]).name
            self.emb_store = EmbeddingStore(emb_store_path, model_name=self.tag_rec_mode.get())

        for img_path in tqdm(self.session_config["data"], desc="Loading image embeddings"):
            self.emb_store.get_image_embedding(img_path, self.session_config["target"])

        for tag in tqdm(self.session_config["tags"], desc="Loading tag embeddings"):
            self.emb_store.get_tag_embedding(tag)

        # perform training
        # =====


if __name__ == "__main__":
    root = tk.Tk()

    app = ImageViewerApp(root)
    root.geometry("+100+10")
    root.state('zoomed')

    root.mainloop()
