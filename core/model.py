from pathlib import Path
import pickle


class Model:
    def __init__(self, model, params, framework, save_path):
        self.model = model
        self.params = params
        self.framework = framework
        self.save_path = Path(save_path)

    def save(self):
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True)
        with open(self.save_path, "wb") as file:
            pickle.dump(self.model, file)

    def __dict__(self):
        self.save()
        return {"path": self.save_path, "params": self.params, "framework": self.framework}

    @classmethod
    def load(cls, data):
        with open(data["path"], "rb") as file:
            model = Model(pickle.load(file), data["params"], data["framework"], data["path"])
        return model