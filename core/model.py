from pathlib import Path
import pickle
import inspect


# todo: self.model is a class name. add the model instance
#  to self._model and save this. it's for the parameter updating
class Model:
    def __init__(self, model, params, framework, embstore_name, save_path):
        self.model = model
        self.params = params
        self.framework = framework
        self.save_path = Path(save_path)
        self.embstore_name = embstore_name

        self._model_obj = None
        self._predictions = []

    def save(self):
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True)
        with open(self.save_path, "wb") as file:
            pickle.dump(self.model, file)

    def get_model_signature(self):
        signature = inspect.signature(self.model.__init__)
        out = {}
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue
            default = param.default
            if default == inspect._empty:
                continue
            out[param_name] = default
        return out

    def __dict__(self):
        self.save()
        return {"path": self.save_path,
                "params": self.params,
                "framework": self.framework,
                "embstore_name": self.embstore_name}

    @classmethod
    def load(cls, data):
        with open(data["path"], "rb") as file:
            model = Model(pickle.load(file),
                          data["params"],
                          data["framework"],
                          data["embstore_name"],
                          data["path"])
        return model

    def fit(self):
        self._model_obj = self.model(**self.params)

    def get_predictions(self):
        return self._predictions
