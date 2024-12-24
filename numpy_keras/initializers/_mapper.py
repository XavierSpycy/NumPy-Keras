from . import main

class _InitializerMapper:
    initializer_class_mapper = {
        "constant": main.Constant,
        "glorot_normal": main.GlorotNormal,
        "glorot_uniform": main.GlorotUniform,
        "he_normal": main.HeNormal,
        "he_uniform": main.HeUniform,
        "ones": main.Ones,
        "random_normal": main.RandomNormal,
        "random_uniform": main.RandomUniform,
        "zeros": main.Zeros,
    }

    def __getitem__(self, name: str):
        if not name.lower() in self.initializer_class_mapper:
            raise ValueError(f"Initializer {name} not found.")
        else:
            return self.initializer_class_mapper[name.lower()]