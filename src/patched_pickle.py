import pickle


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print(module, name)
        return super().find_class(module, name)


def load(file, *argv, **kwargs):
    return Unpickler(file, *argv, **kwargs).load()
