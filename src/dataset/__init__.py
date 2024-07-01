from .abalone import load_abalone
from .diabetes import load_diabetes
from .kin8nm import load_kin8nm
from .openml import load_openml


if __name__ == '__main__':
    data = load_openml("kin8nm")
    print(data[0].shape)