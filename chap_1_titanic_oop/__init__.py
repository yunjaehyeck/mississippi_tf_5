from chap_1_titanic_oop.controller import TitanicController
import warnings
warnings.simplefilter(action= 'ignore', category= FutureWarning)

if __name__ == '__main__':
    c = TitanicController()
    c.test_random_variables()
    c.test_by_sklearn()
    c.submit()

