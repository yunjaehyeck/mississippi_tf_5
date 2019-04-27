from chap_1_titanic_oop.model import TitanicModel
from chap_1_titanic_oop.view import TitanicView

class TitanicController:

    def __init__(self):
        self._m = TitanicModel()
        self._v = TitanicView()
        self._context = './data/'
        self._train = self.createTrain()

    def createTrain(self) -> object:
        m = self._m
        m.context = self._context
        m.fname = 'train.csv'
        t1 = m.new_dframe()
        m.fname = 'test.csv'
        t2 = m.new_dframe()
        train = m.hook_process(t1, t2)
        print('--------- 1 ------------')
        print(train.columns)
        print('--------- 2 ------------')
        print(train.head())
        return train

