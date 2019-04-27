from chap_1_titanic_oop.model import TitanicModel
from chap_1_titanic_oop.view import TitanicView
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class TitanicController:

    def __init__(self):
        self._m = TitanicModel()
        self._v = TitanicView()
        self._context = './data/'
        self._train = self.create_train()

    def create_train(self) -> object:
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

    def create_model(self) -> object:
        train = self._train
        model = train.drop('Survived', axis = 1)
        print('--------- model info ---------')
        print(model.info)
        return model

    def create_dummy(self) -> object:
        train = self._train
        dummy = train['Survived']
        print('--------- dummy info ---------')
        print(dummy.info)
        return dummy

    @staticmethod
    def create_random_variables(train, X_features, Y_features) -> []:
        the_X_features = X_features
        the_Y_features = Y_features
        train2, test2 = train_test_split(train, test_size=0.3, random_state=0)
        train_X = train2[the_X_features]
        train_Y = train2[the_Y_features]
        test_X = test2[the_X_features]
        test_Y = test2[the_Y_features]
        return [train_X, train_Y, test_X, test_Y]

    """
    Learning Part
    """
    def test_random_variables(self) -> str:
        train = self._train
        X_features = ['Pclass', 'Sex', 'Embarked']
        Y_features = ['Survived']
        random_variables = self.create_random_variables(train, X_features, Y_features)
        accuracy = self.accuracy_by_decision_tree(
            random_variables[0],
            random_variables[1],
            random_variables[2],
            random_variables[3]
        )
        return accuracy


    @staticmethod
    def accuracy_by_decision_tree(train_X, train_Y,  test_X, test_Y) -> str:
        tree_model = DecisionTreeClassifier()
        tree_model.fit(train_X.values, train_Y.values)
        dt_prediction = tree_model.predict(test_X)
        accuracy = metrics.accuracy_score(dt_prediction, test_Y)
        return accuracy



















