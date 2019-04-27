import pandas as pd
"""
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
PassengerId 고객아이디      
Survived 생존여부     Survival    0 = No, 1 = Yes
Pclass 승선권 클래스    Ticket class    1 = 1st, 2 = 2nd, 3 = 3rd
Name 이름
Sex  성별  Sex    
Age  나이  Age in years    
SibSp  동반한 형제자매, 배우자 수  # of siblings / spouses aboard the Titanic    
Parch  동반한 부모, 자식 수  # of parents / children aboard the Titanic    
Ticket  티켓 번호  Ticket number    
Fare  티켓의 요금  Passenger fare    
Cabin  객실번호  Cabin number    
Embarked  승선한 항구명  Port of Embarkation  
  C = Cherbourg 쉐부로, Q = Queenstown 퀸스타운, S = Southampton 사우스햄톤
"""
class TitanicModel:
    def __init__(self):
        self._context = None
        self._fname = None
        self._train = None
        self._test = None

    @property
    def context(self) -> object: return self._context

    @context.setter
    def context(self, context): self._context = context

    @property
    def fname(self) -> object: return self._fname

    @context.setter
    def fname(self, fname): self._fname = fname

    @property
    def train(self) -> object: return self._train

    @train.setter
    def train(self, train): self._train = train

    @property
    def test(self) -> object: return self._test

    @test.setter
    def test(self, test): self._test = test

    def new_file(self) -> str:
        return self._context + self._fname

    def new_dframe(self) -> object:
        file = self.new_file()
        return pd.read_csv(file)

    @staticmethod
    def drop_feature(train, test, feature):
        train = train.drop([feature], axis = 1)
        test = test.drop([feature], axis = 1)
        return [train, test]

    @staticmethod
    def embarked_norminal(train, test):
        s_city = train[train['Embarked'] == 'S'].shape[0]  # 스칼라
        c_city = train[train['Embarked'] == 'C'].shape[0]
        q_city = train[train['Embarked'] == 'Q'].shape[0]

        # print("S = ",s_city) #644
        # print("C = ",c_city) #168
        # print("Q = ",q_city) #77

        train = train.fillna({"Embarked": "S"})
        city_mapping = {"S": 1, "C": 2, "Q": 3}
        train['Embarked'] = train['Embarked'].map(city_mapping)
        test['Embarked'] = test['Embarked'].map(city_mapping)

    @staticmethod
    def title_norminal(train, test):
        combine = [train, test]
        for dataset in combine:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
        # print(pd.crosstab(train['Title'],train['Sex']))

        for dataset in combine:
            dataset['Title'] \
                = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
            dataset['Title'] \
                = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            dataset['Title'] \
                = dataset['Title'].replace('Mlle', 'Miss')
            dataset['Title'] \
                = dataset['Title'].replace('Ms', 'Miss')
            dataset['Title'] \
                = dataset['Title'].replace('Mme', 'Mrs')
        train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
        """
            Title  Survived
        0  Master  0.575000
        1    Miss  0.701087
        2      Mr  0.156673
        3     Mrs  0.793651
        4      Ms  1.000000
        5    Rare  0.250000
        6   Royal  1.000000
        """

        title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}
        for dataset in combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)



















