import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
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

    def hook_process(self, train, test) -> object:
        print('--------- 1. Cabin, Ticket 삭제 ----------------')
        t = self.drop_feature(train, test, 'Cabin')
        t = self.drop_feature(t[0],t[1], 'Ticket')
        print('--------- 2. embarked 편집 ----------------')
        t = self.embarked_nominal(t[0],t[1])
        print('--------- 3. title 편집 ----------------')
        t = self.title_nominal(t[0],t[1])
        print('--------- 4. name, PassengerId 삭제 ----------------')
        t = self.drop_feature(t[0],t[1],'Name')
        t = self.drop_feature(t[0],t[1],'PassengerId')
        print('--------- 5. age 편집 ----------------')
        t = self.age_ordinal(t[0],t[1])
        print('--------- 6. age 삭제 ----------------')
        t = self.drop_feature(t[0],t[1],'Age')
        print('--------- 7. fare 편집 ----------------')
        t = self.fare_ordinal(t[0],t[1])
        print('--------- 8. fare 삭제 ----------------')
        t = self.drop_feature(t[0],t[1],'Fare')
        print('--------- 9. sex 편집 ----------------')
        t = self.sex_nominal(t[0],t[1])
        return t[0]



    @staticmethod
    def drop_feature(train, test, feature) -> []:
        train = train.drop([feature], axis = 1)
        test = test.drop([feature], axis = 1)
        return [train, test]

    @staticmethod
    def embarked_nominal(train, test) -> []:
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
        return [train, test]

    @staticmethod
    def title_nominal(train, test) -> []:
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

        return [train, test]

    @staticmethod
    def sex_nominal(train, test) -> []:
        combine = [train, test]
        sex_mapping = {"male": 0, "female": 1}
        for dataset in combine:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)

        return [train, test]

    @staticmethod
    def age_ordinal(train, test) -> []:
        train['Age'] = train['Age'].fillna(-0.5)
        test['Age'] = test['Age'].fillna(-0.5)
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        test['AgeGroup'] = pd.cut(test['Age'], bins, labels=labels)

        age_title_mapping = {0: "Unknown", 1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult",
                             6: "Adult"}
        for x in range(len(train['AgeGroup'])):
            if train["AgeGroup"][x] == "Unknown":
                train["AgeGroup"][x] = age_title_mapping[train['Title'][x]]
        for x in range(len(test['AgeGroup'])):
            if test["AgeGroup"][x] == "Unknown":
                test["AgeGroup"][x] = age_title_mapping[test['Title'][x]]

        age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3,
                       'Student': 4, "Young Adult": 5, "Adult": 6, 'Senior': 7}
        train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

        return [train, test]

    @staticmethod
    def fare_ordinal(train, test) -> []:
        train['FareBand'] = pd.qcut(train['Fare'], 4, labels={1, 2, 3, 4})
        test['FareBand'] = pd.qcut(test['Fare'], 4, labels={1, 2, 3, 4})

        return [train, test]

    @staticmethod
    def create_model_dummy(train) -> []:
       model = train.drop('Survived', axis = 1)
       dummy = train['Survived']
       return [model, dummy]

    """
    Learning Part
    """
    @staticmethod
    def create_random_variables(train) -> str:
        train2, test2 = train_test_split(train, test_size=0.3, random_state=0)
        target_col = ['Pclass', 'Sex', 'Embarked']
        train_X = train2[target_col]
        train_Y = train2['Survived']
        test_X = test2[target_col]
        test_Y = test2['Survived']

        features_one = train_X.values
        target = train_Y.values

        tree_model = DecisionTreeClassifier()
        tree_model.fit(features_one, target)
        dt_prediction = tree_model.predict(test_X)
        accuracy = metrics.accuracy_score(dt_prediction, test_Y)
        print('The accuracy of the Decision Tree is  ', accuracy)
        return accuracy

    @staticmethod
    def create_k_fold():
        k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
        return k_fold

    def accuracy_by_knn(self, model, dummy) -> str:
        clf = KNeighborsClassifier(n_neighbors=13)
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold,
                                n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_dtree(self, model, dummy) -> str:
        print('>>> 결정트리 방식 검증')  # 79.58
        k_fold = self.create_k_fold()
        clf = DecisionTreeClassifier()
        scoring = 'accuracy'
        score = cross_val_score(clf, model, dummy, cv=k_fold,
                                n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_rforest(self, model, dummy) -> str:

        print('>>> 램덤포레스트 방식 검증')  # 82.15
        k_fold = self.create_k_fold()
        clf = RandomForestClassifier(n_estimators=13)  # 13개의 결정트리를 사용함
        scoring = 'accuracy'
        score = cross_val_score(clf, model, dummy, cv=k_fold,
                                n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_nb(self, model, dummy) -> str:
        print('>>> 나이브베이즈 방식 검증')  # 79.57
        clf = GaussianNB()
        k_fold = self.create_k_fold()
        scoring = 'accuracy'
        score = cross_val_score(clf, model, dummy, cv=k_fold,
                                n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_svm(self, model, dummy) -> str:
        k_fold = self.create_k_fold()
        print('>>> SVM 방식 검증')  # 83.05
        clf = SVC()
        scoring = 'accuracy'
        score = cross_val_score(clf, model, dummy, cv=k_fold,
                                n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy














