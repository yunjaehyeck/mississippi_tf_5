import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TitanicView:

    @staticmethod
    def plot_survived_dead(train):
        f, ax = plt.subplots(1, 2, figsize=(18, 8))
        train['Survived'].value_counts().plot.pie(explode=[0, 0.1],
                                                  autopct="%1.1f%%", ax=ax[0], shadow=True)
        ax[0].set_title('Survived')
        ax[0].set_ylabel('')

        sns.countplot('Survived', data=train, ax=ax[1])
        ax[1].set_title('Survived')
        plt.show() # 생존률 38.4% 사망률 61.6%

    @staticmethod
    def plot_sex(train):
        f, ax = plt.subplots(1, 2, figsize=(18, 8))
        train['Survived'][train["Sex"] == 'male'].value_counts().plot.pie(explode=[0, 0.1],
                                                                          autopct="%1.1f%%", ax=ax[0], shadow=True)
        train['Survived'][train["Sex"] == 'female'].value_counts().plot.pie(explode=[0, 0.1],
                                                                            autopct="%1.1f%%", ax=ax[1], shadow=True)
        ax[0].set_title('Survived(Male)')
        ax[1].set_title('Survived(Female)')

        plt.show()

    @staticmethod
    def plot_pclass_sex(train):
        df_1 = [train['Sex'], train['Survived']]
        df_2 = train['Pclass']
        df = pd.crosstab(df_1, df_2, margins=True)
        # print(df.head())
        """
        Pclass             1    2    3  All
        Sex    Survived                    
        female 0           3    6   72   81
               1          91   70   72  233
        male   0          77   91  300  468
               1          45   17   47  109
        All              216  184  491  891
        """
        # Embarked 는 배를 탄 항구
        f, ax = plt.subplots(2, 2, figsize=(20, 15))
        sns.countplot('Embarked', data=train, ax=ax[0, 0])
        ax[0, 0].set_title('No. Of Passengers Boarded')
        sns.countplot('Embarked', hue='Sex', data=train, ax=ax[0, 1])
        ax[0, 1].set_title('Male - Female for Embarked')
        sns.countplot('Embarked', hue='Survived', data=train, ax=ax[1, 0])
        ax[1, 0].set_title('Embarked vs Survived')
        sns.countplot('Pclass', data=train, ax=ax[1, 1])
        ax[1, 1].set_title('Embarked vs PClass')

        plt.show()

    @staticmethod
    def bar_chart(train, feature):
        survived = train[train['Survived'] == 1][feature].value_counts()
        dead = train[train['Survived'] == 0][feature].value_counts()
        df = pd.DataFrame([survived, dead])
        df.index = ['survived', 'dead']
        df.plot(kind='bar', stacked=True, figsize=(10, 5))
        plt.show()








