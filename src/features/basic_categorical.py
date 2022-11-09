from feature_engineering import *
from sklearn.preprocessing import LabelEncoder


class Basic_Categorical(Feature):

    def create_features(self, df):
        class_le = LabelEncoder()
        df = df[['IsHoliday','Type']]
        df['IsHoliday'] = class_le.fit_transform(df['IsHoliday'])
        df['Type'] = class_le.fit_transform(df['Type'])
        return df , df.columns



def main():            
    sample = Basic_Categorical()
    sample.get_dataset()
    sample.run()

if __name__ == '__main__':
    main()