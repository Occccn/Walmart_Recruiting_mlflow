from feature_engineering import *


class Basic_Continuous(Feature):

    def create_features(self, df):
        df = df[['Type' , 'IsHoliday']]
        return df , df.columns



def main():            
    sample = Basic_Continuous()
    sample.get_dataset()
    sample.run()

if __name__ == '__main__':
    main()