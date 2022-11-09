from feature_engineering import *


class Basic_Continuous(Feature):

    def create_features(self, df):
        df = df[['Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment','Size']]
        return df , df.columns



def main():            
    sample = Basic_Continuous()
    sample.get_dataset()
    sample.run()

if __name__ == '__main__':
    main()