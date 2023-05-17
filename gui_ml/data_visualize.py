import pandas as pd



class data_ :

    def read_file(self, filepath):
        
        if str(filepath) == '' :
            return ""
        else :
            return pd.read_csv(str(filepath), index_col=False)
    

    
    def get_column_list(self, df):
        # print(df.columns)

        # df 그대로 안받는 이유는 리스트 형태로 받기위해서 새 리스트에 추가
        columnname_list = []

        for i in df.columns :
            columnname_list.append(i)
        return columnname_list
    
        # 또는 이런 방법을 쓸 수 있다
        # print(list(df.columns))
        # return list(df.columns)