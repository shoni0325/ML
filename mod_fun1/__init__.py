import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Warning 무시
import warnings
warnings.filterwarnings('ignore')




from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression





# 7.titanic ML에서 만든 함수 가져오기

# Null 처리 함수
def fillna_1(df) :
    df['Age'].fillna(df['Age'].mean(), inplace = True)
    df['Cabin'].fillna('N', inplace = True)
    df['Embarked'].fillna('N', inplace = True)
    df['Fare'].fillna(0, inplace = True)

    return df

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df) :
    df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)

    return df

# 레이블 인코딩 수행
def format_features(df) :
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']

    for feature in features :
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])

    return df

# 앞에서 설정한 Data Preprocessing 함수 호출
def transform_features(df) :
    df = fillna_1(df)
    df = drop_features(df)
    df = format_features(df)

    return df


# 8.에서 만든 함수 가져오기
def get_clf_eval_1(y_test, pred) :
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1Score = f1_score(y_test, pred)

    print('오차행렬')
    print(confusion)
    # print('정확도 : {0:.4f}, 정밀도 : {1:.4f}, 재현율 : {2:.4f}, f1Score : {f1_score:.4f}'.format(accuracy, precision, recall, f1_score))
    print(f'정확도 : {accuracy:.4f}, 정밀도 : {precision:.4f}, 재현율 : {recall:.4f}, f1Score : {f1Score:.4f}')


# 위 함수 수정
def get_clf_eval(y_test,pred , pred_proba):
    confusion =confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test,pred)
    precision= precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1score = f1_score(y_test,pred)
    auc = roc_auc_score(y_test, pred_proba)

    print('오차행렬')
    print(confusion)
    print('정확도 : {0:.4f}, 정밀도 : {1:.4f}, 재현율 : {2:.4f} , f1Score:{3:.4f}, AUC:{4:.4f}'.format(accuracy,precision,recall, f1score, auc))




# 그래프 그리기 위한 함수 생성
def roc_curve_plot(y_test, pred_proba_c1) :
    # 임계값에 따른 FPR, TPR 값을 반환 받음
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)

    # ROC Curve를 plot 곡선으로 그림
    plt.plot(fprs, tprs, label = 'ROC')

    # 가운데 대각선 직선을 그림
    plt.plot([0,1], [0,1], 'k--', label = 'Random')

    # FPR X축의 Scale을 0, 1 단위로 변경, X, Y 축명 설정 등
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity)')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()


# 9. 그래프
def precision_recall_curve_plot(y_test=None, pred_proba_c1=None):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출. 
    precisions, recalls, thresholds = precision_recall_curve( y_test, pred_proba_c1)
    
    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')
    
    # threshold 값 X 축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()


# 7.titanic ML에서 만든 함수 가져오기

# Null 처리 함수
def fillna_1(df) :
    df['Age'].fillna(df['Age'].mean(), inplace = True)
    df['Cabin'].fillna('N', inplace = True)
    df['Embarked'].fillna('N', inplace = True)
    df['Fare'].fillna(0, inplace = True)

    return df

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df) :
    df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)

    return df

# 레이블 인코딩 수행
def format_features(df) :
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']

    for feature in features :
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])

    return df

# 앞에서 설정한 Data Preprocessing 함수 호출
def transform_features(df) :
    df = fillna_1(df)
    df = drop_features(df)
    df = format_features(df)

    return df


# 8.에서 만든 함수 가져오기
def get_clf_eval_1(y_test, pred) :
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1Score = f1_score(y_test, pred)

    print('오차행렬')
    print(confusion)
    # print('정확도 : {0:.4f}, 정밀도 : {1:.4f}, 재현율 : {2:.4f}, f1Score : {f1_score:.4f}'.format(accuracy, precision, recall, f1_score))
    print(f'정확도 : {accuracy:.4f}, 정밀도 : {precision:.4f}, 재현율 : {recall:.4f}, f1Score : {f1Score:.4f}')


# 위 함수 수정
def get_clf_eval(y_test,pred , pred_proba):
    confusion =confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test,pred)
    precision= precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1score = f1_score(y_test,pred)
    auc = roc_auc_score(y_test, pred_proba)

    print('오차행렬')
    print(confusion)
    print('정확도 : {0:.4f}, 정밀도 : {1:.4f}, 재현율 : {2:.4f} , f1Score:{3:.4f}, AUC:{4:.4f}'.format(accuracy,precision,recall, f1score, auc))




# 그래프 그리기 위한 함수 생성
def roc_curve_plot(y_test, pred_proba_c1) :
    # 임계값에 따른 FPR, TPR 값을 반환 받음
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)

    # ROC Curve를 plot 곡선으로 그림
    plt.plot(fprs, tprs, label = 'ROC')

    # 가운데 대각선 직선을 그림
    plt.plot([0,1], [0,1], 'k--', label = 'Random')

    # FPR X축의 Scale을 0, 1 단위로 변경, X, Y 축명 설정 등
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity)')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()


# 9. 그래프
def precision_recall_curve_plot(y_test=None, pred_proba_c1=None):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출. 
    precisions, recalls, thresholds = precision_recall_curve( y_test, pred_proba_c1)
    
    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')
    
    # threshold 값 X 축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()



# 11.
# 원본 데이터에 중복된 feature명으로 인하여 신규 버전의 pandas에서 duplicate name 에러를 발생
# 중복 feature명에 대해서 원본 feature명에 '_1또는 2'를 추가로 부여하는 함수인 get_new_feature_name_df() 생성

def get_new_feature_name_df(old_feature_name_df) :
    feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(), columns=['dup_cnt'])
    feature_dup_df = feature_dup_df.reset_index()

    new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df, how='outer')
    new_feature_name_df['column_name'] = new_feature_name_df[['column_name', 'dup_cnt']].apply(lambda x : x[0] + '_' + str(x[1]) if x[1] > 0 else x[0], axis = 1)

    new_feature_name_df = new_feature_name_df.drop(['index'], axis=1)

    return new_feature_name_df


def get_human_dataset() :
    # 각 데이터 파일들은 공백으로 분리되어 있으므로 read_csv에서 공백 문자를  sep으로 할당
    feature_name_df = pd.read_csv("C:/apps/ml_7/datasets/UCI HAR Dataset/UCI HAR Dataset/features.txt", sep='\s+', header=None, names=['column_index', 'column_name'])

    # 중복된 피처명을 수정하는 get_new_feature_name_df()를 이용, 신규 피처명 DataFrame 생성
    new_feature_name_df = get_new_feature_name_df(feature_name_df)

    # DataFrame에 피러명을 컬럼으로 부여하기 위해 리스트 객체로 다시 반환
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()

    # 학습 피러 데이터 셋과 테스트 피처 데이터를 DataFrame으로 로딩. 컬럼명을 feature_name 적용
    X_train = pd.read_csv("C:/apps/ml_7/datasets/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt", sep='\s+', names=feature_name)
    X_test = pd.read_csv("C:/apps/ml_7/datasets/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt", sep='\s+', names=feature_name)

    # 학습 레이블과 테스트 레이블 데이터를 DataFrame으로 로딩하고 컬럼명은 action으로 부여
    y_train = pd.read_csv("C:/apps/ml_7/datasets/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt", sep='\s+', header=None, names=['action'])
    y_test = pd.read_csv("C:/apps/ml_7/datasets/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt", sep='\s+', header=None, names=['action'])

    # 로드된 학습/테스트용 DataFrame으 모두 반환
    return X_train, X_test, y_train, y_test