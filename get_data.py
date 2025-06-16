import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder


def get_categorical_feature_index(x, threshold=5):
    x_num_dim = []
    x_cat_dim = []
    x_cat_cardinalities = []
    for k in range(x.shape[1]):
        count = len(np.unique(x[:, k]))
        if count <= threshold:
            x_cat_dim.append(k)
            x_cat_cardinalities.append(count + 1)
        else:
            x_num_dim.append(k)#
    return x_num_dim, x_cat_dim, x_cat_cardinalities


class AdvancedOrdinalEncoder(OrdinalEncoder):
    def transform(self, X):
        data = super().transform(X)
        for c in range(data.shape[1]):
            c_data = data[:, c]
            c_data[c_data == -1] = 0
        return data


def data_preprocess(X, y):
    x = X.astype(np.float32)
    y = y.astype(np.float32)
    x_num_dim, x_cat_dim, x_cat_cardinalities = get_categorical_feature_index(X, threshold=5)
    x_transformer = ColumnTransformer(
        [('cat_cols', AdvancedOrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), x_cat_dim),
         ('num_cols', StandardScaler(), x_num_dim)])
    x = x_transformer.fit_transform(x).astype(np.float32)
    y_transformer = StandardScaler()
    y = y_transformer.fit_transform(np.reshape(y, (-1, 2)))
    x_cat_dim = [i for i in range(len(x_cat_dim))]
    x_num_dim = [i for i in range(len(x_cat_dim), len(x_cat_dim) + len(x_num_dim))]
    if len(x_cat_cardinalities) == 0:
        x_cat_cardinalities = None
    return x, y, x_num_dim, x_cat_dim, x_cat_cardinalities, y_transformer


def generate_multimodal_data(path="nf.xlsx", method='z', is_df=True, factor=1.0, pca=False,
                             target=['rejection'], delete_smile=True, seed=42, is_rfe=False):
    data = pd.read_excel(path)
    data.fillna(data.mean(), axis=0, inplace=True) # filling
    drop_index = []
    for index in range(len(data)):
        if data.iloc[index]["rejection"] <= 0 \
                or data.iloc[index]["rejection"] >= 1.0 \
                or data.iloc[index]['permeance'] <= 0:
            drop_index.append(index)
    data.drop(index=drop_index, axis=0, inplace=True)
    data = data.drop_duplicates()
    # nf
    if is_rfe:
        rfe_columns = ['membrane_smiles', 'solvent_smiles', 'solute_smiles', 'full_smiles', 'swelling_thickness',
                       'swelling_weight', 'mwco', 'membrane_type', 'process_configuration', 'solute_mw',
                       'solvent_viscosity',
                       'solvent_logp', 'solvent_dt', 'solvent_dp', 'SLogP', 'SMR', 'nAtom', 'nHeavyAtom', 'RotRatio',
                       'TopoPSA_NO', 'TopoPSA', 'Vabc', 'permeance', 'rejection']
        data = data[rfe_columns].copy()
    if delete_smile:
        data.drop(['membrane_smiles', 'solvent_smiles', 'solute_smiles', 'full_smiles'], axis=1, inplace=True)
    # osn
    # if delete_smile:
    #     data.drop(['Solvent Smile', 'Solute smile'], axis=1, inplace=True)

    # 创建一个标签编码器对象
    le = LabelEncoder()
    # nf
    for col in ['membrane_type',
                'process_configuration',
                # 'contact_angle', 'zeta_potential', 'temperature', 'nSpiro', 'nBridgehead',
                #                'nN',
                #                'nS', 'nP', 'nF', 'nCl', 'nBr',
                #                'nI', 'nX', 'nHBDon'
                ]:
        data[col] = le.fit_transform(data[col])
    # data = pd.get_dummies(data)
    # osn
    # le = LabelEncoder()
    #
    # # 对每一列进行顺序编码
    # for col in ['Process configuration']:
    #     data[col] = le.fit_transform(data[col])
    len_data = len(data)
    len_temp = int(factor * len_data)
    data = data[0:len_temp].copy()
    y = data[target] * 100
    x = data.drop(target, axis=1)

    if is_rfe == False:
        scaled_list = ['swelling_thickness', 'swelling_weight', 'mwco',
                       'solute_mw', 'solute_w_conc',
                       'pressure', 'surface_tension', 'solvent_mw', 'solvent_diameter',
                       'solvent_viscosity', 'density', 'solvent_dipole_moment',
                       'solvent_dielectric_constant', 'solvent_hildebrand', 'solvent_logp',
                       'solvent_dt', 'solvent_dp', 'solvent_dh', 'SLogP',
                       'SMR', 'nAtom', 'nHeavyAtom',
                       'nHetero', 'nH', 'nC', 'nO',
                       'nHBAcc', 'nRot', 'RotRatio',
                       'TopoPSA_NO', 'TopoPSA', 'Vabc']
    else:
        scaled_list = ['swelling_thickness',
                       'swelling_weight', 'mwco', 'solute_mw', 'solvent_viscosity',
                       'solvent_logp', 'solvent_dt', 'solvent_dp', 'SLogP', 'SMR', 'nAtom', 'nHeavyAtom', 'RotRatio',
                       'TopoPSA_NO', 'TopoPSA', 'Vabc']
    if method == 'z':
        x[scaled_list] = (x[scaled_list] - x[scaled_list].mean()) / (x[scaled_list].std())
    elif method == 'max_min':
        x[scaled_list] = (x[scaled_list] - x[scaled_list].min()) / (x[scaled_list].max() - x[scaled_list].min())
    else:
        x = x

    y = pd.DataFrame(y, columns=target)

    x.columns = x.columns.astype(str)
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=seed)
    '''
    In the hyperparameter tuning stage, this paper divides the dataset into 
    training set, validation set and test set according to the ratio of 7:1:2, 
    and then firstly determines the approximate range of the optimal parameters 
    through the random parameterization method, and finally determines the optimal 
    parameters through grid search. All models are subjected to 300 sets of 
    hyper-parameter optimization experiments in the validation set, and the random 
    seed adopts a fixed value of 123456 to ensure the fairness as much as possible. 
    After completing the tuning stage, the training set and validation set of the 
    tuning stage are then merged into a training set, and the models are trained 
    under the optimal parameters, and compared on the test set for a comprehensive 
    evaluation of the performance of each model.
    '''
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, shuffle=True, test_size=0.125,
    #                                                   random_state=seed)

    if pca == True and delete_smile == True:
        pca = PCA(n_components=8)
        x_train = pca.fit_transform(x_train.values)  # 对样本进行降维
        x_test = pca.transform(x_test.values)  # 对样本进行降维
        x_train = pd.DataFrame(x_train, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'])
        x_test = pd.DataFrame(x_test, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'])
    elif pca == True and delete_smile == False:
        train_smiles = x_train[['membrane_smiles', 'solvent_smiles', 'solute_smiles', 'full_smiles']]
        test_smiles = x_test[['membrane_smiles', 'solvent_smiles', 'solute_smiles', 'full_smiles']]

        x_train = x_train.drop(['membrane_smiles', 'solvent_smiles', 'solute_smiles', 'full_smiles'], axis=1)
        x_test = x_test.drop(['membrane_smiles', 'solvent_smiles', 'solute_smiles', 'full_smiles'], axis=1)

        pca = PCA(n_components=8)
        x_train = pca.fit_transform(x_train.values)  # 对样本进行降维
        x_test = pca.transform(x_test.values)  # 对样本进行降维

        x_train = pd.DataFrame(x_train, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'],
                               index=train_smiles.index)
        x_test = pd.DataFrame(x_test, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'], index=test_smiles.index)

        x_train = pd.concat([x_train, train_smiles], axis=1)
        x_test = pd.concat([x_test, test_smiles], axis=1)

    if is_df:
        return x_train, x_test, y_train, y_test
    else:
        return x_train.values, x_test.values, y_train.values, y_test.values
