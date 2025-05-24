import pickle
import numpy as np

# Load data for non-defended dataset for CW setting
def LoadDataNoDefCW():

    print("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = '../dataset/ClosedWorld/NoDef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle))

    print("Data dimensions:")
    print("X: Training data's shape : {}".format(X_train.shape))
    print("y: Training data's shape : {}".format(y_train.shape))
    print("X: Validation data's shape : {}".format(X_valid.shape))
    print("y: Validation data's shape : {}".format(y_valid.shape))
    print("X: Testing data's shape : {}".format(X_test.shape))
    print("y: Testing data's shape : {}".format(y_test.shape))

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Load data for non-defended dataset for CW setting
def LoadDataWTFPADCW():

    print("Loading WTF-PAD dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = '../dataset/ClosedWorld/WTFPAD/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WTFPAD.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_WTFPAD.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'X_valid_WTFPAD.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_WTFPAD.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    # Load testing data
    with open(dataset_dir + 'X_test_WTFPAD.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_WTFPAD.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle))

    print("Data dimensions:")
    print("X: Training data's shape : {}".format(X_train.shape))
    print("y: Training data's shape : {}".format(y_train.shape))
    print("X: Validation data's shape : {}".format(X_valid.shape))
    print("y: Validation data's shape : {}".format(y_valid.shape))
    print("X: Testing data's shape : {}".format(X_test.shape))
    print("y: Testing data's shape : {}".format(y_test.shape))

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Load data for non-defended dataset for CW setting
def LoadDataWalkieTalkieCW():

    print("Loading Walkie-Talkie dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = '../dataset/ClosedWorld/WalkieTalkie/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WalkieTalkie.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_WalkieTalkie.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'X_valid_WalkieTalkie.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_WalkieTalkie.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    # Load testing data
    with open(dataset_dir + 'X_test_WalkieTalkie.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_test_WalkieTalkie.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle))

    print("Data dimensions:")
    print("X: Training data's shape : {}".format(X_train.shape))
    print("y: Training data's shape : {}".format(y_train.shape))
    print("X: Validation data's shape : {}".format(X_valid.shape))
    print("y: Validation data's shape : {}".format(y_valid.shape))
    print("X: Testing data's shape : {}".format(X_test.shape))
    print("y: Testing data's shape : {}".format(y_test.shape))

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Load data for non-defended dataset for OW training
def LoadDataNoDefOW_Training():

    print("Loading non-defended dataset for open-world scenario for training")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/NoDef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))


    print("Data dimensions:")
    print("X: Training data's shape : {}".format(X_train.shape))
    print("y: Training data's shape : {}".format(y_train.shape))
    print("X: Validation data's shape : {}".format(X_valid.shape))
    print("y: Validation data's shape : {}".format(y_valid.shape))

    return X_train, y_train, X_valid, y_valid

# Load data for non-defended dataset for OW evaluation
def LoadDataNoDefOW_Evaluation():

    print("Loading non-defended dataset for open-world scenario for evaluation")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/NoDef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_test_Mon_NoDef.pkl', 'rb') as handle:
        X_test_Mon = pickle.load(handle)
    with open(dataset_dir + 'y_test_Mon_NoDef.pkl', 'rb') as handle:
        y_test_Mon = pickle.load(handle)
    with open(dataset_dir + 'X_test_Unmon_NoDef.pkl', 'rb') as handle:
        X_test_Unmon = pickle.load(handle)
    with open(dataset_dir + 'y_test_Unmon_NoDef.pkl', 'rb') as handle:
        y_test_Unmon = pickle.load(handle)

    X_test_Mon = np.array(X_test_Mon)
    y_test_Mon = np.array(y_test_Mon)
    X_test_Unmon = np.array(X_test_Unmon)
    y_test_Unmon = np.array(y_test_Unmon)

    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon

# Load data for WTF-PAD dataset for OW training
def LoadDataWTFPADOW_Training():

    print("Loading WTF-PAD dataset for open-world scenario for training")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/WTFPAD/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WTFPAD.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_WTFPAD.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'X_valid_WTFPAD.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_WTFPAD.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))


    print("Data dimensions:")
    print("X: Training data's shape : {}".format(X_train.shape))
    print("y: Training data's shape : {}".format(y_train.shape))
    print("X: Validation data's shape : {}".format(X_valid.shape))
    print("y: Validation data's shape : {}".format(y_valid.shape))

    return X_train, y_train, X_valid, y_valid

# Load data for WTF-PAD dataset for OW evaluation
def LoadDataWTFPADOW_Evaluation():

    print("Loading WTF-PAD dataset for open-world scenario for evaluation")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/WTFPAD/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_test_Mon_WTFPAD.pkl', 'rb') as handle:
        X_test_Mon = pickle.load(handle)
    with open(dataset_dir + 'y_test_Mon_WTFPAD.pkl', 'rb') as handle:
        y_test_Mon = pickle.load(handle)
    with open(dataset_dir + 'X_test_Unmon_WTFPAD.pkl', 'rb') as handle:
        X_test_Unmon = pickle.load(handle)
    with open(dataset_dir + 'y_test_Unmon_WTFPAD.pkl', 'rb') as handle:
        y_test_Unmon = pickle.load(handle)

    X_test_Mon = np.array(X_test_Mon)
    y_test_Mon = np.array(y_test_Mon)
    X_test_Unmon = np.array(X_test_Unmon)
    y_test_Unmon = np.array(y_test_Unmon)

    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon

# Load data for WalkieTalkie dataset for OW training
def LoadDataWalkieTalkieOW_Training():

    print("Loading Walkie-Talkie dataset for open-world scenario for training")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/WalkieTalkie/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WalkieTalkie.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_train_WalkieTalkie.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(dataset_dir + 'X_valid_WalkieTalkie.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle))
    with open(dataset_dir + 'y_valid_WalkieTalkie.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle))


    print("Data dimensions:")
    print("X: Training data's shape : {}".format(X_train.shape))
    print("y: Training data's shape : {}".format(y_train.shape))
    print("X: Validation data's shape : {}".format(X_valid.shape))
    print("y: Validation data's shape : {}".format(y_valid.shape))

    return X_train, y_train, X_valid, y_valid

# Load data for WTF-PAD dataset for OW evaluation
def LoadDataWalkieTalkieOW_Evaluation():

    print("Loading Walkie-Talkie dataset for open-world scenario for evaluation")
    # Point to the directory storing data
    dataset_dir = '../dataset/OpenWorld/WalkieTalkie/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_test_Mon_WalkieTalkie.pkl', 'rb') as handle:
        X_test_Mon = pickle.load(handle)
    with open(dataset_dir + 'y_test_Mon_WalkieTalkie.pkl', 'rb') as handle:
        y_test_Mon = pickle.load(handle)
    with open(dataset_dir + 'X_test_Unmon_WalkieTalkie.pkl', 'rb') as handle:
        X_test_Unmon = pickle.load(handle)
    with open(dataset_dir + 'y_test_Unmon_WalkieTalkie.pkl', 'rb') as handle:
        y_test_Unmon = pickle.load(handle)

    X_test_Mon = np.array(X_test_Mon)
    y_test_Mon = np.array(y_test_Mon)
    X_test_Unmon = np.array(X_test_Unmon)
    y_test_Unmon = np.array(y_test_Unmon)

    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon

# 加载自定义数据集进行封闭世界评估
def LoadCustomDataCW():
    print("Loading custom dataset for closed-world scenario")
    # 指向存储数据的目录
    dataset_dir = '../dataset/'

    # 加载监控网站数据
    with open(dataset_dir + 'ts_mon.pkl', 'rb') as handle:
        mon_data = pickle.load(handle)
    
    # mon_data是一个字典，包含每个网站的样本
    # 需要将其转换为X（流量方向序列）和y（对应的网站标签）
    X_all = []
    y_all = []
    
    # 遍历每个网站的样本
    for site_id, samples in mon_data.items():
        for sample in samples:
            # 提取方向信息（取符号）
            directions = [np.sign(x) if x != 0 else 1 for x in sample]
            
            # 确保每个样本长度为5000（DF模型的要求）
            if len(directions) > 5000:
                directions = directions[:5000]
            else:
                # 如果样本长度不足5000，用0填充
                directions = directions + [0] * (5000 - len(directions))
            
            X_all.append(directions)
            y_all.append(site_id)
    
    # 将数据转换为numpy数组
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    # 随机打乱数据
    indices = np.random.permutation(len(X_all))
    X_all = X_all[indices]
    y_all = y_all[indices]
    
    # 划分训练集、验证集和测试集（比例：8:1:1）
    train_size = int(len(X_all) * 0.8)
    valid_size = int(len(X_all) * 0.1)
    
    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    
    X_valid = X_all[train_size:train_size+valid_size]
    y_valid = y_all[train_size:train_size+valid_size]
    
    X_test = X_all[train_size+valid_size:]
    y_test = y_all[train_size+valid_size:]
    
    print("Data dimensions:")
    print("X: Training data's shape : {}".format(X_train.shape))
    print("y: Training data's shape : {}".format(y_train.shape))
    print("X: Validation data's shape : {}".format(X_valid.shape))
    print("y: Validation data's shape : {}".format(y_valid.shape))
    print("X: Testing data's shape : {}".format(X_test.shape))
    print("y: Testing data's shape : {}".format(y_test.shape))
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test