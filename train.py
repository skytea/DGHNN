import gcnIO
from torch_geometric.utils import add_self_loops
from utils import *
import os
import numpy as np
from model import DGHNN
from FT_model import FTC
import torch
from troch import nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
import itertools
from torch_geometric.data import Data, DataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 1000
NUM_EPOCH = 3000
NUM_WORKERS = 1
PIN_MEMORY = False
LOAD_MODEL = True

paths = []
paths.append("/xxxxxxxx/CPDB_multiomics.h5")
paths.append("/xxxxxxxx/IREF_multiomics.h5")
paths.append("/xxxxxxxx/IREF_2015_multiomics.h5")
paths.append("/xxxxxxxx/MULTINET_multiomics.h5")
paths.append("/xxxxxxxx/PENCT_multiomics.h5")
paths.append("/xxxxxxxx/STRINGdb_multiomics.h5")
path_name = ["CPDB", "IREF", "IREF_2015", "MULTINET", "PCNET", "STRINGdb"]
pathway_file = "/xxxxxxxx/KEGG_pathway_gene.txt"
mapping_file = "/xxxxxxxx/Homo_sapiens.gene_info"


def train(model, optimizer, criterion, classifier, features, adj_mat, h_matrix, target, index):
    features = torch.tensor(features).float().to(device=DEVICE)
    adj_mat = torch.tensor(adj_mat, dtype=torch.int64).to(device=DEVICE)
    h_matrix = torch.tensor(h_matrix, dtype=torch.int64).to(device=DEVICE)
    target = torch.tensor(target).float().to(device=DEVICE)

    train_loss = 0
    auprc = 0
    auroc = 0

    optimizer.zero_grad()
    model.train()
    classifier.train()

    output = model(features, adj_mat, h_matrix)

    output_, _target_ = output[index], target[index]
    n, dim = output_.size()
    training_times = int(np.ceil(n / BATCH_SIZE))
    output_all = None
    for inner_epoch in range(training_times):
        index_ = [i_temp for i_temp in range(inner_epoch * BATCH_SIZE, min(n, (inner_epoch + 1) * BATCH_SIZE))]
        features_, target_ = output_[index_], _target_[index_]
        output = classifier(features_)
        output_all = output if output_all is None else torch.cat((output_all, output), dim=0)

        loss = criterion(output, target_)
        train_loss += loss.item()

        loss.backward(retain_graph=True)

        torch.cuda.empty_cache()

    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.empty_cache()
    train_loss = train_loss / training_times

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    pred_y = output_all.tolist()
    ground_y = _target_.tolist()
    for i in range(len(pred_y)):
        pred_o = pred_y[i][0]
        pred_two = 1 if pred_o > 0.5 else 0
        if ground_y[i][0] == 1:
            if pred_two == 1:
                TP += 1
            else:
                FN += 1
        else:
            if pred_two == 1:
                FP += 1
            else:
                TN += 1

    try:
        temp_auprc = average_precision_score(ground_y, pred_y)
    except:
        temp_auprc = 0

    auprc += temp_auprc

    try:
        temp_auroc = roc_auc_score(ground_y, pred_y)
    except:
        temp_auroc = 0

    auroc += temp_auroc

    accurancy = np.nan if (TP + FP + TN + FN) == 0 else (TP + TN) / (TP + FP + TN + FN)
    precision = np.nan if (TP + FP) == 0 else (TP) / (TP + FP)
    sensitivity = np.nan if (TP + FN) == 0 else (TP) / (TP + FN)
    specificity = np.nan if (FP + TN) == 0 else (TN) / (FP + TN)
    f1 = np.nan
    try:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    except:
        f1 = np.nan

    print(f"Train Error: \n Avg loss: {test_loss:>4f}, AUPRC:{auprc:>4f}, AUROC:{auroc:>4f}, F1:{f1:>4f}, accurancy:{accurancy:>4f}, precision:{precision:>4f}, sensitivity:{sensitivity:>4f}, specificity:{specificity:>4f} \n")
    result.write(f"Train Error: \n Avg loss: {test_loss:>4f}, AUPRC:{auprc:>4f}, AUROC:{auroc:>4f}, F1:{f1:>4f}, accurancy:{accurancy:>4f}, precision:{precision:>4f}, sensitivity:{sensitivity:>4f}, specificity:{specificity:>4f} \n")


def test(model, criterion, classifier, features, adj_mat, h_matrix, target, index):
    features = torch.tensor(features).float().to(device=DEVICE)
    adj_mat = torch.tensor(adj_mat, dtype=torch.int64).to(device=DEVICE)
    h_matrix = torch.tensor(h_matrix, dtype=torch.int64).to(device=DEVICE)
    target = torch.tensor(target).float().to(device=DEVICE)

    test_loss = 0
    auprc = 0
    auroc = 0
    model.eval()
    classifier.eval()
    with torch.no_grad():
        output = model(features, adj_mat, h_matrix)

        output_, _target_ = output[index], target[index]
        n, dim = output_.size()
        training_times = int(np.ceil(n / BATCH_SIZE))
        output_all = None
        for inner_epoch in range(training_times):
            index_ = [i_temp for i_temp in range(inner_epoch * BATCH_SIZE, min(n, (inner_epoch + 1) * BATCH_SIZE))]
            features_, target_ = output_[index_], _target_[index_]
            output = classifier(features_)
            output_all = output if output_all is None else torch.cat((output_all, output), dim=0)
            loss = criterion(output, target_)
            test_loss += loss.item()
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        test_loss = test_loss / training_times

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        pred_y = output_all.tolist()
        ground_y = _target_.tolist()
        for i in range(len(pred_y)):
            pred_o = pred_y[i][0]
            pred_two = 1 if pred_o > 0.5 else 0
            if ground_y[i][0] == 1:
                if pred_two == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred_two == 1:
                    FP += 1
                else:
                    TN += 1

        try:
            temp_auprc = average_precision_score(ground_y, pred_y)
        except:
            temp_auprc = 0

        auprc += temp_auprc

        try:
            temp_auroc = roc_auc_score(ground_y, pred_y)
        except:
            temp_auroc = 0

        auroc += temp_auroc

        accurancy = np.nan if (TP + FP + TN + FN) == 0 else (TP + TN) / (TP + FP + TN + FN)
        precision = np.nan if (TP + FP) == 0 else (TP) / (TP + FP)
        sensitivity = np.nan if (TP + FN) == 0 else (TP) / (TP + FN)
        specificity = np.nan if (FP + TN) == 0 else (TN) / (FP + TN)
        f1 = np.nan
        try:
            f1 = 2 * precision * sensitivity / (precision + sensitivity)
        except:
            f1 = np.nan

        print(
            f"Test Error: \n Avg loss: {test_loss:>4f}, AUPRC:{auprc:>4f}, AUROC:{auroc:>4f}, F1:{f1:>4f}, accurancy:{accurancy:>4f}, precision:{precision:>4f}, sensitivity:{sensitivity:>4f}, specificity:{specificity:>4f} \n")
        result.write(
            f"Test Error: \n Avg loss: {test_loss:>4f}, AUPRC:{auprc:>4f}, AUROC:{auroc:>4f}, F1:{f1:>4f}, accurancy:{accurancy:>4f}, precision:{precision:>4f}, sensitivity:{sensitivity:>4f}, specificity:{specificity:>4f} \n")

    return test_loss, auprc, auroc


data_list = []
node2idx = {} #init node2idx dict to find commong nodes ammong graphs.
h_node2idx = {}
counter = 0
train_nodes = []
test_nodes = []
val_nodes = []
meta_y = torch.zeros(1000000,1) #init with some default big value
y_list =[]
node_names_list = []
number_nodes_each_graph = []

features_order = ['MF: UCEC', 'MF: BLCA', 'MF: THCA', 'MF: KIRC', 'MF: READ', 'MF: LUAD', 'MF: ESCA', 'MF: LUSC', 'MF: BRCA', 'MF: COAD', 'MF: HNSC', 'MF: KIRP', 'MF: PRAD', 'MF: LIHC', 'MF: STAD', 'MF: CESC',
'METH: UCEC', 'METH: BLCA', 'METH: THCA', 'METH: KIRC', 'METH: READ', 'METH: LUAD', 'METH: ESCA', 'METH: LUSC', 'METH: BRCA', 'METH: COAD', 'METH: HNSC', 'METH: KIRP', 'METH: PRAD', 'METH: LIHC', 'METH: STAD', 'METH: CESC',
'GE: UCEC', 'GE: BLCA', 'GE: THCA', 'GE: KIRC', 'GE: READ', 'GE: LUAD', 'GE: ESCA', 'GE: LUSC', 'GE: BRCA', 'GE: COAD', 'GE: HNSC', 'GE: KIRP', 'GE: PRAD', 'GE: LIHC', 'GE: STAD', 'GE: CESC',
'CNA: UCEC', 'CNA: BLCA', 'CNA: THCA', 'CNA: KIRC', 'CNA: READ', 'CNA: LUAD', 'CNA: ESCA', 'CNA: LUSC', 'CNA: BRCA', 'CNA: COAD', 'CNA: HNSC', 'CNA: KIRP', 'CNA: PRAD', 'CNA: LIHC', 'CNA: STAD', 'CNA: CESC']

meta_x = torch.zeros((100000,64)) #init with some default big value

for path in paths:

    data = gcnIO.load_hdf_data(path, feature_name='features')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names = data

    feature_names = [f.decode("utf-8") for f in feature_names]

    # allign features in all the graphs
    feature_ind = [feature_names.index(f_n) for f_n in features_order]
    feature_names = np.array(feature_names)[feature_ind]

    features = features[:, feature_ind]
    number_nodes_each_graph.append(features.shape[0])

    # construct node2idx dict -> useful to find common nodes among graphs.
    for i in node_names:
        if (tuple(i) not in node2idx):
            node2idx[tuple(i)] = counter
            h_node2idx[i[1]] = counter
            counter += 1

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    y_val = y_val.astype(int)

    y = torch.tensor(y_train).add(torch.tensor(y_test)).add(torch.tensor(y_val))
    y_list.append(y)

    for i, label in enumerate(y):
        idx = node2idx[tuple(node_names[i])]
        meta_x[idx] = torch.from_numpy(features[i])
        if (meta_y[idx] == 0):
            meta_y[idx] = label

    # adjust the train,val,test labels to be in format-> [r,1] instead of [N,1] where r is the number of samples of the set.
    idx_train = [i for i, x in enumerate(train_mask) if x == 1]
    idx_val = [i for i, x in enumerate(val_mask) if x == 1]
    idx_test = [i for i, x in enumerate(test_mask) if x == 1]
    adj = torch.FloatTensor(np.array(adj))
    features = torch.FloatTensor(features)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # labels for the meta graph
    train_nodes.append(node_names[idx_train])
    test_nodes.append(node_names[idx_test])
    val_nodes.append(node_names[idx_val])

    edge_index = (adj > 0).nonzero().t()
    edge_index, _ = add_self_loops(edge_index)

    node_names_list.append(node_names)

    data = Data(x=features, edge_index=edge_index, y=y, node_names=node_names)
    data_list.append(data)

meta_x = meta_x[:len(node2idx)]
number_nodes_each_graph.append(meta_x.shape[0])

node_names = np.concatenate(node_names_list, axis=0)
meta_node_names = np.zeros(len(node2idx), dtype="object")
for k, v in node2idx.items():
    k1, k2 = k
    meta_node_names[v] = k2

meta_y = torch.tensor(meta_y[:len(node2idx)]).type(torch.LongTensor)  # keep meta nodes

gene_name_map = {}
gene_name_map_file = open(mapping_file, 'r').readlines()
for line in gene_name_map_file:
    line = line.split('\t')
    gene_name_map[line[1]] = line[2]

h_edge_index = []
count = 0
pathway = open(pathway_file, 'r').readlines()
for line in pathway:
    line = line.replace('\n', '').split('\t')

    for gene in line[2:]:
        if gene not in gene_name_map:
            continue
        gene_name = gene_name_map[gene]
        if gene_name not in h_node2idx.keys():
            continue
        a = h_node2idx[gene_name]
        h_edge_index.append([a, count])

    count += 1
h_edge_index = np.array(h_edge_index).T

save_dir = '/xxxxxxxxxxxxxx/DGHNN'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for path_index in range(6):
    data_name = path_name[path_index]

    save_dir_path = os.path.join(save_dir, str(data_name))
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    random_seed = [0, 1, 2, 3, 4]
    auprc_ret = []
    auroc_ret = []
    for seed in random_seed:
        set_seed(seed)

        train_set = {tuple(node) for t in train_nodes for node in t}  # we convert to set to remove duplicate notes across graphs
        t2 = {tuple(node) for t in val_nodes for node in t}
        t3 = set()
        for test_nodes_index in range(len(test_nodes)):
            if test_nodes_index == path_index:
                continue
            for node in test_nodes[test_nodes_index]:
                t3.add(tuple(node))
        train_set = train_set.union(t2)
        train_set = train_set.union(t3)

        # add 10% of the training set to the val set
        val_set = set()
        for i, val in enumerate(itertools.islice(train_set, int(len(train_set) * 0.1))):
            val_set.add(val)
        # val_set = {tuple(node) for t in val_nodes for node in t}

        # test_set = {tuple(node) for t in test_nodes for node in t}
        test_set = {tuple(node) for node in test_nodes[path_index]}  # test only in nodes from last graph

        train_set = train_set - train_set.intersection(test_set)  # remove test nodes from training set
        val_set = val_set - val_set.intersection(test_set)  # remove test nodes from val set
        train_set = train_set - val_set.intersection(val_set)  # remove val nodes from train set

        idx_train = torch.tensor([node2idx[i] for i in train_set])
        idx_test = torch.tensor([node2idx[i] for i in test_set])
        idx_val = torch.tensor([node2idx[i] for i in val_set])

        loader = DataLoader(data_list, batch_size=len(data_list))

        batch = next(iter(loader))
        features = batch.x
        edge_index = batch.edge_index

        ############## end

        model = DGHNN(
            in_features=features.shape[1],
            depths=[3, 3, 3, 3, 3, 3, 3, 3, 3],
            n_hidden=[64, 64, 64, 64, 64, 64, 64, 64, 64],
            n_heads=0,
            num_classes=1,
            concat=False,
            dropout=0.1,
            leaky_relu_slope=0.2,
            meta_x=meta_x,
            data=batch,
            node2idx=node2idx,
        ).to(DEVICE)

        classifier = FTC(
            input_dim=128,
            num_classes=1,
            attn_drpout=0.1,
            ff_dropout=0.1,
            dim=16,
            depth=4,
            heads=4,
            dim_head=4,
        ).to(DEVICE)

        optimizer = Adam([
            {'params': model.parameters()},
            {'params': classifier.parameters()},
        ], lr=LEARNING_RATE, weight_decay=1e-5)

        criterion = nn.BCELoss()

        result = open(os.path.join(save_dir_path, 'result' + str(seed) + '.txt'), 'w')

        best_loss = 99999
        best_epoch = 0
        best_auprc = 0
        best_auroc = 0
        save_dict = None

        for epoch in range(NUM_EPOCH):
            print(f"Epoch: {epoch}\n")
            result.write(f"Epoch: {epoch}\n")
            train(model, optimizer, criterion, classifier, features, edge_index, h_edge_index, meta_y, idx_train)
            val_loss, val_auprc, val_auroc = test(model, criterion, classifier, features, edge_index, h_edge_index, meta_y, idx_val)
            test_loss, test_auprc, test_auroc = test(model, criterion, classifier, features, edge_index, h_edge_index, meta_y, idx_test)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_auprc = test_auprc
                best_auroc = test_auroc
                save_dict = model.state_dict()

        print(f"Best Epoch: {best_epoch}\n")
        result.write(f"Best Epoch: {best_epoch}\n")

        save_path = os.path.join(save_dir_path, 'min_loss' + str(seed) + '.pth')
        torch.save(save_dict, save_path)

        result.close()

        auprc_ret.append(best_auprc)
        auroc_ret.append(best_auroc)
    auprc_ret = np.array(auprc_ret)
    mean = auprc_ret.mean()
    std = auprc_ret.std()
    result_auprc = open(os.path.join(save_dir_path, 'result_auprc.txt'), 'w')
    result_auprc.write(str(round(mean, 3)) + '±' + str(round(std, 3)) + '\n')

    auroc_ret = np.array(auroc_ret)
    mean = auroc_ret.mean()
    std = auroc_ret.std()
    result_auprc.write(str(round(mean, 3)) + '±' + str(round(std, 3)) + '\n')

    result_auprc.close()

