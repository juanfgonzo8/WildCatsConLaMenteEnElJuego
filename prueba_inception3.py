##
# Se importan los paquetes
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import cv2
import torch
from tqdm import tqdm_notebook
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

##
#Se establecen los paths

path_csv = '/media/user_home2/vision2020_01/Data/iWildCam2019/train.csv'
path_train = '/media/user_home2/vision2020_01/Data/iWildCam2019/train_images/'

##
# Se importan los datos
train_df_all = pd.read_csv(path_csv)

batch_size = 64
IMG_SIZE = 64
N_EPOCHS = 10
ID_COLNAME = 'file_name'
ANSWER_COLNAME = 'category_id'
TRAIN_IMGS_DIR = path_train
TEST_IMGS_DIR = '../input/test_images/'

train_df, test_df = train_test_split(train_df_all[[ID_COLNAME, ANSWER_COLNAME]],
                                     test_size = 0.25,
                                     shuffle = False
                                    )

CLASSES_TO_USE = train_df_all['category_id'].unique()
NUM_CLASSES = len(CLASSES_TO_USE)
CLASSMAP = dict(
    [(i, j) for i, j
     in zip(CLASSES_TO_USE, range(NUM_CLASSES))
    ]
)
REVERSE_CLASSMAP = dict([(v, k) for k, v in CLASSMAP.items()])

##
# Se crea el modelo
model = models.inception_v3(pretrained='imagenet')

# new_head = torch.nn.Linear(model.classifier.in_features, NUM_CLASSES)
# model.inception_v3.classifier = new_head

# Handle the auxilary net
num_ftrs = model.AuxLogits.fc.in_features
model.AuxLogits.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
# Handle the primary net
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

model.cuda()

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_augmentation = preprocess
val_augmentation = preprocess

##
# Se define la funcion F1
def f1_score(y_true, y_pred, threshold=0.5):
    return fbeta_score(y_true, y_pred, 1, threshold)


def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))

##
# Se crea el DataLoader
class IMetDataset(Dataset):

    def __init__(self,
                 df,
                 images_dir,
                 n_classes=NUM_CLASSES,
                 id_colname=ID_COLNAME,
                 answer_colname=ANSWER_COLNAME,
                 label_dict=CLASSMAP,
                 transforms=None
                 ):
        self.df = df
        self.images_dir = images_dir
        self.n_classes = n_classes
        self.id_colname = id_colname
        self.answer_colname = answer_colname
        self.label_dict = label_dict
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        img_name = img_id  # + self.img_ext
        img_path = os.path.join(self.images_dir, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.answer_colname is not None:
            label = torch.zeros((self.n_classes,), dtype=torch.long)
            label[self.label_dict[cur_idx_row[self.answer_colname]]] = 1.0
            id = self.label_dict[cur_idx_row[self.answer_colname]]

            return img, id

        else:
            return img, img_id

train_dataset = IMetDataset(train_df, TRAIN_IMGS_DIR, transforms = train_augmentation)
test_dataset = IMetDataset(test_df, TRAIN_IMGS_DIR, transforms = val_augmentation)

BS = 32

train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)

def kaggle_commit_logger(str_to_log, need_print = True):
    if need_print:
        print(str_to_log)
    os.system('echo ' + str_to_log)

def cuda(x):
    return x.cuda(non_blocking=True)

##
# Se crean las funciones de train y val
def train_one_epoch(model, train_loader, criterion, optimizer, steps_upd_logging=250):
    model.train();

    total_loss = 0.0

    train_tqdm = tqdm_notebook(train_loader)

    for step, (features, targets) in enumerate(train_tqdm):
        targets = targets.squeeze_()
        features, targets = cuda(features), cuda(targets)

        optimizer.zero_grad()

        logits, aux = model(features)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            logstr = f'Train loss on step {step + 1} was {round(total_loss / (step + 1), 5)}'
            train_tqdm.set_description(logstr)
            kaggle_commit_logger(logstr, need_print=False)

    return total_loss / (step + 1)

def validate(model, valid_loader, criterion, need_tqdm=False):
    model.eval();

    test_loss = 0.0
    TH_TO_ACC = 0.5

    true_ans_list = []
    preds_cat = []

    with torch.no_grad():

        if need_tqdm:
            valid_iterator = tqdm_notebook(valid_loader)
        else:
            valid_iterator = valid_loader

        for step, (features, targets) in enumerate(valid_iterator):

            features, targets = cuda(features), cuda(targets)

            logits, aux = model(features)
            loss = criterion(logits, targets)

            test_loss += loss.item()
            true_ans_list.append(targets)
            preds_cat.append(torch.sigmoid(logits))

        all_true_ans = torch.cat(true_ans_list)
        all_preds = torch.cat(preds_cat)

        print(all_true_ans.shape)
        print(all_preds.shape)
        f1_eval = f1_score(all_true_ans, all_preds).item()

    logstr = f'Mean val f1: {round(f1_eval, 5)}'
    kaggle_commit_logger(logstr)
    return test_loss / (step + 1), f1_eval

##
# Se define el loss y optimizador
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

##
# Se entrena y valida
TRAIN_LOGGING_EACH = 500

train_losses = []
valid_losses = []
valid_f1s = []
best_model_f1 = 0.0
best_model = None
best_model_ep = 0

for epoch in range(1, N_EPOCHS + 1):
    ep_logstr = f"Starting {epoch} epoch..."
    kaggle_commit_logger(ep_logstr)
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, TRAIN_LOGGING_EACH)
    train_losses.append(tr_loss)
    tr_loss_logstr = f'Mean train loss: {round(tr_loss, 5)}'
    kaggle_commit_logger(tr_loss_logstr)

    valid_loss, valid_f1 = validate(model, test_loader, criterion)
    valid_losses.append(valid_loss)
    valid_f1s.append(valid_f1)
    val_loss_logstr = f'Mean valid loss: {round(valid_loss, 5)}'
    kaggle_commit_logger(val_loss_logstr)
    # sheduler.step(valid_loss)

    if valid_f1 >= best_model_f1:
        best_model = model
        best_model_f1 = valid_f1
        best_model_ep = epoch

bestmodel_logstr = f'Best f1 is {round(best_model_f1, 5)} on epoch {best_model_ep}'
kaggle_commit_logger(bestmodel_logstr)

##
# Se guardan las graficas
def save_graph(train_losses,valid_losses,valid_f1s):

    name = 'Graficas' + '/Progress_inception3.png'
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(range(len(train_losses)), train_losses, label='Train')
    ax1.plot(range(len(valid_losses)), valid_losses, label='Val')
    ax1.set_ylim(top=1, bottom=-1)
    ax1.legend()
    ax1.grid()
    ax1.set(title='Prueba')

    ax2 = ax1.twinx()

    ax2.set_ylabel('F1')
    ax2.plot(range(len(valid_f1s)), valid_f1s, 'g', label='Val')
    ax2.set_ylim(top=1, bottom=0)
    ax2.yaxis.grid(linestyle=(0, (1, 10)), linewidth=0.5)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(name, dpi=300)

    plt.close('all')
