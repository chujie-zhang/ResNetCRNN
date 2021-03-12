import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable

import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sn
from matplotlib.ticker import MultipleLocator
import matplotlib
 
print(matplotlib.__version__)
#data_path = 'D:/UCF101/ucf101_jpegs_256/CalotTriangleDissection/'   
test_data_path = 'F:/cholec80_test/CalotTriangleDissection/'
data_path = 'F:/cholec80/CalotTriangleDissection/'

classification_path = './two_classes.pkl'
same_path = "./ResNetCRNN_ckpt/"

# Encoder cnn
cnn_hidden_1, cnn_hidden_2 = 1024, 768
cnn_dims = 512   
size = 224        
dropout = 0.5       

# Decoder rnn
rnnHiddenLayer = 3
rnnHiddenNode = 512
rnn_dim = 256

# hypterparameters
k = 2              #classifications
epochs = 60       
batch_size = 40  
learning_rate = 0.002
logInternal = 10  

# Selected frames in each clips
begin_frame, end_frame, skip_frame = 1, 25, 1


def train(logInternal, model, device, train_loader, optimizer, epoch):
    encoder, decoder = model
    encoder.train()
    decoder.train()

    losses = []
    scores = []
    precisions=[]
    recalls=[]
    f1s=[]
    counts = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        counts += X.size(0)

        optimizer.zero_grad()
        output = decoder(encoder(X))   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        predict_ = torch.max(output, 1)[1]  # predict_ != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), predict_.cpu().data.squeeze().numpy())
        step_precision=metrics.precision_score(y.cpu().data.squeeze().numpy(), predict_.cpu().data.squeeze().numpy(),average='weighted')
        step_recall=metrics.recall_score(y.cpu().data.squeeze().numpy(), predict_.cpu().data.squeeze().numpy(),average='weighted')
        step_f1=metrics.f1_score(y.cpu().data.squeeze().numpy(), predict_.cpu().data.squeeze().numpy(),average='weighted')
        scores.append(step_score)         # computed on CPU
        precisions.append(step_precision)
        recalls.append(step_recall)
        f1s.append(step_f1)
        loss.backward()
        optimizer.step()

        # show information
        #if (batch_idx + 1) % logInternal == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f},  f1: {:.2f}, recall: {:.2f}, precisions: {:.2f}%'.format(
         #       epoch + 1, counts, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score,100 * step_f1, 100 * step_recall, 100 * step_precision))

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f},  f1: {:.2f}, recall: {:.2f}, precisions: {:.2f}%'.format(
            epoch + 1, counts, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score,100 * step_f1, 100 * step_recall, 100 * step_precision))
    '''  
    # save Pytorch models of best record
    torch.save(encoder.state_dict(), os.path.join(same_path, 'encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(decoder.state_dict(), os.path.join(same_path, 'decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(same_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))
    '''
    return losses, scores,precisions,recalls,f1s


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    encoder, decoder = model
    encoder.eval()
    decoder.eval()

    validation_loss = 0
    y_a = []
    y_a_test = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = decoder(encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            validation_loss += loss.item()                 # sum up batch loss
            predict_ = output.max(1, keepdim=True)[1]  # (predict_ != output) get the index of the max log-probability

            # collect all y and predict_ in all batches
            y_a.extend(y)
            y_a_test.extend(predict_)

    validation_loss /= len(test_loader.dataset)
    
    # compute accuracy
    y_a = torch.stack(y_a, dim=0)
    y_a_test = torch.stack(y_a_test, dim=0)
    validation_loss = accuracy_score(y_a.cpu().data.squeeze().numpy(), y_a_test.cpu().data.squeeze().numpy())
    
    #f1
    valid_f1=metrics.f1_score(y_a.cpu().data.squeeze().numpy(), y_a_test.cpu().data.squeeze().numpy(),average='weighted')
    #confusion matrix
    valid_confusionMatrix=confusion_matrix(y_a.cpu().data.squeeze().numpy(), y_a_test.cpu().data.squeeze().numpy())
    #precision
    valid_precision=metrics.precision_score(y_a.cpu().data.squeeze().numpy(), y_a_test.cpu().data.squeeze().numpy(),average='weighted')
    #recall
    valid_recall=metrics.recall_score(y_a.cpu().data.squeeze().numpy(), y_a_test.cpu().data.squeeze().numpy(),average='weighted')
    

    print('\nValidation set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%,  f1: {:.2f}%, recall: {:.2f}%, precisions: {:.2f}%\n'.format(len(y_a), validation_loss, 100* validation_loss,100* valid_f1,100* valid_recall,100* valid_precision))
    
    # save encoder and decoder models
    torch.save(encoder.state_dict(), os.path.join(same_path, 'encoder_epoch{}.pth'.format(epoch + 1))) 
    torch.save(decoder.state_dict(), os.path.join(same_path, 'decoder_epoch{}.pth'.format(epoch + 1)))  
    torch.save(optimizer.state_dict(), os.path.join(same_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      
    print("Epoch {} model saved!".format(epoch + 1))
    
    return validation_loss, validation_loss,valid_f1,valid_confusionMatrix,valid_precision,valid_recall

def test(model, device, optimizer, test_loader):
    # set model as testing mode
    encoder, decoder = model
    #encoder.eval()
    #decoder.eval()

    test_loss = 0
    y_a = []
    y_a_test = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = decoder(encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            predict_ = output.max(1, keepdim=True)[1]  # (predict_ != output) get the index of the max log-probability

            # collect all y and predict_ in all batches
            y_a.extend(y)
            y_a_test.extend(predict_)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    y_a = torch.stack(y_a, dim=0)
    y_a_test = torch.stack(y_a_test, dim=0)
    test_loss = accuracy_score(y_a.cpu().data.squeeze().numpy(), y_a_test.cpu().data.squeeze().numpy())
    
    
    #f1
    valid_f1=metrics.f1_score(y_a.cpu().data.squeeze().numpy(), y_a_test.cpu().data.squeeze().numpy(),average='weighted')
    #confusion matrix
    valid_confusionMatrix=confusion_matrix(y_a.cpu().data.squeeze().numpy(), y_a_test.cpu().data.squeeze().numpy())
    #precision
    valid_precision=metrics.precision_score(y_a.cpu().data.squeeze().numpy(), y_a_test.cpu().data.squeeze().numpy(),average='weighted')
    #recall
    valid_recall=metrics.recall_score(y_a.cpu().data.squeeze().numpy(), y_a_test.cpu().data.squeeze().numpy(),average='weighted')
    
    
    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%,  f1: {:.2f}%, recall: {:.2f}%, precisions: {:.2f}%\n'.format(len(y_a), validation_loss, 100* validation_loss,100* valid_f1,100* valid_recall,100* valid_precision))


    return test_loss, test_loss,valid_f1,valid_confusionMatrix,valid_precision,valid_recall

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap = plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #xlocations = np.array(range(len(labels)))
    #plt.xticks(xlocations, labels, rotation=90)
    #plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plotCM(matrix):
    """classes: a list of class names"""
    # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
    #ax.set_xticklabels([''] + classes, rotation=90)

 
if __name__ == '__main__':
    # Detect devices
    torch.cuda.set_device(0)
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
    
    # Data loading parameters
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': True} if use_cuda else {}


    # load classifications
    with open(classification_path, 'rb') as f:
        classifications_name = pickle.load(f)

    # convert labels to category
    le = LabelEncoder()
    le.fit(classifications_name)

    # show the number of classifications
    list(le.classes_)

    # 1 hot
    error_classifications = le.transform(classifications_name).reshape(-1, 1)
    enc = OneHotEncoder(categories='auto')
    enc.fit(error_classifications)

    errors = []
    fnames = os.listdir(data_path)
    all_error_modes = []
    
    fnames_test = os.listdir(test_data_path)
    all_error_modes_test = []
    errors_test = []
    for f in fnames:
        loc1 = f.find('_e')
        loc2=f.find('_s')
        loc3=f.find('_n')
           
        if f[(loc1 + 1):loc2] == "error_mode_0":  
            errors.append("no_error")
            all_error_modes.append(f) 
        if f[(loc1 + 1):loc2] == "error_mode_7":  
            errors.append("error")
            all_error_modes.append(f) 
        if f[(loc1 + 1):loc2] == "error_mode_8":  
            errors.append("error")
            all_error_modes.append(f) 
        if f[(loc1 + 1):loc2] == "error_mode_9":  
            errors.append("error")
            all_error_modes.append(f) 
        
        if f[(loc1 + 1):loc3] == "error_mode_0":  
            errors.append("no_error")
            all_error_modes.append(f) 
        if f[(loc1 + 1):loc3] == "error_mode_7":  
            errors.append("error")
            all_error_modes.append(f) 
        if f[(loc1 + 1):loc3] == "error_mode_8":  
            errors.append("error")
            all_error_modes.append(f) 
        if f[(loc1 + 1):loc3] == "error_mode_9":  
            errors.append("error")
            all_error_modes.append(f) 
        

        #errors.append(f[(loc1 + 1):loc2])
        
    for f in fnames_test:
        loc1 = f.find('_e')
        loc2=f.find('_s')
        loc3=f.find('_n')
        
        if f[(loc1 + 1):loc2] == "error_mode_0":  
            errors_test.append("no_error")
            all_error_modes_test.append(f) 
        if f[(loc1 + 1):loc2] == "error_mode_7":  
            errors_test.append("error")
            all_error_modes_test.append(f) 
        if f[(loc1 + 1):loc2] == "error_mode_8":  
            errors_test.append("error")
            all_error_modes_test.append(f) 
        if f[(loc1 + 1):loc2] == "error_mode_9":  
            errors_test.append("error")
            all_error_modes_test.append(f) 
        
        if f[(loc1 + 1):loc3] == "error_mode_0":  
            errors_test.append("no_error")
            all_error_modes_test.append(f) 
        if f[(loc1 + 1):loc3] == "error_mode_7":  
            errors_test.append("error")
            all_error_modes_test.append(f) 
        if f[(loc1 + 1):loc3] == "error_mode_8":  
            errors_test.append("error")
            all_error_modes_test.append(f) 
        if f[(loc1 + 1):loc3] == "error_mode_9":  
            errors_test.append("error")
            all_error_modes_test.append(f) 
        
        
  
            

# list all data files
    X_LIST = all_error_modes                  # all video file names
    y_a_list = labels2cat(le, errors)    # all video labels
    
    X_LIST_test = all_error_modes_test                  
    y_a_list_test = labels2cat(le, errors_test)
    
    

    # train, validation split
    train_list, valid_list, train_label, valid_label = train_test_split(X_LIST, y_a_list, test_size=0.125, random_state=42)

    transform = transforms.Compose([transforms.Resize([size, size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    train_set, valid_set, test_set = process_dataset(data_path, train_list, train_label, selected_frames, transform=transform), \
                           process_dataset(data_path, valid_list, valid_label, selected_frames, transform=transform), \
                           process_dataset(test_data_path, X_LIST_test, y_a_list_test, selected_frames, transform=transform)

    
    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)
    test_loader = data.DataLoader(test_set,**params)

    # Create model
    encoder = ResNetEncoder(hidden1=cnn_hidden_1, hidden2=cnn_hidden_2, drop_out=dropout, CNN_DIM=cnn_dims).to(device)
    decoder = RNNDecoder(CNN_DIM=cnn_dims, RNNLayers=rnnHiddenLayer, h_RNN=rnnHiddenNode, 
                         h_Dim=rnn_dim, drop_out=dropout, num_classes=k).to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

   
        crnn_params = list(encoder.module.fc1.parameters()) + list(encoder.module.bn1.parameters()) + \
                      list(encoder.module.fc2.parameters()) + list(encoder.module.bn2.parameters()) + \
                      list(encoder.module.fc3.parameters()) + list(decoder.parameters())

    elif torch.cuda.device_count() == 1:
        print("Using", torch.cuda.device_count(), "GPU!")
   
        crnn_params = list(encoder.fc1.parameters()) + list(encoder.bn1.parameters()) + \
                      list(encoder.fc2.parameters()) + list(encoder.bn2.parameters()) + \
                      list(encoder.fc3.parameters()) + list(decoder.parameters())

    optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)


    # record train results
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_train_f1s = []
    epoch_train_precisions = []
    epoch_train_recalls = []
    
    # record validation results
    epoch_validation_losses = []
    epoch_validation_losss = []
    epoch_test_f1s=[]
    epoch_test_precisions=[]
    epoch_test_recalls=[]
    # start training
    for epoch in range(epochs):
        # train, test model
        train_losses, train_scores,precisions,recalls,f1s = train(logInternal, [encoder, decoder], device, train_loader, optimizer, epoch)
        epoch_validation_loss, epoch_validation_loss,epoch_test_f1,confusionMatrix,valid_precision,valid_recall = validation([encoder, decoder], device, optimizer, valid_loader)

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_train_f1s.append(f1s)
        epoch_train_precisions.append(precisions)
        epoch_train_recalls.append(recalls)
        
        epoch_validation_losses.append(epoch_validation_loss)
        epoch_validation_losss.append(epoch_validation_loss)
        epoch_test_f1s.append(epoch_test_f1)
        epoch_test_precisions.append(valid_precision)
        epoch_test_recalls.append(valid_recall)
        
        
    
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        TRAIN_F1 = np.array(epoch_train_f1s)
        TRAIN_RECALL = np.array(epoch_train_recalls)
        TRAIN_PRECISION = np.array(epoch_train_precisions)
   
        C = np.array(epoch_validation_losses)
        D = np.array(epoch_validation_losss)
        TEST_F1 = np.array(epoch_test_f1s)
        TEST_PRECISION = np.array(epoch_test_precisions)
        TEST_RECALL = np.array(epoch_test_recalls)
        
        
        np.save('./CRNN_epoch_training_losses.npy', A)
        np.save('./CRNN_epoch_training_scores.npy', B)
        np.save('./CRNN_epoch_validation_loss.npy', C)
        np.save('./CRNN_epoch_validation_loss.npy', D)
    
    #to test 
    print('test loss',C)
    print('test accuracy',D)
    min_loss=C.min()
    min_index=np.argmin(C)+1
    print('min_loss',min_loss)
    print('min_index',min_index)
    

    encoder.load_state_dict(torch.load(os.path.join(same_path, 'encoder_epoch{}.pth'.format(min_index))))
    decoder.load_state_dict(torch.load(os.path.join(same_path, 'decoder_epoch{}.pth'.format(min_index))))
    optimizer.load_state_dict(torch.load(os.path.join(same_path, 'optimizer_epoch{}.pth'.format(min_index))))
    print('\nCRNN model reloaded!')
    print('\nThe epoch with the lowest valdiation loss is {}'.format(min_index))
    print('\nTest result:')
    
    
    validation_loss, validation_loss,valid_f1,valid_confusionMatrix,valid_precision,valid_recall = test([encoder, decoder], device, optimizer, test_loader)
    valid_confusionMatrix=valid_confusionMatrix.reshape(valid_confusionMatrix.shape[0],valid_confusionMatrix.shape[1])
    print('confusion size:',valid_confusionMatrix.shape)
    print(valid_confusionMatrix)
    sn.set()
    fig = plt.figure(figsize=(10, 4))
    sn.heatmap(valid_confusionMatrix, annot=True)
    plt.title("confusion matrix")
    plt.xlabel('predict')
    plt.ylabel('true')
    plt.show()
 

    '''
    np.set_printoptions(precision=2)
    cm_normalized = valid_confusionMatrix.astype('float')/valid_confusionMatrix.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plt.show()
    '''
    

    
    #plotCM(valid_confusionMatrix)
    # plot
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.arange(1, epochs + 1), A[:, -2])  # train loss (on epoch end)
    plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc="upper left")
    # 2nd figure
    plt.subplot(122)
    plt.plot(np.arange(1, epochs + 1), B[:, -2])  # train accuracy (on epoch end)
    plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
    plt.title("Accuracy")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'validation'], loc="upper left")
    title = "./fig_UCF101_ResNetCRNN.png"
    plt.savefig(title, dpi=600)
    plt.show()
    
    #3rd figure
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.arange(1, epochs + 1), TRAIN_F1[:, -2])  # train F1
    plt.plot(np.arange(1, epochs + 1), TEST_F1)         # test f1
    plt.title("f1_score")
    plt.xlabel('epochs')
    plt.ylabel('F 1')
    plt.legend(['train', 'validation'], loc="upper left")
    #4TH figure
    plt.subplot(122)
    plt.plot(np.arange(1, epochs + 1), TRAIN_RECALL[:, -2])  # train recall
    plt.plot(np.arange(1, epochs + 1), TEST_RECALL)         #  recall
    plt.title("recall_score")
    plt.xlabel('epochs')
    plt.ylabel('recall')
    plt.legend(['train', 'validation'], loc="upper left")
    title = "./fig_UCF101_ResNetCRNN_f1_recall.png"
    plt.savefig(title, dpi=600)
    plt.show()
    
    #5th figure
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.arange(1, epochs + 1), TRAIN_PRECISION[:, -2])  # train precision
    plt.plot(np.arange(1, epochs + 1), TEST_PRECISION)         #  test precision
    plt.title("precision_score")
    plt.xlabel('epochs')
    plt.ylabel('precision')
    plt.legend(['train', 'validation'], loc="upper left")
    
    title = "./fig_UCF101_ResNetCRNN_precision.png"
    plt.savefig(title, dpi=600)
    # plt.close(fig)
    plt.show()
 