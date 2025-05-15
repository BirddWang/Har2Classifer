from dataset_adni import ADNI
import torch.nn.functional as F
from beta_encoder import BetaEncoder
from SFCN import SFCN, get_Bb_dims
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
from utils import Recorder


class ADNI_Classifier:
    def __init__(self, args):
        self.encoder, self.classifier = self.init_model(args.classifier_state_dict)
        self.dataset = ADNI()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier.to(self.device)
        self.encoder.to(self.device)
        self.recorder = Recorder(log_dir=args.log_dir)
        # training
        self.epochs = args.epochs
        self.lr = args.lr
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.weight_decay = args.weight_decay


    def init_model(self, classifier_state_dict=None):
        encoder = BetaEncoder(in_ch=1, out_ch=5, base_ch=8, final_act='none')
        encoder = encoder.to('cuda')

        classifier = SFCN(get_Bb_dims(7), 1)
        if classifier_state_dict is not None:
            classifier.load_state_dict(classifier_state_dict)

        classifier = classifier.to('cuda')
        return encoder, classifier

    def get_beta(self, img, mask):
        batch_size = img.shape[0]
        slice_size = img.shape[1]
        img_shape = img.shape[2]
        img = img.view(batch_size * slice_size, 1, img_shape, img_shape)
        mask = mask.to('cuda')
        

        logit = reparameterize_logit(self.encoder.encode(img))
        beta = beta_aggregation(logit)

        mask = mask.view(batch_size * slice_size, 1, img_shape, img_shape)
        masked_beta = beta * mask

        masked_beta_3d = masked_beta.view(batch_size, 1, slice_size, img_shape, img_shape)

        return masked_beta_3d


    def encode(self, img, mask):
        img = img.to(self.device)
        mask = mask.to(self.device)
        beta = self.get_beta(img, mask)
        return beta
        

    def sample_predict(self):
        sample = self.dataset[0]
        img = sample['masked_image']
        mask = sample['mask']

        self.encoder.eval()
        with torch.no_grad():
            img = img.to(self.device)
            mask = mask.to(self.device)
            beta = self.get_beta(torch.unsqueeze(img, dim=1), torch.unsqueeze(mask, dim=1))
            print(beta.shape)
            beta = beta.unsqueeze(0).unsqueeze(0)
            print(beta.shape)
            pred = self.classifier(beta)
            pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy()
            print(pred)
            return pred 
    
    def test(self, val_loader, criterion):
        total_loss = 0
        total_acc = 0

        self.classifier.eval()
        with torch.no_grad():
            for batch in val_loader:
                img = batch['masked_image'].to(self.device)
                mask = batch['mask'].to(self.device)
                beta = self.get_beta(img, mask)
                pred = self.classifier(beta)
                label = batch['label'].to(self.device).float()
                loss = criterion(pred.view(-1), label.view(-1))
                total_loss += loss.item()
                pred = torch.sigmoid(pred)
                pred = (pred > 0.5).float()
                acc = (pred == label).float().mean()
                total_acc += acc.item()
        avg_loss = total_loss / len(val_loader)
        avg_acc = total_acc / len(val_loader)
        print(f"Validation Loss: {avg_loss}, Validation Accuracy: {avg_acc}")
        return avg_loss, avg_acc

    def train(self):
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()

        train_num = int(len(self.dataset) * 0.8)
        train_data, val_data = torch.utils.data.random_split(self.dataset, [train_num, len(self.dataset) - train_num])
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.val_batch_size, shuffle=False)
        scaler = torch.amp.grad_scaler.GradScaler()

        self.classifier.train()
        for epoch in tqdm(range(self.epochs)):
            losses = []
            accs = []
            for batch in train_loader:
                optimizer.zero_grad()
                # [B, 192, 224, 224]
                img = batch['masked_image'].to(self.device)
                mask = batch['mask'].to(self.device)
                with torch.no_grad():
                    beta = self.get_beta(img, mask)
                with torch.amp.autocast('cuda'):
                    pred = self.classifier(beta)
                    label = batch['label'].to(self.device).float()
                    loss = criterion(pred.view(-1), label.view(-1))
                scaler.scale(loss).backward()   
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
                pred = torch.sigmoid(pred)
                pred = (pred > 0.5).float()
                acc = (pred == label).float().mean()
                accs.append(acc.item())
            avg_loss = sum(losses) / len(losses)
            avg_acc = sum(accs) / len(accs)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss}, Accuracy: {avg_acc}")
            self.recorder.log_acc(avg_acc, epoch)
            self.recorder.log_loss(avg_loss, epoch)

            val_loss, val_acc = self.test(val_loader, criterion)
            self.recorder.log_val_acc(val_acc, epoch)
            self.recorder.log_val_loss(val_loss, epoch)

            if (epoch + 1) % 10 == 0:
                torch.save(self.classifier.state_dict(), f'checkpoint_epoch_{epoch + 1}.pth')
                print(f"Model saved at epoch {epoch + 1}")
        self.recorder.close()
        torch.save(self.classifier.state_dict(), 'final_model.pth')
        print("Training complete. Model saved as final_model.pth")

#### help functions ####
def reparameterize_logit(logit):
    beta = F.gumbel_softmax(logit, tau=1.0, dim=1, hard=True)
    return beta

def beta_aggregation(logit):
    value_tensor = torch.arange(0, 5).to('cuda')
    value_tensor = value_tensor.view(1, 5, 1, 1).repeat(logit.shape[0], 1, logit.shape[2], logit.shape[3])
    beta = torch.sum(logit * value_tensor.detach(), dim=1, keepdim=True) / 5.0
    return beta
########################



def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='ADNI Classifier')
    parser.add_argument('--classifier_state_dict', type=str, default=None, help='Path to the classifier state dict')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=2, help='Batch size for validation')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for optimizer')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory to save logs')
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = get_args()
    classifier = ADNI_Classifier(args)
    classifier.train()