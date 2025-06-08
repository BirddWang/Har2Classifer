from dataset_adni import ADNI
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall
from beta_encoder import BetaEncoder
from SFCN import SFCN, get_Bb_dims
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
from utils import Recorder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

class EarlyStopping:
    def __init__(self, patience=25, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
        


class ADNI_Classifier:
    def __init__(self, args):
        self.encoder, self.classifier = self.init_model(args.classifier_state_dict)
        self.dataset = ADNI(training=True, data_type='preprocessed')
        print(f"Dataset initialized, length: {len(self.dataset)}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device set to:", self.device)
        self.classifier.to(self.device, non_blocking=True)
        self.encoder.to(self.device, non_blocking=True)
        self.recorder = Recorder(log_dir=f"{args.log_dir}/{self.dataset.data_type}")
        # training
        self.epochs = args.epochs
        self.lr = args.lr
        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.weight_decay = args.weight_decay
        # self.early_stopping = EarlyStopping(patience=25, min_delta=1e-4)
        self.precision = BinaryPrecision(threshold=0.5)
        self.recall = BinaryRecall(threshold=0.5)
        self.f1_score = BinaryF1Score(threshold=0.5)
        self.auroc = BinaryAUROC()


    def init_model(self, classifier_state_dict=None):
        encoder = BetaEncoder(in_ch=1, out_ch=5, base_ch=8, final_act='none')
        encoder = encoder.to('cuda')

        classifier = SFCN(get_Bb_dims(4), 1)
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
        img = img.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        beta = self.get_beta(img, mask)
        return beta
        

    def sample_predict(self):
        sample = self.dataset[0]
        img = sample['masked_image']
        mask = sample['mask']

        self.encoder.eval()
        with torch.no_grad():
            img = img.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            beta = self.get_beta(torch.unsqueeze(img, dim=1), torch.unsqueeze(mask, dim=1))
            print(beta.shape)
            beta = beta.unsqueeze(0).unsqueeze(0)
            print(beta.shape)
            pred = self.classifier(beta)
            pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy()
            print(pred)
            return pred 
    
    def balanced_split(self, train_ratio=0.8):
        ad_paths = [i for i, path in enumerate(self.dataset.img_paths) if self.dataset.labels[path] == 1]
        cn_paths = [i for i, path in enumerate(self.dataset.img_paths) if self.dataset.labels[path] == 0]
        ad_train = ad_paths[:int(train_ratio * len(ad_paths))]
        ad_val = ad_paths[int(train_ratio * len(ad_paths)):]
        
        cn_train = cn_paths[:int(train_ratio * len(cn_paths))]
        cn_val = cn_paths[int(train_ratio * len(cn_paths)):]
        
        train_indices = ad_train + cn_train
        val_indices = ad_val + cn_val

        import random
        random.shuffle(train_indices)
        random.shuffle(val_indices)

        return train_indices, val_indices

    
    def test(self, val_loader, criterion, only_use_beta:bool = False):
        total_loss = 0

        all_preds = []
        all_labels = []
        self.classifier.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                if only_use_beta:
                    img = batch['masked_image'].to(self.device, non_blocking=True)
                    mask = batch['mask'].to(self.device, non_blocking=True)
                    beta = self.get_beta(img, mask)
                    pred = self.classifier(beta).view(-1)
                    label = batch['label'].to(self.device, non_blocking=True).float()
                    loss = criterion(pred, label)
                else:
                    img = batch['image'].to(self.device, non_blocking=True)
                    pred = self.classifier(img).view(-1)
                    label = batch['label'].to(self.device, non_blocking=True).float()
                    loss = criterion(pred, label)
                
                total_loss += loss.item()
                pred = (pred > 0.5).float()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        auc = self.auroc(torch.tensor(all_preds, device=self.device), torch.tensor(all_labels, device=self.device))
        precision = self.precision(torch.tensor(all_preds, device=self.device), torch.tensor(all_labels, device=self.device))
        recall = self.recall(torch.tensor(all_preds, device=self.device), torch.tensor(all_labels, device=self.device))
        f1 = self.f1_score(torch.tensor(all_preds, device=self.device), torch.tensor(all_labels, device=self.device))

        # print(f"Validation Loss: {avg_loss}, Validation Accuracy: {avg_acc}")
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=['CN', 'AD']))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        return {
            'loss': avg_loss,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }

    def train(self, warmup_epochs=20, mid_path=None):
        print("Training...")
        print(f"Learning rate: {self.lr}, Train batch size: {self.train_batch_size}, Val batch size: {self.val_batch_size}, Weight decay: {self.weight_decay}")
        # print classifier structure
        print(self.classifier)
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs - warmup_epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
        criterion = torch.nn.BCEWithLogitsLoss()

        if mid_path is not None:
            print(f"Loading beta model from {mid_path}")
            checkpoint = torch.load(mid_path, map_location=self.device)
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.precision.to(self.device, non_blocking=True)
        self.recall.to(self.device, non_blocking=True)
        self.f1_score.to(self.device, non_blocking=True)
        self.auroc.to(self.device, non_blocking=True)

        train_indices, val_indices = self.balanced_split()
        train_data = torch.utils.data.Subset(self.dataset, train_indices)
        val_data = torch.utils.data.Subset(self.dataset, val_indices)

        class_count = [self.dataset.len_cn, self.dataset.len_ad] # [CN, AD]
        class_weights = torch.tensor([1.0 / count for count in class_count], device=self.device)

        train_sample_weights = []
        for idx in train_indices:
            label = self.dataset.labels[self.dataset.img_paths[idx]]
            train_sample_weights.append(class_weights[label].item())
        sampler = torch.utils.data.WeightedRandomSampler(
            train_sample_weights,
            num_samples=len(train_sample_weights),
            replacement=True,  # Sample with replacement
        )
            
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.train_batch_size,
            sampler=sampler,
            num_workers=8,
            pin_memory=True,

        )
        val_loader = torch.utils.data.DataLoader(
            val_data, 
            batch_size=self.val_batch_size, 
            shuffle=False, 
            num_workers=8,
            pin_memory=True,
        )

        # Ensure validation dataset does not use training augmentations
        scaler = torch.amp.grad_scaler.GradScaler()
        self.classifier.train()
        for epoch in tqdm(range(checkpoint['epoch']+1, self.epochs) if checkpoint is not None else range(self.epochs)):
            if epoch < warmup_epochs: print(f"Warmup Epoch {epoch + 1}/{warmup_epochs}")
            else: print(f"Epoch {epoch + 1 - warmup_epochs}/{self.epochs - warmup_epochs}")
            losses = []
            all_preds = torch.empty(0, device=self.device)
            all_labels = []
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                # [B, 192, 224, 224]
                img = batch['image'].to(self.device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    # img is [B, C(1), 192, 224, 224]
                    pred = self.classifier(img).view(-1)
                    label = batch['label'].to(self.device, non_blocking=True).float()
                    loss = criterion(pred, label)

                scaler.scale(loss).backward()   
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
                all_preds = torch.cat((all_preds, pred), dim=0)
                all_labels.extend(label.cpu().numpy())

                if (i+1) % 25 == 0 or i == len(train_loader) - 1:
                    with torch.no_grad():
                        all_preds = torch.sigmoid(all_preds)
                    all_preds = (all_preds > 0.5).float().cpu().numpy()
                    avg_loss = sum(losses) / len(losses)

                    auc = self.auroc(torch.tensor(all_preds, device=self.device), torch.tensor(all_labels, device=self.device))
                    precision = self.precision(torch.tensor(all_preds, device=self.device), torch.tensor(all_labels, device=self.device))
                    recall = self.recall(torch.tensor(all_preds, device=self.device), torch.tensor(all_labels, device=self.device))
                    f1 = self.f1_score(torch.tensor(all_preds, device=self.device), torch.tensor(all_labels, device=self.device))

                    self.recorder.log_scalar('train/loss', avg_loss, epoch * len(train_loader) + i)
                    self.recorder.log_lr(optimizer.param_groups[0]['lr'], epoch * len(train_loader) + i)
                    self.recorder.log_scalar('train/auc', auc, epoch * len(train_loader) + i)
                    self.recorder.log_scalar('train/precision', precision, epoch * len(train_loader) + i)
                    self.recorder.log_scalar('train/recall', recall, epoch * len(train_loader) + i)
                    self.recorder.log_scalar('train/f1_score', f1, epoch * len(train_loader) + i)
                    print(f"Epoch {epoch + 1}/{self.epochs}, Step {i+1}/{len(train_loader)}, Loss: {avg_loss}, AUC: {auc}, \nPrecision: {precision}, Recall: {recall}, F1: {f1}")
                    
                    confusion = confusion_matrix(all_labels, all_preds)
                    print("Confusion Matrix:")
                    print(confusion)
                    losses = []
                    all_preds = torch.empty(0, device=self.device)
                    all_labels = []
                    
            res = self.test(val_loader, criterion)
            self.recorder.log_scalar('val/loss', res['loss'], epoch)
            self.recorder.log_scalar('val/auc', res['auc'], epoch)
            self.recorder.log_scalar('val/precision', res['precision'], epoch)
            self.recorder.log_scalar('val/recall', res['recall'], epoch)
            self.recorder.log_scalar('val/f1_score', res['f1_score'], epoch)
            # if self.early_stopping(res['loss']):
            #     print(f"Early stopping at epoch {epoch + 1}, validation loss did not improve for {self.early_stopping.patience} epochs.")
            #     break
            print(f"Epoch {epoch + 1}/{self.epochs}, Validation Loss: {res['loss']}, AUC: {res['auc']}")
            print(res['loss'])
            
            if (epoch + 1) % 10 == 0:
                # save several things to maintain the corrupted state
                print(f"Saving model at epoch {epoch + 1}")
                torch.save({
                    'epoch': epoch,
                    'classifier_state_dict': self.classifier.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'loss': res['loss'],
                }, f'checkpoints/{self.dataset.data_type}_epoch-{epoch + 1}_complete.pth')
                torch.save(self.classifier.state_dict(), f'checkpoints/{self.dataset.data_type}_epoch-{epoch + 1}.pth')
            scheduler.step()
        self.recorder.close()
        torch.save(self.classifier.state_dict(), f'{self.dataset.data_type}_final_model.pth')

#### help functions for beta ####
def reparameterize_logit(logit):
    beta = F.gumbel_softmax(logit, tau=1.0, dim=1, hard=True)
    return beta

def beta_aggregation(logit):
    value_tensor = torch.arange(0, 5).to('cuda')
    value_tensor = value_tensor.view(1, 5, 1, 1).repeat(logit.shape[0], 1, logit.shape[2], logit.shape[3])
    beta = torch.sum(logit * value_tensor.detach(), dim=1, keepdim=True) / 5.0
    return beta
#################################



def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='ADNI Classifier')
    parser.add_argument('--classifier_state_dict', type=str, default=None, help='Path to the classifier state dict')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=16, help='Batch size for validation')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory to save logs')
    parser.add_argument('--beta_only', action='store_true', help='Use only beta for training and evaluation')
    parser.add_argument('--midi_path', type=str, default='checkpoints/beta_epoch-110_complete.pth', help='Path to the midi model state dict')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    classifier = ADNI_Classifier(args)
    classifier.train(mid_path = args.midi_path)