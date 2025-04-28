import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rasterio
from torch.nn import CrossEntropyLoss
from torchvision.utils import save_image
import torch.optim as optim
from torchmetrics.classification import JaccardIndex
from sklearn.metrics import confusion_matrix
import seaborn as sns

class_colors = np.array([
    [0, 0, 1, 1],  # Blue for class 0 (RGB: 0, 0, 255)
    [1, 0, 0, 1],  # Red for class 1 (RGB: 255, 0, 0)
    [0, 1, 0, 1],  # Green for class 2 (RGB: 0, 255, 0)
    [0.6, 0.3, 0, 1],  # Brown for class 3 (RGB: 153, 77, 0)
    [1, 1, 0.6, 1]   # Light Yellow for class 4 (RGB: 255, 255, 153)
])

def apply_colormap(classes):
    """Apply a custom colormap to the predicted class tensor."""
    class_np = classes.squeeze().cpu().numpy().astype(int)  
    
    color_image = np.zeros((class_np.shape[0], class_np.shape[1], 4))  

    for i in range(5):
        color_image[class_np == i] = class_colors[i]

    return color_image
    
def load_tiff(file_path):
    with rasterio.open(file_path) as src:
        image = src.read()
        meta = src.meta
        nodata = src.nodata

    image = image.astype(np.float32)
    if nodata is not None:
        image[image == nodata] = np.nan
    image_min, image_max = np.nanmin(image), np.nanmax(image)
    image = (image - image_min) / (image_max - image_min)

    boundary_mask = ~np.isnan(image[0])
    boundary_mask = boundary_mask.astype(np.float32)
    image = np.nan_to_num(image, nan=0.0)

    image_tensor = torch.from_numpy(image).unsqueeze(0)  
    boundary_mask_tensor = torch.from_numpy(boundary_mask).unsqueeze(0) 
    return image_tensor, boundary_mask_tensor, meta


image_tensor, boundary_mask_tensor, meta = load_tiff("/home/shovik.roy/vanaja_btep/grayscaleout_2020.tif")

boundary_mask_np = boundary_mask_tensor.squeeze().numpy()  

#plt.figure(figsize=(6, 6))
#plt.imshow(boundary_mask_np, cmap='gray')
#plt.title("Valid Regions (Boundary Mask)")
#plt.axis('off')
#plt.show()


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtractor, self).__init__()

        # Trainable convolutional layers
        self.conv3x3 = nn.Conv2d(in_channels, 16, 3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels, 16, 5, padding=2, bias=False)
        self.conv7x7 = nn.Conv2d(in_channels, 16, 7, padding=3, bias=False)
        self.conv9x9 = nn.Conv2d(in_channels, 16, 9, padding=4, bias=False)

        # Scharr filters (fixed, used for edge detection)
        scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32)
        scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)
        self.scharr_x = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False, groups=in_channels)
        self.scharr_y = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False, groups=in_channels)
        self.scharr_x.weight = nn.Parameter(scharr_x[None, None].repeat(in_channels, 1, 1, 1), requires_grad=False)
        self.scharr_y.weight = nn.Parameter(scharr_y[None, None].repeat(in_channels, 1, 1, 1), requires_grad=False)

        # Laplacian filter (fixed, highlights regions of rapid intensity change)
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.laplacian = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False, groups=in_channels)
        self.laplacian.weight = nn.Parameter(laplacian[None, None].repeat(in_channels, 1, 1, 1), requires_grad=False)

        # Identity layer (passes input as-is to retain original features)
        self.identity = nn.Identity()


    def forward(self, x):
        features = {
            "identity": self.identity(x),
            "conv3x3": self.conv3x3(x),
            "conv5x5": self.conv5x5(x),
            "conv7x7": self.conv7x7(x),
            "conv9x9": self.conv9x9(x),
            "scharr_x": self.scharr_x(x),
            "scharr_y": self.scharr_y(x),
            "laplacian": self.laplacian(x)
        }
        return torch.cat(list(features.values()), dim=1)
        
class NCAUpdate(nn.Module):
    def __init__(self, num_features, num_hidden=[64, 64, 128], num_out=5, kernel_size=(3, 3)):
        super(NCAUpdate, self).__init__()

        
        from convLSTM import ConvLSTM
        self.conv_lstm = ConvLSTM(input_dim=num_features,
                                  hidden_dim=num_hidden,
                                  kernel_size=kernel_size,
                                  num_layers=len(num_hidden),
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)

        self.final_conv = nn.Conv2d(num_hidden[-1], num_out, kernel_size=1)

    def forward(self, x_seq, prev_state, boundary_mask):

        #print(">> Inside NCAUpdate.forward")
        #print("x_seq:", x_seq.shape)
        #print("prev_state:", prev_state.shape)
        #print("boundary_mask:", boundary_mask.shape)

        if torch.isnan(x_seq).any():
            print("NaNs detected in x_seq!")
        if torch.isnan(prev_state).any():
            print("NaNs detected in prev_state!")
        if torch.isnan(boundary_mask).any():
            print("NaNs detected in boundary_mask!")

        lstm_out, _ = self.conv_lstm(x_seq) 
        print("ConvLSTM output shape:", lstm_out[0].shape)

        out_seq = lstm_out[0]  
        dx = self.final_conv(out_seq[:, -1])  

        current_class = torch.argmax(prev_state, dim=1, keepdim=True)
        irreversible_classes = [1]                                                # Built-up
        irreversible_mask = torch.zeros_like(current_class, dtype=torch.float32)
        for cls in irreversible_classes:
            irreversible_mask += (current_class == cls).float()

        update_mask = boundary_mask * (1 - irreversible_mask)
        new_state = prev_state + dx * update_mask
        return new_state

def dice_loss(pred, target, smooth=1e-7):
    """
    pred: [B, C, H, W] logits
    target: [B, H, W] long
    """
    pred_soft = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)  
    intersection = (pred_soft * target_one_hot).sum(dim=dims)
    cardinality = pred_soft.sum(dim=dims) + target_one_hot.sum(dim=dims)
    dice = (2. * intersection + smooth) / (cardinality + smooth)
    return 1 - dice.mean()
    

def tversky_loss(pred, target, alpha=0.3, beta=0.7, smooth=1e-7):
    """
    pred: [B, C, H, W] logits
    target: [B, H, W] long
    alpha: weight for false negatives
    beta: weight for false positives
    """
    pred_soft = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3) 
    TP = (pred_soft * target_one_hot).sum(dim=dims)
    FN = ((1 - pred_soft) * target_one_hot).sum(dim=dims)
    FP = (pred_soft * (1 - target_one_hot)).sum(dim=dims)

    tversky = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
    return 1 - tversky.mean()


def combo_dice_tversky_loss(pred, target, alpha=0.3, beta=0.7, dice_weight=0.5):
    """
    pred: [B, C, H, W] logits
    target: [B, H, W] long
    alpha: Tversky FN weight
    beta: Tversky FP weight
    dice_weight: Weight to balance Dice and Tversky components
    """
    pred_soft = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    TP = (pred_soft * target_one_hot).sum(dim=dims)
    FN = ((1 - pred_soft) * target_one_hot).sum(dim=dims)
    FP = (pred_soft * (1 - target_one_hot)).sum(dim=dims)

    tversky = (TP + 1e-7) / (TP + alpha * FN + beta * FP + 1e-7)
    dice = (2 * TP + 1e-7) / (2 * TP + FP + FN + 1e-7)

    loss = (1 - dice_weight) * (1 - tversky.mean()) + dice_weight * (1 - dice.mean())
    return loss
    
def compute_class_weights(label_tensor, num_classes):
    """
    Compute class weights using inverse frequency.
    
    label_tensor: Tensor of shape [B, H, W] or [H, W]
    num_classes: Total number of classes
    """
    if label_tensor.dim() == 3:
        label_tensor = label_tensor.view(-1)
    elif label_tensor.dim() == 2:
        label_tensor = label_tensor.flatten()

    total_samples = label_tensor.numel()
    class_counts = torch.bincount(label_tensor, minlength=num_classes).float()
    
    weights = total_samples / (class_counts * num_classes + 1e-7)  
    weights = weights / weights.sum()  
    return weights
    

def train_nca_model():
    ''' 
    Set device and prepare the output directory for saving results 
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "/home/shovik.roy/vanaja_btep/btep_final_results"
    os.makedirs(output_dir, exist_ok=True)

    ''' 
    File paths for temporal grayscale TIFF inputs spanning different years 
    '''
    file_paths = [
        "/home/shovik.roy/vanaja_btep/grayscaleout_Delhi_2017.tif",
        "/home/shovik.roy/vanaja_btep/grayscaleout_Delhi_2019.tif",
        "/home/shovik.roy/vanaja_btep/1_grayscaleout.tif",
        "/home/shovik.roy/vanaja_btep/grayscaleout_Delhi_2023.tif"
    ]

    ''' 
    Initialize model components: Feature extractor and NCA update module 
    '''
    sample_img, _, _ = load_tiff(file_paths[0])
    feature_extractor = FeatureExtractor(in_channels=sample_img.shape[1]).to(device)
    with torch.no_grad():
        dummy_feat = feature_extractor(sample_img.to(device))
    nca_model = NCAUpdate(num_features=dummy_feat.shape[1], num_hidden=[64], num_out=5).to(device)

    ''' 
    Optimizer and loss function setup with class weights 
    '''
    optimizer = optim.Adam(nca_model.parameters(), lr=1e-3)
    class_weights = torch.tensor([0.0062, 0.0115, 0.0092, 0.7788, 0.1944], dtype=torch.float).to(device)
    ce_loss_fn = CrossEntropyLoss(weight=class_weights)
    
   
    ce_loss_history = []
    combo_loss_history = []

    ''' 
    Training loop for multiple epochs 
    '''
    epochs = 700
    #num_steps = 5  

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1} ---")
        total_ce, total_combo = 0, 0

        images_seq, masks_seq, targets_seq = [], [], []

        ''' 
        Load input TIFF images, apply feature extractor, and prepare target sequences 
        '''
        for t in range(len(file_paths)):
            img, mask, _ = load_tiff(file_paths[t])
            img, mask = img.to(device), mask.to(device).unsqueeze(1)
            features = feature_extractor(img)
            images_seq.append(features)
            masks_seq.append(mask)

        for t in range(1, len(file_paths)):
            tgt, _, _ = load_tiff(file_paths[t])
            tgt = tgt.to(device)
            tgt_class = (tgt * (nca_model.final_conv.out_channels - 1)).long().squeeze(1)
            targets_seq.append(tgt_class)

        ''' 
        Stack features and masks into a temporal sequence 
        '''
        feature_seq = torch.stack(images_seq[:-1], dim=1)  
        mask_seq = torch.stack(masks_seq[:-1], dim=1)      

        ''' 
        Initialize the urban state from the first frame 
        '''
        initial_state = images_seq[0]
        initial_class = (initial_state[:, 0:1] * (nca_model.final_conv.out_channels - 1)).long()
        urban_state = torch.zeros((1, nca_model.final_conv.out_channels, initial_state.shape[2], initial_state.shape[3]), device=device)
        urban_state.scatter_(1, initial_class, 1.0)


        ''' 
        Forward pass: Predict the updated state using NCA model 
        '''
        updated_state = nca_model(feature_seq, urban_state, mask_seq[:, -1])

        ''' 
        Analyze class distribution in target and prediction for debugging 
        '''
        unique_classes, counts = torch.unique(targets_seq[-1], return_counts=True)
        class_weights = compute_class_weights(targets_seq[-1], 5)
        
        pred_probs = F.softmax(updated_state, dim=1)
        pred_classes = torch.argmax(pred_probs, dim=1)
        pred_unique, pred_counts = torch.unique(pred_classes, return_counts=True)


        if epoch == epochs - 1:
            colored_pred_classes = apply_colormap(pred_classes)
            colored_target_classes = apply_colormap(targets_seq[-1])

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(colored_target_classes)
            plt.title("Target Class Map")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(colored_pred_classes)
            plt.title("Predicted Class Map")
            plt.axis('off')
            

            target_out_path = os.path.join(output_dir, f"epoch{epoch+1}_target.png")
            save_image(torch.tensor(colored_target_classes.transpose(2, 0, 1)).unsqueeze(0).float(), target_out_path)

            pred_out_path = os.path.join(output_dir, f"epoch{epoch+1}_predicted.png")
            save_image(torch.tensor(colored_pred_classes.transpose(2, 0, 1)).unsqueeze(0).float(), pred_out_path)
            plt.close()


        '''(Cross Entropy + Tversky-Dice)'''
        ce_loss = ce_loss_fn(updated_state, targets_seq[-1])
        combo_loss = combo_dice_tversky_loss(updated_state, targets_seq[-1])
        loss = ce_loss + combo_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_ce += ce_loss.item()
        total_combo += combo_loss.item()
    
        '''Save final prediction and model checkpoint at last epoch'''
        pred_classes = torch.argmax(updated_state, dim=1).float() / (nca_model.final_conv.out_channels - 1)
        out_path = os.path.join(output_dir, f"epoch{epoch+1}_pred_seq.png")
        if epoch == epochs - 1:
            save_image(pred_classes, out_path)
            torch.save(nca_model.state_dict(), os.path.join(output_dir, f"nca_epoch{epoch + 1}.pth"))
    
        print(f"Epoch {epoch + 1} - CE: {ce_loss.item():.4f}, Tversky + Dice: {combo_loss.item():.4f}")
    
        ce_loss_history.append(ce_loss.item())
        combo_loss_history.append(combo_loss.item())
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), ce_loss_history, label='Cross-Entropy Loss')
    plt.plot(range(1, epochs+1), combo_loss_history, label='Combo Loss (Dice + Tversky)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves for Cross-Entropy and Combo Loss')
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(loss_curve_path)
    plt.close()



def main():
    print("Starting Neural Cellular Automata training for urban growth prediction...\n")
    train_nca_model()
    print("\nFinished training and saved results successfully.")


'''Loading the trained model for evaluation'''
def validate_nca_model(model_path, file_paths, output_dir="/home/shovik.roy/vanaja_btep/btep_final_results"):
    print("Running validation using trained model...\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Load sample input to initialize the model
    sample_img, _, _ = load_tiff(file_paths[0])
    feature_extractor = FeatureExtractor(in_channels=sample_img.shape[1]).to(device)
    with torch.no_grad():
        dummy_feat = feature_extractor(sample_img.to(device))
    nca_model = NCAUpdate(num_features=dummy_feat.shape[1], num_hidden=[64], num_out=5).to(device)

    # Load model weights
    state_dict = torch.load(model_path, weights_only=True)
    nca_model.load_state_dict(state_dict)

    nca_model.eval()

    # Prepare image and mask sequences
    images_seq, masks_seq, targets_seq = [], [], []
    for t in range(len(file_paths)):
        img, mask, _ = load_tiff(file_paths[t])
        img, mask = img.to(device), mask.to(device).unsqueeze(1)
        features = feature_extractor(img)
        images_seq.append(features)
        masks_seq.append(mask)

    for t in range(1, len(file_paths)):
        tgt, _, _ = load_tiff(file_paths[t])
        tgt = tgt.to(device)
        tgt_class = (tgt * (nca_model.final_conv.out_channels - 1)).long().squeeze(1)
        targets_seq.append(tgt_class)

    # Temporal stack
    feature_seq = torch.stack(images_seq[:-1], dim=1)
    mask_seq = torch.stack(masks_seq[:-1], dim=1)

    # Initial state
    initial_state = images_seq[0]
    initial_class = (initial_state[:, 0:1] * (nca_model.final_conv.out_channels - 1)).long()
    urban_state = torch.zeros((1, nca_model.final_conv.out_channels, initial_state.shape[2], initial_state.shape[3]), device=device)
    urban_state.scatter_(1, initial_class, 1.0)

    # Predict
    with torch.no_grad():
        updated_state = nca_model(feature_seq, urban_state, mask_seq[:, -1])
        final_pred = torch.argmax(F.softmax(updated_state, dim=1), dim=1)
        final_target = targets_seq[-1]

    ''' ========== Evaluation Metrics ========== '''
    def compute_classwise_accuracy(preds, targets, num_classes):
        class_correct = torch.zeros(num_classes)
        class_total = torch.zeros(num_classes)
        for cls in range(num_classes):
            cls_mask = (targets == cls)
            class_correct[cls] = ((preds == cls) & cls_mask).sum().item()
            class_total[cls] = cls_mask.sum().item()
        return {cls: (class_correct[cls] / class_total[cls] if class_total[cls] > 0 else None)
                for cls in range(num_classes)}

    def compute_mean_iou(preds, targets, num_classes):
        jaccard = JaccardIndex(task='multiclass', num_classes=num_classes).to(device)
        return jaccard(preds.view(-1), targets.view(-1)).item()

    class_acc = compute_classwise_accuracy(final_pred.cpu(), final_target.cpu(), num_classes=5)
    mean_iou = compute_mean_iou(final_pred, final_target, num_classes=5)

    print("Validation Results:")
    for cls, acc in class_acc.items():
        print(f"Class {cls}: {'{:.4f}'.format(acc) if acc is not None else 'No data'}")
    print(f"Mean IoU: {mean_iou:.4f}")

    # Save prediction and target maps
    colored_pred = apply_colormap(final_pred)
    colored_target = apply_colormap(final_target)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(colored_target)
    plt.title("Target (Validation)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(colored_pred)
    plt.title("Prediction (Validation)")
    plt.axis('off')
    

    save_image(torch.tensor(colored_target.transpose(2, 0, 1)).unsqueeze(0).float(), os.path.join(output_dir, "val_target.png"))
    save_image(torch.tensor(colored_pred.transpose(2, 0, 1)).unsqueeze(0).float(), os.path.join(output_dir, "val_predicted.png"))
    plt.close()

    # Optional confusion matrix
    cm = confusion_matrix(final_target.view(-1).cpu().numpy(), final_pred.view(-1).cpu().numpy(), labels=list(range(5)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
    plt.title("Confusion Matrix (Validation)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(output_dir, "val_confusion_matrix.png"))
    plt.close()


if __name__ == "__main__":
    main()

    # Validation image sequence paths (can be different from training)
    val_paths = [
        "/home/shovik.roy/vanaja_btep/1_grayscaleout.tif",
        "/home/shovik.roy/vanaja_btep/grayscaleout_Delhi.tif",
        "/home/shovik.roy/vanaja_btep/grayscaleout_Delhi_2023.tif",
        "/home/shovik.roy/vanaja_btep/grayscaleout_Delhi_2024.tif"
    ]
    model_file = "/home/shovik.roy/vanaja_btep/btep_final_results/nca_epoch700.pth"
    validate_nca_model(model_file, val_paths)

