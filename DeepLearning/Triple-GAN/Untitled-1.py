# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np

# %%
# Configuration pour Mac M3
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Utilisation du GPU Metal (M3)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Utilisation du GPU CUDA")
else:
    DEVICE = torch.device("cpu")
    print("Utilisation du CPU")

# Hyperparamètres
BATCH_SIZE = 64
Z_DIM = 100
HIDDEN_DIM = 256
NUM_EPOCHS = 190
NUM_LABELS = 100
ALPHA = 0.3
ALPHA_P = 0.1

# Augmentation des données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# %% [markdown]
# # Définition des Modèles

# %% [markdown]
# ## Générateur (**G**)

# %%
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, HIDDEN_DIM)
        
        self.model = nn.Sequential(
            # Couche de projection initiale
            nn.Linear(Z_DIM + HIDDEN_DIM, HIDDEN_DIM * 4),
            nn.BatchNorm1d(HIDDEN_DIM * 4),
            nn.ReLU(True),
            
            nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM * 8),
            nn.BatchNorm1d(HIDDEN_DIM * 8),
            nn.ReLU(True),
            
            nn.Linear(HIDDEN_DIM * 8, 784),
            nn.Tanh()
        )
        
    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)
        z = torch.cat([z, label_embedding], dim=1)
        img = self.model(z)
        return img.view(-1, 1, 28, 28)


# %% [markdown]
# ## Discriminateur (**D**)

# %%

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, HIDDEN_DIM)
        
        self.model = nn.Sequential(
            nn.Linear(784 + HIDDEN_DIM, HIDDEN_DIM * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(HIDDEN_DIM * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, labels):
        x = x.view(-1, 784)
        label_embedding = self.label_emb(labels)
        x = torch.cat([x, label_embedding], dim=1)
        return self.model(x)

# %% [markdown]
# ## Classificateur (**C**)

# %%
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(HIDDEN_DIM, 10)
        )
        
    def forward(self, x):
        return self.model(x)

# %% [markdown]
# # Entrainement du modéle

# %%
def train_epoch(generator, discriminator, classifier, g_optimizer, d_optimizer, 
                c_optimizer, labeled_loader, unlabeled_loader, epoch):
    generator.train()
    discriminator.train()
    classifier.train()
    
    total_d_loss = 0
    total_g_loss = 0
    total_c_loss = 0
    
    criterion = nn.BCELoss()
    criterion_cls = nn.CrossEntropyLoss()
    
    for (x_l, y_l), (x_u, _) in zip(labeled_loader, unlabeled_loader):
        batch_size = x_l.size(0)
        
        # Transfert vers le device
        x_l, y_l = x_l.to(DEVICE), y_l.to(DEVICE)
        x_u = x_u.to(DEVICE)
        
        # Entraînement du Discriminateur
        d_optimizer.zero_grad()
        
        z = torch.randn(batch_size, Z_DIM).to(DEVICE)
        y_g = torch.randint(0, 10, (batch_size,)).to(DEVICE)
        
        x_g = generator(z, y_g)
        d_real = discriminator(x_l, y_l)
        d_fake = discriminator(x_g.detach(), y_g)
        
        d_loss_real = criterion(d_real, torch.ones_like(d_real))
        d_loss_fake = criterion(d_fake, torch.zeros_like(d_fake))
        d_loss = d_loss_real + d_loss_fake
        
        d_loss.backward()
        d_optimizer.step()
        
        # Entraînement du Générateur
        g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, Z_DIM).to(DEVICE)
        y_g = torch.randint(0, 10, (batch_size,)).to(DEVICE)
        
        x_g = generator(z, y_g)
        d_fake = discriminator(x_g, y_g)
        
        g_loss = criterion(d_fake, torch.ones_like(d_fake))
        
        g_loss.backward()
        g_optimizer.step()
        
        # Entraînement du Classificateur
        c_optimizer.zero_grad()
        
        c_real = classifier(x_l)
        c_loss = criterion_cls(c_real, y_l)
        
        if epoch >= 50:
            c_fake = classifier(x_g.detach())
            c_loss += ALPHA_P * criterion_cls(c_fake, y_g)
        
        c_loss.backward()
        c_optimizer.step()
        
        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()
        total_c_loss += c_loss.item()
    
    return total_d_loss, total_g_loss, total_c_loss

def evaluate(classifier, test_loader):
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = classifier(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return 100. * correct / total

def visualize_results(generator, classifier, n_samples=10):
    generator.eval()
    classifier.eval()
    
    with torch.no_grad():
        plt.figure(figsize=(15, 15))
        
        for label in range(10):
            z = torch.randn(n_samples, Z_DIM).to(DEVICE)
            labels = torch.full((n_samples,), label, dtype=torch.long).to(DEVICE)
            
            fake_images = generator(z, labels)
            fake_images = (fake_images + 1) / 2
            pred_labels = classifier(fake_images).argmax(dim=1)
            
            for i in range(n_samples):
                plt.subplot(10, n_samples, label * n_samples + i + 1)
                img = fake_images[i].cpu().squeeze().numpy()
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                
                if i == 0:
                    color = 'green' if label == pred_labels[i].item() else 'red'
                    plt.title(f'{label}', color=color, pad=2)
        
        plt.tight_layout()
        plt.show()
        plt.close()
# Fonction pour tracer les métriques
def plot_metrics(d_losses, g_losses, c_losses, accuracies):
    epochs = range(1, len(d_losses) + 1)

    plt.figure(figsize=(12, 6))

    # Pertes
    plt.subplot(1, 2, 1)
    plt.plot(epochs, d_losses, label='Discriminator Loss')
    plt.plot(epochs, g_losses, label='Generator Loss')
    plt.plot(epochs, c_losses, label='Classifier Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses During Training')
    plt.legend()

    # Précision
    plt.subplot(1, 2, 2)
    plt.plot(range(10, len(accuracies) * 10 + 1, 10), accuracies, label='Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Classifier Accuracy During Training')
    plt.legend()

    plt.tight_layout()
    plt.show()


# %%
def train():
    # Chargement des données
    mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = MNIST(root='./data', train=False, transform=transform)

    # Séparation des données
    indices = torch.randperm(len(mnist_train))
    labeled_indices = indices[:NUM_LABELS]
    unlabeled_indices = indices[NUM_LABELS:]

    labeled_dataset = Subset(mnist_train, labeled_indices)
    unlabeled_dataset = Subset(mnist_train, unlabeled_indices)

    # Dataloaders
    labeled_loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE)

    # Initialisation des modèles
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    classifier = Classifier().to(DEVICE)

    # Optimiseurs
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    c_optimizer = optim.Adam(classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Entraînement
    best_accuracy = 0
        # Suivi des pertes et précision
    d_losses, g_losses, c_losses, accuracies = [], [], [], []
    try:
        for epoch in range(NUM_EPOCHS):
            d_loss, g_loss, c_loss = train_epoch(
                generator, discriminator, classifier,
                g_optimizer, d_optimizer, c_optimizer,
                labeled_loader, unlabeled_loader, epoch
            )
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            c_losses.append(c_loss)

            if (epoch + 1) % 10 == 0:
                accuracy = evaluate(classifier, test_loader)
                accuracies.append(accuracy)
                print(f'\nÉpoque [{epoch+1}/{NUM_EPOCHS}]')
                print(f'Pertes - D: {d_loss:.4f}, G: {g_loss:.4f}, C: {c_loss:.4f}')
                print(f'Précision: {accuracy:.2f}%')
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    print(f'Nouvelle meilleure précision: {best_accuracy:.2f}%')
                
                print("Génération d'exemples...")
                visualize_results(generator, classifier)
        plot_metrics(d_losses, g_losses, c_losses, accuracies)

    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur.")
    except Exception as e:
        print(f"\nErreur pendant l'entraînement: {str(e)}")
    finally:
        print("\nNettoyage...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    train()

# %%



