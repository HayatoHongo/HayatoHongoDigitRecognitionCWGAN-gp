import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

# Generator definition (same as in training)
class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        input_dim = noise_dim + num_classes
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_emb(labels)), dim=1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Load trained generator
@st.cache_resource
def load_generator(path="generator_epoch_50.pth", device="cpu"):
    model = Generator().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# Image generation
def generate_images(generator, digit, n_samples=5, noise_dim=100, device="cpu"):
    z = torch.randn(n_samples, noise_dim).to(device) * 0.3
    labels = torch.full((n_samples,), digit, dtype=torch.long).to(device)
    with torch.no_grad():
        imgs = generator(z, labels)
    return imgs

# Streamlit UI
st.title("Handwritten Digit Image Generator")
st.markdown("Generate synthetic MNIST-like images using your trained Conditional WGAN-GP model.")

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))
if st.button("Generate Images"):
    with st.spinner("Generating..."):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = load_generator(device=device)
        imgs = generate_images(generator, digit, n_samples=5, device=device)

        st.subheader(f"Generated images of digit {digit}")
        to_pil = ToPILImage()
        cols = st.columns(5)
        for i, img in enumerate(imgs):
            img = to_pil((img + 1) / 2)  # Convert from [-1,1] to [0,1]
            cols[i].image(img.resize((96, 96)), caption=f"Sample {i+1}")
