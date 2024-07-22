import ast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import string

# Parameters
vocab_size = 128  # Assuming ASCII characters
embed_dim = 64

num_classes = 6
batch_size = 32
num_samples = 1000

# Generate synthetic data
def generate_synthetic_data(num_samples):
    input_data = []
    labels = []
    max_len=200

    for _ in range(num_samples):
        seq_len = random.randint(1, max_len)
        input_seq = ''.join(random.choices(string.ascii_lowercase, k=seq_len))
        output_matrix = np.random.randint(2, size=(seq_len, num_classes))
        
        input_data.append(input_seq)
        labels.append(output_matrix)
    
    return input_data, labels, max_len



def load_data_from_file(file_name):
    max_len=0
    input_data = []
    labels = []
    with open(file_name, mode='r') as file:
        # Read all lines from the file
        lines = file.readlines()
    
        # Iterate over the lines in the file
        for line in lines:
            # Split each line into columns using the comma delimiter
            columns = line.strip().split('\t')
            if len(columns[1]) > max_len:
                max_len = len(columns[1])
        
            input_data.append(columns[1])
            matrix = ast.literal_eval(columns[3])
            labels.append(matrix)

    print("max_sequence_length=", max_len)
    return input_data, labels, max_len

# PyTorch Dataset
class CharDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
    
class CharacterModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(CharacterModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
    

# option 1: randomly generated dataset
#input_data, labels, max_sequence_length = generate_synthetic_data(num_samples)

# option 2: read dataset from file
input_file = "/Users/davidchen/dev/PeSTo/tmp/4a4j.tsv"
input_file = "/Users/davidchen/dev/PeSTo/tmp/small_AA_contacts_rr5A_64nn_8192.tsv"

input_data, labels, max_sequence_length = load_data_from_file(input_file)

# Preprocess data
char_to_index = {char: idx for idx, char in enumerate(string.ascii_uppercase)}
input_data_numeric = [[char_to_index[char] for char in seq] for seq in input_data]
input_data_numeric_padded = np.array([np.pad(seq, (0, max_sequence_length - len(seq)), 'constant') for seq in input_data_numeric])

# Pad labels to match max_sequence_length
padded_labels = np.array([np.pad(label, ((0, max_sequence_length - len(label)), (0, 0)), mode='constant', constant_values=0) for label in labels])
dataset = CharDataset(input_data_numeric_padded, padded_labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CharacterModel(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=64, output_dim=num_classes)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}')


# Test prediction
test_seq="AQTINLQLEGMDCTSCASSIERAIAKVPGVQSCQVNFALEQAVVSYHGETTPQILTDAVERAGYHARVL"
test_input = np.array([char_to_index[char] for char in test_seq])
test_input_padded = np.pad(test_input, (0, max_sequence_length - len(test_input)), 'constant')
test_input_tensor = torch.tensor(test_input_padded, dtype=torch.long).unsqueeze(0)

model.eval()
with torch.no_grad():
    predictions = model(test_input_tensor)
    print(test_seq)
    torch.set_printoptions(threshold=100000, edgeitems=3)
    print(predictions.size())

