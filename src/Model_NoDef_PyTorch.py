import torch
import torch.nn as nn
import torch.nn.functional as F

class DFNetPyTorch(nn.Module):
    def __init__(self, input_channels=1, classes=95):
        super(DFNetPyTorch, self).__init__()
        
        # Block1
        self.block1_conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=1, padding='same')
        self.block1_bn1 = nn.BatchNorm1d(32)
        self.block1_elu1 = nn.ELU(alpha=1.0)
        self.block1_conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=1, padding='same')
        self.block1_bn2 = nn.BatchNorm1d(32)
        self.block1_elu2 = nn.ELU(alpha=1.0)
        self.block1_pool = nn.MaxPool1d(kernel_size=8, stride=4, padding=2) # Keras 'same' padding might need adjustment for MaxPool1d
        self.block1_dropout = nn.Dropout(0.1)

        # Block2
        self.block2_conv1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=1, padding='same')
        self.block2_bn1 = nn.BatchNorm1d(64)
        self.block2_relu1 = nn.ReLU()
        self.block2_conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding='same')
        self.block2_bn2 = nn.BatchNorm1d(64)
        self.block2_relu2 = nn.ReLU()
        self.block2_pool = nn.MaxPool1d(kernel_size=8, stride=4, padding=2)
        self.block2_dropout = nn.Dropout(0.1)

        # Block3
        self.block3_conv1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding='same')
        self.block3_bn1 = nn.BatchNorm1d(128)
        self.block3_relu1 = nn.ReLU()
        self.block3_conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding='same')
        self.block3_bn2 = nn.BatchNorm1d(128)
        self.block3_relu2 = nn.ReLU()
        self.block3_pool = nn.MaxPool1d(kernel_size=8, stride=4, padding=2)
        self.block3_dropout = nn.Dropout(0.1)

        # Block4
        self.block4_conv1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8, stride=1, padding='same')
        self.block4_bn1 = nn.BatchNorm1d(256)
        self.block4_relu1 = nn.ReLU()
        self.block4_conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=8, stride=1, padding='same')
        self.block4_bn2 = nn.BatchNorm1d(256)
        self.block4_relu2 = nn.ReLU()
        self.block4_pool = nn.MaxPool1d(kernel_size=8, stride=4, padding=2)
        self.block4_dropout = nn.Dropout(0.1)

        self.flatten = nn.Flatten()
        
        # Dense layers
        # Calculate the flattened size dynamically or set it based on Keras model summary
        # For LENGTH = 5000, after 4 pooling layers (stride 4 each time): 5000 / (4^4) = 5000 / 256 approx 19.5 -> 20 (due to padding)
        # So, flattened_size = 256 * (ceil(5000 / (4*4*4*4))) , check Keras model output shape after flatten
        # From Keras model: pool_size=8, pool_stride_size=4. Length becomes L/4 after each pool.
        # L0 = 5000
        # L1 = ceil(5000/4) = 1250
        # L2 = ceil(1250/4) = 313
        # L3 = ceil(313/4) = 79
        # L4 = ceil(79/4) = 20
        # Flattened size = 256 * 20 = 5120
        self.fc1 = nn.Linear(256 * 20, 512) # Adjust 256 * 20 based on actual output shape after pooling and flatten
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1_relu = nn.ReLU()
        self.fc1_dropout = nn.Dropout(0.7)

        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc2_relu = nn.ReLU()
        self.fc2_dropout = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, classes)
        
        # Initialize weights for linear layers like Keras glorot_uniform
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        # PyTorch Conv1D expects (batch_size, channels, length)
        # Keras by default (batch_size, length, channels)
        # Ensure input x is permuted if necessary before passing to this model
        # x = x.permute(0, 2, 1) # If input is (batch, length, channels)

        # Block 1
        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = self.block1_elu1(x)
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        x = self.block1_elu2(x)
        x = self.block1_pool(x)
        x = self.block1_dropout(x)

        # Block 2
        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = self.block2_relu1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        x = self.block2_relu2(x)
        x = self.block2_pool(x)
        x = self.block2_dropout(x)

        # Block 3
        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_relu2(x)
        x = self.block3_pool(x)
        x = self.block3_dropout(x)

        # Block 4
        x = self.block4_conv1(x)
        x = self.block4_bn1(x)
        x = self.block4_relu1(x)
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = self.block4_relu2(x)
        x = self.block4_pool(x)
        x = self.block4_dropout(x)

        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.fc1_relu(x)
        x = self.fc1_dropout(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.fc2_relu(x)
        x = self.fc2_dropout(x)

        x = self.fc3(x)
        # Softmax is usually applied in the loss function (e.g., CrossEntropyLoss) in PyTorch
        # If you need raw logits, return x. If you need probabilities, use F.softmax(x, dim=1)
        return x

# Example usage (for testing the model structure):
if __name__ == '__main__':
    # NB_CLASSES = 95 # number of outputs = number of classes
    # LENGTH = 5000 # Packet sequence length
    # INPUT_SHAPE = (LENGTH,1) # Keras
    # PyTorch: (batch, channels, length)
    
    num_classes = 95
    seq_length = 5000
    input_channels = 1
    batch_size = 32

    model = DFNetPyTorch(input_channels=input_channels, classes=num_classes)
    
    # Create a dummy input tensor
    # Keras input: (batch_size, seq_length, input_channels)
    # PyTorch input for Conv1D: (batch_size, input_channels, seq_length)
    dummy_input_keras_format = torch.randn(batch_size, seq_length, input_channels)
    dummy_input_pytorch_format = dummy_input_keras_format.permute(0, 2, 1)
    
    print(f"Model: {model}")
    print(f"Dummy input shape (PyTorch format): {dummy_input_pytorch_format.shape}")
    
    output = model(dummy_input_pytorch_format)
    print(f"Output shape: {output.shape}") # Expected: (batch_size, num_classes)

    # Check parameter initialization for fc layers
    print("FC1 weight example:", model.fc1.weight[0, :5])
    print("FC1 bias example:", model.fc1.bias[:5])