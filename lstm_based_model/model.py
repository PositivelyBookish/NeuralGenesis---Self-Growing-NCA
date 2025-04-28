import torch
import torch.nn as nn

class NCA_LSTM(nn.Module):
    def __init__(self, n_channels=16, hidden_channels=128, lstm_hidden_size=64, filter="sobel", fire_rate=0.5, device=None):
        super(NCA_LSTM, self).__init__()

        self.fire_rate = fire_rate
        self.n_channels = n_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.device = device or torch.device("cpu")

        # Define Perception Filters
        if filter == "sobel":
            filter_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            scalar = 8.0
        elif filter == "scharr":
            filter_ = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
            scalar = 16.0
        elif filter == "gaussian":
            filter_ = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
            scalar = 16.0
        elif filter == "laplacian":
            filter_ = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            scalar = 8.0
        elif filter == "mean":
            filter_ = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            scalar = 9.0
        else:
            raise ValueError(f"Unknown filter: {filter}")

        filter_x = filter_ / scalar
        filter_y = filter_.t() / scalar

        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        kernel = torch.stack([identity, filter_x, filter_y], dim=0)
        kernel = kernel.repeat((n_channels, 1, 1))[:, None, ...]
        self.kernel = kernel.to(self.device)

        # LSTM for Temporal Updates
        self.lstm = nn.LSTM(input_size=3 * n_channels, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)

        # Conv Layer for Output
        self.output_layer = nn.Conv2d(lstm_hidden_size, n_channels, kernel_size=1, bias=False)

        # Initialize Conv layer weights to zero
        with torch.no_grad():
            self.output_layer.weight.zero_()

        self.to(self.device)

    def perceive(self, x):
        """
        Perceive information from neighboring cells.
        """
        return nn.functional.conv2d(x, self.kernel, padding=1, groups=self.n_channels)

    def update(self, x, hidden):
    	pre_life_mask = self.get_alive(x)

    # Perceive the local neighborhood
    	y = self.perceive(x)

    # Flatten for LSTM input
    	batch_size, channels, height, width = y.shape
    	y = y.view(batch_size, height * width, channels)  # Correct shape (batch, sequence_length, channels)

    # Initialize hidden state correctly
    	if hidden is None:
        	hidden = (
            	torch.zeros(1, batch_size, self.lstm_hidden_size, device=self.device),
            	torch.zeros(1, batch_size, self.lstm_hidden_size, device=self.device)
        	)

    # Pass through LSTM
    	y, hidden = self.lstm(y, hidden)

    # Reshape back to grid
    	y = y.view(batch_size, self.lstm_hidden_size, height, width)  # Ensure correct shape

    # Compute Updates
    	dx = self.output_layer(y)

    # Stochastic update
    	mask = (torch.rand(x[:, :1, :, :].shape, device=x.device) <= self.fire_rate).float()
    	dx = dx * mask

    # Apply updates
    	new_x = x + dx

    # Ensure only "alive" cells are updated
    	post_life_mask = self.get_alive(new_x)
    	life_mask = (pre_life_mask & post_life_mask).float()

    	return new_x * life_mask, hidden


    @staticmethod
    def get_alive(x):
        """
        Check which cells are alive based on the alpha channel.
        """
        return (nn.functional.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1)

    def forward(self, x, hidden):
        """
        Forward pass.
        """
        return self.update(x, hidden)