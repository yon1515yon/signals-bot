import torch
import torch.nn as nn


class LSTMTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, n_head=4, dropout=0.2, output_size=4):
        super(LSTMTransformerModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout
        )
        lstm_output_size = hidden_size * 2
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=lstm_output_size, nhead=n_head, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(lstm_output_size, output_size)

    def forward(self, src):
        lstm_out, _ = self.lstm(src)
        transformer_out = self.transformer_encoder(lstm_out)
        last_time_step_out = transformer_out[:, -1, :]
        dropped_out = self.dropout(last_time_step_out)
        predictions = self.linear(dropped_out)
        return predictions


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weights = torch.tensor(weights, dtype=torch.float32) if weights is not None else None

    def forward(self, y_pred, y_true):
        if self.weights is not None and self.weights.device != y_pred.device:
            self.weights = self.weights.to(y_pred.device)
        mse_loss = (y_pred - y_true) ** 2
        focal_factor = torch.abs(y_pred - y_true) ** self.gamma
        loss = focal_factor * mse_loss
        if self.weights is not None:
            loss = loss * self.weights
        return torch.mean(loss)


class DrawdownLSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h_0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).requires_grad_()
        c_0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).requires_grad_()
        lstm_out, _ = self.lstm(input_seq, (h_0.detach(), c_0.detach()))
        last_time_step_out = lstm_out[:, -1, :]
        prediction = self.linear(last_time_step_out)
        return prediction
