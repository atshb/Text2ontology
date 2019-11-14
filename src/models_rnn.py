import torch
import torch.nn as nn


'''
'''
class TwinRnnClassifier(nn.Module):

    def __init__(self, f_size, h_size=300, y_size=4, num_cell=2, drop_rate=0.2):
        super(TwinRnnClassifier, self).__init__()
        # あとで使うパラメーター
        self.h_size   = h_size
        self.num_cell = num_cell
        # BiLSTM
        rnn_drop = 0 if num_cell == 1 else drop_rate # ドロップアウトは最終層以外に適応されるので一層の場合は必要なし。
        self.lstm_a = nn.LSTM(f_size, h_size, num_layers=num_cell, batch_first=True, dropout=rnn_drop, bidirectional=True)
        self.lstm_b = nn.LSTM(f_size, h_size, num_layers=num_cell, batch_first=True, dropout=rnn_drop, bidirectional=True)
        # MLP
        self.classifier = nn.Sequential(
            # bidirectional かつ 二つの入力なので hidden size は4倍
            nn.Linear(4*h_size, 4*h_size), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4*h_size, 4*h_size), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4*h_size,   y_size),
        )

    def forward(self, x_a, x_b):
        _, (h_a, _) = self.lstm_a(x_a)
        _, (h_b, _) = self.lstm_b(x_b)
        h = self.concat(h_a, h_b)
        y = self.classifier(h)
        return y

    def concat(self, a, b):
        _, batch_size, _ = a.size()
        # RNNのレイヤー数が１でない場合は最終層の出力だけ利用
        if self.num_cell != 1:
            a = a.view(self.num_cell, 2, batch_size, self.h_size)[-1]
            b = b.view(self.num_cell, 2, batch_size, self.h_size)[-1]
        # 双方向RNNは出力が２つなので連結
        a = torch.cat([e for e in a], dim=1)
        b = torch.cat([e for e in b], dim=1)
        # 二つの出力を連結
        return torch.cat((a, b), dim=1)
'''
CNNで分類
'''
class CnnClassifier(nn.Module):

    def __init__(self, f_size, h_size=300, y_size=4, num_cell=2, drop_rate=0.2):
        super(CnnClassifier, self).__init__()
        # あとで使うパラメーター
        self.h_size   = h_size
        self.num_cell = num_cell
        # CNN
        rnn_drop = 0 if num_cell == 1 else drop_rate # ドロップアウトは最終層以外に適応されるので一層の場合は必要なし。
        self.conv1 = nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size=(3, 20), stride=1),
            nn.ReLU(),
            nn.Conv2d(10,  1, kernel_size=(3, 20), stride=1),
            nn.AdaptiveAvgPool2d((6, 50)),
            nn.Flatten(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d( 1, 10, kernel_size=(3, 20), stride=1),
            nn.ReLU(),
            nn.Conv2d(10,  1, kernel_size=(3, 20), stride=1),
            nn.AdaptiveAvgPool2d((6, 50)),
            nn.Flatten(),
        )

        # MLP
        self.classifier = nn.Sequential(
            # bidirectional かつ 二つの入力なので hidden size は4倍
            nn.Linear(2*h_size, 2*h_size), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(2*h_size, 2*h_size), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(2*h_size,   y_size),
        )

    def forward(self, x_a, x_b):
        h_a = self.conv1(x_a.unsqueeze(1))
        h_b = self.conv2(x_b.unsqueeze(1))
        h = torch.cat((h_a, h_b), dim=1)
        y = self.classifier(h)
        return y
