import torch
import torch.nn as nn


'''
'''
class RnnBlock(nn.Module):

    def __init__(self, f_size, h_size, num_cell, drop_rate):
        super(RnnBlock, self).__init__()
        #
        self.h_size   = h_size
        self.num_cell = num_cell
        # ドロップアウトは最終層以外に適応されるので一層の場合は必要なし。
        drop_rate = 0 if num_cell == 1 else drop_rate
        self.lstm = nn.LSTM(f_size, h_size, num_layers=num_cell,
                            batch_first=True, dropout=drop_rate,
                            bidirectional=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        # RNNのレイヤー数が１でない場合は最終層の出力だけ利用
        if self.num_cell != 1: h = h.view(self.num_cell, 2, -1, self.h_size)[-1]
        h = torch.cat([e for e in h], dim=1)
        return h


'''
'''
class CnnBlock(nn.Module):

    def __init__(self):
        super(CnnBlock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d( 1,  8, kernel_size=16, stride=(1, 4), padding=8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
            nn.Conv2d( 8, 16, kernel_size= 8,                padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
            nn.Conv2d(16, 32, kernel_size= 4,                padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size= 2,                padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


'''
'''
class RnnClassifier(nn.Module):

    def __init__(self, f_size, h_size=300, y_size=4, num_cell=2, drop_rate=0.2):
        super(RnnClassifier, self).__init__()
        # BiLSTM
        self.rnn_a = RnnBlock(f_size, h_size, num_cell, drop_rate)
        self.rnn_b = RnnBlock(f_size, h_size, num_cell, drop_rate)
        # MLP
        self.classifier = nn.Sequential(
            # bidirectional かつ 二つの入力なので hidden size は4倍
            nn.Linear(4*h_size, h_size), nn.ReLU(inplace=True), nn.Dropout(drop_rate),
            nn.Linear(  h_size, h_size), nn.ReLU(inplace=True), nn.Dropout(drop_rate),
            nn.Linear(  h_size, y_size),
        )

    def forward(self, x_a, x_b):
        h_a = self.rnn_a(x_a)
        h_b = self.rnn_b(x_b)
        h = torch.cat((h_a, h_b), dim=1)
        y = self.classifier(h)
        return y


'''
'''
class CnnClassifier(nn.Module):

    def __init__(self, f_size, h_size=300, y_size=4, drop_rate=0.2):
        super(CnnClassifier, self).__init__()
        # 
        self.cnn_a = CnnBlock()
        self.cnn_b = CnnBlock()
        # MLP
        self.classifier = nn.Sequential(
            nn.Linear(2*64*6*6, h_size), nn.ReLU(inplace=True), nn.Dropout(drop_rate),
            nn.Linear(  h_size, h_size), nn.ReLU(inplace=True), nn.Dropout(drop_rate),
            nn.Linear(  h_size, y_size),
        )

    def forward(self, x_a, x_b):
        h_a = self.cnn_a(x_a)
        h_b = self.cnn_b(x_b)
        h = torch.cat((h_a, h_b), dim=1)
        y = self.classifier(h)
        return y


'''
'''
class ParallelClassifier(nn.Module):

    def __init__(self, f_size, h_size=300, y_size=4, num_cell=2, drop_rate=0.2):
        super(ParallelClassifier, self).__init__()
        #
        self.cnn_a = CnnBlock()
        self.cnn_b = CnnBlock()
        #
        self.rnn_a = RnnBlock(f_size, h_size, num_cell, drop_rate)
        self.rnn_b = RnnBlock(f_size, h_size, num_cell, drop_rate)
        # MLP
        in_size = 4*h_size + 2*64*6*6
        self.classifier = nn.Sequential(
            nn.Linear(in_size, h_size), nn.ReLU(inplace=True), nn.Dropout(drop_rate),
            nn.Linear( h_size, h_size), nn.ReLU(inplace=True), nn.Dropout(drop_rate),
            nn.Linear( h_size, y_size),
        )

    def forward(self, x_a, x_b):
        #
        rnn_a = self.rnn_a(x_a)
        rnn_b = self.rnn_b(x_b)
        #
        cnn_a = self.cnn_a(x_a)
        cnn_b = self.cnn_b(x_b)
        #
        h = torch.cat((rnn_a, rnn_b, cnn_a, cnn_b), dim=1)
        y = self.classifier(h)
        return y



'''
lstm(バイじゃない一方向)
'''
class AccelerateClassifier(nn.Module):

    def __init__(self, f_size, h_size=300, y_size=4, num_cell=2, drop_rate=0.2):
        super(AccelerateClassifier, self).__init__()
        # あとで使うパラメーター
        self.h_size   = h_size
        self.num_cell = num_cell
        # BiLSTM
        rnn_drop = 0 if num_cell == 1 else drop_rate # ドロップアウトは最終層以外に適応されるので一層の場合は必要なし。
        self.lstm_a = nn.LSTM(f_size, h_size, num_layers=num_cell, batch_first=True, dropout=rnn_drop)
        self.lstm_b = nn.LSTM(f_size, h_size, num_layers=num_cell, batch_first=True, dropout=rnn_drop)
        # MLP
        self.classifier = nn.Sequential(
            nn.Linear(2*h_size, 2*h_size), nn.ReLU(inplace=True), nn.Dropout(drop_rate),
            nn.Linear(2*h_size, 2*h_size), nn.ReLU(inplace=True), nn.Dropout(drop_rate),
            nn.Linear(2*h_size,   y_size),
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
            a = a.view(self.num_cell, 1, batch_size, self.h_size)[-1]
            b = b.view(self.num_cell, 1, batch_size, self.h_size)[-1]
        # 双方向RNNは出力が２つなので連結
        a = torch.cat([e for e in a], dim=1)
        b = torch.cat([e for e in b], dim=1)
        # 二つの出力を連結
        return torch.cat((a, b), dim=1)
