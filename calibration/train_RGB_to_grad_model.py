import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class GradNet2D(nn.Module):
    def __init__(self):
        super(GradNet2D, self).__init__()
        # 5 input image channel, 32 output channels, 1x1 square convolution
        # kernel
        self.conv1 = nn.Conv2d(5, 32, 1)
        self.conv2 = nn.Conv2d(32, 32, 1)
        self.conv3 = nn.Conv2d(32, 32, 1)
        self.conv4 = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = self.conv4(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def fit(self, X_train, Y_train):
        batch_size = 2
        N = X_train.shape[0]

        idx = list(range(N))
        random.shuffle(idx)

        X_train = X_train[idx]
        Y_train = Y_train[idx]

        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        sum_loss = 0
        total = 0

        for epoch in range(20):
            sum_loss = 0
            total = 0
            for i in range(N // batch_size):
                # clear gradients accumulated on the parameters
                optim.zero_grad()

                # get an input (say we only care inputs sampled from N(0, I))
                # x = torch.randn(batch_size, 8, device=cuda0)  # this has to be on GPU too
                x = torch.tensor(
                    X_train[i * batch_size : (i + 1) * batch_size], dtype=torch.float32
                )
                y = torch.tensor(
                    Y_train[i * batch_size : (i + 1) * batch_size], dtype=torch.float32
                )

                # forward pass
                result = self.forward(x)  # CHANGED: fc => net

                # compute loss
                loss = F.mse_loss(result, y)

                # compute gradients
                loss.backward()

                # let the optimizer do its work; the parameters will be updated in this call
                optim.step()

                sum_loss += loss.item()
                total += 1
                # add some printing
                if (i + 1) % 1 == 0:
                    print(
                        "epoch {}\titeration {}\tloss {:.5f}".format(
                            epoch, i, sum_loss / total
                        )
                    )

    def predict(self, X_test):
        with torch.no_grad():
            # return self.forward(torch.tensor(X_test, dtype=torch.float32)).numpy()
            return self.forward(torch.tensor(X_test, dtype=torch.float32).cuda()).cpu().numpy()


class GradNet(nn.Module):
    def __init__(self):
        super(GradNet, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = x.tanh()
        x = self.fc2(x)
        x = x.tanh()
        x = self.fc3(x)
        x = x.tanh()
        return self.fc4(x)

    def fit(self, X_train, Y_train):
        batch_size = 256
        N = X_train.shape[0]

        idx = list(range(N))
        random.shuffle(idx)

        X_train = X_train[idx]
        Y_train = Y_train[idx]

        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        sum_loss = 0
        total = 0

        for epoch in range(10):
            sum_loss = 0
            total = 0
            for i in range(N // batch_size):
                # clear gradients accumulated on the parameters
                optim.zero_grad()

                # get an input (say we only care inputs sampled from N(0, I))
                # x = torch.randn(batch_size, 8, device=cuda0)  # this has to be on GPU too
                x = torch.tensor(
                    X_train[i * batch_size : (i + 1) * batch_size], dtype=torch.float32
                )
                y = torch.tensor(
                    Y_train[i * batch_size : (i + 1) * batch_size], dtype=torch.float32
                )

                # forward pass
                result = self.forward(x)  # CHANGED: fc => net

                # compute loss
                loss = F.mse_loss(result, y)

                # compute gradients
                loss.backward()

                # let the optimizer do its work; the parameters will be updated in this call
                optim.step()

                sum_loss += loss.item()
                total += 1
                # add some printing
                if (i + 1) % 200 == 0:
                    print(
                        "epoch {}\titeration {}\tloss {:.5f}".format(
                            epoch, i, sum_loss / total
                        )
                    )

    def predict(self, X_test):
        with torch.no_grad():
            return self.forward(torch.tensor(X_test, dtype=torch.float32)).numpy()
