import torch
import torch.nn as nn
import numpy as np

emojis = {
    'hat': '🎩',
    'rat': '🐀',
    'cat': '🐈',
    'flat': '🏢',
    'matt': '👨',
    'cap': '🧢',
    'son': '👦'
}

index_to_emoji = [value for key, value in emojis.items()]

index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']

char_encodings = np.eye(len(index_to_char))

encoding_size = len(char_encodings)

emoji_encoding = np.eye(len(emojis))

x_train = np.array([
    [[char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],
    [[char_encodings[4]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],
    [[char_encodings[5]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],
    [[char_encodings[6]], [char_encodings[7]], [char_encodings[2]], [char_encodings[3]]],
    [[char_encodings[8]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]]],
    [[char_encodings[5]], [char_encodings[2]], [char_encodings[9]], [char_encodings[0]]],
    [[char_encodings[10]], [char_encodings[11]], [char_encodings[12]], [char_encodings[0]]],
])

y_train = np.array([
    [emoji_encoding[0], emoji_encoding[0], emoji_encoding[0], emoji_encoding[0]],
    [emoji_encoding[1], emoji_encoding[1], emoji_encoding[1], emoji_encoding[1]],
    [emoji_encoding[2], emoji_encoding[2], emoji_encoding[2], emoji_encoding[2]],
    [emoji_encoding[3], emoji_encoding[3], emoji_encoding[3], emoji_encoding[3]],
    [emoji_encoding[4], emoji_encoding[4], emoji_encoding[4], emoji_encoding[4]],
    [emoji_encoding[5], emoji_encoding[5], emoji_encoding[5], emoji_encoding[5]],
    [emoji_encoding[6], emoji_encoding[6], emoji_encoding[6], emoji_encoding[6]]
])

x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)


class ManyToOneLSTMModel(nn.Module):

    def __init__(self, encoding_size):
        super(ManyToOneLSTMModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x,
             y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


model = ManyToOneLSTMModel(encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    for i in range(len(x_train)):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()


def generate_emoji(string):
    y = -1
    model.reset()
    for i in range(len(string)):
        char_index = index_to_char.index(string[i])
        y = model.f(torch.tensor([[char_encodings[char_index]]], dtype=torch.float))
    print(index_to_emoji[y.argmax(1)])


generate_emoji('rt')
generate_emoji('rats')
generate_emoji('cp')
generate_emoji('mt')

