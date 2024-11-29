from torchinfo import summary
from torch import nn

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.fc1 = nn.Linear(64 * 28 * 28, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		x = nn.functional.relu(self.conv1(x))
		x = nn.functional.relu(self.conv2(x))
		x = x.view(-1, 64 * 28 * 28)
		x = nn.functional.relu(self.fc1(x))
		x = self.fc2(x)
		return x

model = ConvNet()

print("torch print:\n\n", model, "\n")

print ("torchinfo summary:\n")
summary(model, (1, 28, 28))