import torch 
X = torch.tensor([1,2,3,4,5,6,7,8], dtype=torch.float32)
Y = torch.tensor([2,4,6,8,10,12,14,16],dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32,requires_grad=True)

def forward(x):
    return w * x

def loss(y,y_pred):
    return((y_pred - y)**2).mean()
X_test = 5.0

print(f"Prediction befor training :F({X_test})={forward(X_test).item():.3f}")


