import torch

def train(model, optimizer, loss_fn, feature_tensor, edge_index, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(feature_tensor, edge_index)
        loss = loss_fn(output)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
