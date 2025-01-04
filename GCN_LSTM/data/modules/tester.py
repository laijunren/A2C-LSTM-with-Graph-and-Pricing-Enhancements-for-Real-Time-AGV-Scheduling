import torch

def test(model, feature_tensor, edge_index):
    model.eval()
    with torch.no_grad():
        output = model(feature_tensor, edge_index)
    return output
