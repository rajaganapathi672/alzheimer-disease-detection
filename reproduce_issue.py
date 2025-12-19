import torch
from models.models import NetWork

def test_model_dimensions():
    # Common MRI dimensions (often around 256^3 or 128^3)
    # Let's try 128x128x128 first as it's a reasonable cropped size
    input_size = (1, 1, 128, 128, 128) 
    x = torch.randn(input_size)
    age_id = torch.zeros(1, dtype=torch.long) # Dummy age
    
    print(f"Testing model with input size: {input_size}")
    
    model = NetWork(in_channel=1, feat_dim=1024, expansion=4)
    print("Model instantiated.")
    
    try:
        output = model(x, None)
        print("Forward pass successful.")
        print(f"Output shape: {output.shape}")
    except RuntimeError as e:
        print("Forward pass failed!")
        print(e)

if __name__ == "__main__":
    test_model_dimensions()
