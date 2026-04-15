import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging

def save_checkpoint(epoch, model, optimizer, checkpoint_dir, model_name):
    """Saves model and optimizer state."""
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_latest.pth")
    
    # Optional: Keep a historical record, but to save space we overwrite 'latest'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    logging.info(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_dir, model_name, device):
    """Loads checkpoint if it exists."""
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_latest.pth")
    
    if os.path.exists(checkpoint_path):
        logging.info(f"Found checkpoint at {checkpoint_path}. Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch
    else:
        logging.info("No checkpoint found. Starting fresh.")
        return 0

def train_model(model, train_loader, val_loader, num_epochs, device, checkpoint_dir, model_name, learning_rate=1e-3, no_resume=False):
    """Training loop for the model."""
    
    # 1. Define Loss and Optimizer (Update ONLY parameters that require grad - i.e. final layer)
    criterion = nn.CrossEntropyLoss()
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=learning_rate)
    
    # 2. Checkpoint Resume
    start_epoch = 0
    if not no_resume:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_dir, model_name, device)

    # 3. Main Loop
    model = model.to(device)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i+1) % 100 == 0:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        # Validate at the end of epoch
        val_acc = validate(model, val_loader, device)
        train_acc = 100 * correct / total
        
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] completed. Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save Checkpoint
        save_checkpoint(epoch, model, optimizer, checkpoint_dir, model_name)
        
    return model

def validate(model, val_loader, device):
    """Evaluate model on validation dataloader."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total
