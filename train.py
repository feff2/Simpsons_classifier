from imports import *


def calculate_loss(model, loader, criterion):
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
    return loss


def train(dataloaders, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    accuracy = {"train": [], "val": []}
    loss_dict = {"train": [], "val": []}
    start_time = 0
    end_time = 0
    
    for epoch in range(num_epochs):
        for k, dataloader in dataloaders.items():
            start_time = time.time()
            epoch_correct = 0
            epoch_all = 0
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                if k == "train":
                    model.train()
                    optimizer.zero_grad()
                    outp = model(x_batch)
                else:
                    model.eval()
                    with torch.no_grad():
                        outp = model(x_batch)
                preds = outp.argmax(-1)
                correct = (preds.detach() == y_batch).sum(dim=0)
                all = y_batch.size(0)
                epoch_correct += correct.item()
                epoch_all += all
                if k == "train":
                    loss = criterion(outp, y_batch)
                    loss.backward()
                    optimizer.step()
            end_time = time.time()
            if k == "train":
                print(f"Epoch: {epoch+1}")
                loss_dict[k].append(calculate_loss(model,dataloaders[k], criterion))
                print(f"Train_loss: {loss_dict[k][-1]}")
            print(f"Loader: {k}. Accuracy: {epoch_correct/epoch_all}. Time_mins:{(end_time - start_time)/60} ")
            accuracy[k].append(epoch_correct/epoch_all)
        if k == 'val' in dataloaders:
            loss_dict[k].append(calculate_loss(model,dataloaders[k], criterion))
            print(f"Val_loss: {loss_dict[k][-1]}")
        scheduler.step(loss_dict["val"][-1])


    del x_batch
    del y_batch
    torch.cuda.empty_cache()

    return None