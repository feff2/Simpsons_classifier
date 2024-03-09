from imports import *

def predict(model, test_loader):
    with torch.no_grad():
        logits = []

        for inputs in test_loader:
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)

    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs


def make_f1_score(model,dataset):

    n=len(dataset)
    idxs = list(map(int, np.random.uniform(0, n, n)))
    imgs = [dataset[id][0].unsqueeze(0) for id in idxs]

    probs_ims = predict(model, imgs)
    y_pred = np.argmax(probs_ims, -1)

    actual_labels = [dataset[id][1] for id in idxs]
    preds_class = list(y_pred)
    
    actual_labels = [dataset[id][1] for id in idxs]
    preds_class = list(y_pred)

    f1_micro = round(f1_score(actual_labels, preds_class, average='micro'), 3)

    return f1_micro