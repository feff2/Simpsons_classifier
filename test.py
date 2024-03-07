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

    num_classes = len(list(set(actual_labels)))
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(num_classes):  
        fpr[i], tpr[i], _ = roc_curve(actual_labels, probs_ims[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return f1_micro