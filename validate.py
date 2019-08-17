def validate(network, criterion, loader):
    network.eval()
    print("Validation - ",)
    print("Validation - ", file=open(train_log, 'a+'))
    with torch.no_grad():
        running_loss = 0.0
        running_acc = 0.0
        for images, masks in loader:
            images = images.cuda()
            masks = masks.cuda()
            predictions = network(images)
            loss = criterion(masks, predictions)
            running_loss += loss.item()
            running_acc += dice_coeff(predictions, masks)

        avg_loss = running_loss/len(loader)
        avg_acc = running_acc/len(loader)
        log(avg_loss, avg_acc)

    network.train()
