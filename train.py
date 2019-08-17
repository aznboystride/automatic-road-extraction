def train(network, optimizer, criterion, train_loader, valid_loader,
          epoch=1, weights=None, loss=100.0, save_name=None
         ):
    global lr
    min_loss = loss
    bad_epochs = 0
    for ep in range(epoch, epochs+1):
        running_loss = 0.0
        running_acc = 0.0
        print("epoch ({}/{})\t{}".format(ep,epochs,
                                     datetime.datetime.now(
                                         pytz.timezone("US/Pacific")).strftime("%m-%d-%Y - %I:%M %p"))
                                     )
        print("epoch ({}/{})\t{}".format(ep,epochs,
                                     datetime.datetime.now(
                                         pytz.timezone("US/Pacific")).strftime("%m-%d-%Y - %I:%M %p")),
                                     file=open(train_log, 'a+'))
        
        batch = 1
        for images, masks in train_loader:
            images = images.cuda()
            masks = masks.cuda()
            optimizer.zero_grad()
            predictions = network(images)
            loss = criterion(masks, predictions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += dice_coeff(predictions, masks)
            if batch % 30 == 0:
                print("batch ({}/{})\tloss = {}\tacc={}".format(batch, len(train_loader),
                                                               running_loss / batch, running_acc / batch))
            batch += 1
            del images, masks, predictions, loss
        
        avg_loss = running_loss/len(train_loader)
        avg_acc = running_acc/len(train_loader)
        if avg_loss < min_loss:
            log(avg_loss, avg_acc, True)
            min_loss = avg_loss
            if save_name != None: save_model(ep, network, optimizer, avg_loss, save_name)
            bad_epochs = 0
        else:
            log(avg_loss, avg_acc)
            bad_epochs += 1
            if bad_epochs == 3:
                if lr < 5e-7:
                    print("Learning rate < 5e-7... stopping early")
                    print("Learning rate < 5e-7... stopping early", file=open("log/train.log"))
                    break
                
                checkpoint = load_model(save_name)
                
                network.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                lr /= 5.0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                print("Learning rate updated from {} -> {}".format(lr*5, lr))
                print("Learning rate updated from {} -> {}".format(lr*5, lr), file=open("log/train.log", 'a+'))
                bad_epochs = 0
        validate(network, criterion, valid_loader)
