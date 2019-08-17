def test_threshold(network, image_path, label_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0 
    image = np.expand_dims(image, axis=0)
    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).float()
    pred = network(image)
    pred = pred.squeeze(0).squeeze(0)
    pred1 = (pred > 0.5).cpu().float().numpy() * 255
    pred2 = (pred > 0.4).cpu().float().numpy() * 255
    pred3 = (pred > 0.3).cpu().float().numpy() * 255
    pred4 = (pred > 0.2).cpu().float().numpy() * 255
    pred5 = (pred > 0.1).cpu().float().numpy() * 255
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label = (label >= 128) * 255
    
    a1 = jaccard_acc(torch.from_numpy(pred1), torch.from_numpy(label))
    a2 = jaccard_acc(torch.from_numpy(pred2), torch.from_numpy(label))
    a3 = jaccard_acc(torch.from_numpy(pred3), torch.from_numpy(label))
    a4 = jaccard_acc(torch.from_numpy(pred4), torch.from_numpy(label))
    a5 = jaccard_acc(torch.from_numpy(pred5), torch.from_numpy(label))


def test(network, root_in, root_out, t):
    dir_list = os.listdir(root_in)
    if not os.path.isdir(root_out):
        os.mkdir(root_out)
    network.eval()
    a1 = a2 = a3 = a4 = a5 = 0.0
    with torch.no_grad():
        for count, filename in enumerate(dir_list,1):
            if filename.endswith('png'):
                continue
            path_in = os.path.join(root_in, filename)
            image = cv2.cvtColor(cv2.imread(path_in), cv2.COLOR_BGR2RGB)
            a,b,c,d,e = test_threshold(network, path_in, os.path.join(root_in, filename.split('_')[0] + "_mask.png"))
            a1 += a
            a2 += b
            a3 += c
            a4 += d
            a5 += e
            
            if count % 20 == 0:
                print("finished ({}/{})".format(count, len(dir_list)))
                
        l = len(dir_list)
        print("a1 = {}\ta2 = {}\ta3 = {}\ta4 = {}\ta5 = {}".format(a1/l,a2/l,a3/l,a4/l,a5/l))
