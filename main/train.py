#train 
import os
def train(args) :
    #prepare paths
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #set device
    iscuda = torch.cuda.is_available()
    if iscuda :
        device = torch.device("cuda")
        print("using GPU")
    else :
        device = torch.device("cpu")
        print("using CPU")    
        
    # initiate model
    model = Model()
    print(model)
    if(torch.cuda.device_count() > 1) :
        num_gpu = torch.cuda.device_count()  
        model = nn.DataParallel(model)
    else :
        num_gpu = torch.cuda.device_count()
    model.to(device)
    
    # defining optimizer, loss
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr).to(device)
    
    # dataloader & dataset
    mydataset = dataset()
    dataloadersTrain = torch.utils.data.DataLoader(mydataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    #summary writer
    writer = SummaryWriter(args.checkpoint_dir)
    #about ready to start !
    global_step = 0
    if args.resume != None:
        global_step = attempt_to_restore(model, optimizer, args.checkpoint_dir, bool(device == "cuda"))
    #start training
    for epochs in range(args.epochs):
        train_dataloader = dataloader()
        start = time.time()
        for x, label in train_dataloader:
            x = x.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(x)
            # 預留位置01
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        
        time_used = time.time() - start
        print("Step: {} --loss: {:.3f} --Time: {:.2f} seconds".format(global_step, loss, time_used))
        global_step += 1
        
        if global_step % args.checkpoint_step == 0:
            save_checkpoint(args.checkpoint_dir, model, optimizer, global_step)

        if global_step % args.summary_step == 0:
            writer.add_scalar('{}'.format(key), loss.get_item(), global_step)
#set args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float,default=0.001)
parser.add_argument('--epochs',type=int, default=500000)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--checkpoint_dir', default='/logdir')
parser.add_argument('--resume',default=None)
parser.add_argument('--checkpoint_step',type=int,default=1000)
parser.add_argument('--summary_step',type=int,default=100)
args = parser.parse_args()
train(args)
     
