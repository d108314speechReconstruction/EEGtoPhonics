def save_checkpoint(check_point_dir, model,
                    optimizer, step, ema=None):
    checkpoint_path = os.path.join(check_point_dir, "model.ckpt-{}.pt".format(step))

    torch.save({"model": model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "global_step": step
                }, checkpoint_path)

    print("Saved checkpoint: {}".format(checkpoint_path))

    with open(os.path.join(checkpoint_dir, 'checkpoint'), 'w') as f:
        f.write("model.ckpt-{}.pt".format(step))

def attempt_to_restore(model, optimizer,checkpoint_dir, use_cuda):
    
    checkpoint_list = os.path.join(checkpoint_dir, 'checkpoint')

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, "{}".format(checkpoint_filename))
        
        print("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]
    
    else:
        global_step = 0

    return global_step
    
def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)

    return checkpoint
