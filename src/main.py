from dataset import get_dataloader, get_square_dataloader
from model import get_model
import torch
from engine import train_one_epoch, evaluate
from visualize import save_coco_predictions
import os


def validate_and_save(model, data_loader):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.train(False)
    model.to(device)

    images = []
    predictions = []
    targets = []
    
    for image_batch, target_batch in data_loader:
        imgs = list(img.to(device) for img in image_batch)
        with torch.no_grad():
            preds = model(imgs)

        for img, pred, target in zip(imgs, preds, target_batch):
            images.append(img.to("cpu"))
            predictions.append(pred)
            targets.append(target)
    save_dir = save_coco_predictions(images, predictions, targets)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

def main():
    channels = [0, 1, 2]
    batch_size = 8
    train_dataloader = get_square_dataloader(vids=[i for i in range(16) if i != 5], batch_size=batch_size, channels=channels, train=True, num_workers=6)
    test_dataloader = get_square_dataloader(vids=[16, 17, 18], batch_size=batch_size, channels=channels)
    val_dataloader = get_square_dataloader(vids=[5], batch_size=1)
    model = get_model(None, None)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = "cpu"
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.003, momentum=0.9, weight_decay=0.0005)

    model.to(device)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, test_dataloader, device=device)

    # images, target = next(iter(train_dataloader))
    # # print(model(img, target))
    # images = list(img.to(device) for img in images)
    # model.train(False)
    # model.to(device)
    # print(target)
    # print(model(images))
    validate_and_save(model, val_dataloader)

if __name__ == "__main__":
    main()
    # model = get_model(None, None)
    # val_dataloader = get_dataloader(vids=[18], batch_size=2)
    # validate_and_save(model, val_dataloader)