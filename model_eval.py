import torch
from ModelNet40Dataset import ModelNet40Dataset
from torch.utils.data import DataLoader

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.load("epoch_0_model.pt")

    root = '/home/zheruiz/datasets/modelnet40_normal_resampled/'
    batch_size = 1
    test_data = ModelNet40Dataset(root=root, split='test')
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        for n_batch, (src, target, R_gt, t_gt) in enumerate(test_loader):
            # mini batch
            src, target, R_gt, t_gt = src.to(device), target.to(device), R_gt.to(device), t_gt.to(device)
            t_init = torch.zeros(1, 3)
            src_keypts, target_vcp = model.test(src, target, R_gt, t_init)

            loss = deepVCP_loss(src_keypts, target_vcp, R_gt, t_gt, alpha=0.5)

            loss_test += [loss.item()]

if __name__ == "__main__":
    main()