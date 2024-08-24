import os

from PIL import Image
from torch.utils import data
from torchvision import transforms

from models.Common import RGB2YCrCb

to_tensor = transforms.Compose([transforms.ToTensor()])


class MSRS_data(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'Inf':
                self.inf_path = temp_path
            else:
                self.vis_path = temp_path

        self.name_list = os.listdir(self.inf_path)
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]

        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')
        vis_image = Image.open(os.path.join(self.vis_path, name))
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name

    def __len__(self):
        return len(self.name_list)
