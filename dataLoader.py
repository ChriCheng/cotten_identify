import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ThreadDataset(Dataset):
    def __init__(self, root_dir, dataset_type='gray', split='train', transform=None, seed=42):
        """
        丝线数据集加载器

        :param root_dir: 包含 'gray' 和 'colorful' 文件夹的总目录路径
        :param dataset_type: 数据集类型接口，可选 'gray' 或 'colorful'
        :param split: 数据划分，可选 'train' (6张) 或 'test' (4张)
        :param transform: PyTorch 的图像变换/预处理操作
        :param seed: 随机种子，确保每次运行训练集和测试集的划分不会重叠和改变
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.split = split
        self.transform = transform

        # 验证接口输入
        if self.dataset_type not in ['gray', 'colorful']:
            raise ValueError("dataset_type 必须是 'gray' 或 'colorful'")
        if self.split not in ['train', 'test']:
            raise ValueError("split 必须是 'train' 或 'test'")

        # 目标工作目录 (例如: /path/to/data/gray)
        self.data_dir = os.path.join(root_dir, dataset_type)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"找不到目录: {self.data_dir}")

        self.image_paths = []
        self.labels = []

        # 获取所有颜色类别（子文件夹名），并进行排序以保证标签索引稳定
        self.classes = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 遍历每个颜色的文件夹
        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)

            # 获取该颜色下的所有图片文件 (应该是 10 张)
            images = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            # 使用局部随机数生成器，确保每个类别的打乱是一致且可复现的
            # 这样保证在 split='train' 和 split='test' 时，随机序列一样，不会发生数据泄露
            rng = random.Random(seed)
            rng.shuffle(images)

            # 按照 6:4 划分
            if self.split == 'train':
                selected_images = images[:6]  # 取前 6 张作为训练集
            else:
                selected_images = images[6:10]  # 取后 4 张作为测试集

            # 存入列表
            for img_name in selected_images:
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 读取图片并转换为 RGB
        image = Image.open(img_path).convert('RGB')

        # 应用预处理操作（如 Resize, ToTensor 等）
        if self.transform:
            image = self.transform(image)

        return image, label


# =====================================================================
# 下面是如何使用该 DataLoader 的接口示例
# =====================================================================
def create_dataloaders(root_dir, dataset_type, batch_size=32):
    """
    创建一个快捷的函数来同时获取 train 和 test 的 DataLoader
    """
    # 定义基础的图像预处理 (您可根据实际需求增加数据增强)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像统一调整为 224x224
        transforms.ToTensor(),  # 转换为 Tensor，并归一化到 [0,1]
    ])

    # 1. 实例化数据集 (使用 dataset_type 接口区分 gray 还是 colorful)
    train_dataset = ThreadDataset(root_dir=root_dir, dataset_type=dataset_type, split='train', transform=transform)
    test_dataset = ThreadDataset(root_dir=root_dir, dataset_type=dataset_type, split='test', transform=transform)

    # 2. 包装成 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    DATA_ROOT = "./cotton_data"

    # 【需求1】：我想加载灰色系的丝线做训练
    print("正在加载 Gray 数据集...")
    gray_train_loader, gray_test_loader, gray_classes = create_dataloaders(
        root_dir=DATA_ROOT,
        dataset_type='gray',  # <-- 这里就是控制灰/彩色的接口
        batch_size=16
    )
    print(f"Gray 类别数量: {len(gray_classes)}")
    print(f"Gray 训练集总图片数: {len(gray_train_loader.dataset)} (每类6张)")
    print(f"Gray 测试集总图片数: {len(gray_test_loader.dataset)} (每类4张)\n")

    # 【需求2】：我想切换到彩色系的丝线做训练
    print("正在加载 Colorful 数据集...")
    colorful_train_loader, colorful_test_loader, colorful_classes = create_dataloaders(
        root_dir=DATA_ROOT,
        dataset_type='colorful',  # <-- 一键切换到彩色数据集
        batch_size=16
    )
    print(f"Colorful 类别数量: {len(colorful_classes)}")
    print(f"Colorful 训练集总图片数: {len(colorful_train_loader.dataset)}")
    print(f"Colorful 测试集总图片数: {len(colorful_test_loader.dataset)}")