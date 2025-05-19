import torchreid
import os
import torch
from glob import glob

class UAVTrackDataset(torchreid.data.ImageDataset):
    dataset_dir = 'reid_data'

    def __init__(self, root='reid_data', verbose=True, **kwargs):
        train_dir = os.path.join(root, 'train')

        self.train = self._process_dir(train_dir, relabel=True)
        self.query = self._process_dir('reid_data/query', relabel=False)
        self.gallery = self._process_dir('reid_data/gallery', relabel=False)


        super(UAVTrackDataset, self).__init__(self.train, self.query, self.gallery, **kwargs)


    def _process_dir(self, dir_path, relabel=False):
        pid_container = set()
        for folder in os.listdir(dir_path):
            pid_container.add(folder)
        pid2label = {pid: idx for idx, pid in enumerate(sorted(pid_container))}

        data = []
        for pid in os.listdir(dir_path):
            pid_path = os.path.join(dir_path, pid)
            if not os.path.isdir(pid_path):
                continue
            for img_path in glob(os.path.join(pid_path, "*.jpg")):
                if relabel:
                    label = pid2label[pid]
                else:
                    label = int(pid) if pid.isdigit() else hash(pid) % 10000  # fallback
                camid = 0  # dummy camid
                data.append((img_path, label, camid))
        return data



torchreid.data.register_image_dataset('uavtrack', UAVTrackDataset)
num_classes = len(os.listdir('reid_data/train'))

# === Build model (e.g. OSNet) ===
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,  # will be inferred
    loss='softmax',
    pretrained=True
)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
# === Setup engine ===
engine = torchreid.engine.ImageSoftmaxEngine(
    model=model,
    datamanager=torchreid.data.ImageDataManager(
        root='reid_data',
        sources=['uavtrack'],
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop']
    ),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.0003),
    scheduler=torch.optim.lr_scheduler.StepLR(torch.optim.Adam(model.parameters(), lr=0.0003), step_size=20, gamma=0.1),
)

# === Train ===
engine.run(
    save_dir='log/reid_model',
    max_epoch=25,
    eval_freq=5,
    print_freq=300,
    test_only=False
)
