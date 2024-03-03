import torch.nn as nn

from Utils.Gaze_estimator.modules import resnet50

class gaze_network(nn.Module):
    def __init__(self, use_face=False, num_glimpses=1):
        super(gaze_network, self).__init__()
        self.gaze_network = resnet50(pretrained=True)

        self.gaze_fc = nn.Sequential(
            nn.Linear(2048, 2),
        )

    def forward(self, x):

        feature = self.gaze_network(x) #[1,3,img_size,img_size] -> torch.Size([1, 2048, 1, 1])
        feature = feature.view(feature.size(0), -1) #torch.Size([1, 2048, 1, 1]) -> torch.Size([1, 2048])
        gaze = self.gaze_fc(feature) #torch.Size([1, 2])

        return gaze
