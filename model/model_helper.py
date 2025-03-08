import importlib
import torch.nn as nn
import numpy as np
import cv2
import os

def visualize_feature_maps(outputs, img_path, name, save_path, index):

    features = outputs[index].squeeze(0)
    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.5 + img * 0.5)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(save_path + name, superimposed_img)

class ModelBuilder(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder, self).__init__()
        self._sync_bn = net_cfg["sync_bn"]
        self._num_classes = net_cfg["num_classes"]

        self.encoder_r = self._build_encoder(net_cfg["encoder"])
        self.encoder_d = self._build_encoder(net_cfg["encoder"])

        self.decoder = self._build_decoder(net_cfg["decoder"])
        self.decoder_d = self._build_decoder(net_cfg['cdecoder'])#depth解码器
        self.decoder_r = self._build_decoder(net_cfg['cdecoder'])  # rgb解码器

    def _build_encoder(self, enc_cfg):
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"])
        return encoder

    def _build_decoder(self, dec_cfg):
        dec_cfg["kwargs"].update(
            {
                "num_classes": self._num_classes,
            }
        )
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    def forward(self, x, y, name):
        feat = self.encoder_r(x)
        feat_d = self.encoder_d(y)
        outs = self.decoder(feat, feat_d)

        #交换特征
        feat[1], feat_d[1] = feat_d[1], feat[1]
        outr = self.decoder_r(feat)
        outd = self.decoder_d(feat_d)
        outs['outr'] = outr['pred']
        outs['outd'] = outd['pred']
        return outs
