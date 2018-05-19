# pytorch-feature-extractor
Template for extracting features for various image based networks in Pytorch. 


For input of size 224 x 224

| Model       | Inference Time(secs) (CPU) | Feature Dims             |
| ----------- | -------------------------- | ------------------------ |
| vgg16       | 0.538311243057251          | torch.Size([512, 7, 7])  |
| vgg19       | 0.7164433002471924         | torch.Size([512, 7, 7])  |
| alexnet     | 0.027948856353759766       | torch.Size([256, 6, 6])  |
| resnet18    | 0.13416695594787598        | torch.Size([512])        |
| resnet34    | 0.21873879432678223        | torch.Size([512])        |
| resnet50    | 0.33829498291015625        | torch.Size([2048])       |
| resnet101   | 0.5382320880889893         | torch.Size([2048])       |
| resnet152   | 0.7319691181182861         | torch.Size([2048])       |
| densenet121 | 0.5038180351257324         | torch.Size([1024, 7, 7]) |
| densenet161 | 0.9816281795501709         | torch.Size([2208, 7, 7]) |
| densenet169 | 0.6185808181762695         | torch.Size([1664, 7, 7]) |
| densenet201 | 0.8141469955444336         | torch.Size([1920, 7, 7]) |

