## Image Captioning

### Overview:

In this project, the objective was to generate captions for images using a model built on an encoder-decoder architecture. The encoder component of the model leverages the Torchvision Resnet101 pretrained model, while the decoder portion is implemented using LSTM.

#### Dataset:

The model was trained using the Flicker8k Dataset

#### Validation and inference:

The trained model demonstrated the capability to provide reasonable descriptions, achieving a test score of 32.32.

### Demo

Below are a few sample inferences generated by the trained model:
1. Greedy: <start> Two people sit on a boat near the water . <end>

![44129946_9eeb385d77](https://github.com/lahari21/Image_Captioning/assets/62760117/b1a4f6a0-3928-4d1b-83a3-08bafe8227fe)

2. Greedy: <start> A brown dog is jumping into a pool with a tennis ball in its mouth . <end>

![44856031_0d82c2c7d1](https://github.com/lahari21/Image_Captioning/assets/62760117/44e71d2f-399d-43a0-a42e-1a15216594af)

3. Greedy: <start> A man on a skateboard is on a skateboard . <end>

![47870024_73a4481f7d](https://github.com/lahari21/Image_Captioning/assets/62760117/2dfc869d-8a2b-44a1-b736-ecae8565ba35)

4. <start> A man on a surfboard is riding on a boat . <end>

![49553964_cee950f3ba](https://github.com/lahari21/Image_Captioning/assets/62760117/1a77c04e-7e5a-4774-8416-c17d5b80a226)


#### References
> [1] https://github.com/yurayli/image-caption-pytorch/tree/b517cddcd2e11fdb017faa1eb67a7990558b4040
> [2] Show and Tell: A Neural Image Caption Generator (https://arxiv.org/abs/1411.4555)
