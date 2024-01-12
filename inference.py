from data_utils import *
from solver import *
from model import *

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

loader = prepare_loader('/Users/laharikethinedi/Desktop/Blog/caption/Flicker8k_Dataset/', batch_size=128, data_part='test')
data = loader.dataset
vocab_size = len(data.idx_to_word)

model = CaptionModel_B(2048, 50, 160, vocab_size, num_layers=1)
model.load_state_dict(torch.load('/Users/laharikethinedi/Desktop/Blog/caption/image-caption-pytorch-master/output/im_caption_37.147_0.322_epoch_11.pth.tar'))
solver = NetSolver(data, model)

def preprocess(im):
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    
    transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(rgb_mean, rgb_std),
                ])
    return transform(im).unsqueeze(0)

def sample_and_plot(im_id):
    im = Image.open(im_id)
    im_t = preprocess(im).to(device)
    feature = extract_features(cnn, im_t).squeeze().unsqueeze(0)

    s1 = solver.sample(feature, search_mode='greedy')
    s1 = decode_captions(s1, data.idx_to_word)

    print('Greedy:', s1[0])
    plt.imshow(np.array(im))
    plt.axis('off')

im_id = '/Users/laharikethinedi/Desktop/Blog/caption/36422830_55c844bc2d.jpg'
sample_and_plot(im_id)