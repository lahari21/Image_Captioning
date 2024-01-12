from data_utils import *
from solver import *
from model import *

trn_feat_path = '/Users/laharikethinedi/Desktop/Blog/caption/image-caption-pytorch-master/output/train_feat_arrays.pkl' #'/features/train_feat_arrays.pkl'
trn_cap_path = '/Users/laharikethinedi/Desktop/Blog/caption/image-caption-pytorch-master/output/train_cap_tokens.pkl'#'/captions/train_cap_tokens.pkl'
test_feat_path = '/Users/laharikethinedi/Desktop/Blog/caption/image-caption-pytorch-master/output/test_feat_array.pkl' #'/features/test_feat_array.pkl'
test_cap_path = '/Users/laharikethinedi/Desktop/Blog/caption/image-caption-pytorch-master/output/test_cap_tokens.pkl'#'/captions/test_cap_tokens.pkl'

# image_path = '/Users/laharikethinedi/Desktop/Blog/caption/Flicker8k_Dataset/' #'/flikr8k/'
# model_path = '/model/'
# output_path = '/Users/laharikethinedi/Desktop/Blog/caption/image-caption-pytorch-master/output' #'/output/'

data = load_features_tokens(trn_feat_path, trn_cap_path, test_feat_path, test_cap_path)
vocab_size = len(data['idx_to_word'])

model = CaptionModel_B(2048, 50, 160, vocab_size, num_layers=1)
model.load_state_dict(torch.load('/Users/laharikethinedi/Desktop/Blog/caption/image-caption-pytorch-master/output/im_caption_37.147_0.322_epoch_11.pth.tar'))
solver = NetSolver(data, model)
# solver.check_bleu('train', num_samples=2000, check_loss=1)
val_loss, val_bleu = solver.check_bleu_pre('test', num_samples=2000, check_loss=True)

print('{"metric": "Val. BLEU.", "value": %.4f}' % (val_bleu))
print('{"metric": "Val. Loss", "value": %.4f}' % (val_loss))