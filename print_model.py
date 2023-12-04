import torch
from torchview import draw_graph
from basic_cnn import BasicModel
from spikingjelly.activation_based import functional


#ALPHA = 3.7132080089425044
#TAU = 2.180830180029865
batch_size = 2
# model = BasicModel(seq_num=50, num_channels=3, optimizer="Adam", lr=0.001, alpha=ALPHA, tau=TAU).cuda()
# functional.set_step_mode(model, step_mode='m')
# model.load_state_dict(torch.load('best_basic_model_snn_sj0929.pth'))
model = BasicModel(num_channels=3, optimizer="Adam", lr=0.001).cuda()
model.load_state_dict(torch.load('best_basic_model_cnn_relu.pth'))
model_graph = draw_graph(model, input_size=(batch_size, 128), device='meta')
model_graph.visual_graph