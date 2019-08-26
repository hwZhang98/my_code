import run
from utils import read_corpus, batch_iter
from nmt_model import NMT
import sys

def multi_parameter_tuning(args):
    lrs = [1e-2,1e-3,5e-3,1e-4,5e-4]
    hidden_sizes = [128,256,512]
    lr_decays = [0.9,0.7,0.5]
    iter = 0
    valid_metric = {}               # 存储各个模型ppl的值
    dev_data_src = read_corpus(args['dev_source'], source='src')
    dev_data_tgt = read_corpus(args['dev_target'], source='tgt')
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    for i in lrs:
        for j in hidden_sizes:
            for k in lr_decays:
                print('第%d次测试================================================='%iter)
                arg_test = args
                arg_test['lr'],arg_test['hidden_size'],arg_test['lr_decay'] = i,j,k
                arg_test['save_to'] = 'model_'+'lr_'+str(i)+'hd_size_'+str(j)+'lr_dys_'+str(k)+'.bin'
                run.train(arg_test)
                model = NMT.load(arg_test['save_to'])
                dev_ppl = run.evaluate_ppl(model, dev_data, batch_size=128)  # dev batch size can be a bit larger
                valid_metric[arg_test['save_to']] = dev_ppl
                print(arg_test['save_to'],'  validation: iter %d, dev. ppl %f' % (iter, dev_ppl), file=sys.stderr)
                iter += 1
    model = min(valid_metric,key=valid_metric.get())
    print('best_model is %s ,ppl is %f'%(model,valid_metric[model]))