import argparse, os, shutil


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

pwd = os.path.dirname(os.path.realpath(__file__))

# Experiment arguments
parser.add_argument('--pwd', default=pwd, help='path of the current folder.')
parser.add_argument('--patient_list', default=f'{pwd}/data/patient_all.txt', help='path to the file which contains the patient IDs and labels.')
parser.add_argument('--split_path', default=f'{pwd}/data/', help='path to the file which contains the split of train and val.')
parser.add_argument('--exp_name', default='test', help='The name for this experiment') 
parser.add_argument('--feature_dir', default='/media/ruoyu/Data/Ruoyu/Data/TCGA/Patches/10x_256/data_all/cnn_feat/res18_imagenet_all_info/', help='The path to the folder which contains extracted features.')
parser.add_argument('--output_dir', default=f'{pwd}/output/', help='The output path.') 

# Training arguments
parser.add_argument('--input_dim', type=int, default=512, help='The dimension of input feature.')
parser.add_argument('--hidden_dim', type=int, default=512, help='The dimension of the hidden layer.')
parser.add_argument('--epoch', type=int, default=10, help='The number of epoch.')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate.')
parser.add_argument('--lr_step', type=int, default=3, help='learning rate steps.')
parser.add_argument('--margin_inter', type=float, default=0.5, help='Margin between negative sample and positive.')
parser.add_argument('--margin_intra', type=float, default=0.1, help='Margin between negative samples.')
parser.add_argument('--k_ratio', type=float, default=0.1, help='Choose top K%\ scores for aggregation.')
parser.add_argument('--sample_max_num', type=int, default=1000, help='The maximum sample number allowed within a bag given the GPU limitation.')


def save_args(args):
    shutil.copy(f'{args.pwd}/tripletMIL_training.py', args.output_dir)
    shutil.copy(f'{args.pwd}/arguments.py', args.output_dir)
    shutil.copy(f'{args.pwd}/utils.py', args.output_dir)
    shutil.copy(f'{args.pwd}/data_utils.py', args.output_dir)
