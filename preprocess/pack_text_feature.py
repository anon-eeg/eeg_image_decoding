"""
Package text feature maps from obtain_text_feature.py output
Aggregates individual feature vectors into a single array for PCA/training
"""
import argparse
import numpy as np
import os
from tqdm import tqdm

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--llm_model', default='Qwen', choices=['Qwen', 'llava', 'blip'], type=str)
	parser.add_argument('--encoder', default='cn-clip', choices=['ViT-H-14', 'cn-clip'], type=str)
	parser.add_argument('--pretrained', default=True, type=bool)
	parser.add_argument('--output_root', default='./EEG2image', type=str)
	return parser.parse_args()


def get_encoder_suffix(encoder_name):
	if encoder_name == 'cn-clip':
		return 'newclip_cn'
	return 'vit_h_14'

def aggregate_features(feature_dir, encoder_suffix):
	feature_list = []
	feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
	feature_files.sort()

	for feature_file in tqdm(feature_files, desc=f'Loading from {os.path.basename(feature_dir)}'):
		if encoder_suffix in feature_file:
			feature_data = np.load(os.path.join(feature_dir, feature_file))
			feature_list.append(feature_data)

	return np.array(feature_list) if feature_list else None


args = parse_args()

print('>>> Package and aggregate text feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


seed = 20200220
np.random.seed(seed)
encoder_suffix = get_encoder_suffix(args.encoder)

print(f'\nAggregating features for {args.llm_model} + {args.encoder}')
print(f'Looking for feature files with suffix: {encoder_suffix}')

save_dir = os.path.join(
	args.output_root,
	'DNN_feature_maps',
	'pca_feature_maps',
	args.llm_model,
	f'pretrained-{args.pretrained}'
)
os.makedirs(save_dir, exist_ok=True)

partitions = ['training_images', 'test_images']

for partition in partitions:
	feature_base_dir = os.path.join(
		args.output_root,
		'DNN_feature_maps',
		'full_feature_maps',
		args.llm_model,
		f'pretrained-{args.pretrained}',
		partition
	)

	if not os.path.isdir(feature_base_dir):
		print(f'Warning: Directory not found {feature_base_dir}, skipping...')
		continue

	print(f'\nProcessing {partition}...')
	feats = aggregate_features(feature_base_dir, encoder_suffix)

	if feats is not None:
		file_name = f'{args.llm_model}_feature_maps_{partition}_{encoder_suffix}'
		save_path = os.path.join(save_dir, f'{file_name}.npy')
		np.save(save_path, feats)
		print(f'Saved aggregated features to {save_path}')
		print(f'  Shape: {feats.shape}')
	else:
		print(f'No features found matching suffix {encoder_suffix} in {partition}')
