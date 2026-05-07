"""
obtain image description texts and encode them into feature maps using text encoders
"""

import argparse
import os
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BlipForConditionalGeneration, BlipProcessor, LlavaForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--pretrained', default=True, type=bool)
	parser.add_argument('--project_dir', default='./Things_EEG2/', type=str)
	parser.add_argument('--output_root', default='./EEG2image', type=str)
	parser.add_argument('--llm_model', default='Qwen', choices=['Qwen', 'llava', 'blip'], type=str)
	parser.add_argument('--encoder', default='cn-clip', choices=['ViT-H-14', 'cn-clip'], type=str)
	parser.add_argument('--gpu', default='0', type=str)
	return parser.parse_args()


def setup_env(gpu):
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	seed = 20200220
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	return device


def load_text_encoder(encoder_name, device):
	if encoder_name == 'cn-clip':
		from cn_clip.clip import load_from_name
		import cn_clip.clip as clip

		encoder_model, _ = load_from_name('RN50')
		encoder_model = encoder_model.to(device)
		encoder_model.eval()
		tokenizer = clip.tokenize

		def encode_text(text):
			with torch.no_grad():
				tokens = tokenizer([text]).to(device)
				features = encoder_model.encode_text(tokens)
				features = features / features.norm(dim=-1, keepdim=True)
			return features.cpu().numpy()

		return encode_text, 'newclip_cn'

	model_id = 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
	encoder_model, _, _ = open_clip.create_model_and_transforms(model_id)
	encoder_model = encoder_model.to(device)
	encoder_model.eval()
	tokenizer = open_clip.get_tokenizer(model_id)

	def encode_text(text):
		with torch.no_grad():
			tokens = tokenizer([text]).to(device)
			features = encoder_model.encode_text(tokens)
			features = features / features.norm(dim=-1, keepdim=True)
		return features.cpu().numpy()

	return encode_text, 'vit_h_14'


def load_caption_model(name):
	torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

	if name == 'Qwen':
		model_name = 'Qwen/Qwen2-VL-7B-Instruct'
		processor = AutoProcessor.from_pretrained(model_name)
		model = Qwen2VLForConditionalGeneration.from_pretrained(
			model_name, torch_dtype=torch_dtype, device_map='auto'
		)

		return model, processor

	if name == 'llava':
		model_name = 'llava-hf/llava-1.5-7b-hf'
		processor = AutoProcessor.from_pretrained(model_name)
		model = LlavaForConditionalGeneration.from_pretrained(
			model_name,
			torch_dtype=torch_dtype,
			device_map='auto',
		)
		return model, processor

	model_name = 'Salesforce/blip-image-captioning-base'
	processor = BlipProcessor.from_pretrained(model_name)
	model = BlipForConditionalGeneration.from_pretrained(
		model_name,
		torch_dtype=torch_dtype,
		device_map='auto',
	)
	return model, processor


def generate_caption(image, label_text, llm_name, model, processor):
	prompt = f'Describe only what is directly visible in the image of {label_text} in one short sentence.'

	if llm_name == 'Qwen':
		messages = [
			{
				'role': 'user',
				'content': [
					{'type': 'image', 'image': image},
					{'type': 'text', 'text': prompt},
				],
			}
		]
		chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
		inputs = processor(text=[chat_text], images=[image], return_tensors='pt').to(model.device)
		with torch.no_grad():
			output_ids = model.generate(**inputs, max_new_tokens=70, repetition_penalty=1.2, no_repeat_ngram_size=3)
		response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
		return response

	if llm_name == 'llava':
		llava_prompt = f'<image>\n{prompt}'
		inputs = processor(text=llava_prompt, images=image, return_tensors='pt')
		for key in inputs:
			if isinstance(inputs[key], torch.Tensor):
				inputs[key] = inputs[key].to(model.device)
		with torch.no_grad():
			output_ids = model.generate(
				input_ids=inputs.get('input_ids'),
				pixel_values=inputs.get('pixel_values'),
				image_sizes=inputs.get('image_sizes'),
				attention_mask=inputs.get('attention_mask'),
				max_new_tokens=70,
				do_sample=False,
			)
		response = processor.decode(output_ids[0], skip_special_tokens=True).strip()
		return response

	inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device)
	with torch.no_grad():
		output_ids = model.generate(**inputs, max_new_tokens=50)
	response = processor.decode(output_ids[0], skip_special_tokens=True)
	return response


def clean_response(response, label_text):
	if not response:
		return f'A photo of {label_text}.'

	if 'assistant' in response:
		response = response.split('assistant')[-1].strip()

	prompt_text = f'Describe only what is directly visible in the image of {label_text} in one short sentence:'
	response = response.replace(prompt_text, '').strip()
	response = response[:210].strip()

	if not response:
		return f'A photo of {label_text}.'
	return response


def list_images(part_dir):
	image_list = []
	for root, _, files in os.walk(part_dir):
		for file_name in files:
			if file_name.endswith('.jpg') or file_name.endswith('.JPEG'):
				image_list.append(os.path.join(root, file_name))
	image_list.sort()
	return image_list


def load_existing_texts(save_text_dir, partition, image_count):
	responses = []
	for i in range(image_count):
		text_path = os.path.join(save_text_dir, f'{partition}_{i + 1:07d}.txt')
		text_data = np.loadtxt(text_path, dtype=str)
		responses.append(str(text_data))
	return responses


def save_feature(save_dir, partition, idx, encoder_suffix, feats):
	file_name = f'{partition}_{idx + 1:07d}_{encoder_suffix}'
	np.save(os.path.join(save_dir, file_name), feats)


def main():
	args = parse_args()
	device = setup_env(args.gpu)

	print('Extract feature maps - text pipeline <<<')
	print('\nInput arguments:')
	for key, val in vars(args).items():
		print('{:16} {}'.format(key, val))

	encode_text, encoder_suffix = load_text_encoder(args.encoder, device)
	caption_model, caption_processor = load_caption_model(args.llm_model)

	img_set_dir = os.path.join(args.project_dir, 'Image_set')
	img_partitions = os.listdir(img_set_dir)

	for partition in img_partitions:
		part_dir = os.path.join(img_set_dir, partition)
		image_list = list_images(part_dir)

		save_text_dir = os.path.join(args.output_root, 'Description', args.llm_model, partition)
		save_dir = os.path.join(
			args.output_root,
			'DNN_feature_maps',
			'full_feature_maps',
			args.llm_model,
			f'pretrained-{args.pretrained}',
			partition,
		)
		os.makedirs(save_text_dir, exist_ok=True)
		os.makedirs(save_dir, exist_ok=True)

		use_existing_texts = len(os.listdir(save_text_dir)) == len(image_list) and len(image_list) > 0
		if use_existing_texts:
			print(f'Description texts of {partition} already exist, only encoding texts.')
			responses = load_existing_texts(save_text_dir, partition, len(image_list))
			for i, response_text in tqdm(enumerate(responses), total=len(responses)):
				feats = encode_text(str(response_text))
				save_feature(save_dir, partition, i, encoder_suffix, feats)
			continue

		print(f'Start to extract description texts of {partition}...')
		for i, image_path in tqdm(enumerate(image_list), total=len(image_list)):
			image = Image.open(image_path).convert('RGB')
			label_text = image_path.split('/')[-2][6:]

			raw_response = generate_caption(image, label_text, args.llm_model, caption_model, caption_processor)
			response = clean_response(raw_response, label_text)
			print(f'[{i}] {label_text}: {response}')

			text_path = os.path.join(save_text_dir, f'{partition}_{i + 1:07d}.txt')
			with open(text_path, 'w') as f:
				f.write(response + '\n')

			feats = encode_text(response)
			save_feature(save_dir, partition, i, encoder_suffix, feats)

if __name__ == '__main__':
	main()