# This is a port of the Pytorch AlexNet training into Tensorflow.
# I am using the tf.estimator API

import argparse
from models import model
import data_utils
import tensorflow as tf
import os
import sys

tf.logging.set_verbosity(tf.logging.INFO)
MODELS = ['AlexNet']

def get_AlexNet_experiment(args):
	"""
	Function for creating an experiment using the AlexNet model on ImageNet
	"""
	train_input_fn = data_utils.get_input_fn(
		data_dir=os.path.join(args.data_dir,'train'),
		num_epochs=args.num_epochs,
		batch_size=args.batch_size,
		shuffle=True)

	val_input_fn = data_utils.get_input_fn(
		data_dir=os.path.join(args.data_dir,'val'),
		num_epochs=1,
		batch_size=2*args.batch_size,
		shuffle=False)

	net = model.AlexNet(
		num_classes=1000,
		scope='ImageNet_AlexNet')

	config = tf.contrib.learn.RunConfig(
		log_device_placement=False,
		gpu_memory_fraction=0.98,
		tf_random_seed=1234,
		save_summary_steps=50,
		save_checkpoints_secs=300,
		keep_checkpoint_max=10000,
		keep_checkpoint_every_n_hours=10000,
		log_step_count_steps=10,
	)

	estimator = tf.estimator.Estimator(
		model_fn=net.get_model_fn(),
		model_dir=args.model_dir,
		config=config,
		params={'learning_rate': args.lr}
	)

	experiment = tf.contrib.learn.Experiment(
		estimator=estimator,
		train_input_fn=train_input_fn,
		eval_input_fn=val_input_fn,
		eval_metrics=None,
		train_steps=None,
		eval_steps=None,
		train_monitors=[],
		min_eval_frequency=1000,
		eval_delay_secs=240
	)

	return experiment


def main(args):

	parser = argparse.ArgumentParser(description='TF ImageNet Training')
	parser.add_argument('-m', '--model', required=True, choices=MODELS,                 help='Select which model to train')
	parser.add_argument('-dd', '--data_dir', required=True,
					   help='Data directory to load input images')
	parser.add_argument('-md', '--model_dir', required=True,
					   help='Model directory to save files related to models')
	parser.add_argument('-bs', '--batch_size', default=64, type=int,
					   help='Batch size of ImageNet data')
	parser.add_argument('--num_epochs', default=90, type=int,
					   help='Number of training epochs')
	parser.add_argument('--lr', default=0.1)

	args = parser.parse_args(args)

	if args.model=='AlexNet':
		experiment = get_AlexNet_experiment(args)
	else:
		raise NotImplementedError()

	experiment.train_and_evaluate()


if __name__=='__main__':
	main(sys.argv[1:])



