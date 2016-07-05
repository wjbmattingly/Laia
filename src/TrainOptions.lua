local M = { }

function M.parse(arg)
  cmd = torch.CmdLine()

  cmd:text()
  cmd:text('Train a model with CTC')
  cmd:text()

  cmd:text('Batcher options:')
  cmd:option('-batch_size', 40, 'Batch size')
  cmd:option('-max_epoch_samples', -1,
	     'Maximum number of training samples to process in each epoch; ' ..
             '-1 to disable this maximum')
  --[[cmd:option('-curriculum_learning_init', 3,
	     'Initial value of the curriculum learning parameter. ' ..
	     'This parameter will be linearly decreased in each epoch until' ..
	     ' it reaches 0 in -curriculum_learning_epochs.')
  cmd:option('-curriculum_learning_epochs', 5,
             'Use curriculum learning for the first n epochs') ]]--
  cmd:text()

  cmd:text('Loss options:')
  cmd:option('-adversarial_epsilon', 0.007,
	     'Maximum differences in the adversarial samples')
  cmd:option('-adversarial_weight', 0.0,
	     'Weight of the loss adversarial samples')
  cmd:option('-grad_clip', 3, 'Gradient clipping')
  cmd:option('-weight_l1_decay', 1e-6, 'L1 weight decay penalty strength')
  cmd:option('-weight_l2_decay', 1e-6, 'L2 weight decay penalty strength')
  cmd:text()

  cmd:text('Optimizer options:')
  cmd:option('-alpha',0.95, 'RMS Prop alpha value to use')
  cmd:option('-learning_rate', 0.001, 'Learning rate to use')
  cmd:option('-learning_rate_decay', 0.97, 'Learning rate decay to use')
  cmd:option('-learning_rate_decay_after', 10,
             'Start learning rate decay after this epoch')
  cmd:option('-max_epochs', -1,
	     'Number of epochs to run; -1 to disable this maximum')
  cmd:option('-max_no_improv_epochs', 15,
	     'Stop training after this number of epochs without improvement;' ..
             ' -1 to disable this maximum')
  cmd:text()

  cmd:text('Other options:')
  cmd:option('-gpu', 0, 'Which gpu to use. -1 = use CPU')
  cmd:option('-output_path', '.', 'Directory to store ouput')
  cmd:option('-seed', 0x12345, 'Random number generator seed to use')
  cmd:text()

  cmd:text('Arguments:')
  cmd:argument('initial_checkpoint', 'Path to the input model for training')
  cmd:argument('training',
      'HDF5 file containing the training set (from prepare_XXXX.sh)')
  cmd:argument('validation',
      'HDF5 file containing the validation set (from prepare_XXXX.sh)')
  cmd:text()

  local opt = cmd:parse(arg or {})
  return opt
end

return M
