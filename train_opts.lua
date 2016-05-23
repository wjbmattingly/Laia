local M = { }

function M.parse(arg)

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Train a DCNN-LSTM-CTC model.')
  cmd:text()
  cmd:text('Options')

  -- Core ConvNet settings
  cmd:option('-backend', 'cudnn', 'nn|cudnn')
  
  -- Data input/ouput settings
  cmd:argument('training',
      'HDF5 file containing the training set (from prepare_XXXX.sh)')
  cmd:argument('validation',
      'HDF5 file containing the validation set (from prepare_XXXX.sh)')
  --
  cmd:option('-checkpoint_start_from', '',
      'Load model from a checkpoint instead of random initialization.')
  --
  cmd:option('-output_path', '.',
      'Directory to store ouput')

  -- Model settings
  cmd:argument('-vocab_size',  
      'Number of characters (see original_symbols.txt)')
  cmd:argument('-sample_height',  
      'Height of the samples')

  -- Loss function
  cmd:option('-weight_l1_decay', 1e-6, 'L1 weight decay penalty strength')
  cmd:option('-weight_l2_decay', 1e-6, 'L2 weight decay penalty strength')
  cmd:option('-grad_clip', 3, 'Gradient clipping')

  -- Optimization
  cmd:option('-batch_size', 40, 'Batch size')
  cmd:option('-learning_rate', 0.001, 'Learning rate to use')
  cmd:option('-learning_rate_decay', 0.97, 'Learning rate decay to use')
  cmd:option('-learning_rate_decay_after', 10, 'Learning rate decay to use')
  cmd:option('-alpha',0.95, 'RMS Prop alpha value to use')
  cmd:option('-curriculum_learning_after', 3, 'Epoch in which Curriculum Learning starts to work')
  cmd:option('-curriculum_learning_update', 10, 
      'Update Curriculum Learning lambda every n epoch')
  cmd:option('-max_epochs', -1, 'Number of iterations to run; -1 to run forever')

  -- Layers stuff
  cmd:option('-drop_prob', 0.5, 'Dropout strength throughout the model')
  --cmd:option('-leakyrelu_a', 1/100., 'Negval from LeakyReLU')

  -- Misc
  cmd:option('-seed', 1234, 'Random number generator seed to use')
  cmd:option('-gpu', 0, 'Which gpu to use. -1 = use CPU')

  cmd:text()
  local opt = cmd:parse(arg or {})
  return opt
end

return M
