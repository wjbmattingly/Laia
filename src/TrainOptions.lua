require 'src.ImageDistorter'

local M = { }

function M.parse(arg)
  cmd = torch.CmdLine()

  cmd:text()
  cmd:text('Train a model with CTC')
  cmd:text()

  cmd:text('Batcher options:')
  cmd:option('-batch_size', 40, 'Batch size')
  cmd:option('-min_width', 0, 'Minimum image width for batches')
  cmd:option('-width_factor', true, 'Make width a factor of the max pooling reduction')
  cmd:option('-num_samples_epoch', -1,
	     'Number of training samples to process in each epoch; ' ..
             '-1 sets this value to the number of samples in the training ' ..
             'dataset')
  --[[cmd:option('-curriculum_learning_init', 3,
	     'Initial value of the curriculum learning parameter. ' ..
	     'This parameter will be linearly decreased in each epoch until' ..
	     ' it reaches 0 in -curriculum_learning_epochs.')
  cmd:option('-curriculum_learning_epochs', 5,
             'Use curriculum learning for the first n epochs') ]]--
  cmd:text()

  cmd:text('Early stop options:')
  cmd:option('-early_stop_criterion', 'valid_cer', 'Use this criterion for ' ..
             'early stop; values: train_cer, train_loss, valid_cer, valid_loss')
  cmd:option('-max_epochs', -1,
	     'Maximum number of epochs to run; -1 to disable this maximum')
  cmd:option('-max_no_improv_epochs', 15,
	     'Stop training after this number of epochs without a ' ..
             'significant improvement in the early stop criterion; ' ..
             '-1 to disable this maximum')
  cmd:option('-min_relative_improv', 0.0,
             'Only relative improvements in the early stop criterion ' ..
             'greater than this amount are considered significant; ' ..
	     'ej: 0.05 for 5%')
  cmd:text()

  cmd:text('Loss options:')
  cmd:option('-adversarial_epsilon', 0.007,
	     'Maximum differences in the adversarial samples')
  cmd:option('-adversarial_weight', 0.0,
	     'Weight of the loss adversarial samples')
  cmd:option('-grad_clip', 5, 'Gradient clipping; -1 to disable clipping')
  cmd:option('-weight_l1_decay', 1e-6, 'L1 weight decay penalty strength')
  cmd:option('-weight_l2_decay', 1e-6, 'L2 weight decay penalty strength')
  cmd:text()

  cmd:text('Optimizer options:')
  cmd:option('-alpha',0.95, 'RMS Prop alpha value to use')
  cmd:option('-learning_rate', 0.001, 'Learning rate to use')
  cmd:option('-learning_rate_decay', 0.97, 'Learning rate decay to use')
  cmd:option('-learning_rate_decay_after', 10,
             'Start learning rate decay after this epoch')
  cmd:text()

  --cmd:text('Image distorter options:')
  --ImageDistorter.addCmdOptions(cmd)
  --cmd:text()

  cmd:text('Other options:')
  cmd:option('-gpu', 0, 'Which gpu to use. -1 = use CPU')
  --cmd:option('-output_path', '.', 'Directory to store ouput')
  cmd:option('-output_model', '', 'Write output model to this file instead ' ..
	     'of overwritting the input model')
  cmd:option('-output_progress', '', 'Write the progress of training ' ..
	     'after each epoch to this text file')
  cmd:option('-seed', 0x12345, 'Random number generator seed to use')
  --[[cmd:option('-save_module_input', '',
	     'Save the input and the gradient respect the input of each ' ..
	     'module in the list as an image')
  --]]
  cmd:option('-cer_trim', -1, 'For computing CER, removes leading, trailing and repetitions of given symbol number (i.e. space)')
  cmd:text()

  cmd:text('Arguments:')
  cmd:argument('model', 'Path to the input model or checkpoint for training')
  cmd:argument('symbols_table', 'list of training symbols')
  cmd:argument('training', 'list of images for training')
  cmd:argument('training_gt', 'training transcripts')
  cmd:argument('validation', 'list of images for validation')
  cmd:argument('validation_gt', 'validation transcripts')
  cmd:text()

  local opt = cmd:parse(arg or {})
  assert(opt.batch_size > 0, 'Batch size must be greater than 0')
  --[[
  opt.save_module_input = string.split(opt.save_module_input)
  opt.save_module_input = table.map(opt.save_module_input, function(x)
      local t = table.map(string.split(x, '[^.]+'), tonumber)
      table.extend_with_last_element(t, 2)
      return t
  end)
  --]]

  return opt
end

return M
