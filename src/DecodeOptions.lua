local M = { }

function M.parse(arg)
  cmd = torch.CmdLine()

  cmd:text()
  cmd:text('Decode using a DCNN-LSTM-CTC model.')
  cmd:text()

  cmd:text('Options:')
  cmd:option('-batch_size', 40, 'Batch size')
  cmd:option('-min_width', 0, 'Minimum image width for batches')
  cmd:option('-gpu', 0, 'Which gpu to use. -1 = use CPU')
  cmd:option('-seed', 0x12345, 'Random number generator seed to use')
  cmd:option('-symbols_table', '', 'Symbols table (original_symbols.txt)')
  cmd:text()

  cmd:text('Arguments:')
  cmd:argument('model', 'Path to the neural network model file')
  cmd:argument('data', 'Path to the list of images')
  cmd:text()
  local opt = cmd:parse(arg or {})
  assert(opt.batch_size > 0, 'Batch size must be greater than 0')
  return opt
end

return M
