local M = { }

function M.parse(arg)
  cmd = torch.CmdLine()

  cmd:text()
  cmd:text('Decode using a DCNN-LSTM-CTC model.')
  cmd:text()
  cmd:text('Options')

  cmd:argument('model', 'Path to the neural network model file')
  cmd:argument('data', 'Path to the dataset HDF5 file')

  -- -- Misc
  cmd:option('-symbols_table', '', 'Symbols table (original_symbols.txt)')
  cmd:option('-batch_size', 40, 'Batch size')
  cmd:option('-gpu', 0, 'Which gpu to use. -1 = use CPU')
  
  cmd:text()
  local opt = cmd:parse(arg or {})
  return opt
end

return M
