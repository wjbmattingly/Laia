local M = {}

function M.setup(opt)
  local model
  if opt.checkpoint_start_from == '' then
    print('Initializing the model from scratch ...')
    model = createModel(opt.sample_height, opt.vocab_size)
  else
    print('Initializing the model from ' .. opt.checkpoint_start_from)
    model = torch.load(opt.checkpoint_start_from)
  end

  -- Find all Dropout layers and set their probabilities
  local dropout_modules = model:findModules('nn.Dropout')
  for i, dropout_module in ipairs(dropout_modules) do
    dropout_module.p = opt.drop_prob
  end
  -- -- Find all LeakyReLU layers and set their negval value
  -- local leakyrelu_modules = model:findModules('nn.LeakyReLU')
  -- for i, leakyrelu_modules in ipairs(leakyrelu_modules) do
  --   leakyrelu_modules.negval = opt.drop_prob
  -- end

  -- If using gpu
  if opt.gpu >= 0 then
    model:cuda()
  else
    model:float()
  end

  if opt.backend == 'cudnn' then
    cudnn.convert(model, cudnn)
  end

  return model
end

return M
