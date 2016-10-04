local WeightDecayRegularizer, Parent =
  torch.class('laia.WeightDecayRegularizer', 'laia.Regularizer')

function WeightDecayRegularizer:__init()
  Parent.__init(self)
  self._opt.weight_l1_decay = 0
  self._opt.weight_l2_decay = 0
end

-- loss  = the loss value to regularize
-- model = model being optimized
function WeightDecayRegularizer:regularize(loss, model)
  -- Handful variables
  local w1 = self._opt.weight_l1_decay
  local w2 = self._opt.weight_l2_decay
  -- This assumes model:getParameters() was called before to flatten the
  -- parameters of the model.
  local parameters, gradParameters = laia.getFlatParameters(model)
  -- L1 Regularization
  if w1 > 0 then
    -- J += w1 * sum_i |P(i)|
    loss = loss + w1 * torch.abs(parameters):sum()
    -- dJ/dP(i) += w1 * sign(P(i))
    gradParameters:add(w1, torch.sign(parameters))
  end
  -- L2 Regularization
  if w2 > 0 then
    -- J += w_l2 * 0.5 * sum_i P(i)^2
    loss = loss + w2 * 0.5 * torch.pow(parameters, 2):sum()
    -- dJ/dP(i) += w_l2 * P(i)
    gradParameters:add(w2, parameters)
  end
  return loss
end

function WeightDecayRegularizer:registerOptions(parser)
  parser:option('--weight_l1_decay',
		'L1 regularization factor, applied to ALL trainable parameters',
		0, tonumber)
  parser:option('--weight_l2_decay',
		'L2 regularization factor, applied to ALL trainable parameters',
		0, tonumber)
end

function WeightDecayRegularizer:checkOptions()
  assert(self._opt.weight_l1_decay >= 0,
	 ('Weight L1 decay must be >= 0 (value = %s)'):format(
	   self._opt.weight_l1_decay))
  assert(self._opt.weight_l2_decay >= 0,
	 ('Weight L2 decay must be >= 0 (value = %s)'):format(
	   self._opt.weight_l2_decay))
end
