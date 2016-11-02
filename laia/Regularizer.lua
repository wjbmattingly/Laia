local Regularizer = torch.class('laia.Regularizer')

function Regularizer.__init()
  self._opt = {}
end

-- Return the regularized loss, parameters and gradParameters are updated
-- directly
function Regularizer:regularize(...)
  return nil
end

function Regularizer:registerOptions(parser)
end

function Regularizer:setOptions(opt)
  if opt then table.update(self._opt, opt, false) end
  self:checkOptions()
end

function Regularizer:checkOptions()
end
