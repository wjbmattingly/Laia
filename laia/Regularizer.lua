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

function Regularizer:setOptions(opts)
  table.update_values(self._opt, opts)
  self:checkOptions()
end

function Regularizer:checkOptions()
end
