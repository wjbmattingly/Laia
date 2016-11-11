local ClassWithOptions = torch.class('laia.ClassWithOptions')

function ClassWithOptions:__init(opt)
  self._opt = opt or {}
end

function ClassWithOptions:setOptions(opt)
  if opt then table.update(self._opt, opt, false) end
  self:checkOptions()
  return self
end

function ClassWithOptions:getOptions()
  return self._opt
end

function ClassWithOptions:registerOptions(parser, advanced)
  error('Not implemented!')
end

function ClassWithOptions:checkOptions()
  error('Not implemented!')
end

return ClassWithOptions
