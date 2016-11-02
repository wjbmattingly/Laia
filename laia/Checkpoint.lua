local Checkpoint = torch.class('laia.Checkpoint')

-- Utility class to save/load Laia checkpoints from/to disk.
--
-- Checkpoints store information about (among other things):
-- Best / last model during training
-- Best / last epoch
-- Best / last accuracy performance
-- RNG state
-- Training state
-- etc
--
-- It is important that when a model is binded to a checkpoint through
-- setModelBest(model) or setModelLast(model), the model is cloned to the CPU
-- and all possible cudnn modules are replaced by the nn equivalents.
-- This allows to use the checkpoint on other systems that don't have these
-- packages installed.
--
-- Next, an example of how to use this class to keep checkpoints of the
-- training process and when the model is improved.
--
-- local checkpoint = laia.Checkpoint()
-- checkpoint:load('checkpoint.t7')
-- local model = checkpoint:getModelLast():cuda()
-- local best_accuracy = checkpoint:getPerformanceBest()
-- for epoch=(checkpoint:getEpochLast()+1),100 do
--   TrainEpoch(model)
--   local a = ComputeAccuracy(model)
--
--   -- If the accuracy improves, make a checkpoint of the model.
--   if best_accuracy == nil or best_accuracy < a then
--     checkpoint:setEpochBest(epoch)
--     checkpoint:setModelBest(model)
--     checkpoint:setPerformanceBest(a)
--     best_accuracy = a
--   end
--
--   -- Save the model to disk when it improves (getEpochBest() == epoch) or
--   -- every 10 epochs (epoch % 10 == 0).
--   if checkpoint:getEpochBest() == epoch or epoch % 10 == 0 then
--     checkpoint:setEpochLast(epoch)
--     checkpoint:setModelLast(model)
--     checkpoint:setPerformanceLast(a)
--     checkpoint:save('checkpoint.t7')
--   end
-- end

function Checkpoint:__init()
end

function Checkpoint:clear()
  for k, _ in pairs(self) do
    self[k] = nil
  end
end

-- Getter/Setter for the best Epoch
function Checkpoint:getEpochBest()
  return self._epoch_best or 0
end
function Checkpoint:setEpochBest(epoch)
  self._epoch_best = epoch
end

-- Getter/Setter for the last Epoch
function Checkpoint:getEpochLast()
  return self._epoch_last or 0
end
function Checkpoint:setEpochLast(epoch)
  self._epoch_last = epoch
end

-- Getter/Setter for the model config (i.e: arguments of create_model.lua)
function Checkpoint:getModelConfig()
  return self._model_config
end
function Checkpoint:setModelConfig(cfg)
  self._model_config = cfg
end

-- Getter/Setter for the model config (i.e: arguments of train.lua)
function Checkpoint:getTrainConfig()
  return self._train_config
end
function Checkpoint:setTrainConfig(cfg)
  self._train_config = cfg
end

-- Getter/Setter for the best Model
function Checkpoint:getModelBest()
  return self._model_best
end
function Checkpoint:setModelBest(model)
  -- By default, use all the possible nn layers to maximize portability.
  if cudnn then model = cudnn.convert(model, nn) end
  -- Always keep a deep copy of the best model on the CPU
  self._model_best = Checkpoint._cloneToCPU(model)
  self._model_best:clearState()
end

-- Getter/Setter for the last Model
function Checkpoint:getModelLast()
  return self._model_last
end
function Checkpoint:setModelLast(model)
  -- By default, use all the possible nn layers to maximize portability.
  if cudnn then model = cudnn.convert(model, nn) end
  -- Always keep a deep copy of the last model on the CPU
  self._model_last = Checkpoint._cloneToCPU(model)
  self._model_last:clearState()
end

-- Getter/Setter for the best performance
function Checkpoint:getPerformanceBest()
  return self._performance_best
end
function Checkpoint:setPerformanceBest(performance)
  self._performance_best = performance
end

-- Getter/Setter for the last performance
function Checkpoint:getPerformanceLast()
  return self._performance_last
end
function Checkpoint:setPerformanceLast(performance)
  self._performance_last = performance
end

-- Getter/Setter for RMSPropState
function Checkpoint:getRMSPropState()
  return self._rmsprop
end
function Checkpoint:setRMSPropState(rmsprop)
  self._rmsprop = rmsprop
end

-- Getter/Setter for RNGState
function Checkpoint:getRNGState()
  return self._rngstate
end
function Checkpoint:setRNGState(rngstate)
  self._rngstate = rngstate
end

-- Save checkpoint to file
function Checkpoint:save(filename)
  torch.save(filename, self)
end

-- Load checkpoint from file
function Checkpoint:load(filename)
  self:clear()
  local checkpoint = torch.load(filename)
  if torch.isTypeOf(checkpoint, 'laia.Checkpoint') then
    for k, v in pairs(checkpoint) do
      self[k] = v
    end
  elseif torch.isTypeOf(checkpoint, 'nn.Module') then
    self:setEpochBest(0)
    self:setEpochLast(0)
    self:setModelBest(checkpoint)
    self:setModelLast(checkpoint)
  elseif torch.type(checkpoint) == 'table' then
    self:setEpochBest(checkpoint.epoch or 0)
    self:setEpochLast(checkpoint.epoch or 0)
    self:setModelBest(checkpoint.model)
    self:setModelLast(checkpoint.model)
    self:setModelConfig(checkpoint.model_opt)
    self:setTrainConfig(checkpoint.train_opt)
    self:setRNGState(checkpoint.rng_state)
    self:setRMSPropState(checkpoint.rmsprop)
  else
    laia.log.fatal('Wrong checkpoint data in %q!', filename)
  end
end

local _cloneToCPU_typemap = {
  ['torch.CudaByteTensor'] = 'torch.ByteTensor',
  ['torch.CudaCharTensor'] = 'torch.CharTensor',
  ['torch.CudaShortTensor'] = 'torch.ShortTensor',
  ['torch.CudaIntTensor'] = 'torch.IntTensor',
  ['torch.CudaLongTensor'] = 'torch.LongTensor',
  ['torch.CudaTensor'] = 'torch.FloatTensor',
  ['torch.CudaDoubleTensor'] = 'torch.DoubleTensor',
  -- Data on the CPU does not change type.
  ['torch.ByteTensor'] = 'torch.ByteTensor',
  ['torch.CharTensor'] = 'torch.CharTensor',
  ['torch.ShortTensor'] = 'torch.ShortTensor',
  ['torch.IntTensor'] = 'torch.IntTensor',
  ['torch.LongTensor'] = 'torch.LongTensor',
  ['torch.FloatTensor'] = 'torch.FloatTensor',
  ['torch.DoubleTensor'] = 'torch.DoubleTensor'
}

function Checkpoint._cloneToCPU(obj, tensorCache)
  tensorCache = tensorCache or {}
  if torch.type(obj) == 'table' then
    local new_obj = {}
    for k, v in pairs(obj) do
      new_obj[k] = Checkpoint._cloneToCPU(v, tensorCache)
    end
    obj = new_obj
  elseif
    torch.isTypeOf(obj, 'nn.Module') or
    torch.isTypeOf(obj, 'nn.Criterion')
  then
    local new_obj = {}
    torch.setmetatable(new_obj, torch.typename(obj))
    for k, v in pairs(obj) do
      new_obj[k] = Checkpoint._cloneToCPU(v, tensorCache)
    end
    new_obj._type = _cloneToCPU_typemap[obj._type]
    obj = new_obj
  elseif torch.isTensor(obj) then
    if tensorCache[obj] then
      obj = tensorCache[obj]
    else
      local old_type = torch.typename(obj)
      local new_type = _cloneToCPU_typemap[old_type]
      local new_obj = torch.Tensor():type(new_type)
      if obj:storage() then
	local storage_type = new_type:gsub('Tensor', 'Storage')
	local storage_key = torch.pointer(obj:storage())
	if not tensorCache[storage_key] then
	  local new_storage = torch.getmetatable(storage_type).new()
	  if obj:storage():size() > 0 then
	    new_storage:resize(obj:storage():size()):copy(obj:storage())
	  end
	  tensorCache[storage_key] = new_storage
	end
	assert(torch.type(tensorCache[storage_key]) == storage_type)
	new_obj:set(
	  tensorCache[storage_key],
	  obj:storageOffset(),
	  obj:size(),
	  obj:stride()
	)
      end
      assert(torch.type(new_obj) == new_type)
      tensorCache[obj] = new_obj
      obj = new_obj
    end
  end
  return obj
end

return Checkpoint
