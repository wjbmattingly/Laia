require 'laia.EpochCheckpoint'

local Checkpoint = torch.class('laia.Checkpoint')

-- Utility class to save/load Laia checkpoints from/to disk.
--
-- Checkpoints store information about (among other things):
-- Best / last epoch checkpoints
-- RNG state
-- Training state
-- Model commandline parameters (arguments and options passed to create_model)
-- Training commandline parameters (arguments and options passed to train_ctc)
-- etc
--
-- Next, an example of how to use this class to keep checkpoints of the
-- training process and when the model is improved.
--
-- local checkpoint = laia.Checkpoint()
-- checkpoint:load('checkpoint.t7')
-- local model = checkpoint:Last():getModel():cuda()
-- local best_cer = checkpoint:Best():getSummary('valid').cer
-- for epoch=(checkpoint:Last():getEpoch()+1),100 do
--   TrainEpoch(model, ...)
--   local summary = ComputeSummary(model, ...)
--
--   -- If the accuracy improves, make a checkpoint of the model.
--   if best_cer == nil or best_cer > summary.cer then
--     checkpoint:Best():setEpoch(epoch)
--     checkpoint:Best():setModel(model)
--     checkpoint:Best():addSummary('valid', summary)
--     best_cer = summary.cer
--   end
--
--   -- Save the model to disk when it improves (getEpochBest() == epoch) or
--   -- every 10 epochs (epoch % 10 == 0).
--   if checkpoint:Best():getEpoch() == epoch or epoch % 10 == 0 then
--     checkpoint:Last():setEpoch(epoch)
--     checkpoint:Last():setModel(model)
--     checkpoint:Last():addSummary('valid', summary)
--     checkpoint:save('checkpoint.t7')
--   end
-- end

function Checkpoint:__init()
  self._best = laia.EpochCheckpoint()
  self._last = laia.EpochCheckpoint()
end

function Checkpoint:clear()
  for k, _ in pairs(self) do
    self[k] = nil
  end
  self._best = laia.EpochCheckpoint()
  self._last = laia.EpochCheckpoint()
end

function Checkpoint:Best()
  return self._best
end
function Checkpoint:Last()
  return self._last
end

-- Getter/Setter for the model config (i.e: arguments of create_model.lua)
function Checkpoint:getModelConfig()
  return self._model_config
end
function Checkpoint:setModelConfig(cfg)
  self._model_config = cfg
  return self
end

-- Getter/Setter for the model config (i.e: arguments of train.lua)
function Checkpoint:getTrainConfig()
  return self._train_config
end
function Checkpoint:setTrainConfig(cfg)
  self._train_config = cfg
  return self
end

-- Getter/Setter for RMSPropState
function Checkpoint:getRMSPropState()
  return self._rmsprop
end
function Checkpoint:setRMSPropState(rmsprop)
  self._rmsprop = rmsprop
  return self
end

-- Getter/Setter for RNGState
function Checkpoint:getRNGState()
  return self._rngstate
end
function Checkpoint:setRNGState(rngstate)
  self._rngstate = rngstate
  return self
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
    self:Best():setEpoch(0):setModel(checkpoint)
    self:Last():setEpoch(0):setModel(checkpoint)
  elseif torch.type(checkpoint) == 'table' then
    self:setModelConfig(checkpoint.model_opt)
    self:setTrainConfig(checkpoint.train_opt)
    self:setRNGState(checkpoint.rng_state)
    self:setRMSPropState(checkpoint.rmsprop)
    self:Best():setEpoch(checkpoint.epoch or 0):setModel(checkpoint.model)
    self:Last():setEpoch(checkpoint.epoch or 0):setModel(checkpoint.model)
  else
    laia.log.fatal('Wrong checkpoint data in %q!', filename)
  end
  return self
end

return Checkpoint
