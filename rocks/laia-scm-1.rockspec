package = "Laia"
version = "scm-1"

source = {
  url = "git://github.com/jpuigcerver/Laia.git",
}

description = {
  summary = "Laia: A deep learning toolkit for HTR based on Torch",
  detailed = [[
  ]],
  homepage = "https://github.com/jpuigcerver/Laia",
  license = "MIT"
}

dependencies = {
  "torch >= 7.0",
  "cutorch",
  "nn",
  "cunn",
  "cudnn",
  "warp-ctc",
  "optim",
  "xlua",
  "lua-term",
  "luaposix",
  "lbase64",
  "etlua",
  "imgdistort"
}

build = {
  type = "builtin",
  modules = {
    ["laia"] = "laia/init.lua",
    ["laia.argparse"] = "laia/argparse.lua",
    ["laia.log"] = "laia/log.lua",
    ["laia.utilities"] = "laia/utilities.lua",
    ["laia.AdversarialRegularizer"] = "laia/AdversarialRegularizer.lua",
    ["laia.CachedBatcher"] = "laia/CachedBatcher.lua",
    ["laia.CTCTrainer"] = "laia/CTCTrainer.lua",
    ["laia.CurriculumBatcher"] = "laia/CurriculumBatcher.lua",
    ["laia.DecodeOptions"] = "laia/DecodeOptions.lua",
    ["laia.ImageDistorter"] = "laia/ImageDistorter.lua",
    ["laia.Monitor"] = "laia/Monitor.lua",
    ["laia.RandomBatcher"] = "laia/RandomBatcher.lua",
    ["laia.Regularizer"] = "laia/Regularizer.lua",
    ["laia.Statistics"] = "laia/Statistics.lua",
    ["laia.TrainOptions"] = "laia/TrainOptions.lua",
    ["laia.WeightDecayRegularizer"] = "laia/WeightDecayRegularizer.lua",
    ["laia.nn.MDRNN"] = "laia/nn/MDRNN.lua",
    ["laia.nn.NCHW2WND"] = "laia/nn/NCHW2WND.lua"
  },
  install = {
    bin = {
      "create_model.lua",
      "train.lua",
      "decode.lua",
      "netout.lua"
    }
  }
}
