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
  "imgdistort"
}

build = {
  type = "builtin",
  modules = {
    ["laia"] = "laia/init.lua",
    ["laia.argparse"] = "laia/argparse.lua",
    ["laia.CachedBatcher"] = "laia/CachedBatcher.lua",
    ["laia.CurriculumBatcher"] = "laia/CurriculumBatcher.lua",
    ["laia.DecodeOptions"] = "laia/DecodeOptions.lua",
    ["laia.ImageDistorter"] = "laia/ImageDistorter.lua",
    ["laia.log"] = "laia/log.lua",
    ["laia.MDRNN"] = "laia/MDRNN.lua",
    ["laia.Monitor"] = "laia/Monitor.lua",
    ["laia.RandomBatcher"] = "laia/RandomBatcher.lua",
    ["laia.TrainOptions"] = "laia/TrainOptions.lua",
    ["laia.utilities"] = "laia/utilities.lua"
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
