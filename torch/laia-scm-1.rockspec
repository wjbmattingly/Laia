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
  license = "TBD"
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
  "luaposix"
}

build = {
  type = "builtin",
  modules = {
    ["src.CachedBatcher"] = "src/CachedBatcher.lua",
    ["src.CurriculumBatcher"] = "src/CurriculumBatcher.lua",
    ["src.DecodeOptions"] = "src/DecodeOptions.lua",
    ["src.ImageDistorter"] = "src/ImageDistorter.lua",
    ["src.MDRNN"] = "src/MDRNN.lua",
    ["src.Model-VGG_A"] = "src/Model-VGG_A.lua",
    ["src.RandomBatcher"] = "src/RandomBatcher.lua",
    ["src.TrainOptions"] = "src/TrainOptions.lua",
    ["src.WidthBatcher"] = "src/WidthBatcher.lua",
    ["src.utilities"] = "src/utilities.lua"
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
