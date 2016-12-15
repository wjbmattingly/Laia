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
  license = "MIT",
  maintainer = [[
  Joan Puigcerver <joapuipe@upv.es>
  Dani Martin-Albo <damarsi1@upv.es>
  Mauricio Villegas <mauvilsa@upv.es>
  ]]
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
    ["laia.AdversarialRegularizer"] = "laia/AdversarialRegularizer.lua",
    ["laia.CachedBatcher"] = "laia/CachedBatcher.lua",
    ["laia.Checkpoint"] = "laia/Checkpoint.lua",
    ["laia.ClassWithOptions"] = "laia/ClassWithOptions.lua",
    ["laia.CTCTrainer"] = "laia/CTCTrainer.lua",
    ["laia.CurriculumBatcher"] = "laia/CurriculumBatcher.lua",
    ["laia.EpochCheckpoint"] = "laia/EpochCheckpoint.lua",
    ["laia.EpochSummarizer"] = "laia/EpochSummarizer.lua",
    ["laia.ImageDistorter"] = "laia/ImageDistorter.lua",
    ["laia.Monitor"] = "laia/Monitor.lua",
    ["laia.ProgressTable"] = "laia/ProgressTable.lua",
    ["laia.RandomBatcher"] = "laia/RandomBatcher.lua",
    ["laia.SignalHandler"] = "laia/SignalHandler.lua",
    ["laia.Statistics"] = "laia/Statistics.lua",
    ["laia.Version"] = "laia/Version.lua",
    ["laia.WeightDecayRegularizer"] = "laia/WeightDecayRegularizer.lua",
    ["laia.nn.ImageColumnSequence"] = "laia/nn/ImageColumnSequence.lua",
    ["laia.nn.MDRNN"] = "laia/nn/MDRNN.lua",
    ["laia.util.argparse"] = "laia/util/argparse.lua",
    ["laia.util.base"] = "laia/util/base.lua",
    ["laia.util.cudnn"] = "laia/util/cudnn.lua",
    ["laia.util.decode"] = "laia/util/decode.lua",
    ["laia.util.format"] = "laia/util/format.lua",
    ["laia.util.io"] = "laia/util/io.lua",
    ["laia.util.log"] = "laia/util/log.lua",
    ["laia.util.math"] = "laia/util/math.lua",
    ["laia.util.rand"] = "laia/util/rand.lua",
    ["laia.util.string"] = "laia/util/string.lua",
    ["laia.util.table"] = "laia/util/table.lua",
    ["laia.util.torch"] = "laia/util/torch.lua",
    ["laia.util.types"] = "laia/util/types.lua"
  },
  install = {
    bin = {
      "laia-create-model",
      "laia-decode",
      "laia-force-align",
      "laia-netout",
      "laia-train-ctc"
    }
  }
}
