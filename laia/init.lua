require 'torch'
require 'cutorch'
require 'nn'
require 'cudnn'

laia = {}
laia.log = require('laia.log')
laia.log.loglevel = 'warn'

require('laia.utilities')
require('laia.CachedBatcher')
require('laia.RandomBatcher')
require('laia.ImageDistorter')
require('laia.MDRNN')
require('laia.Monitor')
require('laia.NCHW2HND')

return laia
