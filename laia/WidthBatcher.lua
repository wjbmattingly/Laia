local WidthBatcher, Parent = torch.class('laia.WidthBatcher',
					  'laia.CachedBatcher')

function WidthBatcher:__init(opt)
  Parent.__init(self, opt)
end

function WidthBatcher:load(img_list, gt_file, symbols_table)
  Parent.load(self, img_list, gt_file, symbols_table)
  self:sort()
end

function WidthBatcher:sort()
  local aux = {}
  for i=1,self._num_samples do
    local img = image.load(self._imglist[i], self._channels)
    local h, w = img:size(2), img:size(3)
    table.insert(aux, {h, w, self._samples[i], self._imglist[i]})
  end
  table.sort(aux, function(a, b) return a[2] < b[2] end)
  for i=1,#aux do
    self._samples[i] = aux[i][3]
    self._imglist[i] = aux[i][4]
  end
  self._idx = 0
  self:clearCache()
end

function WidthBatcher:epochReset(epoch_opt)
  self._idx = 0
end

return WidthBatcher
