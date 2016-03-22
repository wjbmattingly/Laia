--[[
   a = {1, 2, 3, 4}
   table.reduce(a, function(x) return 2* x; end)
   {2, 4, 6, 8}
--]]
table.map = function(t, fn)
   local t2 = {}
   for i, v in pairs(t) do
      t2[i] = fn(v)
   end
   return t2
end

--[[
   a = {1, 2, 3, 4}
   table.reduce(a, operator.add, 0)
   10
--]]
table.reduce = function(t, fn, v0)
   local res = v0
   for _, v in pairs(t) do
      res = fn(res, v)
   end
   return res
end

--[[
   Compute a histogram of the absolute values of the elements in a tensor,
   using a logarithmic scale.
   The function returs first a table containing the histogram (the number of
   elements in each bin), and secondly a table containing the thresholds for
   each bin.
--]]
torch.loghistc = function(x, nbins, bmin, bmax)
   x = torch.abs(x)
   nbins = nbins or 10
   bmin = bmin or torch.min(x)
   bmax = bmax or torch.max(x)
   if bmin == 0.0 then bmin = 1E-16 end
   assert(nbins > 1)
   assert(bmax > bmin)
   local hist = { }
   local bins = { }
   local step = math.exp(math.log(bmax / bmin) / (nbins - 1))
   table.insert(bins, bmin)
   table.insert(hist, torch.le(x, bmin):sum())
   for i=1,(nbins-1) do
      local pb = bmin * math.pow(step, i - 1)
      local cb = pb * step
      table.insert(bins, cb)
      table.insert(hist, torch.cmul(torch.gt(x, pb), torch.le(x, cb)):sum())
   end
   -- Due to numerical errors, we may miss some elements in the last bin.
   local cumhist = table.reduce(hist, operator.add, 0)
   local missing = x:storage():size() - cumhist
   if missing > 0 then
      hist[#hist] = hist[#hist] + missing
   end
   return hist, bins
end

torch.sumarizeMagnitudes = function(x, mass, nbins, bmin, bmax, log_scale)
   mass = mass or 0.75
   log_scale = log_scale or true
   local hist, bins
   if log_scale then
      hist, bins = torch.loghistc(x, nbins, bmin, bmax)
   else
      x = torch.abs(x)
      bmin = bmin or torch.min(x)
      bmax = bmax or torch.max(x)
      hist = torch.totable(torch.histc(x, nbins, bmin, bmax))
      bins = torch.totable(torch.range(bmin, bmax, (bmax - bmin) / (nbins - 1)))
   end
   local n = x:storage():size()
   local aux = {}
   for i=1,#hist do table.insert(aux, {i, hist[i]})  end
   table.sort(aux,
	      function(a, b)
		 return a[2] > b[2] or (a[2] == b[2] and a[1] < b[1])
              end)
   local cum = 0
   local mini = #aux
   local maxi = 0
   for i=1,#aux do
      cum = cum + aux[i][2]
      if mini > aux[i][1] then mini = aux[i][1] end
      if maxi < aux[i][1] then maxi = aux[i][1] end
      if cum >= mass * n then break end
   end
   if mini < 2 then
      return torch.min(torch.abs(x)), bins[maxi], cum / n
   else
      return bins[mini - 1], bins[maxi], cum / n
   end
end
