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
   Print a histogram of the absolute values of the elements in a tensor.
--]]
torch.loghistc = function(x, bins)
   local function num_in_range(x, a, b)
      return torch.cmul(torch.ge(x, a), torch.lt(x, b)):sum()
   end
   x = torch.abs(x)
   local xmin = torch.min(x)
   local xmax = torch.max(x)


   bins = bins or 10
   local n = x:storage():size()
   return string.format('[0, 1E-6) = %.2f\n[1E-6,1E-4) = %.2f\n[1E-4,1E-2) = %.2f\n[1E-2,1E0) = %.2f\n' ..
			'[1E0,1E+1) = %.2f\n[1E+1,1E+2) = %.2f\n[1E+2,inf) = %.2f',
			100 * num_in_range(x, 0, 1E-6) / n,
			100 * num_in_range(x, 1E-6, 1E-4) / n,
			100 * num_in_range(x, 1E-4, 1E-2) / n,
			100 * num_in_range(x, 1E-2, 1) / n,
			100 * num_in_range(x, 1, 10) / n,
			100 * num_in_range(x, 10, 100) / n,
			100 * num_in_range(x, 100, 100000000) / n)
end
