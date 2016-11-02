-- Extension functions for Lua tables.

-- Apply a function to each value of the given table and return a new table
-- with the result of the evaluation as the values.
--
-- Examples:
-- a = {1, 2, 3, 4}
-- table.map(a, function(x) return 2* x end)
-- {2, 4, 6, 8}
function table.map(t, fn)
  local res = {}
  for i, v in pairs(t) do
    res[i] = fn(v)
  end
  return res
end

-- Apply function of two arguments cumulatively to the items of the table,
-- so as to reduce the table to a single value.
--
-- If the table is an array, reduces the table from left to right, as
-- in foldl.
--
-- Examples:
-- a = {1, 2, 3, 4}
-- table.reduce(a, operator.add, 0)
-- 10
-- table.reduce(a, function(acc, x) return acc + x end, 0)
-- 10
function table.reduce(t, fn, v0)
  local res = v0
  for _, v in pairs(t) do
    res = fn(res, v)
  end
  return res
end

-- Apply function of two arguments cumulatively to the items of the table,
-- from left to right, so as to reduce the table to a single value.
--
-- Examples:
-- a = {1, 2, 3}
-- table.foldl(t, operator.sub, 5)
-- -1
table.foldl = table.reduce

-- Apply function of two arguments cumulatively to the items of the table,
-- from right to left, so as to reduce the table to a single value.
--
-- Examples:
-- a = {1, 2, 3}
-- table.foldr(t, operator.sub, 5)
-- -3
function table.foldr(t, fn, v0)
  local res = v0
  for i=#t,1,-1 do
    res = fn(t[i], res)
  end
  return res
end

-- Return true if all the values of a given table evaluate to true, or if the
-- table is empty. Optionally, a boolean function can be given as a second
-- argument to evaluate the table values.
--
-- Examples:
-- table.all({})
-- true
-- table.all({false, false})
-- false
-- table.all({true, true})
-- true
-- table.all({true, false})
-- false
-- table.all({2, 4, 6}, function(x) return x % 2 == 0 end)
-- true
function table.all(t, f)
  f = f or function(x) return x end
  return table.reduce(t, function(p, x) return p and f(x) end, true)
end

-- Return true if any of the values of a given table evaluate to true, or if the
-- table is empty. Optionally, a boolean function can be given as a second
-- argument to evaluate the table values.
--
-- Examples:
-- table.any({})
-- false
-- table.any({false, false})
-- false
-- table.any({true, true})
-- true
-- table.any({true, false})
-- true
-- table.any({2, 4, 6}, function(x) return x % 2 ~= 0 end)
-- false
function table.any(t, f)
  f = f or function(x) return x end
  return table.reduce(t, function(p, x) return p or f(x) end, false)
end

-- This function updates the destination (dst) table with the values from the
-- source (src) table. The third argument, can be used to add new keys to
-- the destination table (the default) or to ignore keys from the source
-- table that were not in the destination.
--
-- Notice that this function modifies the destination table, it does not
-- create a new table. Anyhow, the destination table is also returned.
--
-- Examples:
-- a = {a = 1, b = 2, c = 3}
-- table.update(a, {a = -1, d = -4})
-- {
--   a = -1,
--   b = 2,
--   c = 3,
--   d = -4
-- }
-- print(a)
-- {
--   a = -1,
--   b = 2,
--   c = 3,
--   d = -4
-- }
-- table.update(a, {d = 0, e = 5}, false)
-- {
--   a = -1,
--   b = 2,
--   c = 3,
--   d = 0
-- }
function table.update(dst, src, add)
  add = add == nil or add
  for k,v in pairs(src) do
    if add or dst[k] ~= nil then
      dst[k] = v
    end
  end
  return dst
end

-- Append the last element of a table array to the end. An optional argument
-- can be used to specify how many times the last element should be appended.
-- If the table is empty, does nothing.
-- This function modifies the input table, but returns it as well.
--
-- Examples:
-- a = {}
-- table.append_last(a)
-- {}
-- a = {1, 2, 3}
-- table.append_last(a)
-- {1, 2, 3, 3}
-- table.append_last(a, 2)
-- {1, 2, 3, 3, 3, 3}
-- print(a)
-- {1, 2, 3, 3, 3, 3}
function table.append_last(t, n)
  if #t == 0 then return t end
  n = n or 1
  for i=1,n do
    table.insert(t, t[#t])
  end
  return t
end

-- Given a table where values are positive (or zero) weights, select a random
-- key with a probability proportional to its value. An optional argument can
-- be passed with the sum of the values precomputed, otherwise the function
-- will compute it.
--
-- If the table is empty will nil will be returned. If all weights are 0, the
-- distribution is assumed to be uniform.
--
-- Examples:
-- table.weighted_choice({})
-- nil
-- table.weighted_choice({0.0, 1.0})
-- 2
-- table.weighted_choice({a = 0.0, b = 1.0, c = 0.0})
-- b
function table.weighted_choice(t, z)
  assert(table.all(t, function(x) return type(x) == 'number' and x >= 0 end),
	 'All values in the input table must be greater than or equal to 0!')
  z = z or table.reduce(t, function(acc, x) return acc + x end, 0.0)
  if z > 0 then
    -- The weights in the table define a valid probability distribution.
    local cut_likelihood = math.random() * z
    local cum_likelihood = 0
    for k, w in pairs(t) do
      if cum_likelihood + w >= cut_likelihood then
	return k
      end
      cum_likelihood = cum_likelihood + w
    end
    error('This point should not be reached.')
  else
    -- Reservoir sampling, when all weights are 0.0 or the table contains
    -- no elements.
    local n = 0
    local k = nil
    local i = nil
    while true do
      i = next(t, i)
      if i == nil then
	return k
      end
      n = n + 1
      if math.random() < 1.0 / n then
	k = i
      end
    end
  end
end
