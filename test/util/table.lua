require 'laia.util.table'

local testsuite = torch.TestSuite()
local tester = torch.Tester()

function testsuite.map()
  local f = function(x) return 2 * x end
  tester:eq(table.map({}, f), {})
  tester:eq(table.map({1, 2, 3, 4}, f), {2, 4, 6, 8})
  tester:eq(table.map({a = 1, b = 2, c = 3, d = 4}, f),
	    {a = 2, b = 4, c = 6, d = 8})
end

function testsuite.reduce()
  local f = function(acc, x) return acc + x end
  tester:eq(table.reduce({}, f, 99), 99)
  tester:eq(table.reduce({1, 2, 3, 4}, f, 0), 10)
  tester:eq(table.reduce({a = 1, b = 2, c = 3, d = 4}, f, 0), 10)
end

function testsuite.foldl()
  local f = function(x, y) return x - y end
  tester:eq(table.foldl({}, f, 99), 99)
  tester:eq(table.foldl({1, 2, 3}, f, 5), -1)
end

function testsuite.foldr()
  local f = function(x, y) return x - y end
  tester:eq(table.foldr({}, f, 99), 99)
  tester:eq(table.foldr({1, 2, 3}, f, 5), -3)
end

function testsuite.all()
  tester:eq(table.all({}), true)
  tester:eq(table.all({false, false}), false)
  tester:eq(table.all({true, true}), true)
  tester:eq(table.all({true, false}), false)
  local f = function(x) return x % 2 == 0 end
  tester:eq(table.all({2, 4, 6}, f), true)
  tester:eq(table.all({2, 5, 6}, f), false)
  tester:eq(table.all({1, 3, 7}, f), false)
end

function testsuite.any()
  tester:eq(table.any({}), false)
  tester:eq(table.any({false, false}), false)
  tester:eq(table.any({true, true}), true)
  tester:eq(table.any({true, false}), true)
  local f = function(x) return x % 2 == 0 end
  tester:eq(table.any({2, 4, 6}, f), true)
  tester:eq(table.any({2, 5, 6}, f), true)
  tester:eq(table.any({1, 3, 7}, f), false)
end

function testsuite.update()
  local a = {a = 1, b = 2, c = 3}
  tester:eq(table.update(a, {a = -1, d = -4}),
	    { a = -1, b = 2, c = 3, d = -4 })
  tester:eq(a, { a = -1, b = 2, c = 3, d = -4 })
  tester:eq(table.update(a, {d = 0, e = 5}, false),
	    { a = -1, b = 2, c = 3, d = 0 })
  tester:eq(a, { a = -1, b = 2, c = 3, d = 0 })
end

function testsuite.append_last()
  local a = {}
  tester:eq(table.append_last(a), {})
  tester:eq(a, {})
  local b = {1, 2, 3}
  tester:eq(table.append_last(b), {1, 2, 3, 3})
  tester:eq(b, {1, 2, 3, 3})
  tester:eq(table.append_last(b, 2), {1, 2, 3, 3, 3, 3})
  tester:eq(b, {1, 2, 3, 3, 3, 3})
end

function testsuite.weighted_choice()
  tester:eq(table.weighted_choice({}), nil)
  tester:eq(table.weighted_choice({0.0, 1.0, 0.0}), 2)
  tester:eq(table.weighted_choice({a = 0.0, b = 99.0, c = 0.0}), 'b')

  -- Function to perform Chi-Square test
  local function chi2(t, reps, expected)
    local z = table.reduce(t, function(x, y) return x + y end, 0)
    local nk = table.map(t, function(_) return 0 end)
    local ek = expected or table.map(t, function(x) return x * reps end)
    for r=1,reps do
      local k = table.weighted_choice(t, z)
      nk[k] = nk[k] + 1
    end
    -- Compute Chi-Squared statistic
    local s = 0
    for k, e in pairs(ek) do
      local d = nk[k] - e
      s = s + (d * d) / e
    end
    return s
  end

  -- 11.345 is the critical value of the Chi-Squared at 99% with 3 degrees
  -- of freedom. If this fails, we a 99% sure that the weighted_choice is
  -- broken.
  local thr = 11.345
  tester:assertlt(chi2({0.1, 0.2, 0.3, 0.4}, 10000), thr)
  tester:assertlt(chi2({a = 0.1, b = 0.2, c = 0.3, d = 0.4}, 10000), thr)
  tester:assertlt(chi2({0, 0, 0, 0}, 10000, {2500, 2500, 2500, 2500}), thr)
  tester:assertlt(chi2(
		    {a = 0, b = 0, c = 0, d = 0}, 10000,
		    {a = 2500, b = 2500, c = 2500, d = 2500}),
		  thr)
end

tester:add(testsuite)
tester:run()
