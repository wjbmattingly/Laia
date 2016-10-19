require 'laia.util.math'

local testsuite = torch.TestSuite()
local tester = torch.Tester()


function testsuite.isnan()
  tester:eq(math.isnan(0 / 0), true)
  tester:eq(math.isnan(1 / 0), false)
  tester:eq(math.isnan(- 1 / 0), false)
  tester:eq(math.isnan(1.2), false)
end

function testsuite.isinf()
  tester:eq(math.isinf(0 / 0), false)
  tester:eq(math.isinf(1 / 0), true)
  tester:eq(math.isinf(- 1 / 0), true)
  tester:eq(math.isinf(1.2), false)
end

function testsuite.isfinite()
  tester:eq(math.isfinite(0 / 0), false)
  tester:eq(math.isfinite(1 / 0), false)
  tester:eq(math.isfinite(- 1 / 0), false)
  tester:eq(math.isfinite(1.2), true)
end

tester:add(testsuite)
tester:run()
