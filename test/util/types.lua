require 'laia.util.types'

local testsuite = torch.TestSuite()
local tester = torch.Tester()

function testsuite.isboolean()
  tester:assert(laia.isboolean(true))
  tester:assert(laia.isboolean(false))
  tester:assert(not laia.isboolean(0))
  tester:assert(not laia.isboolean(1))
end

function testsuite.isint()
  tester:assert(laia.isint(0))
  tester:assert(laia.isint(5))
  tester:assert(laia.isint(-5))
  tester:assert(not laia.isint(-5.5))
  tester:assert(not laia.isint(5.5))
end

function testsuite.toboolean()
  tester:eq(laia.toboolean(false), false)
  tester:eq(laia.toboolean(true), true)
  tester:eq(laia.toboolean(0), false)
  tester:eq(laia.toboolean(1), true)
  tester:eq(laia.toboolean('1'), true)
  tester:eq(laia.toboolean('T'), true)
  tester:eq(laia.toboolean('true'), true)
  tester:eq(laia.toboolean('TRUE'), true)
  tester:eq(laia.toboolean('0'), false)
  tester:eq(laia.toboolean('F'), false)
  tester:eq(laia.toboolean('false'), false)
  tester:eq(laia.toboolean('FALSE'), false)
  tester:eq(laia.toboolean(''), nil)
  tester:eq(laia.toboolean('foo'), nil)
  tester:eq(laia.toboolean(-99), nil)
  tester:eq(laia.toboolean(99), nil)
  tester:eq(laia.toboolean({}), nil)
end

function testsuite.toint()
  tester:eq(laia.toint(25), 25)
  tester:eq(laia.toint('25'), 25)
  tester:eq(laia.toint('-99'), -99)
  tester:eq(laia.toint(''), nil)
  tester:eq(laia.toint('foo'), nil)
  tester:eq(laia.toint({}), nil)
  tester:eq(laia.toint(false), nil)
  tester:eq(laia.toint(true), nil)
end

function testsuite.tolistnum()
  tester:eq(laia.tolistnum(''), {})
  tester:eq(laia.tolistnum('1'), {1})
  tester:eq(laia.tolistnum('1,-2,3'), {1, -2, 3})
  tester:eq(laia.tolistnum('1.3,-2.2,3.1'), {1.3, -2.2, 3.1})
  -- Wrong formats for list of numbers
  tester:eq(laia.tolistnum('1 2 3'), nil)
  tester:eq(laia.tolistnum('1,2 3'), nil)
end

function testsuite.tolistint()
  tester:eq(laia.tolistint(''), {})
  tester:eq(laia.tolistint('1'), {1})
  tester:eq(laia.tolistint('1,-2,3'), {1, -2, 3})
  -- Wrong formats for list of INTEGERS
  tester:eq(laia.tolistint('1.3,-2.2,3.1'), nil)
  tester:eq(laia.tolistint('1 2 3'), nil)
  tester:eq(laia.tolistint('1,2 3'), nil)
end

tester:add(testsuite)
tester:run()
