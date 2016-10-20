require 'laia.util.string'

local testsuite = torch.TestSuite()
local tester = torch.Tester()

function testsuite.split()
  tester:eq(string.split('hello my friend'), {'hello', 'my', 'friend'})
  tester:eq(string.split('  hello   my  friend  '), {'hello', 'my', 'friend'})
  tester:eq(string.split('hello,my,friend', '[^,]+'), {'hello', 'my', 'friend'})
  tester:eq(string.split(',,,hello  ,  my  ,  friend,,,', '[^,%s]+'),
	    {'hello', 'my', 'friend'})
end

tester:add(testsuite)
tester:run()
