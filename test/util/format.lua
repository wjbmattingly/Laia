require 'laia.util.format'

local testsuite = torch.TestSuite()
local tester = torch.Tester()

function testsuite.sec_to_dhms()
  tester:eq(laia.sec_to_dhms(2.5), '2s')
  tester:eq(laia.sec_to_dhms(5), '5s')
  tester:eq(laia.sec_to_dhms(59), '59s')
  tester:eq(laia.sec_to_dhms(60), '1m')
  tester:eq(laia.sec_to_dhms(61), '1m1s')
  tester:eq(laia.sec_to_dhms(3599), '59m59s')
  tester:eq(laia.sec_to_dhms(3600), '1h')
  tester:eq(laia.sec_to_dhms(3600 + 60), '1h1m')
  tester:eq(laia.sec_to_dhms(3600 + 59), '1h59s')

  tester:eq(laia.sec_to_dhms(86399), '23h59m59s')
  tester:eq(laia.sec_to_dhms(86400), '1d')
  tester:eq(laia.sec_to_dhms(86401), '1d1s')

  tester:eq(laia.sec_to_dhms(86400 + 3600 + 60 + 1), '1dh1h1m1s')



  --tester:eq(laia.sec_to_dhms(3601), '1m1s')

end

tester:add(testsuite)
tester:run()
