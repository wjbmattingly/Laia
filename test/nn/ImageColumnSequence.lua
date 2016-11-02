require 'nn'

-- Require only the strictly necessary, not the whole laia package.
laia = { nn = {} }
require 'laia.nn.ImageColumnSequence'

local testsuite = torch.TestSuite()
local tester = torch.Tester()

function testsuite.forward()
  local N, C, H, W = 2, 3, 5, 4
  local x = torch.rand(N, C, H, W)
  local y = laia.nn.ImageColumnSequence():forward(x)
  tester:eq(y:size(1), W)
  tester:eq(y:size(2), N)
  tester:eq(y:size(3), H * C)
  for j=1,W do
    for n=1,N do
      local y_nj = y:sub(j, j, n, n, 1, H * C):squeeze()
      local x_nj = x:sub(n, n, 1, C, 1, H, j, j):squeeze():t():contiguous()
      tester:eq(x_nj:view(H * C), y_nj)
    end
  end
end

function testsuite.backward()
  local N, C, H, W = 2, 3, 5, 4
  local m = laia.nn.ImageColumnSequence()
  local x = torch.rand(N, C, H, W)
  local y = m:forward(x)
  -- Backward the output, so that the dx must be equal to x.
  local dx = m:backward(x, y)
  tester:eq(x, dx)
end

function testsuite.check_nn_equivalent()
  local N, C, H, W = 2, 3, 5, 4
  local x = torch.rand(N, C, H, W)
  local dy = torch.rand(W, N , H * C)
  local m1 = laia.nn.ImageColumnSequence()
  local m2 = nn.Sequential()
    :add(nn.Transpose({2,4},{1,2}))
    :add(nn.Contiguous())
    :add(nn.Reshape(-1, H * C, true))
  tester:eq(m1:forward(x), m2:forward(x))
  tester:eq(m1:backward(x, dy), m2:backward(x, dy))
end

tester:add(testsuite)
tester:run()
