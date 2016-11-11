require ('laia.util.base')

-- Set torch and cutorch manual seed
function laia.manualSeed(seed)
  torch.manualSeed(seed)
  if cutorch then cutorch.manualSeed(seed) end
end

-- Get torch/cutorch RNG State
function laia.getRNGState()
  local state = {}
  state.torch = torch.getRNGState()
  if cutorch then state.cutorch = cutorch.getRNGState() end
  return state
end

-- Set torch/cutorch RNG State
function laia.setRNGState(state)
  if not state then return end
  if state.torch then
    torch.setRNGState(state.torch)
  else
    laia.log.error('No torch RNG state found!')
  end
  if cutorch then
    if state.cutorch then
      cutorch.setRNGState(state.cutorch)
    else
      laia.log.error('No cutorch RNG state found!')
    end
  end
end

-- Sample from a von Mises distribution, using rejection sampling
-- See: https://en.wikipedia.org/wiki/Von_Mises_distribution
--      https://en.wikipedia.org/wiki/Rejection_sampling
function laia.rand_von_Mises(m, k)
  local function lpdf(x, m, k)
    return k * math.cos(x - m) - math.log(2 * math.pi) - laia.log_bessel_I0(k)
  end
  local max, x = lpdf(m, m, k)
  repeat
    x = math.pi * (2.0 * torch.uniform() - 1.0)
  until math.log(torch.uniform()) + max < lpdf(x, m, k)
  return x
end

-- Compute log of the modified Bessel function of order 0, to avoid problems
-- with large values of x.
-- Approximation from: http://www.advanpix.com/2015/11/11/rational-approximations-for-the-modified-bessel-function-of-the-first-kind-i0-computations-double-precision
function laia.log_bessel_I0(x)
  -- Compute polynomial of degree n with coefficients stored in a
  -- y = \sum_{i=0}^n a_i x^i
  local function Poly(a, x)
    local y = a[1]
    for i=2,#a do y = y + a[i] * math.pow(x, i - 1) end
    return y
  end
  local P = {
    1.0000000000000000000000801e+00,
    2.4999999999999999999629693e-01,
    2.7777777777777777805664954e-02,
    1.7361111111111110294015271e-03,
    6.9444444444444568581891535e-05,
    1.9290123456788994104574754e-06,
    3.9367598891475388547279760e-08,
    6.1511873265092916275099070e-10,
    7.5940584360755226536109511e-12,
    7.5940582595094190098755663e-14,
    6.2760839879536225394314453e-16,
    4.3583591008893599099577755e-18,
    2.5791926805873898803749321e-20,
    1.3141332422663039834197910e-22,
    5.9203280572170548134753422e-25,
    2.0732014503197852176921968e-27,
    1.1497640034400735733456400e-29
  }
  local Q = {
    3.9894228040143265335649948e-01,
    4.9867785050353992900698488e-02,
    2.8050628884163787533196746e-02,
    2.9219501690198775910219311e-02,
    4.4718622769244715693031735e-02,
    9.4085204199017869159183831e-02,
   -1.0699095472110916094973951e-01,
    2.2725199603010833194037016e+01,
   -1.0026890180180668595066918e+03,
    3.1275740782277570164423916e+04,
   -5.9355022509673600842060002e+05,
    2.6092888649549172879282592e+06,
    2.3518420447411254516178388e+08,
   -8.9270060370015930749184222e+09,
    1.8592340458074104721496236e+11,
   -2.6632742974569782078420204e+12,
    2.7752144774934763122129261e+13,
   -2.1323049786724612220362154e+14,
    1.1989242681178569338129044e+15,
   -4.8049082153027457378879746e+15,
    1.3012646806421079076251950e+16,
   -2.1363029690365351606041265e+16,
    1.6069467093441596329340754e+16
  }
  local ax = math.abs(x)
  if ax < 7.75 then
    local z = ax * ax * 0.25
    return math.log(z * Poly(P, z) + 1)
  else
    return math.log(Poly(Q, 1.0 / ax)) + ax - 0.5 * math.log(ax)
  end
end
