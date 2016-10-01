local image = require 'image'
local base64 = require 'base64'
local ffi = require 'ffi'
local etlua = require 'etlua'

laia = laia or {}
local Monitor = torch.class('laia.Monitor')

-- This table contains for each module (e.g. nn.SpatialConvolution) the
-- function that will generate the corresponding HTML file.
Monitor._htmlModule = {}

Monitor._HTMLTemplate = etlua.compile([[
<!DOCTYPE html>
<html>
<head>
  <title>Laia Monitor</title>
  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <link rel="stylesheet" href="https://code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css" type="text/css">
  <script src="https://code.jquery.com/jquery-1.12.4.js" type="text/javascript"></script>
  <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js" type="text/javascript"></script>
  <script src="https://cdn.plot.ly/plotly-latest.min.js" type="text/javascript"></script>
  <script src="http://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.js"></script>
  <link rel="stylesheet" href="http://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.css" />
  <script type="text/javascript">
    $(function() {
      /* Progress plots ... */
      $("#plots").tabs({collapsible: true});
      <% for pn, pv in pairs(plots) do -%>
      var data_<%= pn %> = [
      <% for cn, cv in pairs(pv.curves) do -%>
        { x: [ <% for x,_ in ipairs(cv.data) do %><%= x %>, <% end %> ],
          y: [ <% for _,y in ipairs(cv.data) do %><%= y %>, <% end %> ],
          mode: 'lines+markers',
          <% if cv.name then -%>
          name: '<%= cv.name %>'
          <% else -%>
          name: '<%= cn %>'
          <% end -%>
        },
      <% end -%>
      ];
      var layout_<%= pn %> = {
        width: 700,
        height: 450,
        <% if pv.xlabel then -%>
        xaxis : { title: '<%= pv.xlabel %>' },
        <% end %>
        <% if pv.ylabel then -%>
        yaxis : { title: '<%= pv.ylabel %>' },
        <% end -%>
      };
      Plotly.newPlot('plot-<%= pn %>', data_<%= pn %>, layout_<%= pn %>);
      <% end -%>
    });
  </script>
</head>
<body>
  <div id="plots">
    <ul>
    <% for pn, pv in pairs(plots) do -%>
      <li><a href="#plot-<%= pn %>"><%= pv.name or pn %></a></li>
    <% end -%>
    </ul>
    <% for pn, _ in pairs(plots) do -%>
    <div id="plot-<%= pn %>"></div>
    <% end -%>
  </div>
</body>
</html>
]])
assert(Monitor._HTMLTemplate ~= nil, 'Wrong HTML template!')

Monitor._HTMLTemplate_Snapshot = etlua.compile([[

]])

function Monitor:__init(output_file, plots)
  self._output_file = output_file
  self._plots = plots or {}
  for pn, pl in pairs(self._plots) do
    assert(pl.curves ~= nil and next(pl.curves) ~= nil,
	   string.format('Plot %q does not contain any curve!', pn))
    for cn, cv in pairs(pl.curves) do
      if not cv.data then cv.data = {} end
    end
  end
end

function Monitor._ScaleImage(x)
  assert(x:nDimension() == 4)
  local N, C, H, W = x:size(1), x:size(2), x:size(3), x:size(4)
  assert(C == 1 or C == 3)
  local mD = math.max(H, W)
  if mD < 32 then
    local nW = math.ceil(W * (32 / mD))
    local nH = math.ceil(H * (32 / mD))
    local x2 = torch.FloatTensor(N, C, nH, nW)
    for n=1,N do image.scale(x2[n], x[n]) end
    return x2
  else
     return x
  end
end

function Monitor._NCHWImage(x)
  assert(x:nDimension() == 4)
  local N = x:size(1)
  local C = x:size(2)
  local H = x:size(3)
  local W = x:size(4)
  if C ~= 1 and C ~= 3 then
    -- Convert from NxCxHxW to (NC)x1xHxW, to support other number of channels
    -- than 1 or 3.
    x = x:view(N * C, 1, H, W)
  end
  return image.toDisplayTensor{input=Monitor._ScaleImage(x), nrow=C, padding=2}
end

function Monitor._CHWImage(x)
  assert(x:nDimension() == 3)
  local C = x:size(1)
  local H = x:size(2)
  local W = x:size(3)
  if C ~= 1 and C ~= 3 then
    -- Convert from CxHxW to Cx1xHxW, to support other number of channels
    -- than 1 or 3.
    x = x:view(C, 1, H, W)
  end
  return image.toDisplayTensor{input=Monitor._ScaleImage(x), nrow=C, padding=2}
end

function Monitor._LNDImage(x)
  assert(x:nDimension() == 3)
  local L = x:size(1)
  local N = x:size(2)
  local D = x:size(3)
  x = x:permute(3, 1, 2):contiguous():view(N, 1, D, L)
  return image.toDisplayTensor{input=Monitor._ScaleImage(x), nrow=1, padding=2}
end

function Monitor._NDImage(x)
  assert(x:nDimension() == 2)
  local N = x:size(1)
  local D = x:size(2)
  x = x:view(N, 1, 1, D)
  return image.toDisplayTensor{input=Monitor._ScaleImage(x), nrow=1, padding=0}
end

function Monitor._NImage(x)
  x = x:view(x:nElement(), 1, 1, 1)
  return image.toDisplayTensor{input=Monitor._ScaleImage(x), nrow=1, padding=0}
end

function Monitor._Encode_Base64_JPEG(x)
  x = image.compressJPG(x)
  return base64.encode(ffi.string(torch.data(x), x:nElement()))
end

function Monitor._getModuleName(mdl)
  local mdl_n = torch.type(mdl)
  if mdl.name ~= nil and type(mdl.name) == 'function' then
    mdl_n = mdl:name() .. ' (' .. mdl_n .. ')'
  elseif mdl.name ~= nil then
    mdl_n = mdl.name .. ' (' .. mdl_n .. ')'
  end
  return mdl_n
end

function Monitor._getModuleHTML(mdl)
  if mdl.html ~= nil and type(mdl.html) == 'function' then
    return mdl:html()
  elseif mdl.html ~= nil then
    return mdl.html
  elseif Monitor._htmlModule[torch.type(mdl)] ~= nil then
    return Monitor._htmlModule[torch.type(mdl)](mdl)
  else
    laia.log.debug(string.format('HTML for module %q is ignored!\n',
				 torch.type(mdl)))
    return nil
  end
end

function Monitor:updateSnapshot(input, mdl, hyp, ref)
  assert(torch.isTypeOf(mdl, 'nn.Module'))
  if self._output_file == nil or self._output_file == '' then return end
  if mdl.html ~= nil then
    self._snapshot_html = mdl:html()
  elseif Monitor._htmlModule[torch.type(mdl)] ~= nil then
    self._snapshot_html = Monitor._htmlModule[torch.type(mdl)](mdl)
  else
    laia.log.debug(string.format('HTML for module %q is ignored!\n',
   				 torch.type(mdl)))
  end

  -- local html = string.format([[
  -- <div id="snapshot">
  --   <div>
  --     <h4>Input</h4>
  --     <img src="data:image/jpg;base64,%s" />
  --   </div>
  -- ]], Monitor._Encode_Base64_JPEG(Monitor._NCHWImage(input:float())))

  -- if mdl['html'] ~= nil then
  --   html = html .. mdl:html()
  -- elseif Monitor._htmlModule[torch.type(mdl)] ~= nil then
  --   html = html .. Monitor._htmlModule[torch.type(mdl)](mdl)
  -- else
  --   laia.log.debug(string.format('HTML for module %q is ignored!\n',
  -- 				 torch.type(mdl)))
  -- end

  -- html = html .. [[
  -- </div>  <!-- END of snapshot -->
  -- ]]
  -- self._snapshot_html = html
  self:writeHTML()
end

function Monitor._TensorHistogram(x, nbins)
  nbins = nbins or 100
  local bmin = torch.min(x)
  local bmax = torch.max(x)
  local hist = torch.totable(torch.histc(x, nbins, bmin, bmax))
  local step = (bmax - bmin) / nbins
  local bins = torch.totable(torch.range(bmin + step / 2, bmax - s / 2, step))
  if #bins < nbins then table.insert(bins, bmax - s / 2) end
  assert(#bins == #hist)
  return bins, hist
end

function Monitor:writeHTML()
  if self._output_file == nil or self._output_file == '' then
    laia.log.warn('Monitor will not generate any HTML...')
    return
  end

  local f_html = io.open(self._output_file, 'w')
  assert(f_html ~= nil, string.format('Error creating HTML file %q',
  				      self._output_file))
  f_html:write(Monitor._HTMLTemplate({plots = self._plots,
				      snapshot_html = self._snapshot_html}))
  f_html:close()
end

function Monitor:updatePlots(...)
  for pn, pv in pairs(...) do
    assert(self._plots[pn] ~= nil, string.format('Unknown plot %q', pn))
    for cn, cv in pairs(pv) do
      assert(self._plots[pn].curves[cn] ~= nil,
	     string.format('Unknown curve %q in plot %q', cn, pn))
      table.insert(self._plots[pn].curves[cn].data, tonumber(cv))
    end
  end
  self:writeHTML()
end

function Monitor._HTML_Weight_Bias_Module(
    output, gradInput, weight, gradWeight, bias, gradBias)
  -- Generate HTML
  return string.format([[
    <div>
      <h4>Output:</h4>
      <div><img src="data:image/jpg;base64,%s" /></div>
    </div>
    <div>
      <h4>gradInput:</h4>
      <img src="data:image/jpg;base64,%s" />
    </div>
    <div>
      <h4>Bias / gradBias:</h4>
      <img src="data:image/jpg;base64,%s" />
      <img src="data:image/jpg;base64,%s" />
    </div>
    <div>
      <h4>Weight / gradWeight:</h4>
      <img src="data:image/jpg;base64,%s" />
      <img src="data:image/jpg;base64,%s" />
    </div>
  ]],
    Monitor._Encode_Base64_JPEG(output),
    Monitor._Encode_Base64_JPEG(gradInput),
    Monitor._Encode_Base64_JPEG(bias),
    Monitor._Encode_Base64_JPEG(gradBias),
    Monitor._Encode_Base64_JPEG(weight),
    Monitor._Encode_Base64_JPEG(gradWeight))
end

function Monitor._HTML_Parameters_Module(
    output, gradInput, param, gradParam)
  -- Generate HTML
  return string.format([[
    <div class="accordion">
      <h4>Output:</h4>
      <div><img src="data:image/jpg;base64,%s" /></div>
    </div>
    <div class="accordion">
      <h4>gradInput</h4>
      <div><img src="data:image/jpg;base64,%s" /></div>
    </div>
    <div class="accordion">
      <h4>param / gradParam:</h4>
      <img src="data:image/jpg;base64,%s" />
      <img src="data:image/jpg;base64,%s" />
    </div>
  ]],
    Monitor._Encode_Base64_JPEG(output),
    Monitor._Encode_Base64_JPEG(gradInput),
    Monitor._Encode_Base64_JPEG(param),
    Monitor._Encode_Base64_JPEG(gradParam))
end



Monitor._HTMLTemplate_Container = etlua.compile([[
<div data-role="collapsible">
  <h2><%= name %></h2>
  <% for _, html in pairs(children) do -%>
  <% if html ~= nil then -%>
  <%- html -%>
  <% end -%>
  <% end -%>
</div>
]])
assert(Monitor._HTMLTemplate_Container ~= nil)

Monitor._HTMLTemplate_Histogram = etlua.compile([[
<div id="<%= id %>"></div>
<script type="text/javascript">
  var data_<%= id %> = [
    x: [ <% for _,b in ipairs(bins) do %><%= b %>, <% end %> ],
    y: [ <% for _,v in ipairs(values) do %><%= v %>, <% end %> ],
    type: 'scatter',
    fill: 'tozeroy',
    showlegend: false
  ];
  var layout_<%= id %> = {
    width: <%= width or 350 %>,
    height: <%= height or 350 %>
  };
  Plotly.newPlot('<%= id %>', data_<%= id %>, layout_<%= id %>);
</script>
]])
assert(Monitor._HTMLTemplate_Histogram ~= nil)


Monitor._HTMLTemplate_Basic = etlua.compile([[
<%
  local output_b, output_v = laia.Monitor._TensorHistogram(mdl.output)
  local gradInput_b, gradInput_v = laia.Monitor._TensorHistogram(mdl.gradInput)
  local output_id = string.format('histogram_output_%d', torch.pointer(mdl))
  local gradInput_id = string.format('histogram_gradInput_%d', torch.pointer(mdl))
%>
<div data-role="collapsible">
  <h2><%= laia.Monitor._getModuleName(mdl) %></h2>
  <div>
    <div>
      <h3>Output</h3>
      <%- laia.Monitor._HTMLTemplate_Histogram({id = output_id, bins = output_b, values = output_v}) -%>
    </div>
    <div>
      <h3>gradInput</h3>
      <%- laia.Monitor._HTMLTemplate_Histogram({id = gradInput_id, bins = gradInput_b, values = gradInput_v}) -%>
    </div>
  </div>
</div>
]])
assert(Monitor._HTMLTemplate_Basic ~= nil)
-- function Monitor._HTML_Basic_Module(output, gradInput)
--   return string.format([[
--     <div>
--       <h4>Output</h4>
--       <div><img src="data:image/jpg;base64,%s" /></div>
--     </div>
--     <div>
--       <h4>gradInput:</h4>
--       <div><img src="data:image/jpg;base64,%s" /></div>
--     </div>
--   ]],
--     Monitor._Encode_Base64_JPEG(output),
--     Monitor._Encode_Base64_JPEG(gradInput))
-- end




--
-- HTML functions for the 'nn' package modules
--

-- nn.Container
Monitor._htmlModule['nn.Container'] = function(mdl)
  local ch_html = {}
  for i=1,mdl:size() do
    local h = Monitor._getModuleHTML(mdl:get(i))
    if h ~= nil then table.insert(ch_html, h) end
  end
  return Monitor._HTMLTemplate_Container({name = Monitor._getModuleName(mdl),
					  children = ch_html})
end

-- nn.Sequential
Monitor._htmlModule['nn.Sequential'] =
  Monitor._htmlModule['nn.Container']

-- nn.Tanh
Monitor._htmlModule['nn.Tanh'] = function(mdl)
  return Monitor._HTMLTemplate_Basic({mdl = mdl})
end

-- nn.ReLU
Monitor._htmlModule['nn.ReLU'] = Monitor._htmlModule['nn.Tanh']

-- nn.LeakyReLU
Monitor._htmlModule['nn.LeakyReLU'] = Monitor._htmlModule['nn.Tanh']

-- nn.PReLU
Monitor._htmlModule['nn.PReLU'] = Monitor._htmlModule['nn.Tanh']

-- nn.RReLU
Monitor._htmlModule['nn.RReLU'] = Monitor._htmlModule['nn.Tanh']

-- nn.SoftPlus
Monitor._htmlModule['nn.SoftPlus'] = Monitor._htmlModule['nn.Tanh']

-- nn.SpatialConvolution
Monitor._htmlModule['nn.SpatialConvolution'] = nil
-- function(mdl)
--   local output = Monitor._NCHWImage(mdl.output:float())
--   local gradInput = Monitor._NCHWImage(mdl.gradInput:float())
--   local param, gParam = mdl:parameters()
--   local weight  = Monitor._NCHWImage(param[1]:float())
--   local gWeight = Monitor._NCHWImage(gParam[1]:float())
--   local bias    = Monitor._NImage(param[2]:float())
--   local gBias   = Monitor._NImage(gParam[2]:float())
--   return Monitor._HTML_Weight_Bias_Module(output, gradInput,
-- 					  weight, gWeight, bias, gBias)
-- end

-- nn.Linear
Monitor._htmlModule['nn.Linear'] = nil
-- function(mdl)
--   local output = mdl.output:float()
--   local gradInput = mdl.gradInput:float()
--   if output:nDimension() == 2 then
--     output = Monitor._NDImage(output)
--     gradInput = Monitor._NDImage(gradInput)
--   else
--     output = Monitor._NImage(output)
--     gradInput = Monitor._NImage(gradInput)
--   end
--   local param, gParam = mdl:parameters()
--   local weight  = Monitor._NDImage(param[1]:float())
--   local gWeight = Monitor._NDImage(gParam[1]:float())
--   local bias    = Monitor._NImage(param[2]:float())
--   local gBias   = Monitor._NImage(gParam[2]:float())
--   return Monitor._HTML_Weight_Bias_Module(output, gradInput,
-- 					  weight, gWeight, bias, gBias)
-- end

-- nn.Dropout
Monitor._htmlModule['nn.Dropout'] = nil
-- function(mdl)
--   local output = mdl.output:float()
--   local gradInput = mdl.gradInput:float()
--   if output:nDimension() == 4 then
--     output = Monitor._NCHWImage(output)
--     gradInput = Monitor._NCHWImage(gradInput)
--   elseif output:nDimension() == 3 then
--     output = Monitor._LNDImage(output)
--     gradInput = Monitor._LNDImage(gradInput)
--   elseif output:nDimension() == 2 then
--     output = Monitor._NDImage(output)
--     gradInput = Monitor._NDImage(gradInput)
--   else
--     output = Monitor._NImage(output)
--     gradInput = Monitor._NImage(gradInput)
--   end
--   return Monitor._HTML_Basic_Module(output, gradInput)
-- end

-- nn.SpatialDropout
Monitor._htmlModule['nn.SpatialDropout'] =
  Monitor._htmlModule['nn.Dropout']

-- nn.SpatialMaxPooling
Monitor._htmlModule['nn.SpatialMaxPooling'] = nil
-- function(mdl)
--   local output = mdl.output:float()
--   local gradInput = mdl.gradInput:float()
--   if output:nDimension() == 4 then
--     output = Monitor._NCHWImage(output)
--     gradInput = Monitor._NCHWImage(gradInput)
--   else
--     output = Monitor._CHWImage(output)
--     gradInput = Monitor._CHWImage(gradInput)
--   end
--   return Monitor._HTML_Basic_Module(output, gradInput)
-- end

-- nn.SpatialAveragePooling
Monitor._htmlModule['nn.SpatialAveragePooling'] =
  Monitor._htmlModule['nn.SpatialMaxPooling']

-- nn.SpatialDilatedMaxPooling
Monitor._htmlModule['nn.SpatialDilatedMaxPooling'] =
  Monitor._htmlModule['nn.SpatialMaxPooling']

-- nn.SpatialFractionalMaxPooling
Monitor._htmlModule['nn.SpatialFractionalMaxPooling'] =
  Monitor._htmlModule['nn.SpatialMaxPooling']

-- nn.SpatialAdaptiveMaxPooling
Monitor._htmlModule['nn.SpatialAdaptiveMaxPooling'] =
  Monitor._htmlModule['nn.SpatialMaxPooling']


--
-- HTML functions for the 'cudnn' package modules
--

-- cudnn.Tanh
Monitor._htmlModule['cudnn.Tanh'] = Monitor._htmlModule['nn.Tanh']

-- cudnn.ReLU
Monitor._htmlModule['cudnn.ReLU'] = Monitor._htmlModule['nn.Tanh']

-- cudnn.SpatialConvolution
Monitor._htmlModule['cudnn.SpatialConvolution'] =
  Monitor._htmlModule['nn.SpatialConvolution']

-- cudnn.SpatialMaxPooling
Monitor._htmlModule['cudnn.SpatialMaxPooling'] =
  Monitor._htmlModule['nn.SpatialMaxPooling']

-- cudnn.BLSTM
Monitor._htmlModule['cudnn.BLSTM'] = nil
-- function(mdl)
--   local output = mdl.output:float()
--   local gradInput = mdl.gradInput:float()
--   local tWeight, tGradWeight = mdl:parameters()
--   assert(output:nDimension() == 3)
--   assert(#tWeight == 1 and #tGradWeight == 1)
--   local N = tWeight[1]:nElement()
--   local nrow = math.ceil(math.sqrt(N))
--   local weight  = tWeight[1]:float():view(N, 1, 1, 1)
--   local gWeight = tGradWeight[1]:float():view(N, 1, 1, 1)
--   output    = Monitor._LNDImage(output)
--   gradInput = Monitor._LNDImage(gradInput)
--   weight    = image.toDisplayTensor{input=weight, nrow=nrow, padding=0}
--   gWeight   = image.toDisplayTensor{input=gWeight, nrow=nrow, padding=0}
--   return Monitor._HTML_Parameters_Module(output, gradInput, weight, gWeight)
-- end

-- cudnn.BGRU
Monitor._htmlModule['cudnn.BGRU'] = Monitor._htmlModule['cudnn.BLSTM']
