require 'torch'
require 'nn'
require 'optim'
require 'data'

local opt = require 'opts'


-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

torch.setdefaulttensortype('torch.FloatTensor')

paths.dofile('data.lua')

trainlen = torch.sum(mn)
print(string.format('total data: %d', trainlen))

----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution


local netG = nn.Sequential()
-- input is (nc) x 128 x 128
netG:add(SpatialConvolution(nc, ngf, 4, 4, 2, 2, 1, 1))
netG:add(nn.LeakyReLU(0.2, true))
-- state size: (ngf) x 64 x 64
netG:add(SpatialConvolution(ngf, ngf*2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf*2) x 32 x 32
netG:add(SpatialConvolution(ngf*2, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf*4) x 16 x 16
if opt.loadSize==128 then
    netG:add(SpatialConvolution(ngf * 4, ngf * 4, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ngf*4) x 8 x 8
end
netG:add(SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ngf*8) x 4 x 4

-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
if opt.loadSize==128 then
    netG:add(SpatialFullConvolution(ngf * 4, ngf * 4, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
    -- state size: (ngf*4) x 16 x 16
end
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 32 x 32
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 64 x 64
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 128 x 128
netG:apply(weights_init)



local netD = nn.Sequential()
-- input is (nc) x 128 x 128
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 64 x 64
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 32 x 32
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 16 x 16
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 8 x 8
if opt.loadSize==128 then
    netD:add(SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 4 x 4
end
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(-1, 1))
-- state size: 1
netD:apply(weights_init)



local netA = nn.Sequential()
-- input is (nc*2) x 128 x 128
netA:add(SpatialConvolution(nc*2, ndf, 4, 4, 2, 2, 1, 1))
netA:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 64 x 64
netA:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netA:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 32 x 32
netA:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netA:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 16 x 16
netA:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netA:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 8 x 8
if opt.loadSize==128 then
    netA:add(SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1))
    netA:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 4 x 4
end
netA:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netA:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netA:add(nn.View(-1,1))
-- state size: 1
netA:apply(weights_init)


local criterion = nn.BCECriterion()

print('netG:',netG)
print('netA:',netA)
print('netD:',netD)

if opt.load_cp > 0 then
    epoch = opt.load_cp
    require 'cunn'
    require 'cudnn'
    netG = torch.load('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7')
    netD = torch.load('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7')
    netA = torch.load('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_A.t7')
end




local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

local input_img = torch.Tensor(opt.batchSize, 3, opt.loadSize, opt.loadSize)
local ass_label = torch.Tensor(opt.batchSize, 3, opt.loadSize, opt.loadSize)
local noass_label = torch.Tensor(opt.batchSize, 3, opt.loadSize, opt.loadSize)
local label = torch.Tensor(opt.batchSize, 1)

if opt.gpu >= 0 then
    require 'cunn'
    cutorch.setDevice(opt.gpu+1)
    input_img = input_img:cuda()
    ass_label = ass_label:cuda()
    noass_label = noass_label:cuda()
    label = label:cuda()

    if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(netG, cudnn)
      cudnn.convert(netD, cudnn)
      cudnn.convert(netA, cudnn)
    end
    netD:cuda();           
    netG:cuda();           
    netA:cuda();   
    criterion:cuda();
end


local parametersD, gradParametersD = netD:getParameters()
local parametersA, gradParametersA = netA:getParameters()
local parametersG, gradParametersG = netG:getParameters()


local function load_data()
    data_tm:reset(); data_tm:resume()
    local batch = getbatch(opt.loadSize)
    input_img:copy(batch[{{},3,{},{},{}}]:squeeze())
    ass_label:copy(batch[{{},1,{},{},{}}]:squeeze())
    noass_label:copy(batch[{{},2,{},{},{}}]:squeeze())
    data_tm:stop()
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()
   -- train with real   
   label:fill(real_label)
   local output = netD:forward(ass_label)
  
   local errD_real1 = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(ass_label, df_do)
   
   -- train with real (not associated)
   label:fill(real_label)
   local output = netD:forward(noass_label)
    
   local errD_real2 = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(noass_label, df_do)
    
   -- train with fake
   local fake = netG:forward(input_img)
   label:fill(fake_label)
   local output = netD:forward(fake)
    
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(fake, df_do)

   errD = (errD_real1 + errD_real2 + errD_fake)/3
   return errD, gradParametersD:mul(1/3)
end


-- create closure to evaluate f(X) and df/dX of domain discriminator
local fAx = function(x)
   gradParametersA:zero()
    
   local assd = torch.cat(input_img, ass_label, 2)
   local noassd = torch.cat(input_img, noass_label, 2)
   local fake = netG:forward(input_img)
   local faked = torch.cat(input_img, fake, 2)

   -- train with associated   
   label:fill(real_label)
   local output = netA:forward(assd)

   local errA_real1 = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netA:backward(assd, df_do)

   -- train with not associated
   label:fill(fake_label)
   local output = netA:forward(noassd)
   
   local errA_real2 = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netA:backward(noassd, df_do)

   -- train with fake
   label:fill(fake_label)
   local output = netA:forward(faked)
   
   local errA_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netA:backward(faked, df_do)

   errA = (errA_real1 + errA_real2 + errA_fake)/3
   return errA, gradParametersA:mul(1/3)
end


-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()
   --[[ the three lines below were already executed in fDx, so save computation
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   
   local fake = netG:forward(input_img)
   local output = netD:forward(fake)
   
   label:fill(real_label) -- fake labels are real for generator cost

   errGD = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput(fake, df_do)
   netG:backward(input_img, df_dg)
   
   local faked = torch.cat(input_img, fake, 2)
   local output = netA:forward(faked)
   label:fill(real_label) -- fake labels are real for generator cost
   errGA = criterion:forward(output, label)    
   local df_do = criterion:backward(output, label)
   local df_dg2 = netA:updateGradInput(faked, df_do)
   -- print(df_dg2:size())
   local df_dg = df_dg2[{{},{4,6}}]
   -- print(df_dg:size()) 
   netG:backward(input_img, df_dg)
   errG = (errGA + errGD)/2
   return errG, gradParametersG:mul(1/2)
end

-- utility functions
local function grid_png(nsample, imsize, img_arr, iter)
    os.execute(string.format('mkdir -p save/%d', opt.loadSize))
    png_grid = torch.Tensor(3, nsample*imsize*3, nsample*imsize*3):zero()
    cnt = 1
    for h = 1, imsize*nsample*3, imsize*3 do
        for w = 1, imsize*nsample*3, imsize do
            png_grid[{{},{h,h+imsize*3-1},{w, w+imsize-1}}]:copy(img_arr[{{cnt},{},{},{}}]:add(1):div(2):squeeze())
            cnt = cnt+1
        end
    end
    image.save(string.format('save/%d/%d.jpg', opt.loadSize, iter/opt.save_png_every), png_grid)
end

if opt.display then 
    disp = require 'display' 
    disp.configure({hostname=opt.display_ip, port=opt.display_port})
end


if opt.optimizer == 'adam' 
    then optimizer = optim.adam
    else optimizer = optim.sgd
end

result = {}
local disp_config = {
  title = "error over time",
  labels = {"samples", "errD", "errG", "errA"},
  ylabel = "error",
  win=opt.display_id*2,
}

-- train
os.execute('mkdir -p log')
logger = optim.Logger(string.format('log/%s_%d_result.log', opt.name, opt.loadSize))
logger:setNames{'Epoch', 'Time', 'ErrG', 'ErrD', 'ErrA'}
g_cnt = 0
for epoch = 1, opt.nTotalEpoch do
    epoch_tm:reset()
    for i = 1, trainlen do
        g_cnt = g_cnt+1             -- increase g_cnt by 1.
        tm:reset()
        load_data()
      
        -- Update D network
        optimizer(fDx, parametersD, optimStateD)
        -- Update A network
        optimizer(fAx, parametersA, optimStateA)
        -- Update G network
        optimizer(fGx, parametersG, optimStateG)
        
        -- display
        if g_cnt % opt.display_every == 0 then
            local fake = netG:forward(input_img)
            local real = ass_label
            disp.image(torch.cat(fake,real,3):cat(input_img,3), {win=opt.display_id, title=opt.name})
            
        end
        -- logging
        if g_cnt % opt.logging_every == 0 then
            print(string.format('Epoch: [%d][%8d/%8d] | Time: %.3f | DataTime:%.3f | Err_G: %.4f | Err_D: %.4f | Err_A: %.4f', epoch, i, trainlen, tm:time().real, data_tm:time().real, errG and errG or -1, errD and errD or -1, errA and errA or -1))
            table.insert(result, {g_cnt, errD, errG, errA})
            disp.plot(result, disp_config)
            -- record using torch logger.
            logger:add({epoch, tm:time().real, errG, errD, errA})
        end
        -- save png
        if g_cnt %  opt.save_png_every == 0 then
            local fake = netG:forward(input_img)
            local real = ass_label
            img_arr = torch.cat(fake,real,3):cat(input_img,3)
            if opt.loadSize==128 then
                grid_png(3, opt.loadSize, img_arr, i)
            elseif opt.loadSize==64 then
                grid_png(6, opt.loadSize, img_arr, i)
            end
        end
        
    end
    os.execute(string.format('mkdir -p checkpoints/%d', opt.loadSize))
    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersA, gradParametersA = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil
    torch.save('checkpoints/'.. opt.loadSize .. '/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
    torch.save('checkpoints/' .. opt.loadSize .. '/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
    torch.save('checkpoints/' ..opt.loadSize .. '/' .. opt.name .. '_' .. epoch .. '_net_A.t7', netA:clearState())
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersA, gradParametersA = netA:getParameters()
    parametersG, gradParametersG = netG:getParameters()
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end



