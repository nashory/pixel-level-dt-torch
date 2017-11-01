require 'image'

function loadImage(path, loadSize)
   local input = image.load(path, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   local iW = input:size(3)
   local iH = input:size(2)
   if iW > iH then
      input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
   else
      input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
   end
   return input:mul(2):csub(1)
end


cloth_table = torch.load('cloth_table.t7')
models_table = torch.load('models_table.t7')

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end


cn = tablelength(cloth_table)
mn = torch.Tensor(cn)

for k,v in pairs(models_table) do
    mn[k] = tablelength(v)
end

function add_padding(im, imsize)
    local iW = im:size(3)
    local iH = im:size(2)
    local iC = im:size(1)
    local pad = nil
    local out = nil
    if iW > iH then
        out = torch.Tensor(iC, iW, iW):fill(1)
        pad = math.floor((iW-iH)/2.0)
        out[{{},{pad+1, iH+pad},{1, iW}}]:copy(im)
    else
        out = torch.Tensor(iC, iH, iH):fill(1)
        pad = math.floor((iH-iW)/2.0)
        out[{{},{1,iH},{pad+1, iW+pad}}]:copy(im)
    end
    return image.scale(out, imsize, imsize)
end


function getbatch(imsize)
    batch = torch.Tensor(128,3,3,imsize,imsize):zero()
    local loadSize = {imsize,imsize}
    for i = 1,128 do
        seed = torch.random(1, 100000) -- fix seed
        gen = torch.Generator()
        torch.manualSeed(gen, i*seed)
        r1 = torch.random(gen,1,cn)
        r2 = torch.random(gen,1,cn)
        r3 = torch.random(gen,1,mn[r1])

        path1 = cloth_table[r1]
        path2 = cloth_table[r2]
        path3 = models_table[r1][r3]

        img1 = loadImage(path1, loadSize)
        img2 = loadImage(path2, loadSize)
        img3 = loadImage(path3, loadSize)
        
        -- preprocessing
        img1 = add_padding(img1, imsize)
        img2 = add_padding(img2, imsize)
        img3 = add_padding(img3, imsize)

        batch[i][1] = img1
        batch[i][2] = img2
        batch[i][3] = img3
    end
    return batch
end

